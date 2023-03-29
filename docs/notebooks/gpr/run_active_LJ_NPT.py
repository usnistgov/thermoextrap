import argparse
import glob
import sys

import numpy as np
import sims_cassandra
import sympy as sp
import xarray as xr
from pymbar import timeseries
from run_active_LJ_GEMC import transform_lognorm
from scipy import linalg, special

import thermoextrap
from thermoextrap.gpr_active import active_utils


def parse_bool(astr):
    if astr in ["True", "true", "Yes", "yes", "Y", "y"]:
        return True
    elif astr in ["False", "false", "No", "no", "N", "n"]:
        return False
    else:
        raise ValueError("Provided string %s is not convertible to boolean." % astr)


class StatePsat:
    """
    Object that provides GPR input information in similar way that ExtrapModel might through
    input_GP_from_state, but much simpler. When called, provides data organized such that
    it is ready for input to a GPR.
    """

    def __init__(self, lnPsat, dlnPsat_dbeta, beta):
        self.lnPsat = lnPsat
        self.dlnPsat_dbeta = dlnPsat_dbeta
        self.beta = beta

    def __call__(self):
        # If lnPsat values all same, no variance, so do not use, only use derivatives
        if np.all(self.lnPsat == self.lnPsat[0]):
            x = np.array([self.beta, 1])
            y = np.array([[np.average(self.dlnPsat_dbeta)]])
            cov = np.array([[np.var(self.dlnPsat_dbeta)]])
        else:
            x = np.vstack([self.beta * np.ones(2), np.arange(2)]).T
            raw_y = np.vstack([self.lnPsat, self.dlnPsat_dbeta]).T
            y = np.average(raw_y, axis=0)[:, None]
            cov = np.cov(raw_y.T)[None, ...]
        return x, y, cov


class DataWrapPsat(active_utils.DataWrapper):
    """
    Custom DataWrapper for saturation pressure information. For compatibility, uses all of
    the same variable names as DataWrapper object, but uses them in a different way.
    Only x_files will really be used, with sim_info_files and cv_bias_files ignored.
    x_files are expected to have the log saturation pressure, U_box1, H_box1, U_box2, H_box2
    as the columns, with only a single row. If the saturation pressure is computed from
    repreated simulations of GEMC or flat-histogram GCMC, and hence exhibits non-zero
    variance, it will be treated as a valid data point. If it is the fixed pressure used
    for NPT simulations and is the same in all x_files, then only the first derivative
    information will be computed and provided to the custom state object produced by
    build_state.
    """

    def get_data(self):
        """
        Loads all necessary data files for computing derivatives. Will not subsample
        uncorrelated indices, because collects single row of information from each
        file, which should represent an independent set of simulations.
        """
        dat = []
        for f in self.x_files:
            dat.append(np.loadtxt(f))
        dat = np.vstack(dat)
        return dat

    def build_state(self, max_order=None):
        """
        Builds a state that provides derivs.
        """
        # Max order will be ignored - only compute 0th and 1st derivatives
        del max_order

        # Compute derivatives, expecting columns in data of ln(Psat), U1, H1, U2, H2
        # Where 1 and 2 are indicating vapor and liquid phases, in that order
        data = self.get_data()
        lnPsat = data[:, 0]
        dH = data[:, 2] - data[:, 4]
        dU = data[:, 1] - data[:, 3]
        pdV = dH - dU
        dlnPsat_dbeta = -dH / (self.beta * pdV)

        # Create state and return
        return StatePsat(lnPsat, dlnPsat_dbeta, self.beta)


class DensityGPModel:
    """
    Class to hold model for density as a function of reciprocal temperature (beta).
    Idea is to use this to better guess density at which NPT simulations take place
    for adaptive Gibbs-Duhem integration, and then update based on results of simulations.
    Will use all derivative information at initial densities, but then only use densities
    themselves at updates, since derivatives from NPT simulations do not follow coexistence
    line.
    """

    def __init__(self, x_input, y_input, cov_input, transform_func=transform_lognorm):
        self.x_input = x_input
        self.y_input = y_input
        self.cov_input = cov_input
        self.transform_func = transform_func
        self.gp = active_utils.create_base_GP_model((x_input, y_input, cov_input))
        active_utils.train_GPR(self.gp)

    def __call__(self, beta):
        gpr_pred = self.gp.predict_f(np.vstack([beta, np.zeros_like(beta)]).T)
        pred_mu, pred_std, pred_conf_int = self.transform_func(
            beta,
            gpr_pred[0].numpy(),
            gpr_pred[1].numpy(),
        )
        # Need to return dictionary with keyword argument for function providing info for
        # Here, replacing densities in sim_VLE_NPT
        return {"densities": np.squeeze(pred_mu)}

    def update(self, beta, sim_info_files, cv_bias_files, x_files):
        # Only interested in sim_info_files since that is only file with densities
        dens_info = []
        for f in sim_info_files:
            dens_info.append(np.loadtxt(f)[:, [7, 16]])
        dens_info = np.vstack(dens_info)

        # Subsample based on autocorrelation time and compute means and variances
        mu = []
        var = []
        for k in range(dens_info.shape[1]):
            this_g = timeseries.statisticalInefficiency(dens_info[:, k])
            timeseries.subsampleCorrelatedData(np.arange(dens_info.shape[0]), this_g)
            # Take logarithm to ensure density cannot go negative
            # Note that default transformation function handles this, modeling log(dens)
            # and transforming back to density for prediction
            mu.append(np.average(np.log(dens_info[:, k])))
            var.append(np.var(np.log(dens_info[:, k])))
        mu = np.array(mu)[None, :]
        var = np.array(var)[None, :]

        # Add on to original inputs and update GP model
        self.x_input = np.vstack([self.x_input, np.array([[beta, 0.0]])])
        self.y_input = np.vstack([self.y_input, mu])
        new_cov = []
        for k in range(self.cov_input.shape[0]):
            new_cov.append(linalg.block_diag(self.cov_input[k], var[0, k]))
        self.cov_input = np.array(new_cov)
        self.gp = active_utils.create_base_GP_model(
            (self.x_input, self.y_input, self.cov_input)
        )
        active_utils.train_GPR(
            self.gp, start_params=[p.numpy() for p in self.gp.trainable_parameters]
        )


def find_local_mins(x, pad=20):
    # Finds local minima using a sliding window of size pad on each side of a central index
    mins = []

    for i in range(1, len(x) - 1):
        # Set lower index to current minus padding, or 0, whichever is greater
        lower = i - pad
        if lower < 0:
            lower = 0
        # Set upper index to current plus padding, or size of x, whichever is smaller
        upper = i + pad + 1
        if upper > len(x):
            upper = len(x)
        # Find minimum over this window
        this_window = x[lower:upper]
        this_min = np.min(this_window)
        # First check if flat... in that case, don't add and print warning
        if np.all(this_window == this_min):
            print(
                "Warning: have flat region between indices %i to %i; excluding from local minima."
                % (lower, upper)
            )
            continue
        # Next check if only this index is minimum, in which case it is a local minimum
        elif x[i] == this_min:
            mins.append(i)

    return mins


def find_phase_boundary(lnPi, pad=20):
    # Phase boundary will be at a local minimum, if one exists
    while pad < len(lnPi) // 2:
        local_mins = find_local_mins(lnPi, pad=pad)
        if len(local_mins) > 1:
            print("Warning: >1 local minima found! local_mins=%s" % str(local_mins))
            print("Expanding padding and trying again")
            pad += 20
        else:
            break
    if len(local_mins) > 1 and pad > len(lnPi) // 2:
        raise AttributeError(
            ">1 local minima found, even with padding of full width of lnPi"
        )

    # Looking for a local minimum in between two local maxima, ideally
    # Still useful to look for maxima either way
    local_maxs = find_local_mins(-lnPi, pad=pad)
    if len(local_maxs) > 2:
        print(
            "Warning: >2 local maxima found! May have more than 2 phases. local_maxes=%s"
            % str(local_maxs)
        )
        print("Only using first and last local maxima")
        local_maxs = [local_maxs[0], local_maxs[-1]]

    if len(local_mins) == 0:
        # Can only have single local maximum if have no local minimum
        if len(local_maxs) > 1:
            print(
                "Warning: >1 local maxima with no local minima! local_maxes=%s"
                % str(local_maxs)
            )
            print(
                "Will just use halfway point in lnPi as approximate phase boundary and continue"
            )
            pb = len(lnPi) // 2

        # Have only global minimum at one end or the other (tipped landscape)
        curr_min = np.argmin(lnPi)
        if curr_min == 0:
            # Have case where global min is 0, so set phase boundary to 1
            pb = len(lnPi) // 2
        elif curr_min == (len(lnPi) - 1):
            # Alternatively, global min could be at end of lnPi
            # But if true, can have two cases depending on if have a local maximum
            # Either way, will set pb to 1
            # This choice presents discontinuities in loss objective for establishing VLE
            # But will type out both cases as a sanity check, since should be exclusive
            if len(local_maxs) == 0:
                # If no local min or max, just monotonically decreases
                pb = len(lnPi) // 2
            else:
                if curr_min > local_maxs[0]:
                    # If global min is beyond a single local maximum, still set phase boundary to 1
                    pb = len(lnPi) // 2
    else:
        # Check to make sure have at least one local maximum if have local minimum
        #         if len(local_maxs) == 0:
        #             raise AttributeError("Have local minimum, but no local maxima - need to expand range to find VLE!")
        # Can have similar issue as above if have single local maximum and single minimum
        # But don't bother checking for that
        pb = local_mins[0]

    return pb


def main(inp_files, output_dir, mcf_file, ff_file, pdb_file, no_stop):
    T_lo = 0.7
    T_hi = 1.2
    beta_lo = 1 / T_lo
    beta_hi = 1 / T_hi
    act_sim_lo = -6.250
    act_sim_hi = -3.000

    # Loop over edge betas and collect information for GP inputs
    # Will need x_info, y_info (derivatives), and cov_info (covariance on derivatives)
    x_info = []
    y_info = []
    cov_info = []
    N_cutoff = 481  # Need upper cutoff for particle number
    box_vol = 512.0  # And box volume

    for b, act in zip([beta_lo, beta_hi], [act_sim_lo, act_sim_hi]):
        # Loading data
        file_prefix = "beta_{:f}/lj.t{}.n12m6.v512.rc3.b.r".format(
            b,
            str("%1.2f" % (1 / b)).replace(".", ""),
        )
        lnPi_files = sorted(glob.glob("%s*.lnpi.dat" % file_prefix))
        this_dat = [np.loadtxt(f) for f in lnPi_files]
        this_N = np.vstack([d[:N_cutoff, 0] for d in this_dat])
        this_N = xr.DataArray(this_N, dims=["rec", "n"])
        this_lnPi = np.vstack([d[:N_cutoff, 1] for d in this_dat])
        this_lnPi = xr.DataArray(this_lnPi, dims=["rec", "n"])
        # Renormalize from simulation chemical potential to VLE
        this_sim_mu = act / b
        vle_files = sorted(glob.glob("%s*.sat.dat" % file_prefix))
        this_vle_mu = np.array([np.loadtxt(f, skiprows=1)[0] / b for f in vle_files])
        this_vle_mu = xr.DataArray(this_vle_mu, dims=["rec"])
        this_lnPi = this_lnPi + this_N * (this_vle_mu - this_sim_mu) * b
        # Load in energies
        energy_files = sorted(glob.glob("%s*.energy.dat" % file_prefix))
        u_moms = np.array([np.loadtxt(f)[:N_cutoff, :] for f in energy_files])
        u_moms[..., 0] = np.ones_like(u_moms[..., 0])
        u_moms = xr.DataArray(u_moms, dims=["rec", "n", "umom"])

        # Add in -mu*N at VLE to get full potential used for calculating derivatives
        # And combine powers of -mu*N and U appropriately so just have powers of (U - mu*N)
        power_facs = xr.DataArray(np.arange(u_moms.shape[-1]), dims=["umom"])
        muN_moms = (-this_vle_mu * this_N) ** power_facs
        pot_moms = [xr.ones_like(this_N)]
        for n in range(1, u_moms.shape[-1]):
            this_sum = 0.0
            for k in range(n + 1):
                this_sum += special.binom(n, k) * muN_moms[..., n - k] * u_moms[..., k]
            pot_moms.append(this_sum)
        pot_moms = xr.concat(pot_moms, dim="umom")
        pot_moms = pot_moms.transpose("rec", "n", "umom")
        # For derivatives, will also need N multiplied by moments (U-mu*N)^k
        xpot_moms = this_N * pot_moms / box_vol
        # And weights on each N from lnPi
        this_w = np.exp(this_lnPi - this_lnPi.max("n"))

        # Loop over each lnPi and find phase boundary, then compute derivatives
        this_derivs = []
        for i, lnp in enumerate(this_lnPi):
            pb = find_phase_boundary(lnp)
            # Phase 1
            # Compute average moments over numbers of particles, weighted by lnPi
            avg_pot_moms1 = (
                pot_moms[
                    [
                        i,
                    ],
                    :pb,
                ]
                * this_w[
                    [
                        i,
                    ],
                    :pb,
                ]
            ).sum("n")
            avg_pot_moms1 /= this_w[
                [
                    i,
                ],
                :pb,
            ].sum("n")
            avg_xpot_moms1 = (
                xpot_moms[
                    [
                        i,
                    ],
                    :pb,
                ]
                * this_w[
                    [
                        i,
                    ],
                    :pb,
                ]
            ).sum("n")
            avg_xpot_moms1 /= this_w[
                [
                    i,
                ],
                :pb,
            ].sum("n")
            data1 = thermoextrap.DataCentralMoments.from_ave_raw(
                u=avg_pot_moms1, xu=avg_xpot_moms1, central=True
            )
            state1 = thermoextrap.beta.factory_extrapmodel(
                beta=b, data=data1, post_func=lambda x: sp.log(x)
            )
            derivs1 = state1.derivs(norm=False)
            # Phase 2
            avg_pot_moms2 = (
                pot_moms[
                    [
                        i,
                    ],
                    pb:,
                ]
                * this_w[
                    [
                        i,
                    ],
                    pb:,
                ]
            ).sum("n")
            avg_pot_moms2 /= this_w[
                [
                    i,
                ],
                pb:,
            ].sum("n")
            avg_xpot_moms2 = (
                xpot_moms[
                    [
                        i,
                    ],
                    pb:,
                ]
                * this_w[
                    [
                        i,
                    ],
                    pb:,
                ]
            ).sum("n")
            avg_xpot_moms2 /= this_w[
                [
                    i,
                ],
                pb:,
            ].sum("n")
            data2 = thermoextrap.DataCentralMoments.from_ave_raw(
                u=avg_pot_moms2, xu=avg_xpot_moms2, central=True
            )
            state2 = thermoextrap.beta.factory_extrapmodel(
                beta=b, data=data2, post_func=lambda x: sp.log(x)
            )
            derivs2 = state2.derivs(norm=False)
            this_derivs.append(xr.concat([derivs1, derivs2], dim="val"))

        # Concatenate derivatives along 'rec' dimension, then compute statistics
        this_derivs = xr.concat(this_derivs, dim="rec")
        this_derivs = this_derivs.transpose("order", "rec", "val")
        # Average goes in y_info
        y_info.append(this_derivs.mean(dim="rec").values)
        # Compute covariance matrix over each phase's derivatives
        this_cov = [np.cov(this_derivs[:, :, k]) for k in range(this_derivs.shape[-1])]
        cov_info.append(np.array(this_cov))
        # Input locations for GP are beta and derivative orders in second column
        x_info.append(
            np.vstack(
                [b * np.ones(this_derivs.shape[0]), np.arange(this_derivs.shape[0])]
            ).T
        )

    # Finish assembling input for GP over densities
    x_info = np.vstack(x_info)
    y_info = np.vstack(y_info)
    noise_cov_mat = []
    for k in range(y_info.shape[1]):
        noise_cov_mat.append(linalg.block_diag(*[cov[k, ...] for cov in cov_info]))
    noise_cov_mat = np.array(noise_cov_mat)

    # Create and train GP model for density
    dens_model = DensityGPModel(x_info, y_info, noise_cov_mat)

    # Loop over betas again and create DataWrapPsat objects
    dat_in = []
    for b in [beta_lo, beta_hi]:
        info_files = None
        bias_files = None
        x_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/vle_info*.txt"))
        dat_in.append(
            DataWrapPsat(
                info_files,
                bias_files,
                b,
                x_files=x_files,
            )
        )

    sim_wrap = active_utils.SimWrapper(
        sims_cassandra.sim_VLE_NPT,
        inp_files,
        None,
        "sim_info_out",
        "cv_bias_out",
        kw_inputs={
            "mcf_file": mcf_file,
            "ff_file": ff_file,
            "pdb_file": pdb_file,
        },
        data_class=DataWrapPsat,
        post_process_func=sims_cassandra.pull_psat_info,
        post_process_out_name="vle_info",
        pre_process_func=dens_model,
    )

    # Define shared update and stop arguments
    update_stop_kwargs = {"d_order_pred": 0}
    max_iter = 10

    update_func = active_utils.UpdateAdaptiveIntegrate(
        tol=0.001, **update_stop_kwargs, save_dir=output_dir, save_plot=True
    )

    # Define stopping criteria
    metrics = [
        active_utils.MaxRelGlobalVar(1e-02),
        active_utils.MaxAbsRelGlobalDeviation(1e-02),
    ]
    if no_stop:
        metrics.append(active_utils.MaxIter())
    stop_func = active_utils.StopCriteria(metrics, **update_stop_kwargs)

    active_utils.active_learning(
        dat_in,
        sim_wrap,
        update_func,
        stop_criteria=stop_func,
        base_dir=output_dir,
        max_iter=max_iter,
        alpha_name="beta",
        save_history=True,
        num_state_repeats=4,
        use_predictions=True,
    )

    # Good idea to save the density model
    dens_input_x = dens_model.gp.data[0].numpy()
    dens_input_y = dens_model.gp.data[1].numpy()
    dens_input_x = np.concatenate(
        [dens_input_x[:, :1] / dens_model.gp.x_scale_fac, dens_input_x[:, 1:]], axis=-1
    )
    dens_input_y = dens_input_y * (dens_model.gp.x_scale_fac ** dens_input_x[:, 1:])
    dens_input_y = dens_input_y * dens_model.gp.scale_fac
    dens_input_cov = dens_model.gp.likelihood.cov.copy()
    dens_input_cov = dens_input_cov * dens_model.gp.x_scale_fac ** (
        np.add(*np.meshgrid(dens_input_x[:, 1:], dens_input_x[:, 1:]))
    )
    dens_input_cov = dens_input_cov * (
        np.expand_dims(
            dens_model.gp.scale_fac,
            axis=tuple(range(dens_model.gp.scale_fac.ndim, dens_input_cov.ndim)),
        )
        ** 2
    )
    dens_pred = dens_model.gp.predict_f(
        np.vstack([np.linspace(beta_hi, beta_lo, 1000), np.zeros(1000)]).T
    )
    dens_pred_mu, dens_pred_std, dens_pred_conf_int = dens_model.transform_func(
        None, dens_pred[0].numpy(), dens_pred[1].numpy()
    )
    dens_pred_mu = np.squeeze(dens_pred_mu)
    dens_pred_std = np.squeeze(dens_pred_std)
    dens_params = np.array([p.numpy() for p in dens_model.gp.trainable_parameters])
    np.savez(
        "dens_model_info.npz",
        input_x=dens_input_x,
        input_y=dens_input_y,
        input_cov=dens_input_cov,
        pred_mu=dens_pred_mu,
        pred_std=dens_pred_std,
        pred_conf_int=dens_pred_conf_int,
        params=dens_params,
    )


if __name__ == "__main__":
    print("Received raw arguments: ")
    print(sys.argv)
    print("\n\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inp_files",
        nargs=3,
        type=str,
        help=".inp files in order for NVT equil, GEMC equil, and GEMC prod",
    )
    parser.add_argument(
        "--dir", default="./", type=str, help="directory to write to, default ./"
    )
    parser.add_argument("--mcf", default="LJ.mcf", type=str, help=".mcf file")
    parser.add_argument("--ff", default="LJ.ff", type=str, help=".ff file")
    parser.add_argument("--pdb", default="LJ.pdb", type=str, help=".pdb file")
    parser.add_argument(
        "--no_stop",
        default=False,
        type=parse_bool,
        help="whether to allow stopping criteria go until max iter",
    )

    args = parser.parse_args()

    main(args.inp_files, args.dir, args.mcf, args.ff, args.pdb, args.no_stop)

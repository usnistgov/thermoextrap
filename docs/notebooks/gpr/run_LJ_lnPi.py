import glob
import os

import gpflow
import joblib
import lnpy
import numpy as np
import xarray as xr

import thermoextrap
from thermoextrap.gpr_active import active_utils

# Define some constants that we'll use throughout

N_cutoff = 481  # N value beyond which ln_Pi becomes unreliable, so won't use

# Change path below to set which data set to use
# SRS_data was used for the main text, higher_order_LJ_lnPi_data for Fig. S4 in the SI
SRS_base_dir = os.path.expanduser("~/bin/thermo-extrap/docs/notebooks/gpr/SRS_data")
# SRS_base_dir = os.path.expanduser("~/bin/thermo-extrap/docs/notebooks/gpr/higher_order_LJ_lnPi_data")


# Necessary functions
def get_sim_activity(Tr):
    if Tr == 0.700:
        return -6.250
    elif Tr == 0.730 or Tr == 0.740:
        return -5.500
    elif Tr == 0.770:
        return -5.800
    elif Tr == 0.850:
        return -4.800
    elif Tr == 0.950:
        return -4.100
    elif Tr == 1.100:
        return -3.380
    elif Tr == 1.200:
        return -3.000


def load_lnPi_info(Tr, ref_mu=-4.0, run_num=None):
    file_prefix = "{}/lj.t{}.n12m6.v512.rc3.b.r".format(
        SRS_base_dir,
        str("%1.2f" % Tr).replace(".", ""),
    )

    U_moms = np.array(
        [
            np.loadtxt(f)[:N_cutoff]
            for f in sorted(glob.glob("%s*.energy.dat" % file_prefix))
        ]
    )
    # For first column of U information (which is currently N), set to ones, which is zeroth moment
    U_moms[:, :, 0] = np.ones(U_moms.shape[:-1])
    lnPis = np.array(
        [
            np.loadtxt(f)[:N_cutoff, 1]
            for f in sorted(glob.glob("%s*.lnpi.dat" % file_prefix))
        ]
    )
    N_vals = np.array(
        [
            np.loadtxt(f)[:N_cutoff, 0]
            for f in sorted(glob.glob("%s*.lnpi.dat" % file_prefix))
        ]
    )
    mu = Tr * get_sim_activity(
        Tr
    )  # Tr is kB*T/eps, activity is mu/kB*T, so getting mu/eps

    # Convert to x_arrays with proper labeling
    U_moms = xr.DataArray(U_moms, dims=["rec", "n", "umom"])
    # For lnPi, adjust to a reference mu value
    # And subtract off N=0 bin
    lnPis = lnPis + (
        (1.0 / Tr) * (ref_mu - mu) * N_vals
    )  # Multiply by 1/Tr since want beta*mu
    lnPis = lnPis - lnPis[:, :1]
    lnPis = xr.DataArray(lnPis, dims=["rec", "n"])
    # For mu need to add extra axis called comp
    ref_mu = xr.DataArray(ref_mu * np.ones((U_moms.shape[0], 1)), dims=["rec", "comp"])

    return {
        "energy": U_moms,
        "lnPi": lnPis,
        "mu": ref_mu,
        "mu_sim": mu,
        "beta": 1.0 / Tr,
    }


class StatelnPi:
    def __init__(self, x, y, cov):
        self.x = x
        self.y = y
        self.cov = cov

    def __call__(self):
        return self.x, self.y, self.cov


def state_from_info_dict(info, d_o=None):
    meta_lnpi = thermoextrap.lnpi.lnPiDataCallback(
        info["lnPi"],  # .isel(n=slice(1, None)),
        info["mu"],
        dims_n=["n"],
        dims_comp="comp",
    )
    data_lnpi = thermoextrap.DataCentralMoments.from_ave_raw(
        u=info["energy"],  # .isel(n=slice(1, None)),
        xu=None,
        x_is_u=True,
        central=True,
        meta=meta_lnpi,
    )
    state_lnpi = thermoextrap.lnpi.factory_extrapmodel_lnPi(
        beta=info["beta"], data=data_lnpi
    )

    # Within thermoextrap.lnpi, must have N=0 included, but that will cause issues with GP
    # (because variance at N=0 will be 0)
    # So providing custom state wrapper that, instead of providing state itself,
    # provides x, y, and derivative info directly
    # Essentially this is a simple, custom version of input_GP_from_state
    if d_o is None:
        d_o = info["energy"].sizes["umom"] - 1
    alphas = state_lnpi.alpha0 * np.ones((d_o + 1, 1))
    x_data = np.concatenate([alphas, np.arange(d_o + 1)[:, None]], axis=1)

    derivs = state_lnpi.derivs(norm=False, order=d_o).mean("rec")
    derivs = derivs.isel(n=slice(1, None)).values
    resamp_derivs = state_lnpi.derivs(norm=False, order=d_o)
    resamp_derivs = resamp_derivs.isel(n=slice(1, None)).values

    y_data = derivs
    cov_data = []
    for k in range(resamp_derivs.shape[-1]):
        cov_data.append(np.cov(resamp_derivs[..., k]))
    cov_data = np.array(cov_data)

    return StatelnPi(x_data, y_data, cov_data)


def tag_phases(list_of_phases):
    """Simple tag_phases callback

    This looks at the local maximum of each lnPiMasked object.

    If location of maximum < len(data)/2 -> phase = 0
    else -> phase = 1

    """
    if len(list_of_phases) > 2:
        raise ValueError("bad tag function")
    argmax0 = np.array([xx.local_argmax()[0] for xx in list_of_phases])

    if len(list_of_phases) == 2:
        return np.argsort(argmax0)
    else:
        return np.where(argmax0 <= list_of_phases[0].shape[0] / 2, 0, 1)


def get_VLE_info(lnPi, ref_activity, beta, vol=512.0, efac=10.0):
    # Create lnpy lnPiMasked object based on given lnPi and reference activity
    lnp = lnpy.lnPiMasked.from_data(
        lnz=ref_activity,
        lnz_data=ref_activity,
        data=lnPi,
        state_kws={"beta": beta, "volume": vol},
    )

    # Create function to partition phases and function to build them at many mu
    phase_creator = lnpy.PhaseCreator(
        nmax=2,
        nmax_peak=4,
        ref=lnp,
        merge_kws={"efac": efac},
        segment_kws={"peaks_kws": {"min_distance": 50}},
        tag_phases=tag_phases,
    )
    build_phases = phase_creator.build_phases_mu([None])

    # Turn off parallel here
    with lnpy.set_options(joblib_use=False, tqdm_use=False):
        # Set range of activities to search
        lnzs = np.linspace(-8, -2, 30)

        # And collection of lnPi's at different activities partitioned into phases
        c = lnpy.lnPiCollection.from_builder(
            lnzs=lnzs,
            build_phases=build_phases,
            unstack=False,
        )

        # Try to find the spinodal and binodal, but may not be able to
        # May get unlucky with particular sample with too many wiggles, etc.
        try:
            # Find the spinodal
            spino, spino_info = c.spinodal(
                phase_ids=2,
                build_phases=build_phases,
                inplace=False,
                as_dict=False,
                build_kws=dict(efac=efac * 0.5),
                efac=efac * 0.5,
            )

            # Find the binodal
            bino, bino_info = c.binodal(
                spinodals=spino,  # lnz_min=lnzs[0], lnz_max=lnzs[-1],
                phase_ids=[0, 1],
                build_phases=build_phases,
                inplace=False,
                as_dict=False,
                build_kws=dict(efac=efac * 0.5),
            )

            return bino

        except:
            return None


def get_sat_props(lnPi, ref_activity, Tr, lnPi_vars=None, N_boot=100):
    # If provided with variances, bootstrap uncertainties by TRAINING a GP model
    # and sampling from it. Train the model only on the N dimension, having already
    # used the learned temperature dependence from the other GP. Will use the means as the
    # observable values and lnPi_stds as the noise values in the likelihood covariance matrix
    # (covariance matrix will be diagonal, but will learn correlations with GP process)
    # Effectively learning function in N and how confident we are in this

    if lnPi_vars is not None:
        # Create data for GP model over N dimension
        # No derivatives here, so zeros for second column
        x_input = np.vstack(
            [np.arange(1, lnPi.shape[0]), np.zeros(lnPi.shape[0] - 1)]
        ).T
        y_input = np.reshape(lnPi, (-1, 1))[1:, :]
        cov_input = np.diag(np.squeeze(lnPi_vars)[1:])
        # Note that ignoring modeling of N=0 bin... will be zero no matter what, with std of 0
        # That can throw off matrix, so ignore for modeling and sampling purposes

        # Create model, but make sure to set power scale to zero and constrain it
        # Don't want to modify/learn anything about covariance matrix here
        this_gp = active_utils.create_base_GP_model(
            (x_input, y_input, cov_input),
            likelihood_kwargs={
                "p": 0.0,
                "transform_p": None,
                "constrain_p": True,
            },
        )
        active_utils.train_GPR(this_gp)

        # Want the distribution of functions given the provided inputs
        # But evaluating at the input locations, so x=x* and want full covariance
        this_pred = this_gp.predict_f(this_gp.data[0], full_cov=True)
        # And want uncertainty in model plus noise, so add noise covariance to output
        this_mean = this_pred[0][:, 0].numpy()
        this_cov = np.squeeze(this_pred[1]) + cov_input

        # print(np.max(this_cov), np.max(cov_input))

        # Draw random samples from multivariate normal based on GP output
        rng = np.random.default_rng()
        boot_lnPi = rng.multivariate_normal(mean=this_mean, cov=this_cov, size=N_boot)

        # Add N=0 bin back in
        boot_lnPi = np.concatenate([np.zeros((N_boot, 1)), boot_lnPi], axis=-1)

        # print(this_gp.parameters)
        # plt.plot(lnPi)
        # plt.plot(this_mean)
        # plt.plot(boot_lnPi[0, :])
        # plt.show()

        # Compute VLE info in parallel (much faster)
        boot_vle_info = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(get_VLE_info)(blp, ref_activity, 1 / Tr) for blp in boot_lnPi
        )

        boot_props = []
        for this_vle_info in boot_vle_info:
            # If did not detect phase coexistence, spinodal calculation will fail
            # Then get_VLE_info will return None
            if this_vle_info is None:
                continue
            lnz = this_vle_info.xge.lnz.isel(component=0, sample=0).values
            densities = this_vle_info.xge.dens.isel(component=0).values
            pressures = this_vle_info.xge.pressure().values
            this_boot_props = np.hstack([lnz, densities, pressures])
            # If spinodal was successful but binodal not, may end up with 1 phase still
            # Weird, but in that case, don't want to include, just move on
            # (expect single lnz, 2 densities, and 2 pressures
            if len(this_boot_props) != 5:
                continue
            boot_props.append(this_boot_props)
        boot_props = np.array(boot_props)
        print(boot_props.shape)
        props = np.median(boot_props, axis=0)
        props_conf_int = np.percentile(
            boot_props, [2.50, 97.5], axis=0, method="median_unbiased"
        )
        return props, props_conf_int

    else:
        this_vle_info = get_VLE_info(lnPi, ref_activity, 1 / Tr)
        lnz = this_vle_info.xge.lnz.isel(component=0, sample=0).values
        densities = this_vle_info.xge.dens.isel(component=0).values
        pressures = this_vle_info.xge.pressure().values
        return np.hstack([lnz, densities, pressures])


def main():
    # Load data for comparison
    raw_dat = np.loadtxt("%s/SRS_LJ_VLE_data.txt" % SRS_base_dir)
    1.0 / raw_dat[::-1, 0]
    raw_dat[::-1, 5]
    raw_dat[::-1, [1, 3]]
    raw_dat[::-1, -2]

    # Load in data needed to train GP
    ref_T = [0.7, 1.2]

    ref_mu = np.average([get_sim_activity(t) * t for t in ref_T])
    print("Reference chemical potential: %f" % ref_mu)

    # To show impact of using different derivative order information, loop over orders
    # Use parameters from previous order
    # (Training not stable from default params for higher orders...)
    # (due to p=10.0 being too high...)
    # (higher order moments can be large with high variance, and get bigger when apply p>0)
    # (leading to overflows)
    curr_params = [1.0, 1.0, 10.0]  # Start with defaults
    for d_order in range(1, 6):
        lnpi_info = [load_lnPi_info(t, ref_mu=ref_mu) for t in ref_T]
        state_list = [state_from_info_dict(i, d_o=d_order) for i in lnpi_info]

        # Create GP model
        gp_model = active_utils.create_GPR(state_list, start_params=curr_params)

        gpflow.utilities.print_summary(gp_model)

        # Now look at predictions with GP model
        test_T = np.array([0.70, 0.74, 0.85, 1.10, 1.20])
        test_beta = 1.0 / test_T
        gp_pred = gp_model.predict_f(np.vstack([test_beta, np.zeros_like(test_beta)]).T)
        gp_pred_mu = gp_pred[0].numpy()
        gp_pred_std = np.sqrt(gp_pred[1])
        # Add N=0 bin back in
        gp_pred_mu = np.concatenate(
            [np.zeros((gp_pred_mu.shape[0], 1)), gp_pred_mu], axis=-1
        )
        gp_pred_std = np.concatenate(
            [np.zeros((gp_pred_std.shape[0], 1)), gp_pred_std], axis=-1
        )
        gp_pred_conf_int = np.array(
            [gp_pred_mu - 2.0 * gp_pred_std, gp_pred_mu + 2.0 * gp_pred_std]
        )
        N_vals = np.arange(gp_pred_mu.shape[1])

        # Save GP model info and predictions
        gp_input_x = gp_model.data[0].numpy()
        gp_input_y = gp_model.data[1].numpy()
        gp_input_x = np.concatenate(
            [gp_input_x[:, :1] / gp_model.x_scale_fac, gp_input_x[:, 1:]], axis=-1
        )
        gp_input_y = gp_input_y * (gp_model.x_scale_fac ** gp_input_x[:, 1:])
        gp_input_y = gp_input_y * gp_model.scale_fac
        gp_input_cov = gp_model.likelihood.cov.copy()
        gp_input_cov = gp_input_cov * gp_model.x_scale_fac ** (
            np.add(*np.meshgrid(gp_input_x[:, 1:], gp_input_x[:, 1:]))
        )
        gp_input_cov = gp_input_cov * (
            np.expand_dims(
                gp_model.scale_fac,
                axis=tuple(range(gp_model.scale_fac.ndim, gp_input_cov.ndim)),
            )
            ** 2
        )
        gp_params = np.array([p.numpy() for p in gp_model.trainable_parameters])
        np.savez(
            "gp_model_info_order%i.npz" % d_order,
            N=N_vals,
            input_x=gp_input_x,
            input_y=gp_input_y,
            input_cov=gp_input_cov,
            pred_mu=gp_pred_mu,
            pred_std=gp_pred_std,
            pred_conf_int=gp_pred_conf_int,
            params=gp_params,
        )

        # Update current parameters to use for next order training
        curr_params = gp_params.tolist()

        # For expanded set of temperatures, make predictions of lnPi, then use to predict
        # VLE properties
        extra_T = np.linspace(0.7, 1.2, 1000)
        extra_betas = 1.0 / extra_T
        extra_pred = gp_model.predict_f(
            np.vstack([extra_betas, np.zeros_like(extra_betas)]).T
        )

        # Collect saturation properties
        gp_sat_props = []
        gp_sat_props_conf_ints = []
        for i, t in enumerate(extra_T):
            print(i, t)
            this_lnPi = extra_pred[0][i, :].numpy()
            this_props, this_prop_conf_int = get_sat_props(
                this_lnPi,
                ref_mu / t,
                t,
                lnPi_vars=extra_pred[1][i, :].numpy(),
                N_boot=1000,
            )
            gp_sat_props.append(this_props)
            gp_sat_props_conf_ints.append(this_prop_conf_int)

        # Should be lnz, density_gas, density_liquid, pressure_gas, pressure_liquid
        gp_sat_props = np.array(gp_sat_props)
        gp_sat_props_conf_ints = np.array(gp_sat_props_conf_ints)

        # Save saturation property predictions
        np.savez(
            "sat_props_GPR_order%i.npz" % d_order,
            props=gp_sat_props,
            conf_ints=gp_sat_props_conf_ints,
        )


if __name__ == "__main__":
    main()

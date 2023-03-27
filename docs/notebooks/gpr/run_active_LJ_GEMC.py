import argparse
import glob
import os
import sys

import numpy as np
import sims_cassandra
import sympy as sp
from scipy import interpolate

import thermoextrap
from thermoextrap.gpr_active import active_utils


def parse_bool(astr):
    if astr in ["True", "true", "Yes", "yes", "Y", "y"]:
        return True
    elif astr in ["False", "false", "No", "no", "N", "n"]:
        return False
    else:
        raise ValueError("Provided string %s is not convertible to boolean." % astr)


def make_ground_truth_dens():
    Tr_data = np.array(
        [
            0.7,
            0.72,
            0.74,
            0.76,
            0.78,
            0.8,
            0.82,
            0.84,
            0.86,
            0.88,
            0.9,
            0.92,
            0.94,
            0.96,
            0.98,
            1.0,
            1.02,
            1.04,
            1.06,
            1.08,
            1.1,
            1.12,
            1.14,
            1.16,
            1.18,
            1.2,
        ]
    )
    dens_lo_data = np.array(
        [
            0.0019956,
            0.0025624,
            0.0032425,
            0.0040495,
            0.0049974,
            0.0061007,
            0.0073745,
            0.0088349,
            0.010499,
            0.012384,
            0.01451,
            0.016897,
            0.019569,
            0.022549,
            0.025868,
            0.029556,
            0.03365,
            0.038192,
            0.043233,
            0.048831,
            0.055061,
            0.062015,
            0.069813,
            0.078615,
            0.088649,
            0.1003,
        ]
    )
    dens_hi_data = np.array(
        [
            0.84341,
            0.83492,
            0.82635,
            0.81764,
            0.80879,
            0.79981,
            0.7907,
            0.78147,
            0.7721,
            0.76256,
            0.75284,
            0.74291,
            0.73276,
            0.7224,
            0.71182,
            0.70094,
            0.68974,
            0.67817,
            0.66618,
            0.65373,
            0.64075,
            0.62715,
            0.6128,
            0.59753,
            0.58115,
            0.56329,
        ]
    )
    beta_data = 1.0 / Tr_data
    ground_truth_lo = interpolate.interp1d(beta_data, dens_lo_data, kind="cubic")
    ground_truth_hi = interpolate.interp1d(beta_data, dens_hi_data, kind="cubic")

    def ground_truth(beta):
        beta = np.array(beta)
        if len(beta.shape) < 2:
            beta = np.reshape(beta, (-1, 1))
        return np.hstack([ground_truth_lo(beta), ground_truth_hi(beta)])

    return ground_truth


class DataWrapDensities(active_utils.DataWrapper):
    """
    Identical to traditional DataWrapper, but different build_state() method so
    that can handle densities, which can't be negative, so model their natural
    logarithm and transform back.
    """

    def build_state(self, all_data=None, max_order=6):
        """Builds a thermoextrap data object for the data described by this wrapper class.
        If all_data is provided, should be list or tuple of (potential energies, X) to
        be used, where X should be appropriately weighted if the simulation is biased.
        Here, adds argument to factory_extrapmodel to model log(data).
        """
        if all_data is None:
            all_data = self.get_data()
        u_vals = all_data[0]
        x_vals = all_data[1]
        weights = all_data[2]
        state_data = thermoextrap.DataCentralMomentsVals.from_vals(
            uv=u_vals, xv=x_vals, w=weights, order=max_order
        )
        state = thermoextrap.beta.factory_extrapmodel(
            beta=self.beta, data=state_data, post_func=lambda x: sp.log(x)
        )
        return state


# Because deal with logs of densities, need to transform back
# (may influence active learning, so want to make decisions in space we care about)
# But need special transform function for that
# Since predict Gaussian for ln(x), then x has log-normal distribution
def transform_lognorm(x, y, y_var):
    # Calculate median, not mean (better sense of distribution)
    median = np.exp(y)
    # For uncertainty, use standard deviation of log-normal distribution
    uncert = np.sqrt(np.exp(y_var) - 1) * np.exp(y + 0.5 * y_var)
    # Confidence intervals corresponding to two sigma of original Gaussian around median
    conf = [np.exp(y - 2.0 * np.sqrt(y_var)), np.exp(y + 2.0 * np.sqrt(y_var))]
    return median, uncert, conf


def main(
    inp_files,
    output_dir,
    lowT,
    highT,
    mcf_file,
    ff_file,
    pdb_file,
    update_type,
    avoid_repeats,
    no_stop,
):
    # Working in all reduced units, so if reduced temperatures provided, good to go
    beta1 = 1.0 / lowT
    beta2 = 1.0 / highT

    dat_in = []
    for b in [beta1, beta2]:
        if os.path.isdir(f"{output_dir}/beta_{b:f}"):
            info_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/sim_info_out*.txt"))
            bias_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/cv_bias_out*.txt"))
            x_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/dens_out*.txt"))
            dat_in.append(
                DataWrapDensities(
                    info_files,
                    bias_files,
                    b,
                    x_files=x_files,
                    u_col=-1,
                    x_col=[1, 2],
                    n_frames=2000,
                )
            )
        else:
            dat_in.append(b)

    sim_wrap = active_utils.SimWrapper(
        sims_cassandra.sim_VLE_GEMC,
        inp_files,
        None,
        "sim_info_out",
        "cv_bias_out",
        kw_inputs={
            "mcf_file": mcf_file,
            "ff_file": ff_file,
            "pdb_file": pdb_file,
        },
        data_class=DataWrapDensities,
        data_kw_inputs={
            "u_col": -1,
            "x_col": [1, 2],
            "n_frames": 2000,
        },
        post_process_func=sims_cassandra.pull_density_info,
        post_process_out_name="dens_out",
    )

    # Define shared update and stop arguments
    update_stop_kwargs = {
        "avoid_repeats": avoid_repeats,
        "d_order_pred": 0,
        "transform_func": transform_lognorm,
    }
    max_iter = 7

    ground_truth_dens = make_ground_truth_dens()

    if update_type == "ALM":
        update_func = active_utils.UpdateALMbrute(
            **update_stop_kwargs,
            compare_func=ground_truth_dens,
            save_dir=output_dir,
            save_plot=True,
        )
    elif update_type == "Space":
        update_func = active_utils.UpdateSpaceFill(
            **update_stop_kwargs,
            compare_func=ground_truth_dens,
            save_dir=output_dir,
            save_plot=True,
        )
    elif update_type == "Random":
        update_func = active_utils.UpdateRandom(
            **update_stop_kwargs,
            compare_func=ground_truth_dens,
            save_dir=output_dir,
            save_plot=True,
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
        "lowT", type=float, help="low temperature to start at (in Kelvin)"
    )
    parser.add_argument(
        "highT", type=float, help="high temperature to start at (in Kelvin)"
    )
    parser.add_argument(
        "--dir", default="./", type=str, help="directory to write to, default ./"
    )
    parser.add_argument("--mcf", default="LJ.mcf", type=str, help=".mcf file")
    parser.add_argument("--ff", default="LJ.ff", type=str, help=".ff file")
    parser.add_argument("--pdb", default="LJ.pdb", type=str, help=".pdb file")
    parser.add_argument(
        "--update",
        default="ALM",
        type=str,
        choices=["ALM", "Space", "Random"],
        help="type of update to use; can be ALM, Space, or Random",
    )
    parser.add_argument(
        "--avoid_repeats",
        default=False,
        type=parse_bool,
        help="whether to randomize grid and avoid repeats or not",
    )
    parser.add_argument(
        "--no_stop",
        default=False,
        type=parse_bool,
        help="whether to allow stopping criteria go until max iter",
    )

    args = parser.parse_args()

    main(
        args.inp_files,
        args.dir,
        args.lowT,
        args.highT,
        args.mcf,
        args.ff,
        args.pdb,
        args.update,
        args.avoid_repeats,
        args.no_stop,
    )

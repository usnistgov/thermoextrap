import argparse
import glob
import os
import sys

import numpy as np
import sims_poly_feasst
from openmm import unit

from thermoextrap.gpr_active import active_utils


def parse_bool(astr):
    if astr in ["True", "true", "Yes", "yes", "Y", "y"]:
        return True
    elif astr in ["False", "false", "No", "no", "N", "n"]:
        return False
    else:
        raise ValueError("Provided string %s is not convertible to boolean." % astr)


def transform_Cv(x, y, y_var):
    scale_fac = (10.0**x) / np.log(10.0)
    out = scale_fac * y
    y_std = scale_fac * np.sqrt(y_var)
    conf = [out - 2.0 * y_std, out + 2.0 * y_std]
    return out, y_std, conf


def main(
    struc_file,
    sys_file,
    output_dir,
    lowT,
    highT,
    obs_type,
    update_type,
    avoid_repeats,
    no_stop,
):
    lowT = lowT * unit.kelvin
    highT = highT * unit.kelvin

    beta1 = 1.0 / (lowT * unit.MOLAR_GAS_CONSTANT_R)
    beta1 = beta1.value_in_unit(unit.kilojoules_per_mole ** (-1))
    beta2 = 1.0 / (highT * unit.MOLAR_GAS_CONSTANT_R)
    beta2 = beta2.value_in_unit(unit.kilojoules_per_mole ** (-1))

    dat_in = []
    for b in [beta1, beta2]:
        if os.path.isdir(f"{output_dir}/beta_{b:f}"):
            info_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/polymer_out*.txt"))
            bias_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/cv_bias_out*.txt"))
            if obs_type == "Rg":
                x_files = None
            else:
                x_files = sorted(glob.glob(f"{output_dir}/beta_{b:f}/U_info*.txt"))
            dat_in.append(
                active_utils.DataWrapper(
                    info_files,
                    bias_files,
                    b,
                    x_files=x_files,
                    u_col=3,
                    n_frames=2000,
                )
            )
        else:
            dat_in.append(b)

    if obs_type == "Rg":
        post_process_func = None
        post_process_out_name = None
    else:
        post_process_func = sims_poly_feasst.calc_raw_U
        post_process_out_name = "U_info"
    sim_wrap = active_utils.SimWrapper(
        sims_poly_feasst.poly_sim_NVT,
        struc_file,
        sys_file,
        "polymer_out",
        "cv_bias_out",
        kw_inputs={
            "steps_sim": 2e6,
            "steps_per": 1e3,
        },
        data_kw_inputs={
            "u_col": 3,
            "n_frames": 2000,
        },
        post_process_func=post_process_func,
        post_process_out_name=post_process_out_name,
    )

    # Define shared update and stop arguments
    update_stop_kwargs = {"log_scale": True, "avoid_repeats": avoid_repeats}
    if obs_type == "Cv":
        update_stop_kwargs["d_order_pred"] = 1
        update_stop_kwargs["transform_func"] = transform_Cv
        max_iter = (
            31  # 32 equal intervals (space fill); needs more b/c higher uncertainty
        )
    else:
        update_stop_kwargs["d_order_pred"] = 0
        max_iter = 15  # 16 equal intervals (space fill)

    if update_type == "ALM":
        update_func = active_utils.UpdateALMbrute(**update_stop_kwargs, save_plot=True)
    elif update_type == "Space":
        update_func = active_utils.UpdateSpaceFill(**update_stop_kwargs, save_plot=True)
    elif update_type == "Random":
        update_func = active_utils.UpdateRandom(**update_stop_kwargs, save_plot=True)

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
        log_scale=True,
        num_state_repeats=5,
        save_history=True,
    )


if __name__ == "__main__":
    print("Received raw arguments: ")
    print(sys.argv)
    print("\n\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "struc_file", type=str, help="structure file (e.g., pdb or gro)"
    )
    parser.add_argument("sys_file", type=str, help="system file (e.g., top")
    parser.add_argument(
        "lowT", type=float, help="low temperature to start at (in Kelvin)"
    )
    parser.add_argument(
        "highT", type=float, help="high temperature to start at (in Kelvin)"
    )
    parser.add_argument(
        "--dir", default="./", type=str, help="directory to write to, default ./"
    )
    parser.add_argument(
        "--observable",
        default="Rg",
        type=str,
        choices=["Rg", "Cv", "U"],
        help="observable to learn behavior of; can be Rg, Cv, or U",
    )
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
        args.struc_file,
        args.sys_file,
        args.dir,
        args.lowT,
        args.highT,
        args.observable,
        args.update,
        args.avoid_repeats,
        args.no_stop,
    )

import glob
import os

import numpy as np
import xarray as xr
from scipy import interpolate
from simtk import unit

import thermoextrap
from thermoextrap.gpr_active import active_utils

# Define some global constants
kBT_over_eps = 0.12500e01
eps_over_kB = 0.10000e03  # Assuming units of Kelvin
kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.joules / unit.kelvin)
eps = eps_over_kB * kB
beta = 1.0 / (eps * kBT_over_eps)
sigma = 0.30000e01  # Angstroms
lnZeta = -0.268000e01  # Equal to beta*mu for the bulk reference simulation
muRef = lnZeta * kBT_over_eps  # In units of the LJ epsilon


# Define all the functions we will use
def get_lnPi(
    mu,
    file_str=os.path.expanduser(
        "~/GPR_Extrapolation/Adsorption/sw_t125/t125.h0900_l10.ew_50.r*.lnpi.dat"
    ),
):
    files = glob.glob(file_str)
    dat = np.array([np.loadtxt(f) for f in files])
    Nvals = dat[:, :, 0]
    lnPi = dat[:, :, 1]
    # lnPi_err = np.var(dat[:, :, 1], axis=0)
    # Adjust to desired chemical potential
    lnPi = lnPi + (1.0 / kBT_over_eps) * (mu - muRef) * Nvals
    return Nvals, lnPi


def get_bulk_P(mu):
    # mu, chemical potential, is in units of the LJ epsilon

    # Load data
    refFile = os.path.expanduser(
        "~/GPR_Extrapolation/Adsorption/sw_t125/t125.v729.b.lnpi.dat"
    )
    Nvals, lnPi = get_lnPi(mu, file_str=refFile)
    lnPi = (
        lnPi - lnPi[:, :1]
    )  # Remove value at zero particles, making this value exactly 1 as it should be

    # Define volume
    V = 729.0  # *(sigma**3) #Angstroms^3 #currently V is V/sigma**3
    # V = (V*(unit.angstrom**3)).value_in_unit(unit.meters**3)

    # Compute log-sum-exp of lnPi
    maxVal = np.max(lnPi)
    betaPV = np.log(np.sum(np.exp(lnPi - maxVal))) + maxVal
    P = (
        betaPV * kBT_over_eps / V
    )  # / (beta*V) #Should be pascals since have J/m^3 #Now reduced units

    return P


class Adsorption_DataWrapper(active_utils.DataWrapper):
    def __init__(self, mu):
        self.lnPi_files = os.path.expanduser(
            "~/GPR_Extrapolation/Adsorption/sw_t125/t125.h0900_l10.ew_50.r*.lnpi.dat"
        )
        self.beta = mu  # Calling it beta for compatibility, but it's really mu

    def load_U_info(self):
        raise NotImplementedError

    def load_CV_info(self):
        raise NotImplementedError

    def load_x_info(self):
        raise NotImplementedError

    def get_data(self):
        # Instead of returning U, X, and weights, return -beta*N, N, and weights for each N
        # Averaging those should give the appropriate moments in N and beta*N
        N, lnPi = get_lnPi(self.beta, file_str=self.lnPi_files)
        lnPi = lnPi - lnPi[:, :1]
        maxProb = np.amax(lnPi, axis=1, keepdims=True)
        weights = np.exp(lnPi - maxProb)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        beta_N = xr.DataArray((1.0 / kBT_over_eps) * N, dims=["rep", "n"])
        N = xr.DataArray(N[:, :, None], dims=["rep", "n", "val"])
        weights = xr.DataArray(weights, dims=["rep", "n"])
        return -beta_N, N, weights

    def build_state(self, all_data=None, max_order=6):
        if all_data is None:
            all_data = self.get_data()
        u_vals = all_data[0]
        x_vals = all_data[1]
        weights = all_data[2]
        # Build raw moments array
        moments = np.ones((x_vals["rep"].shape[0], 2, max_order + 1))
        for i in range(2):
            for j in range(max_order + 1):
                moments[:, i, j] = (
                    ((x_vals**i) * (u_vals**j) * weights)
                    .sum("n", keep_attrs=False)
                    .values.flatten()
                )
        moments = xr.DataArray(
            moments[:, np.newaxis, :, :], dims=["rec", "val", "xmom", "umom"]
        )
        state_data = thermoextrap.DataCentralMoments.from_raw(moments)
        state = thermoextrap.beta.factory_extrapmodel(self.beta, state_data)
        return state


class SimulateAdsorption:
    def __init__(self, sim_func=None):
        self.sim_func = sim_func  # Will not perform any simulations

    def run_sim(self, unused, mu, n_repeats=None):
        # All this does is creates an Adsorption_DataWrapper object at the specified chemical potential and returns it
        del unused
        return Adsorption_DataWrapper(mu)


def run_active(
    active_dir,
    ground_truth_func,
    change_points=False,
    init_mu=[-12.0, 3.8],
    max_order=2,
):
    # Create directory for this run
    os.mkdir(active_dir)

    # Define update functions - will share kwargs, but will use different types
    update_kwargs = {
        "save_plot": True,
        "save_dir": active_dir,
        "compare_func": ground_truth_func,
        "avoid_repeats": True,  # Can't reduce uncertainty with "more sims" here
    }
    if "alm" in active_dir:
        update_func = active_utils.UpdateALMbrute(**update_kwargs)
    elif "space" in active_dir:
        update_func = active_utils.UpdateSpaceFill(**update_kwargs)
    elif "rand" in active_dir:
        update_func = active_utils.UpdateRandom(**update_kwargs)
    else:
        raise ValueError(
            "Must have 'alm', 'space', or 'rand' in active_dir argument or cannot pick update strategy."
        )

    # Set up list of metrics to compute
    metrics = [
        active_utils.MaxVar(1e-03),
        active_utils.MaxRelVar(1e-02),
        active_utils.AvgVar(1e-03),
        active_utils.AvgRelVar(1e-02),
        active_utils.MSD(1e-01),
        active_utils.MaxAbsRelDeviation(1e-02),
        active_utils.AvgAbsRelDeviation(1e-02),
        active_utils.ErrorStability(1e-02),
        active_utils.MaxRelGlobalVar(1e-02),
        active_utils.MaxAbsRelGlobalDeviation(1e-02),
        active_utils.MaxIter(),
    ]
    # And stopping criteria function wraps metrics
    stop_func = active_utils.StopCriteria(metrics)

    # Use change_points kernel if specified
    if change_points:
        base_kwargs = {"kernel": active_utils.ChangeInnerOuterRBFDerivKernel}
    else:
        base_kwargs = {}

    dat_list, train_history = active_utils.active_learning(
        init_mu,
        SimulateAdsorption(),
        update_func,
        stop_criteria=stop_func,
        max_iter=50,
        alpha_name="mu",
        max_order=max_order,
        base_dir=active_dir,
        save_history=True,
        gp_base_kwargs=base_kwargs,
    )

    return dat_list, train_history


def main():
    # To look at convergence, compute absorption curves with uncertainties at many mu values
    mu_plot = np.linspace(-12.0, 3.8, 100)
    np.array([get_bulk_P(m) for m in mu_plot])
    ads_vals = np.zeros((mu_plot.shape[0], 3))
    ads_derivs = np.zeros((mu_plot.shape[0], 3))

    for i, m in enumerate(mu_plot):
        this_dat = SimulateAdsorption().run_sim(None, m)
        this_coefs = this_dat.build_state(max_order=1).derivs(norm=False)
        this_means = this_coefs.mean("rec").values
        this_stds = np.sqrt(
            this_coefs.var("rec").values / this_coefs.sizes["rec"]
        )  # Want std in mean, not just std
        resamp_stds = np.sqrt(
            this_dat.build_state(max_order=1)
            .resample(nrep=100)
            .derivs(norm=False)
            .var("rep")
            .values
        )
        ads_vals[i, 0] = this_means[0, 0]
        ads_vals[i, 1] = this_stds[0, 0]
        ads_vals[i, 2] = resamp_stds[0, 0]
        ads_derivs[i, 0] = this_means[1, 0]
        ads_derivs[i, 1] = this_stds[1, 0]
        ads_derivs[i, 2] = resamp_stds[1, 0]

    # Create a function we can use as the ground truth to compare to
    ground_truth_ads = interpolate.interp1d(mu_plot, ads_vals[:, 0], kind="cubic")

    # Select initial end states to find behavior within
    init_mu = [-12.0, 3.8]

    # Define maximum order of derivatives to use with GP models
    max_order = 2

    ####################### GP models with change points kernels ########################

    # Update by selecting point with maximum variance (uncertainty) predicted by model
    dat_list_cp_alm, train_hist_cp_alm = run_active(
        "changepoints_alm",
        ground_truth_ads,
        change_points=True,
        init_mu=init_mu,
        max_order=max_order,
    )

    # Update by selecting point to fill space
    dat_list_cp_space, train_hist_cp_space = run_active(
        "changepoints_space",
        ground_truth_ads,
        change_points=True,
        init_mu=init_mu,
        max_order=max_order,
    )

    ####################### GP models with standard RBF kernels ########################

    # Update by selecting point with maximum variance (uncertainty) predicted by model
    dat_list_out, train_history = run_active(
        "alm",
        ground_truth_ads,
        change_points=False,
        init_mu=init_mu,
        max_order=max_order,
    )

    # Update by selecting point to fill space
    dat_list_space, train_history_space = run_active(
        "space_fill",
        ground_truth_ads,
        change_points=False,
        init_mu=init_mu,
        max_order=max_order,
    )

    # Update by selecting point randomly
    dat_list_rand, train_history_rand = run_active(
        "random",
        ground_truth_ads,
        change_points=False,
        init_mu=init_mu,
        max_order=max_order,
    )

    # Want to know how data influences loss and optimal parameters
    # Ideally, should converge to same parameters regardless of specific input points
    # But how much data is needed for this?
    # And know that there are local minima, but does specific data set change that situation?
    train_labels = ["ALM", "Random", "Space"]
    for i, dat_list in enumerate([dat_list_out, dat_list_rand, dat_list_space]):
        print("%s data set:" % train_labels[i])
        this_gp = active_utils.create_GPR(
            [dat.build_state(max_order=max_order) for dat in dat_list]
        )
        default_loss = this_gp.training_loss()
        default_params = tuple([par.numpy() for par in this_gp.trainable_parameters])
        print(
            "\t Unbiased opt from default: loss %f, l %f, var %f, p %f"
            % ((default_loss,) + default_params)
        )

        for j, train_hist in enumerate(
            [train_history, train_history_rand, train_history_space]
        ):
            for k, tpar in enumerate(train_hist["params"][-1]):
                this_gp.trainable_parameters[k].assign(tpar)
            this_loss = this_gp.training_loss()
            print(
                "\t W/ params from %s: loss %f, l %f, var %f, p %f"
                % (
                    (train_labels[j].ljust(10), this_loss)
                    + tuple(train_hist["params"][-1])
                )
            )

        print("\n")


if __name__ == "__main__":
    main()

"""
GPR utilities (:mod:`~thermoextrap.gpr_active.active_utils`)
------------------------------------------------------------
"""
import glob
import multiprocessing
import os
import time

import gpflow
import numpy as np
import sympy as sp

# import tensorflow as tf
import xarray as xr
from pymbar import timeseries
from scipy import integrate, linalg, special

from .. import DataCentralMomentsVals, ExtrapModel
from .. import beta as xpan_beta
from .gp_models import (
    ConstantMeanWithDerivs,
    DerivativeKernel,
    HeteroscedasticGPR,
    LinearWithDerivs,
)

# from typing import Optional


def get_logweights(bias):
    """
    Given values of the biasing potential for each configuration, calculates the weights
    for averaging over those configurations for the biased ensemble so that the
    average represents the unbiased ensemble.
    """
    bias_max = np.max(bias)
    log_denom = (
        np.log(np.sum(np.exp(bias - bias_max))) + bias_max
    )  # - np.log(bias.shape[0])
    logw = (
        bias - log_denom
    )  # - np.log(bias.shape[0]) # Acts as weight summing to 1 this way
    return logw


def input_GP_from_state(state, n_rep=100, log_scale=False):
    """
    Builds input for GP model up to specified order from ExtrapModel object of thermoextrap.
    If log_scale, adjust x inputs and derivatives to reflect taking the logarithm of x.

    Parameters
    ----------
    state : ExtrapModel
      object containing derivative information
    n_rep : int, default=100
        Number of bootstrap draws of data to perform to compute variances
    log_scale : bool, default=False
        Whether or not to apply a log scale in the input locations
        (i.e., compute derivatives of dy/dlog(x) instead of dy/dx)

    Returns
    -------
    x_data : object
        input locations, likely state points (e.g., temperature, pressure, etc.),
        augmented with derivative order of each observation, as required for GP
        models
    y_data : object
        Output data, which includes both function values and derivative information
    cov_data : object
        covariance matrix between function observations, including derivative
        observations; note that this will be block-diagonal since it is expected
        that the state objects at different conditions are based on information
        from independent simulations, while the observations at different derivative
        orders from a single simulation are correlated
    """
    # Prepare x data
    alphas = state.alpha0 * np.ones((state.order + 1, 1))
    if log_scale:
        alphas = np.log10(alphas)
    x_data = np.concatenate([alphas, np.arange(state.order + 1)[:, None]], axis=1)

    if isinstance(state.data, DataCentralMomentsVals):
        derivs = state.derivs(norm=False).values
        resamp_derivs = state.resample(nrep=n_rep).derivs(norm=False)
    else:
        # Above with DataCentralMomentsVals is for simulation snapshots
        # Below is if things are pre-computed, so have multiple simulations
        # With multiple simulations, have mean of observable for each along 'rec' dim
        # So just want variance along that 'rec' dimension
        # resample() does not necessarily only resample along the 'rec' dimension
        # So to be consistent with above single simulation case, just compute
        # variance along 'rec' dimension by not doing resampling
        derivs = state.derivs(norm=False).mean("rec").values
        resamp_derivs = state.derivs(norm=False)

    if log_scale:
        # Need to use Faa di Bruno's formula
        log_derivs = np.zeros_like(derivs)
        log_derivs[0, :] = derivs[0, :]
        resamp_log_derivs = xr.zeros_like(resamp_derivs)
        resamp_log_derivs[dict(order=0)] = resamp_derivs[dict(order=0)].values
        for n in range(1, derivs.shape[0]):
            for k in range(1, n + 1):
                bell_fac = sp.bell(
                    n, k, state.alpha0 * (np.log(10.0) ** np.arange(1, n - k + 2))
                )
                log_derivs[n, :] = log_derivs[n, :] + derivs[k, :] * bell_fac
                resamp_log_derivs[dict(order=n)] = (
                    resamp_log_derivs[dict(order=n)].values
                    + resamp_derivs[dict(order=k)].values * bell_fac
                )
        y_data = log_derivs
        # Compute full covariance matrix
        # Note loop over output dimensions since will have independent covariance for each
        cov_data = []
        for k in range(resamp_log_derivs.shape[-1]):
            cov_data.append(np.cov(resamp_log_derivs.values[..., k]))
        cov_data = np.array(cov_data)
    else:
        y_data = derivs
        cov_data = []
        for k in range(resamp_derivs.shape[-1]):
            cov_data.append(np.cov(resamp_derivs.values[..., k]))
        cov_data = np.array(cov_data)

    return x_data, y_data, cov_data


class DataWrapper:
    """
    Class to keep track of metadata around the data. Data will not be stored here, but
    this class will define how data is loaded and processed. If want to change the
    column indices, simply have to call get_data, then build_state in two separate
    steps rather than just calling build_state. Handles multiple files, but only
    under the assumption that all biases are fixed after the initial simulation.
    If the biases change, then need to use MBAR to reweight everything. This is
    feasible, but difficult if trying to provide all samples to GPR so it can
    bootstrap. When to point where can provide all derivatives and uncertainties
    directly, can switch over to using MBAR and computeExpectations.

    Parameters
    ----------
    sim_info_files : list of str
        list of files containing simulation information, such as potential
        energy timeseries
    cv_bias_files : list of str
        list of files containing the collective variable, or quantity of interest
        for the active learning procedure; a bias along this quantity or
        another CV may also be included for a simulation with enhanced sampling
    beta : float
        the reciprocal temperature (1/(kB*T)) of the simulations in this data set
    x_files : list of str, optional
        the files containing the quantity of interest for the active
        learning procedure; default is to assume this is the CV in cv_bias_files
        and that these are not necessary
    n_frames : int, default=10000
        number of frames from the END of simulation(s), or the files read in, to
        use for computations; allows exclusion of equilibration periods
    u_col : int, default=2
        column of sim_info_files in which potential energy is found
    cv_cols : list, default=`[1,2]`
        columns of cv_bias_files in which the CV and bias are found
    x_col : list, default=`[1]`
        list of columns from x_files to pull out; can be multiple if
        the are multiple outputs of interest
    """

    def __init__(
        self,
        sim_info_files,
        cv_bias_files,
        beta,
        x_files=None,
        n_frames=10000,
        u_col=2,
        cv_cols=[1, 2],
        x_col=[
            1,
        ],
    ):
        self.sim_info_files = sim_info_files
        self.cv_bias_files = cv_bias_files
        self.beta = beta
        self.x_files = x_files
        self.n_frames = n_frames
        self.u_col = u_col
        self.cv_cols = cv_cols
        if isinstance(x_col, (int, float)):
            x_col = [
                int(x_col),
            ]
        self.x_col = x_col

    def load_U_info(self):
        """Loads potential energies from a list of files."""
        U = []
        for f in self.sim_info_files:
            U.append(np.loadtxt(f)[-self.n_frames :, self.u_col])
        # If eventually using MBAR, will want to vstack instead
        U = np.hstack(U)
        return U

    def load_CV_info(self):
        """
        Loads data from a file specifying CV coordinate and added bias at each frame.
        Assumes that the first value in col_ind is the index of the CV coordinate column and the
        second is the index for the bias.
        """
        cv_vals = []
        cv_bias = []
        for f in self.cv_bias_files:
            cv_info = np.loadtxt(f)[-self.n_frames :, self.cv_cols]
            cv_vals.append(cv_info[:, 0])
            cv_bias.append(cv_info[:, 1])
        cv_vals = np.hstack(cv_vals)
        cv_bias = np.hstack(cv_bias)
        return cv_vals, cv_bias

    def load_x_info(self):
        """Loads observable data."""
        x = []
        for f in self.x_files:
            x.append(np.loadtxt(f)[-self.n_frames :, self.x_col])
        x = np.vstack(x)
        return x

    def get_data(self):
        """
        Loads data from files needed to generate data classes for thermoextrap.
        Will change significantly if using MBAR on trajectories with different biases.
        """
        tot_pot = self.load_U_info()
        cv, bias = self.load_CV_info()
        # If the cv we bias along is the x of interest for extrapolation, return that
        # Otherwise, load x generically (if x_file specified)
        if self.x_files is not None:
            x = self.load_x_info()
        else:
            x = cv[:, None]
        # Need to subtract bias from total potential energy
        pot = tot_pot - bias
        # Extract statistical inefficiencies for x and pot and pick largest
        # Can have multidimensional outputs, i.e., multiple columns of x
        # Handle by looping and taking maximum
        g_x = 0.0
        g_cross = 0.0
        for k in range(x.shape[1]):
            this_g_x = timeseries.statisticalInefficiency(x[:, k])
            this_g_cross = timeseries.statisticalInefficiency(x[:, k], pot)
            if this_g_x > g_x:
                g_x = this_g_x
            if this_g_cross > g_cross:
                g_cross = this_g_cross
        g_pot = timeseries.statisticalInefficiency(pot)
        g_max = np.max([g_x, g_pot, g_cross])
        # Get indices of uncorrelated data and subsample everything
        uncorr_inds = timeseries.subsampleCorrelatedData(np.arange(x.shape[0]), g_max)
        x = x[uncorr_inds, :]
        bias = bias[uncorr_inds]
        pot = pot[uncorr_inds]
        # Get weights to remove bias during averaging
        logw = get_logweights(self.beta * bias)
        w = np.exp(logw)
        # Convert to xarray objects with dimensions named appropriately
        pot = xr.DataArray(pot, dims=["rec"])
        x = xr.DataArray(x, dims=["rec", "val"])
        return pot, x, w

    def build_state(self, all_data=None, max_order=6):
        """
        Builds a thermoextrap data object for the data described by this wrapper class.
        If all_data is provided, should be list or tuple of (potential energies, X) to
        be used, where X should be appropriately weighted if the simulation is biased.
        """
        if all_data is None:
            all_data = self.get_data()
        u_vals = all_data[0]
        x_vals = all_data[1]
        weights = all_data[2]
        state_data = DataCentralMomentsVals.from_vals(
            uv=u_vals, xv=x_vals, w=weights, order=max_order
        )
        state = xpan_beta.factory_extrapmodel(self.beta, state_data)
        return state


class SimWrapper:
    """
    Wrapper around simulations to spawn similar simulations easily and keep
    track of all parameter values.

    Parameters
    ----------
    sim_func : callable
        function that runs a new simulation
    struc_name : str
        name of structure file inputs to simulation
    sys_name : str
        name of system or topological file inputs to simulation
    info_name : str
        name of information file for simulation to produce
    bias_name : str
        name of file with CV values and bias for simulation to produce
    kw_inputs : dict, optional
        additional keyword inputs to the simulation
    data_class : object
        class (e.g., :class:`DataWrapper`) to use for wrapping simulation output data;
        data will be wrapped before returned to the active learning algorithm
    post_process_func : callable, optional
        Function for post-processing simulation outputs but before
        wrapping in data_class
    post_process_out_name : str, optional
        name of output files produced by post_process_func
    post_process_kw_inputs : dict, optional
        additional dictionary of arguments for the
        post_process_func
    pre_process_func : callable, optional
        function to apply before a simulation run in order to produce extra
        keyword arguments for the simulation run; useful if have an extra model
        that predicts info needed by or helpful for the simulation
    """

    def __init__(
        self,
        sim_func,
        struc_name,
        sys_name,
        info_name,
        bias_name,
        kw_inputs={},
        data_kw_inputs={},
        data_class=DataWrapper,
        post_process_func=None,
        post_process_out_name=None,
        post_process_kw_inputs={},
        pre_process_func=None,
    ):
        self.sim_func = sim_func  # Function for running a simulation
        self.struc_file = (
            struc_name  # Name of structure file for positions and topology
        )
        self.sys_file = sys_name  # Name of system file for setting up, say, OpenMM
        self.info_name = info_name  # Name of simulation info file produced
        self.bias_name = bias_name  # Name of simulation bias file produced
        self.kw_inputs = (
            kw_inputs  # Dictionary of other key-word args for the simulation
        )
        self.kw_inputs[
            "info_name"
        ] = self.info_name  # Restricts naming convention for sim_func
        self.kw_inputs["bias_name"] = self.bias_name
        self.data_kw_inputs = data_kw_inputs
        self.data_class = data_class
        # If have post-processing step, need to handle that
        # Will also constrain structure of inputs for post-processing function
        self.pp_func = post_process_func
        self.pp_out_name = post_process_out_name
        self.pp_kw_inputs = post_process_kw_inputs

        self.pre_func = pre_process_func  # Pre-processing function (predict extra args)

    def run_sim(self, sim_dir, alpha, n_repeats=1, **extra_kwargs):
        """
        Runs simulation(s) and returns an object of type self.data_class pointing to
        right files. By default only one, but will run n_repeats in parallel if specified.
        """
        # Create directory if does not exist
        if not os.path.isdir(sim_dir):
            os.mkdir(sim_dir)

        # Pre-processing function output
        if self.pre_func is not None:
            pre_output = self.pre_func(alpha)
            extra_kwargs = {**extra_kwargs, **pre_output}

        # Determine numbering of runs so can keep track if parallel
        curr_sim_num = len(glob.glob(os.path.join(sim_dir, self.info_name + "*")))

        # Will now kick off n_repeats simulations in parallel
        job_list = []
        for i in range(n_repeats):
            # Passing extra_kwargs, which could be model information/predictions as dictionary
            # Sim functions using extra_kwargs should take model_pred and model_std as inputs
            p = multiprocessing.Process(
                target=self.sim_func,
                args=(self.struc_file, self.sys_file, alpha),
                kwargs={
                    "file_prefix": sim_dir,
                    "sim_num": curr_sim_num + i,
                    **self.kw_inputs,
                    **extra_kwargs,
                },
            )
            p.start()
            job_list.append(p)
            # Pause for a bit to make sure time-based random number seeds different
            time.sleep(5)

        for p in job_list:
            p.join()

        # Check that jobs finished successfully
        for p in job_list:
            if p.exitcode != 0:
                raise RuntimeError(
                    "At least one parallel simulation did not terminate cleanly."
                )

        # Loop over post-processing functions (not costly, so no need for parallel)
        if self.pp_func is not None:
            for i in range(n_repeats):
                self.pp_func(
                    sim_dir,
                    self.pp_out_name,
                    sim_num=curr_sim_num + i,
                    **self.pp_kw_inputs,
                )
            sim_x_files = sorted(
                glob.glob(os.path.join(sim_dir, self.pp_out_name + "*"))
            )
        else:
            sim_x_files = None

        # Retrieve all files in lists and create self.data_class object
        sim_info_files = sorted(glob.glob(os.path.join(sim_dir, self.info_name + "*")))
        sim_bias_files = sorted(glob.glob(os.path.join(sim_dir, self.bias_name + "*")))
        sim_dat = self.data_class(
            sim_info_files,
            sim_bias_files,
            alpha,
            x_files=sim_x_files,
            **self.data_kw_inputs,
        )

        # If have pre-processing function provide sim info to update it
        # Ignore if pre_func has no update() method, though
        if self.pre_func is not None:
            try:
                self.pre_func.update(alpha, sim_info_files, sim_bias_files, sim_x_files)
            except AttributeError:
                pass

        return sim_dat


# FIXME: replace l with v


def make_matern_expr(p):
    """
    Creates a sympy expression for the Matern kernel of order p.

    Parameters
    ----------
    p : int
        order of Matern kernel

    Returns
    -------
    expr : Expr
    kern_params : dict
        parameters matching naming in sympy expression
    """
    d = sp.symbols("d")
    k = sp.var("k")
    poly_part = sp.Sum(
        (sp.factorial(p + k) / (sp.factorial(k) * sp.factorial(p - k)))
        * (2 * sp.sqrt(float(2 * p + 1)) * d) ** (p - k),
        (k, 0, p),
    ).doit()
    poly_part = poly_part * sp.factorial(p) / sp.factorial(2 * p)
    exp_part = sp.exp(-sp.sqrt(float(2 * p + 1)) * d)
    full_expr = sp.simplify(poly_part * exp_part)
    l = sp.symbols("l", real=True)  # noqa: E741
    x1 = sp.symbols("x1", real=True)
    x2 = sp.symbols("x2", real=True)
    distance = sp.sqrt((x1 / l - x2 / l) ** 2)
    var = sp.symbols("var", real=True)
    kern_params = {
        "var": [1.0, {"transform": gpflow.utilities.positive()}],
        "l": [1.0, {"transform": gpflow.utilities.positive()}],
    }
    return var * full_expr.subs(d, distance), kern_params


def make_rbf_expr():
    """
    Creates a sympy expression for an RBF kernel.

    Returns
    -------
    expr : Expr
    kern_params : dict
        parameters matching naming in sympy expression
    """
    var = sp.symbols("var", real=True)
    l = sp.symbols("l", real=True)  # noqa: E741
    x1 = sp.symbols("x1", real=True)
    x2 = sp.symbols("x2", real=True)
    rbf_kern_expr = var * sp.exp(-0.5 * (x1 / l - x2 / l) ** 2)
    kern_params = {
        "var": [1.0, {"transform": gpflow.utilities.positive()}],
        "l": [1.0, {"transform": gpflow.utilities.positive()}],
    }
    return rbf_kern_expr, kern_params


def make_poly_expr(p):
    """
    Creates a sympy expression for a polynomial kernel.

    Parameters
    ----------
    p : int
        order of polynomial

    Returns
    -------
    expr : Expr
    kern_params : dict
        parameters matching naming in sympy expression
    """
    var = sp.symbols("var", real=True)
    l = sp.symbols("l", real=True)  # noqa: E741
    x1 = sp.symbols("x1", real=True)
    x2 = sp.symbols("x2", real=True)
    poly_kern_expr = (var * x1 * x2 + l) ** p
    kern_params = {
        "var": [1.0, {"transform": gpflow.utilities.positive()}],
        "l": [1.0, {"transform": gpflow.utilities.positive()}],
    }
    return poly_kern_expr, kern_params


# Below implements compactly-supported piecewise polynomial kernel
# (See Rasmussen and Williams)
# But only for first order - not that useful, was just experimenting
# def make_compact_support_poly_expr():
#     var = sp.symbols("var")
#     l = sp.symbols("l")
#     x1 = sp.symbols("x1")
#     x2 = sp.symbols("x2")
#     f = var * (1 - sp.exp(-1 / (x1 / l - x2 / l)**2))
#     cspp_kern_expr = sp.Piecewise((1, sp.Eq(x1, x2)), (f, True))
#     kern_params = {"var" : [1.0, {"transform": gpflow.utilities.positive()}],
#                    "l" : [1.0, {"transform": gpflow.utilities.positive()}],
#                   }
#     return cspp_kern_expr, kern_params


class RBFDerivKernel(DerivativeKernel):
    """
    For convenience, create a derivative kernel specific to RBF function.
    Use it most often, so convenient to have.
    """

    def __init__(self, **kwargs):
        kern_expr, kern_params = make_rbf_expr()
        super().__init__(kern_expr, 1, kernel_params=kern_params, **kwargs)


class ChangeInnerOuterRBFDerivKernel(DerivativeKernel):
    """
    Implements a change-points kernel via logistic switching functions (as in GPflow's
    ChangePoints kernel), but only for two points, where two instead of three kernels
    are utilized: one for the outer region and one for the inner. Both kernels are
    RBF kernels with a shared variance parameter. The resulting kernel is differentiable,
    inheriting DerivativeKernel. Two points where the kernel changes may be specified,
    c1 and c2, meaning that the outer kernel is used for x <= c1 and x >= c2, while the
    inner kernel is used for c1 < x < c2.

    Parameters
    ----------
    c1 : float
        first change point
    c2 : float
        second change point
    **kwargs
        Extra Arguments to :class:`~thermoextrap.gpr_active.gp_models.DerivativeKernel`


    See Also
    --------
    ~thermoextrap.gpr_active.gp_models.DerivativeKernel
    """

    def __init__(self, c1=-7.0, c2=-2.0, **kwargs):
        x1 = sp.symbols("x1", real=True)
        x2 = sp.symbols("x2", real=True)

        var = sp.symbols("var", real=True)
        l_out = sp.symbols("l_out", real=True)
        l_in = sp.symbols("l_in", real=True)
        kern_expr_outer = var * sp.exp(-0.5 * (x1 / l_out - x2 / l_out) ** 2)
        kern_expr_inner = var * sp.exp(-0.5 * (x1 / l_in - x2 / l_in) ** 2)
        kern_params = {
            "var": [1.0, {"transform": gpflow.utilities.positive()}],
            "l_out": [1.0, {"transform": gpflow.utilities.positive()}],
            "l_in": [1.0, {"transform": gpflow.utilities.positive()}],
        }

        x = sp.symbols("x")
        s = sp.symbols("s")
        c = sp.symbols("c")
        # tanh better behaved with sympy than coding up logistic for sigmoid
        sig_expr = 0.5 * (1.0 + sp.tanh(s * (x - c)))
        low_change_expr = (1.0 - sig_expr.subs(x, x1)) * (1.0 - sig_expr.subs(x, x2))
        hi_change_expr = sig_expr.subs(x, x1) * (sig_expr.subs(x, x2))

        c1 = sp.symbols("c1")
        c2 = sp.symbols("c2")
        full_expr = (
            kern_expr_outer * low_change_expr.subs(c, c1)
            + hi_change_expr.subs(c, c1) * kern_expr_inner * low_change_expr.subs(c, c2)
            + hi_change_expr.subs(c, c2) * kern_expr_outer
        )

        kern_params["s"] = [
            10.0,
            {"transform": gpflow.utilities.positive(), "trainable": False},
        ]
        kern_params["c1"] = [-7.0, {"trainable": False}]
        kern_params["c2"] = [-2.0, {"trainable": False}]

        super().__init__(full_expr, 1, kernel_params=kern_params, **kwargs)


def create_base_GP_model(
    gpr_data,
    d_order_ref=0,
    shared_kernel=True,
    kernel=RBFDerivKernel,
    mean_func=None,
    likelihood_kwargs={},
):
    """
    Creates just the base GP model without any training,just sets up sympy and
    GPflow. kernel can either be a kernel object, in which case it is assumed
    you know what you're doing and shared_kernel will be ignored (will not wrap
    in SharedIndependent or SeparateIndependent and thus if the kernel is not a
    subclass of MultioutputKernel, HeteroscedasticGPR will wrap it in a
    SharedIndependent kernel). If shared_kernel is False and a kernel object is
    provided, only a warning will be printed, so beware. If only a class is
    provided for kernel, then shared_kernel is respected - a class is necessary
    so that if the kernel is not to be shared, separate instances can be
    created. Note that if a class is provided for kernel, it must be initiated
    without any passed parameters - this is easy to set up with a wrapper class
    as for RBFDerivKernel, the default.

    Parameters
    ----------
    gpr_data : tuple
        a tuple of input locations, output data, and the noise covariance matrix
    d_order_ref : int, default=0
        derivative order to treat as the reference for constructing mean
        functions; PROBABLY BEST TO REMOVE THIS OPTION UNTIL HAVE MORE
        SOPHISTICATED MEAN FUNCTIONS - DEFAULT BEHAVIOR DEFENDS AGAINST
        SITUATION WITH NO ZEROTH ORDER DERIVATIVES, BUT DOES NOT CREATE
        MEANINGFUL MEAN FUNCTION IN THAT CASE (JUST ZEROS)
    shared_kernel : bool, default=True
        whether or not the kernel will be shared across output dimensions
    kernel : object
        Defaults to RBFDerivKernel.  Kernel to use in GP model.
    mean_func : callable, optional
        mean function to use for GP model
    likelihood_kwargs : dict, optional
        keyword arguments to pass to the likelihood model

    Returns
    -------
    gpr: :class:`thermoextrap.gpr_active.gp_models.HeteroscedasticGPR`
        Note, that this is an untrained model.
    """
    # Will be helpful to know where have zero-order derivatives in data
    # Or more generally may want to specify which order to pay attention to rather than just 0
    ref_d_bool = gpr_data[0][:, 1] == d_order_ref

    # Have played with scaling x data to change length scales without major success
    x_scale = 1.0

    # Create mean function, if not provided
    if mean_func is None:
        # But only if working with zero-order data - mean functions not fancy enough yet
        # Would be as easy as fitting polynomial object (numpy or scipy) and integrals/diffs
        if d_order_ref == 0:
            # By default, fit linear model to help get rid of monotonicity
            if len(np.unique(gpr_data[0][ref_d_bool, :1])) > 2:
                mean_func = LinearWithDerivs(
                    x_scale * gpr_data[0][ref_d_bool, :1], gpr_data[1][ref_d_bool, :]
                )
            # Linear mean only meaningful if have at least 3 different input locations
            # If just have 1 or 2, just fit constant mean function
            else:
                mean_func = ConstantMeanWithDerivs(gpr_data[1][ref_d_bool, :])
        else:
            mean_func = ConstantMeanWithDerivs(
                np.zeros_like(gpr_data[1][ref_d_bool, :])
            )

    # For multiple output kernels, helpful to scale all outputs so have similar variance
    # (over data, that is, so scale by variance in all y for each output dimension)
    # Also helps ensure kernel variance will be close to 1, so close to starting value
    # But can only do if have at least 2 values... really 3 for computing std, but
    # doing it with 2 won't hurt, will just scale by mean, technically
    if len(np.unique(gpr_data[0][ref_d_bool, :1])) > 1:
        std_scale = np.std(
            gpr_data[1][ref_d_bool, :]
            - mean_func(x_scale * gpr_data[0][ref_d_bool, :]),
            axis=0,
        )
    else:
        std_scale = 1.0

    # If already provided fully prepared kernel, just pass along
    if not isinstance(kernel, type):
        full_kernel = kernel
        # But warn if the kernel is not MultioutputKernel and shared_kernel=False
        if (
            not issubclass(type(kernel), gpflow.kernels.MultioutputKernel)
            and not shared_kernel
        ):
            print(
                "WARNING: A kernel object (not class) of %s has been provided. Since this is not a subclass of gpflow.kernels.MultioutputKernel, it will be wrapped in a SharedIndependent kernel by HeteroscedasticGPR. However, you have set shared_kernel=False, so this may not be the behavior you wanted."
                % str(kernel)
            )
    else:
        if shared_kernel:
            full_kernel = gpflow.kernels.SharedIndependent(
                kernel(), output_dim=gpr_data[1].shape[-1]
            )
        else:
            full_kernel = gpflow.kernels.SeparateIndependent(
                [kernel() for k in range(gpr_data[1].shape[-1])]
            )

    gpr = HeteroscedasticGPR(
        gpr_data,
        kernel=full_kernel,
        scale_fac=std_scale,
        x_scale_fac=x_scale,
        mean_function=mean_func,
        likelihood_kwargs=likelihood_kwargs,
    )

    return gpr


def train_GPR(gpr, record_loss=False, start_params=None):
    """
    Trains a given gpr model for n_opt steps.
    Actually uses scipy wrapper in gpflow, which seems faster.
    If starting parameter values are provided in start_params, should be
    iterable with numpy array or float values (e.g., in tuple or list).

    Parameters
    ----------
    gpr : object
        The GPR model to train
    record_loss : bool, default=False
        Whether or not to record the output of the optimizer and return it
    start_params : dict, optional
        Parameters to also try as starting points for the optimization; if these
        are provided as a list or numpy array, two optimizations are performed,
        one from the current GPR model parameters, and one with the GPR model
        parameters set to start_params; the optimization result with the lowest
        loss function value is selected and the GPR parameters are set to those
        values
    """
    optim = gpflow.optimizers.Scipy()
    loss_info = optim.minimize(
        gpr.training_loss, gpr.trainable_variables, method="L-BFGS-B", compile=False
    )

    # If provided with starting parameters, also do optimization starting with them
    if start_params is not None:
        # Record optimized parameters with default starting values
        optim_params = [tpar.numpy() for tpar in gpr.trainable_parameters]

        # Set values to provided starting values
        for j in range(len(gpr.trainable_parameters)):
            gpr.trainable_parameters[j].assign(start_params[j])

        # Perform optimization starting with provided values
        loss_info_new = optim.minimize(
            gpr.training_loss, gpr.trainable_variables, method="L-BFGS-B", compile=False
        )

        # Make sure one or both losses are not NaN
        check_nan = np.isnan([loss_info.fun, loss_info_new.fun])
        if np.all(check_nan):
            print(
                "All optimizations resulted in NaN with respective loss information: "
            )
            print(loss_info)
            print(loss_info_new)
            raise ValueError("Had NaNs in loss!")

        # If optimization with default values better, reassign values back
        # Checked above if BOTH losses gave NaNs
        # Comparing np.nan to anything provides False
        # So if loss_info.fun is NaN, no problems because loss_info_new.fun can't be
        # But if loss_info_new.fun is NaN, do want to change parameters back
        # In other words, update parameters if default loss less OR if new loss is NaN
        if (loss_info.fun < loss_info_new.fun) or (check_nan[1]):
            for j, tpar in enumerate(optim_params):
                gpr.trainable_parameters[j].assign(tpar)
        # Otherwise, change loss information that's potentially returned and leave params
        else:
            loss_info = loss_info_new

    if record_loss:
        return loss_info
    else:
        return None


def create_GPR(state_list, log_scale=False, start_params=None, base_kwargs={}):
    """
    Generates and trains a GPR model based on a list of ExtrapModel objects or a
    StateCollection object from thermoextrap. If a list of another type of
    object, such as a custom state function, will simply call it and expect to
    return GPR input data.

    Parameters
    ----------
    state_list : list of ExtrapModel
        Each at different conditions.
    log_scale : bool, default=False
        whether or not to compute derivatives with respect to x or the logarithm
        of x, where x is the input location
    start_params : dict, optional
        Starting parameter values to consider during optimization
    base_kwargs : dict, optional
        Additional dictionary of keyword arguments to pass to create_base_GP_model

    Returns
    -------
    gpr : :class:`thermoextrap.gpr_active.gp_models.HeteroscedasticGPR`
        Trained model.
    """

    # Loop over states and collect information needed for GP
    x_data = []
    y_data = []
    cov_data = []
    for s in state_list:
        if isinstance(s, ExtrapModel):
            this_x_data, this_y_data, this_cov_data = input_GP_from_state(
                s, log_scale=log_scale
            )
        else:
            this_x_data, this_y_data, this_cov_data = s()
        x_data.append(this_x_data)
        y_data.append(this_y_data)
        cov_data.append(this_cov_data)

    # Put state information together
    x_data = np.vstack(x_data)
    y_data = np.vstack(y_data)
    # Derivatives from same simulation correlated, between independent
    # And different outputs are also independent in their likelihood model (covariance matrix)
    # So loop to treat each dimension separately
    noise_cov_mat = []
    for k in range(y_data.shape[1]):
        noise_cov_mat.append(linalg.block_diag(*[cov[k, ...] for cov in cov_data]))
    noise_cov_mat = np.array(noise_cov_mat)
    data_input = (x_data, y_data, noise_cov_mat)

    # Create GPR
    gpr = create_base_GP_model(data_input, **base_kwargs)
    # And train
    train_GPR(gpr, start_params=start_params)

    #     #Want to catch tf.python.framework.errors_impl.InvalidArgumentError
    #     #Associated with invalid Cholesky decomposition, typically associated with off derivatives
    #     #So keep decreasing order until works, and if order=0 fails throw error
    #     order = int(np.max(x_data[:, 1]))
    #     while order >= 0:
    #
    #         try:
    #             #Create GPR
    #             gpr = create_base_GP_model(data_input, **base_kwargs)
    #             #And train
    #             train_GPR(gpr, start_params=start_params)
    #             break
    #         except tf.errors.InvalidArgumentError as error:
    #             print(error)
    #             print('Invalid Cholesky decomposition, likely due to poorly converged derivatives.')
    #             print('Reducing order to %i and trying again.'%(order-1))
    #             x_data = np.vstack([x_data[i:i+order, :]
    #                                 for i in range(0, x_data.shape[0], order+1)])
    #             y_data = np.vstack([y_data[i:i+order, :]
    #                                 for i in range(0, y_data.shape[0], order+1)])
    #             noise_cov_mat = linalg.block_diag(*[c[:order, :order] for c in cov_data])
    #             data_input = (x_data, y_data, noise_cov_mat)
    #             order = order - 1
    #
    #         if order < 0:
    #             raise ValueError('Cholesky decomposition has failed at order=0. Something is wrong.')

    return gpr


# Need functions for transforming GP outputs
# The rules are the following: it must take as input x, y, and y_std of the GP output
# Then it should return the median of the transformed distribution, an uncertainty estimate,
# which for a Gaussian should be the std, but for other distributions could be std or some
# confidence width, and finally upper and lower bounds of a confidence interval
# (preferably around 95%)
# Here only implement the simplest (and default) transformation, the identity transform
# (which also computes std given variance and upper and lower confidence interval values)
def identityTransform(x, y, y_var):
    y_std = np.sqrt(y_var)
    conf_int = [y - 2.0 * y_std, y + 2.0 * y_std]
    return y, y_std, conf_int


# The following functions will be useful in both update function and stopping criteria classes
# So creating a class that just implements those to inherit
class UpdateStopABC:
    """
    Class that forms basis for both update and stopping criteria classes, which both need to
    define transformation functions and create grids of alpha values.

    """

    def __init__(
        self,
        d_order_pred=0,
        transform_func=identityTransform,
        log_scale=False,
        avoid_repeats=False,
    ):
        """
        Parameters
        ----------
        d_order_pred : int, default=0
            Derivative order at which predictions should be made
        transform_func : identityTransform function
            For transforming GP model output
            should take the x, y, and y_var (input, output, and variance
            in y) of GP as input; output should be transformed mu (or
            median, which is better), transformed uncertainty (could be
            std or confidence interval width), and finally the confidence
            interval itself, which is best for plotting.
        log_scale : bool, default=False
            Whether to use log scale for input (x) or not.
        avoid_repeats : bool, default=False
            Whether or not to randomize grid of new locations.
        """
        self.d_order_pred = d_order_pred
        self.transform_func = transform_func
        self.log_scale = log_scale
        self.avoid_repeats = avoid_repeats

    def create_alpha_grid(self, alpha_list):
        """
        Given a list of alpha values used in the GP model, creates a grid of values
        to evaluate the GP model at. This grid, alpha_grid is returned along with values
        of possible points to select to add to the GP model, alpha_select. Depending
        on the update strategy, these may be different points (e.g., if using integrated
        uncertainty, point to add may be different than grid used to evaluate integrated
        variance).
        """
        # Set up grid of points based on adjustable parameter values passed to GPR
        alpha_min = np.min(alpha_list)
        alpha_max = np.max(alpha_list)
        if self.log_scale:
            alpha_min = np.log10(alpha_min)
            alpha_max = np.log10(alpha_max)
        alpha_grid = np.linspace(alpha_min, alpha_max, 1000)

        # If want to avoid repeats, randomize a bit and exclude end points
        alpha_select = alpha_grid.copy()
        if self.avoid_repeats:
            alpha_select += np.hstack(
                [
                    [0.0],
                    2.0
                    * (alpha_grid[1] - alpha_grid[0])
                    * (np.random.random(len(alpha_grid) - 2) - 0.5),
                    [0.0],
                ]
            )
            alpha_select = alpha_select[1:-1]
        return alpha_grid, alpha_select

    def get_transformed_GP_output(self, gpr, x_vals):
        """Returns output of GP and transforms it, evaluating GP using predict_f at alpha values."""
        # Could use predict_y instead
        # But cannot unless have model for noise at new points, so just work with predict_f
        # If do have reason to work with predict_y, inherit this class and modify this method

        # For x_vals, need extra dimension so can work with multidimensional y
        # And need for creating GP input anyway
        x_vals = x_vals[:, None]
        gpr_pred = gpr.predict_f(
            np.concatenate(
                [x_vals, self.d_order_pred * np.ones((x_vals.shape[0], 1))], axis=1
            )
        )
        gpr_mu, gpr_std, gpr_conf_int = self.transform_func(
            x_vals,
            gpr_pred[0].numpy(),
            gpr_pred[1].numpy(),
        )
        # gpr_mu = self.transform_func(x_vals, gpr_pred[0].numpy())
        # gpr_std = self.transform_func(x_vals, np.sqrt(gpr_pred[1].numpy()))
        return gpr_mu, gpr_std, gpr_conf_int


class UpdateFuncBase(UpdateStopABC):
    """
    Base update function class defining structure and implementing basic methods.
    This class will be callable and will use the do_update() function to perform updates,
    which means that for new classes inheriting from this do_update() must be implemented.
    The update function should typically take two arguments: the GP model and the list of
    alpha (or x) input values that the model is based on.

    Parameters
    ----------
    show_plot : bool, default=False
        Whether or not to show a plot after each update
    save_plot : bool, default=False
        Whether or not to save a plot after each update
    save_dir : str or path-like, default='./'
        Directory to save figures.
    compare_func : callable, optional
        Function to compare to for plotting, like ground truth if it is known.
    """

    def __init__(
        self,
        show_plot=False,
        save_plot=False,
        save_dir="./",
        compare_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init just sets up how update function should behave
        # If want to change behavior on call, can adjust these variables
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.save_dir = save_dir
        self.compare_func = compare_func

    def do_plotting(self, x, y, err, alpha_list):
        """
        Plots output used to select new update point.
        err is expected to be length 2 list with upper and lower confidence intervals.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if self.compare_func is not None:
            compare_y = self.compare_func(x[:, None])
        # Need loop to handle multiple outputs if have them
        for k in range(y.shape[1]):
            ax.plot(x, y[:, k])
            ax.fill_between(x, err[0][:, k], err[1][:, k], alpha=0.2)
            if self.compare_func is not None:
                ax.plot(x, compare_y[:, k], "k--")
        # Use compare_func output to set range of plot
        if self.compare_func is not None:
            compare_min = np.min(compare_y)
            compare_max = np.max(compare_y)
            compare_range = compare_max - compare_min
            ax.set_ylim(
                (compare_min - 0.05 * compare_range, compare_max + 0.05 * compare_range)
            )
        # Plot points where collected data for GPR
        y_lims = ax.get_ylim()
        y_range = y_lims[1] - y_lims[0]
        ax.set_ylim((y_lims[0] - 0.10 * y_range, y_lims[1]))
        ax.plot(
            alpha_list,
            (y_lims[0] - 0.05 * y_range) * np.ones_like(alpha_list),
            marker="^",
            color="k",
            linestyle="",
        )
        if self.log_scale:
            ax.set_xlabel(r"log$_{10}$Alpha")
        else:
            ax.set_xlabel(r"Alpha")
        ax.set_ylabel(r"GP output")
        fig.tight_layout()
        if self.save_plot:
            # save_dir = os.path.split(data_list[-1].sim_info_files[0])[0]
            num_figs = len(glob.glob("%s/GP_v_alpha*.png" % self.save_dir))
            fig.savefig("%s/GP_v_alpha%i.png" % (self.save_dir, num_figs))
        if self.show_plot:
            plt.show()

    def do_update(self, gpr, alpha_list):
        raise NotImplementedError(
            "Must implement this function for specific update scheme"
        )

    def __call__(self, gpr, alpha_list):
        new_alpha, pred_mu, pred_std = self.do_update(gpr, alpha_list)

        if self.log_scale:
            new_alpha = 10.0 ** (new_alpha)

        return new_alpha, pred_mu, pred_std


# TODO: update structure to inherit docstrings.  Can simplify a bunch of stuff.


class UpdateALMbrute(UpdateFuncBase):
    """
    Performs active learning with a GPR to select new location for performing simulation.
    This is called "Active Learning Mackay" in the book by Grammacy (Surrogates, 2022).
    Selection is based on maximizing uncertainty over the interval, which is done with brute
    force evaluation on a grid of points (this is cheap compared to running simulations or
    training the GP model). It is possible to select a point that has already been run and
    add more data there, but will not move outside the range of data already collected.  A
    function for transforming the GPR prediction may also be provided, taking the inputs
    and prediction as arguments, in that order. This will not affect the GPR model, but
    will change the active learning outcomes, as will changing the derivative order of the
    outcome. Note that the transform should only involve addition or scaling (linear
    operations) such that the uncertainty can also be adjusted in the same way.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do_update(self, gpr, alpha_list):
        # Create grid of alpha values to interogate GP model and select new values
        alpha_grid, alpha_select = self.create_alpha_grid(alpha_list)

        # Obtain predictions and uncertainties at all grid points
        gpr_mu, gpr_std, gpr_conf = self.get_transformed_GP_output(gpr, alpha_select)

        # Plot if desired
        if self.save_plot or self.show_plot:
            self.do_plotting(alpha_select, gpr_mu, gpr_conf, alpha_list)

        # Find maximum uncertainty
        # But for multidimensional output, make relative to dimension-wise variance
        # For 1D data, won't change selection since divide by constant
        # But for multidimensional, makes dimensions comparable
        d_bool = gpr.data[0].numpy()[:, 1] == self.d_order_pred
        std_y = np.std(gpr.data[1].numpy()[d_bool, ...] * gpr.scale_fac, axis=0)
        max_err = np.max(gpr_std / std_y)

        # May have multiple equivalent maxima because uncertainty is flat over a region
        # In that case, take point in between
        # But watch out for case of multiple separate maximums or separate plateaus
        # Will take first maximum or plateau if multiple, but with a plateau takes ~halfway
        # For case of plateau, not possible to have "wiggles" due to smoothness conditions
        # (at least with an RBF kernel)
        max_inds = np.where((gpr_std / std_y) == max_err)
        # Select dimension with most maxima, breaking ties with first such dimension
        dim_vals, dim_counts = np.unique(max_inds[1], return_counts=True)
        dim_max = dim_vals[np.argmax(dim_counts)]
        max_inds = np.sort(max_inds[0][(max_inds[1] == dim_max)])
        if max_inds.size == 1:
            new_alpha = alpha_select[max_inds[0]]
            out_mu = gpr_mu[max_inds[0], ...]
            out_std = gpr_std[max_inds[0], ...]
        else:
            max_set = [max_inds[0]]
            for ind in max_inds[1:]:
                if ind == max_set[-1] + 1:
                    max_set.append(ind)
                else:
                    break
            new_ind = max_set[len(max_set) // 2]
            new_alpha = alpha_select[new_ind]
            out_mu = gpr_mu[new_ind, ...]
            out_std = gpr_std[new_ind, ...]

        return new_alpha, out_mu, out_std


class UpdateRandom(UpdateFuncBase):
    """
    Select point randomly along a grid based on previously sampled points.
    This does not require training a GP model, but one is trained anyway for plotting, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do_update(self, gpr, alpha_list):
        # Create grid of alpha values to interogate GP model and select new values
        alpha_grid, alpha_select = self.create_alpha_grid(alpha_list)

        # Obtain predictions and uncertainties at all grid points
        # Don't actually need to here, but nice to see progress
        gpr_mu, gpr_std, gpr_conf = self.get_transformed_GP_output(gpr, alpha_select)

        # Plot if desired
        if self.save_plot or self.show_plot:
            self.do_plotting(alpha_select, gpr_mu, gpr_conf, alpha_list)

        # Randomly select new alpha
        new_ind = np.random.choice(alpha_select.shape[0])
        new_alpha = alpha_select[new_ind]
        out_mu = gpr_mu[new_ind, ...]
        out_std = gpr_std[new_ind, ...]

        return new_alpha, out_mu, out_std


class UpdateSpaceFill(UpdateFuncBase):
    """
    Select point as far as possible from previously sampled points.
    This will just be halfway between for two points. For situations where
    multiple locations work equally well, locations are chosen randomly.
    This does not require training a GP model, but one is trained anyway for plotting, etc.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do_update(self, gpr, alpha_list):
        # Create grid of alpha values to interogate GP model and select new values
        alpha_grid, alpha_select = self.create_alpha_grid(alpha_list)

        # Obtain predictions and uncertainties at all grid points
        # Don't actually need to here, but nice to see progress
        gpr_mu, gpr_std, gpr_conf = self.get_transformed_GP_output(gpr, alpha_select)

        # Plot if desired
        if self.save_plot or self.show_plot:
            self.do_plotting(alpha_select, gpr_mu, gpr_conf, alpha_list)

        # Sort alpha_list and check size of interval in between pairs
        sorted_alpha = np.sort(alpha_list)
        if self.log_scale:
            sorted_alpha = np.log10(sorted_alpha)
        intervals = sorted_alpha[1:] - sorted_alpha[:-1]

        # Pick largest interval, breaking ties by random selection
        # Note avoiding round-off issues with isclose rather than ==
        max_int = np.max(intervals)
        max_int_inds = np.where(np.isclose(intervals, max_int))[0]
        sel_int_ind = np.random.choice(max_int_inds)

        # Take alpha as halfway point in this interval
        new_alpha = sorted_alpha[sel_int_ind] + 0.5 * intervals[sel_int_ind]

        # For outputs, return gpr predictions at closest point
        new_ind = np.argmin(abs(alpha_select - new_alpha))
        out_mu = gpr_mu[new_ind, ...]
        out_std = gpr_std[new_ind, ...]

        return new_alpha, out_mu, out_std


class UpdateAdaptiveIntegrate(UpdateFuncBase):
    """
    Select point as far as possible from previously sampled points, but within
    specified error tolerance based on model relative uncertainty predictions.
    If all values in the interval satisfy the tolerance, the furthest point from
    all others will be chosen, as in a space-filling update.

    Parameters
    ----------
    tol : float, default=0.005
        tolerance threshold to stay under when finding next point; this is
        defined as the relative uncertainty, or the GPR-predicted standard
        deviation divided by the absolute value of the GPR-predicted mean
    """

    def __init__(self, tol=0.005, **kwargs):
        super().__init__(**kwargs)
        self.tol = tol

    def do_update(self, gpr, alpha_list):
        # Create grid of alpha values to interogate GP model and select new values
        alpha_grid, alpha_select = self.create_alpha_grid(alpha_list)

        # Obtain predictions and uncertainties at all grid points
        # Don't actually need to here, but nice to see progress
        gpr_mu, gpr_std, gpr_conf = self.get_transformed_GP_output(gpr, alpha_select)

        # Plot if desired
        if self.save_plot or self.show_plot:
            self.do_plotting(alpha_select, gpr_mu, gpr_conf, alpha_list)

        # Determine relative uncertainties based on model
        rel_uncert = gpr_std / abs(gpr_mu)

        # If log_scale, convert alpha_list after making copy
        alpha_vals = np.array(alpha_list).copy()
        if self.log_scale:
            alpha_vals = np.log10(alpha_vals)

        # For each point in alpha_vals, find furthest distance within relative uncertainty
        max_ind = 0
        max_dist = -1
        for a_val in alpha_vals:
            close_ind = np.argmin(abs(alpha_select - a_val))
            # Make sure relative uncertainty at input value is below threshold
            # Not really needed since same as condition in while
            # But good to have two checks, and makes logic simpler since won't end up
            # changing max_ind or max_dist
            if np.any(rel_uncert[close_ind] >= self.tol):
                continue
            # Since know less than tolerance at this point, find closest point where crosses
            curr_inds = [close_ind, close_ind]
            while np.all(rel_uncert[curr_inds, :] < self.tol):
                if curr_inds[0] > 0:
                    curr_inds[0] -= 1
                if curr_inds[1] < (alpha_select.shape[0] - 1):
                    curr_inds[1] += 1
                # If reach end of interval without satisfying tolerance, break
                # Will handle how deals with this later
                if (curr_inds[0] == 0) and (
                    curr_inds[1] == (alpha_select.shape[0] - 1)
                ):
                    break
            # Find which index is furthest and its distance
            this_dists = abs(alpha_select[curr_inds] - alpha_select[close_ind])
            further_ind = np.argmax(this_dists)
            this_new_ind = curr_inds[further_ind]
            this_new_dist = this_dists[further_ind]
            if this_new_dist > max_dist:
                max_ind = this_new_ind
                max_dist = this_new_dist

        # Have to handle case where no current points satisfy tolerance
        if max_dist == -1:
            raise RuntimeError(
                "No points used to train GP model satisfy tolerance, meaning more simulation is needed at those points to perform adaptive sampling within tolerance."
            )

        # And handle case where an end-point is chosen (because all points satisfy tolerance)
        if (max_ind == 0) or (max_ind == (alpha_select.shape[0] - 1)):
            print(
                "Tolerance satisfied for all points in interval. Selecting new point with space-filling design."
            )
            # Will now pick largest interval
            sorted_alpha = np.sort(alpha_vals)
            intervals = sorted_alpha[1:] - sorted_alpha[:-1]
            max_int = np.max(intervals)
            max_int_inds = np.where(np.isclose(intervals, max_int))[0]
            sel_int_ind = np.random.choice(max_int_inds)
            # Take alpha as halfway point in this interval
            new_alpha = sorted_alpha[sel_int_ind] + 0.5 * intervals[sel_int_ind]
        # Otherwise, select new alpha as described
        else:
            new_alpha = alpha_select[max_ind]

        # For outputs, return gpr predictions at closest point
        new_ind = np.argmin(abs(alpha_select - new_alpha))
        out_mu = gpr_mu[new_ind, ...]
        out_std = gpr_std[new_ind, ...]

        return new_alpha, out_mu, out_std


class UpdateALCbrute(UpdateFuncBase):
    """
    EXPERIMENTAL! MAY BE USEFUL IN FUTURE WORK, BUT NOT NOW!

    Performs active learning with a GPR to select new location for performing simulation.
    This is called "Active Learning Cohn" in the book by Grammacy (Surrogates, 2022).
    Selection is based on maximizing INTEGRATED uncertainty, which is done with brute
    force evaluation on a grid of points (this is cheap compared to running simulations or
    training the GP model). It is possible to select a point that has already been run and add
    more data there, but will not move outside the range of data already collected. The
    provided data should be a list of DataWrapper objects. A function for transforming the
    GPR prediction may also be provided, taking the inputs and prediction as arguments, in
    that order. This will not affect the GPR model, but will change the active learning
    outcomes.

    ONLY EXPECTED TO WORK WITH A FULLY HETEROSCEDASTIC GP MODEL WHERE THERE IS A MODEL,
    PERHAPS A SEPARATE GP PROCESS, FOR THE BEHAVIOR OF THE NOISE ACROSS INPUT LOCATIONS.

    (the trivial case is that the noise does not vary with input location, in which case
    this will also work, but if providing heteroscedastic noise but no model to predict
    new noise, this will not work)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do_update(self, gpr, alpha_list):
        # Create grid of alpha values to interogate GP model and select new values
        alpha_grid, alpha_select = self.create_alpha_grid(alpha_list)

        # Obtain predictions and uncertainties at all grid points
        # Note that get_transformed_GP_output using predict_f, not predict_y
        # This follows IMSPE from Gramacy's book (Surrogates) and is a better idea
        # With predict_y, adding a new X point just gets the current noise GP prediction
        # Has no reason to change this, even though may be wrong
        # This then heavily biases away from these points because high uncertainty points
        # do not help reduce overall integrated uncertainty
        # Working with predict_f, still get benefit of estimating uncertainty at new point
        # But encourage better exploration
        gpr_mu, gpr_std, gpr_conf = self.get_transformed_GP_output(gpr, alpha_select)

        # Plot if desired
        if self.save_plot or self.show_plot:
            self.do_plotting(alpha_grid, gpr_mu, gpr_conf, alpha_list)

        # Now need to loop over grid of points and add each one to a GP model
        # Adding the alpha value and predicted uncertainty (through x-dependent likelihood)
        # Keeping parameters from original training
        orig_x, orig_y = gpr.data
        orig_x = orig_x.numpy()
        orig_y = orig_y.numpy()
        max_order = int(np.max(orig_x[:, 1]))
        orig_params = gpr.trainable_parameters
        new_int_std = np.zeros_like(alpha_select)
        for i, val in enumerate(alpha_select):
            # Augment original data
            this_x = np.vstack(
                [val * np.ones(max_order + 1), np.arange(max_order + 1)]
            ).T
            this_x = np.vstack([orig_x, this_x])
            this_y = np.vstack(
                [orig_y, np.ones((max_order + 1, 2))]
            )  # y is just a placeholder
            # Create a model with augmented data
            this_model = create_base_GP_model(this_x, this_y)
            # Set all trainable GP parameters based on original
            for j, tpar in enumerate(orig_params):
                this_model.trainable_parameters[j].assign(tpar.numpy())
            # Predict the uncertainty over the whole grid of points and integrate
            this_pred = this_model.predict_f(
                np.concatenate(
                    [
                        alpha_grid[:, None],
                        self.d_order_pred * np.ones((alpha_grid.shape[0], 1)),
                    ],
                    axis=1,
                )
            )
            # TODO: fix parameter definitions
            this_std = transform_func(  # noqa: F821
                alpha_grid, np.sqrt(np.squeeze(this_pred[1].numpy()))
            )
            new_int_std[i] = integrate.simpson(this_std, x=alpha_grid)

        # Identify point where get minimum integrated uncertainty
        new_ind = np.argmin(new_int_std)
        new_alpha = alpha_select[new_ind]

        # Randomly select new alpha
        new_alpha = np.random.choice(alpha_select)

        return new_alpha


class MetricBase:
    """
    Base class for structure of metrics used for stopping criteria.
    To create a metric, write the calc_metric method.
    Inputs can be history, x_vals, and gp. See below for definition of history.
    x_vals are the values at which the means and variances were evaluated, and
    gp is the actual Gaussian process model. If possible, should avoid using the GP
    in metric functions since will make them slow, but in some cases it is necessary,
    so pass it in for flexibility.

    Parameters
    ----------
    name : str
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    """

    def __init__(self, name, tol):
        """
        Inputs:
        name - name of metric
        tol - tolerance to define stopping criteria.
        """
        super().__init__()
        self.name = name
        self.tol = tol

    def _check_history(self, history):
        if history is None:
            raise ValueError("history is None.")
        elif len(history) != 2:
            raise ValueError(
                "history must be list of length 2 of GP means and variances evaluated with a series of GP models"
            )

    def calc_metric(self, history, x_vals, gp):
        raise NotImplementedError

    def __call__(self, history, x_vals, gp):
        self._check_history(history)
        return self.calc_metric(history, x_vals, gp)


class MaxVar(MetricBase):
    """Metric based on maximum variance of GP output."""

    def __init__(self, tol, name="MaxVar", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)

    def calc_metric(self, history, x_vals, gp):
        gp_std = history[1][-1, ...]
        max_var = np.max(gp_std)
        return max_var


class AvgVar(MetricBase):
    """
    Metric based on average variance of GP output.

    Parameters
    ----------
    name : str, default='AvgVar'
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    **kwargs
        Extrap arguments to :class:`MetricBase`
    """

    def __init__(self, tol, name="AvgVar", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)

    def calc_metric(self, history, x_vals, gp):
        gp_std = history[1][-1, ...]
        avg_var = np.average(gp_std)
        return avg_var


class MaxRelVar(MetricBase):
    """
    Metric based on maximum relative variance of GP output (actually std).


    Parameters
    ----------
    name : str, default='MaxRelVar'
        Name of this metric
    tol : float
        tolerance threshold for defining stopping
    threshold : float, default=1e-12
        checks to make sure GPR-predicted means have absolute value larger than
        this value so that do not divide by zero; if below this value, those
        points are ignored for purposes of calculating metric (set to zero)
    """

    def __init__(self, tol, threshold=1e-12, name="MaxRelVar", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)
        self.threshold = threshold

    def calc_metric(self, history, x_vals, gp):
        gp_mu = history[0][-1, ...].copy()
        gp_std = history[1][-1, ...].copy()
        # For stability, points where gp_mu is small need to be handled differently
        # So for points with gp_mu <= self.threshold, just checks if std is tol of threshold
        small_bool = abs(gp_mu) <= self.threshold
        gp_mu[small_bool] = self.threshold
        max_rel_var = np.max(gp_std / abs(gp_mu))
        return max_rel_var


class MaxRelGlobalVar(MetricBase, UpdateStopABC):
    """
    Metric based on maximum ratio of GP output variance to variance of data input to the
    GP (actually ratio of std devs).

    Parameters
    ----------
    name : str, default='MaxRelGlobalVar'
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    """

    def __init__(self, tol, name="MaxRelGlobalVar", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)

    def calc_metric(self, history, x_vals, gp):
        # Compute full std over y data, not just subtracting mean function as for GP data
        # d_bool = (gp.data[0].numpy()[:, 1] == self.d_order_pred)
        # std_y = np.std(gp.data[1].numpy()[d_bool, ...]*gp.scale_fac, axis=0)
        # Or approximate it with std over GPR-predicted mu values
        # (works better with transformations of output)
        std_y = np.std(history[0][-1, ...])
        gp_std = history[1][-1, ...].copy()
        max_rel_var = np.max(gp_std / std_y)
        return max_rel_var


class AvgRelVar(MetricBase):
    """
    Metric based on average relative variance of GP output.

    Parameters
    ----------
    name : str, default='AvgRelVar'
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    threshold : float, default=1e-12
        checks to make sure GPR-predicted means have absolute value larger than
        this value so that do not divide by zero; if below this value, those
        points are ignored for purposes of calculating metric (set to zero)
    """

    def __init__(self, tol, threshold=1e-12, name="AvgRelVar", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)
        self.threshold = threshold

    def calc_metric(self, history, x_vals, gp):
        gp_mu = history[0][-1, ...].copy()
        gp_std = history[1][-1, ...].copy()
        # For stability, points where gp_mu is small need to be handled differently
        # So for points with gp_mu <= self.threshold, just checks if std is tol of threshold
        small_bool = abs(gp_mu) <= self.threshold
        gp_mu[small_bool] = self.threshold
        avg_rel_var = np.average(gp_std / abs(gp_mu))
        return avg_rel_var


class MSD(MetricBase):
    """
    Metric based on mean squared deviation between GP model outputs.

    Parameters
    ----------
    name : str, default='MSD'
        Name of this metric
    tol : float
        tolerance threshold for defining stopping
    """

    def __init__(self, tol, name="MSD", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)

    def calc_metric(self, history, x_vals, gp):
        gp_mu = history[0][-1, ...]
        if history[0].shape[0] <= 1:
            gp_mu_prev = np.zeros_like(gp_mu)
        else:
            gp_mu_prev = history[0][-2, ...]
        msd = np.average((gp_mu - gp_mu_prev) ** 2)
        return msd


class MaxAbsRelDeviation(MetricBase):
    """
    Metric based on maximum absolute relative deviation between GP model outputs.

    Parameters
    ----------
    name : str, default='MaxAbsRelDev'
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    threshold : float, default=1e-12
        checks to make sure GPR-predicted means have absolute value larger than
        this value so that do not divide by zero; if below this value, those
        points are ignored for purposes of calculating metric (set to zero)
    """

    def __init__(self, tol, threshold=1e-12, name="MaxAbsRelDev", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)
        self.threshold = threshold

    def calc_metric(self, history, x_vals, gp):
        gp_mu = history[0][-1, ...].copy()
        # For stability, points where gp_mu is small need to be handled differently
        # So for points with gp_mu <= self.threshold, just checks if std is tol of threshold
        small_bool = abs(gp_mu) <= self.threshold
        gp_mu[small_bool] = self.threshold
        if history[0].shape[0] <= 1:
            gp_mu_prev = np.ones_like(gp_mu) * self.threshold
        else:
            gp_mu_prev = history[0][-2, ...].copy()
            small_prev = abs(gp_mu_prev) <= self.threshold
            gp_mu_prev[small_prev] = self.threshold
        dev = abs(gp_mu - gp_mu_prev)
        rel_max_dev = np.max(dev / abs(gp_mu))
        return rel_max_dev


class MaxAbsRelGlobalDeviation(MetricBase, UpdateStopABC):
    """
    Metric based on maximum absolute deviation between GP model outputs divided by the
    std of the data.

    Parameters
    ----------
    name : str, default='MaxAbsRelGlobalDeviation'
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    """

    def __init__(self, tol, name="MaxAbsRelGlobalDeviation", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)

    def calc_metric(self, history, x_vals, gp):
        # Compute full std over y data, not just subtracting mean function as for GP data
        # d_bool = (gp.data[0].numpy()[:, 1] == self.d_order_pred)
        # std_y = np.std(gp.data[1].numpy()[d_bool, ...]*gp.scale_fac, axis=0)
        # Or approximate it with std over GPR-predicted mu values
        # (works better with transformations of output)
        std_y = np.std(history[0][-1, ...])
        gp_mu = history[0][-1, ...].copy()
        if history[0].shape[0] <= 1:
            gp_mu_prev = np.zeros_like(gp_mu)
        else:
            gp_mu_prev = history[0][-2, ...].copy()
        dev = abs(gp_mu - gp_mu_prev)
        rel_max_dev = np.max(dev / std_y)
        return rel_max_dev


class AvgAbsRelDeviation(MetricBase):
    """
    Metric based on average absolute relative deviation between GP model outputs.

    Parameters
    ----------
    name : str, default='AvgAbsRelDev'
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    """

    def __init__(self, tol, threshold=1e-12, name="AvgAbsRelDev", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)
        self.threshold = threshold

    def calc_metric(self, history, x_vals, gp):
        gp_mu = history[0][-1, ...].copy()
        # For stability, points where gp_mu is small need to be handled differently
        # So for points with gp_mu <= self.threshold, just checks if std is tol of threshold
        small_bool = abs(gp_mu) <= self.threshold
        gp_mu[small_bool] = self.threshold
        if history[0].shape[0] <= 1:
            gp_mu_prev = np.ones_like(gp_mu) * self.threshold
        else:
            gp_mu_prev = history[0][-2, ...].copy()
            small_prev = gp_mu_prev <= self.threshold
            gp_mu_prev[small_prev] = self.threshold
        dev = abs(gp_mu - gp_mu_prev)
        rel_avg_dev = np.average(dev / abs(gp_mu))
        return rel_avg_dev


class ErrorStability(MetricBase, UpdateStopABC):
    """
    Implements the stopping metric introduced by Ishibashi and Hino (2021).
    Note that for this metric, also inherits UpdateStopABC, so has its own
    parameters for log_scale, d_order_pred, and transform_func, that are
    separate from the StopCriteria it's used in.
    Note that tol should be between 0 and 1, likely at 0.1 or below.
    And not that implementation here, transform_func can only be a
    linear transformation (scale and/or shift) of the GPR output (even though
    generally the transform_func can be more complicated).

    Parameters
    ----------
    name : str
        name of this metric
    tol : float
        tolerance threshold for defining stopping
    """

    def __init__(self, tol, name="ErrorStability", **kwargs):
        super().__init__(tol=tol, name=name, **kwargs)

        # Need to set up normalization - will just use first r calculated
        self.r1 = None

    def calc_metric(self, history, x_vals, gp):
        # Get input data points for current active learning step (current GP model)
        input_x = gp.data[0].numpy()
        input_y = gp.data[1].numpy()
        input_x = np.concatenate(
            [input_x[:, :1] / gp.x_scale_fac, input_x[:, 1:]], axis=-1
        )
        input_y = input_y * (gp.x_scale_fac ** input_x[:, 1:])
        input_y = input_y * gp.scale_fac
        input_cov = gp.likelihood.cov.copy()
        input_cov = input_cov * gp.x_scale_fac ** (
            np.add(*np.meshgrid(input_x[:, 1:], input_x[:, 1:]))
        )
        input_cov = input_cov * (
            np.expand_dims(
                gp.scale_fac, axis=tuple(range(gp.scale_fac.ndim, input_cov.ndim))
            )
            ** 2
        )

        # And select out x points at order we care about
        d_bool = input_x[:, 1] == self.d_order_pred
        pred_x = input_x[d_bool, :]

        # Should only use this metric if have at least 3 points
        # Will be very unstable before then - since "bounded", return 1
        if pred_x.shape[0] <= 2:
            return 1.0

        # Get GP output with full covariance at all input data points
        mu_curr, cov_curr = gp.predict_f(pred_x, full_cov=True)

        # Need to handle application of transform_func carefully
        # y is fine because linearly transformed, but dependence on x is tricky
        # Assume transformation only scales y, so may just have different scaling based on x
        # For means, can just apply transformation
        # But for covariance matrix, need all products of scaling factors
        # Note that transform_func is more general, but here assume does not use variance
        # (otherwise will break since make third input  of just 1.0)
        # and only linearly changes the mean based on x
        # (and remember to only take first output of transform_func, which is mean/median)
        mu_curr = self.transform_func(pred_x[:, :1], mu_curr.numpy(), 1.0)[0]
        transform_scale = self.transform_func(
            pred_x[:, :1], np.ones_like(pred_x[:, :1]), 1.0
        )[0]
        cov_curr = cov_curr * (transform_scale * transform_scale.T)

        # Next need to create new GP with SAME PARAMETERS (so same prior)
        # But should exclude most recently added inputs
        max_order = np.max(input_x[:, 1])
        prev_input = (
            input_x[: -int(max_order + 1), :],
            input_y[: -int(max_order + 1), :],
            input_cov[:, : -int(max_order + 1), : -int(max_order + 1)],
        )
        # SHOULD REALLY MAKE A GP_Model CLASS THAT KEEPS TRACK OF BASE PARAMETERS
        # CAN CALL TO CREATE A NEW GP MODEL, USING create_GPR()
        # IF ALSO GIVE THAT A from_data() METHOD SO THAT CAN CREATE COPY WITH NEW DATA EASILY
        # Can ALSO consider adding function to scale gp input/output to the GP model
        # Would simplify much of __init__ and predict_f and would be helpful outside as well
        # Actually, everything can be recovered from GP model except kernel
        # Well, can get kernel, just not inputs to create it
        # But here, want kernel with same parameters, which get by passing kernel object
        # (that makes prior the same)
        prev_gp = create_base_GP_model(prev_input, kernel=gp.kernel)
        # Even though passed in kernel, need to set trainable parameters (for likelihood)
        gp_params = [p.numpy() for p in gp.trainable_parameters]
        for i, tpar in enumerate(gp_params):
            prev_gp.trainable_parameters[i].assign(tpar)
        # And make prediction with GP with only previous inputs, but at all current inputs
        mu_prev, cov_prev = prev_gp.predict_f(pred_x, full_cov=True)
        mu_prev = self.transform_func(pred_x[:, :1], mu_prev.numpy(), 1.0)[0]
        cov_prev = cov_prev * (transform_scale * transform_scale.T)

        # For metric, calculate the KL divergence between predicted distributions
        inv_cov_curr = np.linalg.inv(
            cov_curr
        )  # May need to make more numerically stable...
        inv_cov_prev = np.linalg.inv(cov_prev)  # Like maybe with Cholesky
        det_cov_curr = np.linalg.det(cov_curr)
        det_cov_prev = np.linalg.det(cov_prev)
        diff_cp = np.expand_dims((mu_curr - mu_prev).T, -1)
        diff_pc = np.expand_dims((mu_prev - mu_curr).T, -1)
        KL_curr_prev = 0.5 * (
            np.trace(inv_cov_curr @ cov_prev, axis1=-2, axis2=-1)
            + np.squeeze(np.transpose(diff_cp, [0, 2, 1]) @ inv_cov_curr @ diff_cp)
            - mu_curr.shape[0]
            + np.log(det_cov_curr)
            - np.log(det_cov_prev)
        )
        KL_prev_curr = 0.5 * (
            np.trace(inv_cov_prev @ cov_curr, axis1=-2, axis2=-1)
            + np.squeeze(np.transpose(diff_pc, [0, 2, 1]) @ inv_cov_prev @ diff_pc)
            - mu_prev.shape[0]
            + np.log(det_cov_prev)
            - np.log(det_cov_curr)
        )
        # For multiple independent output dimensions, need to sum over KLs
        KL_curr_prev = np.sum(KL_curr_prev)
        KL_prev_curr = np.sum(KL_prev_curr)

        # Will run into issues (NaNs) in lambertw if KL is too close to zero...
        # So just adding very small number here
        KL_curr_prev += 1e-20
        KL_prev_curr += 1e-20

        # Calculate "r" for both directions
        lamb_curr_prev = special.lambertw((KL_curr_prev - 1.0) / np.exp(1)).real
        r_curr_prev = np.exp(lamb_curr_prev + 1.0) - 1.0
        lamb_prev_curr = special.lambertw((KL_prev_curr - 1.0) / np.exp(1)).real
        r_prev_curr = np.exp(lamb_prev_curr + 1.0) - 1.0

        # If have not defined normalization yet, do now - hopefully going from 2 to 3 points
        # Note that if very accurate with 2 points, may take overly long to converge
        # So technically not using r1, really using r3 by Ishibashi and Hino's definition
        if self.r1 is None:
            self.r1 = r_curr_prev + r_prev_curr

        out = (r_curr_prev + r_prev_curr) / self.r1
        return out


class MaxIter(MetricBase):
    """
    Metric that always returns False so that will reach maximum number of iterations.
    This can be used with or without other metrics to reach maximum iterations since
    all metrics must be True to reach stopping criteria. Note that do not need to (and
    should not) set the tolerance here.

    Parameters
    ----------
    name : str
        name of this metric
    """

    def __init__(self, name="MaxIter", **kwargs):
        super().__init__(tol=1.0, name=name, **kwargs)

    def calc_metric(self, history, x_vals, gp):
        return self.tol + 1.0  # Always bigger than tol


class StopCriteria(UpdateStopABC):
    """
    Class that calculates metrics used to determine stopping criteria for active learning.
    The key component of this class is a list of metric functions which have names and define
    tolerances. All of the metrics must be less than the tolerance to trigger stopping.

    To perform calculations of metrics, this class keeps track of the history of the GPR
    predictions (necessary for metrics based on deviations since past iterations). This
    history object is stored as a list of array objects, specifically the GPR mean (list
    index 0) and GPR standard deviations (list index 1) with rows in each array being
    different iterations. So GPR prediction of a mean at iteration 3 would be
    history[0][3, ...].

    Parameters
    ----------
    metric_funcs : dict
        A dictionary {name: function} of metric names and associated functions;
        this will be looped over with metrics calculated to determine stopping
    """

    def __init__(self, metric_funcs, **kwargs):
        """
        Inputs:
        metric_funcs - dictionary of (name, function) pairs; just nice to have names.
        """
        # Make sure avoid repeats is False for reproducibility
        kwargs["avoid_repeats"] = False
        super().__init__(**kwargs)

        self.metric_funcs = metric_funcs

        # For any metrics with their own log_scale, etc. attributes, force to
        # be consistent with StopCriteria
        for m in self.metric_funcs:
            if issubclass(type(m), UpdateStopABC):
                m.d_order_pred = self.d_order_pred
                m.transform_func = self.transform_func
                m.log_scale = self.log_scale
                m.avoid_repeats = self.avoid_repeats

        # For many metrics, will need history of past GPR outputs, so maintain this
        # Start as None and add arrays of predictions and uncertainties
        # Can be used generally by any metric, so write metric functions to take history
        self.history = None

    def compute_metrics(self, alpha_grid, history=None, gpr=None):
        """
        Uses current history (default) or one provided to compute all metrics.
        Must provide grid of alpha values as well to input to metrics.
        """
        if history is None:
            history = self.history
        out_dict = {}
        tol_bools = []
        # Compute metrics, which should be functions that take the history and alpha values
        # Keep track of whether tolerance reached for each
        for m in self.metric_funcs:
            this_metric = m(history, alpha_grid, gpr)
            out_dict[m.name] = this_metric
            out_dict[m.name + "_tol"] = m.tol
            tol_bools.append(this_metric <= m.tol)
        return tol_bools, out_dict

    def __call__(self, gpr, alpha_list):
        # Create grid of alpha values to interogate GP model
        # In case avoid_repeats gets set to True somehow, still only take alpha_grid
        # This should not get randomized
        alpha_grid, _ = self.create_alpha_grid(alpha_list)

        # Obtain predictions and uncertainties at all grid points
        gpr_mu, gpr_std, gpr_conf = self.get_transformed_GP_output(gpr, alpha_grid)

        # Update history
        if self.history is None:
            self.history = [gpr_mu[None, ...], gpr_std[None, ...]]
        else:
            self.history[0] = np.concatenate(
                [self.history[0], gpr_mu[None, ...]], axis=0
            )
            self.history[1] = np.concatenate(
                [self.history[1], gpr_std[None, ...]], axis=0
            )

        # Calculate all metrics and whether or not reached tolerance
        tol_bools, out_dict = self.compute_metrics(alpha_grid, gpr=gpr)

        # Only stop if all metrics have simultaneously reached tolerances
        return np.all(tol_bools), out_dict


def active_learning(
    init_states,
    sim_wrapper,
    update_func,
    base_dir="",
    stop_criteria=None,
    max_iter=10,
    alpha_name="alpha",
    log_scale=False,
    max_order=4,
    gp_base_kwargs=None,
    num_state_repeats=1,
    save_history=False,
    use_predictions=False,
):
    """
    Continues adding new points with active learning by running simulations until the
    specified tolerance is reached or the maximum number of iterations is achieved.

    Parameters
    ----------
    init_states : list of :class:`DataWrapper`
    sim_wrapper : :class:`SimWrapper`
        Object for running simulations.
    update_func : callable
        For selecting the next state point.
    base_dir : string
        File path.
        based directory in which active learning run performed and outputs generated
    stop_criteria : callable, optional
        callable taking GP to determine if should stop
    max_iter : int, default=10
        maximum number of iterations to run (new points to add)
    alpha_name : str, default='alpha'
        the changed parameter; MUST match input name for sim_inputs
    log_scale : bool, default=False
        whether or not to use a log scale for alpha
    max_order : int, default=4
        Maximum order to use for derivative observations
    gp_base_kwargs : dict, optional
        dictionary of keyword arguments for create_base_GP_model
        (allows for more advanced specification of GP model)
    num_state_repeats : int, default=1
        Number of simulations to run for each state
        (can help to estimate uncertainty as long as independent)
    save_history : bool, default=False
        If stop_criteria is not None, saves it's history (all
        predictions at each step of active learning protocol).
    use_predictions : bool, default=False
        Whether or not sim_wrapper needs predictions from
        the GP model or not; if True, passes keyword arguments of
        model_pred and model_std (model predicted mu and std) to
        sim_wrapper

    Returns
    -------
    data_list : list of :class:`DataWrapper`
        List of DataWrapper objects describing how to load data (can be used to build states and create_GPR to generate GP model)
    train_history : dict
        Dictionary of information about results at each training
        iteration, like GP predictions, losses, parameters, etc.
    """

    if gp_base_kwargs is None:
        gp_base_kwargs = {}

    if log_scale ^ update_func.log_scale:  # Bitwise XOR
        print(
            "WARNING: Usage of log scale in x for GPs is set to %s but %s for updates. Typically these should match, so make sure you know what you're doing!"
            % (str(log_scale), str(update_func.log_scale))
        )
    if stop_criteria is not None:
        if log_scale ^ stop_criteria.log_scale:
            print(
                "WARNING: Usage of log scale in x for GPs is set to %s but %s for stopping criteria. Typically these should match, so make sure you know what you're doing!"
                % (str(log_scale), str(stop_criteria.log_scale))
            )

    data_list = [None] * len(init_states)
    for i, state in enumerate(init_states):
        if isinstance(state, DataWrapper) or issubclass(type(state), DataWrapper):
            data_list[i] = state
        elif isinstance(state, (int, float)):
            # Run simulation and return DataWrapper object for this state
            # Multiple simulation repeats will be performed in parallel
            data_list[i] = sim_wrapper.run_sim(
                f"{base_dir}/{alpha_name}_{state:f}",
                state,
                n_repeats=num_state_repeats,
            )

    # Will need to keep track of alpha values
    alpha_list = [dat.beta for dat in data_list]

    print("\n\nInitial %s values: " % alpha_name, alpha_list)

    # Also nice to keep track of loss and parameter values
    # Results in more robust parameter optimization, too
    train_history = {"loss": [], "params": []}

    # Also add metrics to training history if have stopping criteria
    if stop_criteria is not None:
        for m in stop_criteria.metric_funcs:
            train_history[m.name] = []

    # Loop over iterations, breaking if reach stopping criteria
    # Go to max_iter+1 so that have final model and its predictions
    for i in range(max_iter + 1):
        # Create GP model with current information, first building ExtrapModel objects
        state_list = [dat.build_state(max_order=max_order) for dat in data_list]
        if i == 0:
            this_GP = create_GPR(
                state_list, log_scale=log_scale, base_kwargs=gp_base_kwargs
            )
        else:
            this_GP = create_GPR(
                state_list,
                log_scale=log_scale,
                base_kwargs=gp_base_kwargs,
                start_params=train_history["params"][-1],
            )
        print("\nCurrent GP info: ")
        gpflow.utilities.print_summary(this_GP)
        # Add to training history
        train_history["loss"].append(this_GP.training_loss().numpy())
        train_history["params"].append(
            [p.numpy() for p in this_GP.trainable_parameters]
        )

        # Check if should stop
        if stop_criteria is not None:
            stop_bool, stop_metrics = stop_criteria(this_GP, alpha_list)
            # Add to training history
            for m in stop_metrics.keys():
                if "tol" not in m:
                    train_history[m].append(stop_metrics[m])
            if stop_bool:
                print("\nStopping criteria satisfied with stopping metrics of: ")
                print(stop_metrics)
                print("\n")
                break
            else:
                print("\nCurrent stopping metrics: ")
                print(stop_metrics)

        if i == max_iter:
            # Don't do update on this loop, just break
            print("\nReached maximum iterations of %i without convergence\n" % max_iter)
            break

        # If stopping criteria not satisfied, select new point
        new_alpha, new_mu, new_std = update_func(this_GP, alpha_list)
        if use_predictions:
            new_model_info = {"model_pred": new_mu, "model_std": new_std}
        else:
            new_model_info = {}

        # Run simulations for current data
        this_data = sim_wrapper.run_sim(
            f"{base_dir}/{alpha_name}_{new_alpha:f}",
            new_alpha,
            n_repeats=num_state_repeats,
            **new_model_info,
        )

        # If we're adding data to a previously sampled state, need to replace in data_list
        # Just in case values SLIGHTLY different, use isclose to check
        # May happen if restart run and recalculate betas as a result
        if np.any(np.isclose(alpha_list, new_alpha)):
            replace_ind = np.where(np.isclose(alpha_list, new_alpha))[0][0]
            data_list[replace_ind] = this_data
        else:
            data_list.append(this_data)
            alpha_list.append(new_alpha)

        print("\nAfter %i updates, %s values are: " % (i + 1, alpha_name))
        print(alpha_list)

    if save_history and (stop_criteria is not None):
        for key in train_history.keys():
            train_history[key] = np.array(train_history[key])
        np.savez(
            "%s/active_history.npz" % base_dir,
            pred_mu=stop_criteria.history[0],
            pred_std=stop_criteria.history[1],
            alpha=np.array(alpha_list),
            **train_history,
        )

    return data_list, train_history

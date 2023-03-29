"""
Routines GPR interpolation models
"""


import gpflow
import numpy as np
import sympy as sp
import tensorflow as tf
import xarray as xr
from gpflow.ci_utils import ci_niter

# from .models import StateCollection
from .core.cached_decorators import gcached
from .core.stack import GPRData, StackedDerivatives, multiindex_to_array

__all__ = ("GPRData", "GPRModel", "StackedDerivatives", "factory_gprmodel")

# First define classes needed for a GPR model
# A general derivative kernel based on a sympy expression


class DerivativeKernel(gpflow.kernels.Kernel):
    """
        Creates a kernel that can be differentiated based on a sympy expression for the kernel.
    Given observations that are tagged with the order of the derivative, builds the appropriate
    kernel. Be warned that your kernel_expr will not be checked to make sure it is positive
    definite, stationary, etc.

    There are rules for kernel_expr and kernel_params that guarantee consistency. First, the
    variable names supplied as keys to kernel_params should match the symbol names in
    kernel_expr. Symbol names for the inputs should be 'x1' and 'x2' (ignoring case). We could
    accept anything as long as 2 symbols are left over after those in kernel_params, but these
    two rules will guarantee that nothing breaks.

    Currently, everything is only intended to work with 1D observables.

        Inputs:
               kernel_expr - sympy expression for the kernel that can be differentiated - must
                             have at least 2 symbols
                             (symbol names should be 'x1' and 'x2', case insensitive, if have
                              only 2)
               obs_dims - number of dimensions for observable input
                          (input should be twice this with obs_dims values then obs_dims
                           derivative labels each row)
               kernel_params - a dictionary of kernel parameters that can be optimized by
                               tensorflow
                               (key should be name, then references list with value then
                                another dict with kwargs for gpflow.Parameter, i.e.,
                                {'variance', [1.0, {'transform':gpflow.utilities.positive()}]}
                                so if you don't want to set any kwargs, just pass empty
                                dictionary
                               NOTE THAT THE KEYS MUST MATCH THE SYMBOL NAMES IN kernel_expr
                               OTHER THAN 'x1' and 'x2'
                               Default is empty dict, so will mine names from kernel_expr and
                               set all parameters to 1.0
    """

    def __init__(
        self, kernel_expr, obs_dims, kernel_params={}, active_dims=None, **kwargs
    ):
        if active_dims is not None:
            print("active_dims set to: ", active_dims)
            print("This is not implemented in this kernel, so setting to 'None'")
            active_dims = None

        super().__init__(active_dims=active_dims, **kwargs)

        # Get the sympy expression for the kernel
        self.kernel_expr = kernel_expr
        # Now need to mine it a little bit to get the adjustable parameters and input variables
        expr_syms = tuple(kernel_expr.free_symbols)
        # Require that have two symbols called x1 and x2, with the rest being parameters
        self.x_syms = []
        self.param_syms = []
        for s in expr_syms:
            if s.name.casefold() == "x1" or s.name.casefold() == "x2":
                self.x_syms.append(s)
            else:
                self.param_syms.append(s)
        # Make sure to sort so clearly define x1 and x2
        list(self.x_syms).sort(key=lambda s: s.name)
        # If have no other symbols (i.e. parameters) there is nothing to optimize!
        if len(self.param_syms) == 0:
            raise ValueError(
                "Provided kernel expression only takes inputs x1 and x2, "
                + "no optimizable parameters!"
            )
        # Make sure that parameters here match those in kernel_params, if it's provided
        if bool(kernel_params):
            if (
                list([s.name for s in self.param_syms]).sort()
                != list(kernel_params.keys()).sort()
            ):
                raise ValueError(
                    "Symbol names in kernel_expr must match keys in " + "kernel_params!"
                )
            # If they are the same, obtain parameter values from kernel_params dictionary
            # Need to set as gpflow Parameter objects so can optimize over them
            for key, val in kernel_params.items():
                setattr(self, key, gpflow.Parameter(val[0], **val[1]))

        # If kernel_params is not provided, set everything to 1.0 by default
        else:
            for s in self.param_syms:
                setattr(self, s.name, gpflow.Parameter(1.0))

        # Set number of observable dimensions
        self.obs_dims = obs_dims

    # Define ARD behavior (if ever want multiple dimensions with different lengthscales)
    @property
    def ard(self) -> bool:
        """
        Whether ARD behavior is active, following gpflow.kernels.Stationary
        """
        return self.lengthscales.shape.ndims > 0

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        x1, d1 = self._split_x_into_locs_and_deriv_info(X)
        x2, d2 = self._split_x_into_locs_and_deriv_info(X2)

        self.x1, self.d1 = x1, d1
        self.x2, self.d2 = x2, d2

        # Output should be a tensor that is len(X) by len(X2) - at least in 1D, not
        # sure what to do otherwise
        # And must be traceable with tensorflow's autodifferentiation
        # (in the inherited kernel parameters)

        # Want full list of all combinations of derivative pairs
        # Definitely only works for 1D data because of way reshaping
        expand_d1 = tf.reshape(
            tf.tile(d1, (1, d2.shape[0])), (d1.shape[0] * d2.shape[0], -1)
        )
        expand_d2 = tf.tile(d2, (d1.shape[0], 1))
        deriv_pairs = tf.stack([expand_d1, expand_d2], axis=1)

        # For convenience, do same with x, but no need to stack
        # Sort of same idea as creating a mesh grid
        expand_x1 = tf.reshape(
            tf.tile(x1, (1, x2.shape[0])), (x1.shape[0] * x2.shape[0], -1)
        )
        expand_x2 = tf.tile(x2, (x1.shape[0], 1))

        # Now need UNIQUE derivative pairs because will be faster to loop over
        unique_pairs = np.unique(deriv_pairs, axis=0)

        # Loop over unique pairs, tracking indices and kernel values for pairs
        k_list = []
        inds_list = []
        for pair in unique_pairs:
            # get the right indices
            this_inds = tf.cast(
                tf.where(tf.reduce_all(deriv_pairs == pair, axis=1))[:, :1], tf.int32
            )
            # # use sympy to obtain right derivative
            # this_expr = sp.diff(
            #     self.kernel_expr,
            #     self.x_syms[0],
            #     int(pair[0]),
            #     self.x_syms[1],
            #     int(pair[1]),
            # )
            # # get lambdified function compatible with tensorflow
            # this_func = sp.lambdify(
            #     (self.x_syms[0], self.x_syms[1], *self.param_syms),
            #     this_expr,
            #     modules="tensorflow",
            # )
            this_func = self._lambda_kernel(int(pair[0]), int(pair[1]))

            # plug in our values for the derivative kernel
            k_list.append(
                this_func(
                    tf.gather_nd(expand_x1, this_inds),
                    tf.gather_nd(expand_x2, this_inds),
                    *[getattr(self, s.name) for s in self.param_syms],
                )
            )
            # also keep track of indices so can dynamically stitch back together
            inds_list.append(this_inds)

        # Stitch back together
        k_list = tf.dynamic_stitch(inds_list, k_list)

        # Reshape to the correct output - will only really work for 1D, I think
        k_mat = tf.reshape(k_list, (x1.shape[0], x2.shape[0]))
        return k_mat

    def K_diag(self, X):
        # Same as for K but don't need every combination, just every x with itself
        x1, d1 = self._split_x_into_locs_and_deriv_info(X)
        unique_d1 = np.unique(d1)
        k_list = []
        inds_list = []
        for d in unique_d1:
            this_inds = tf.cast(tf.where(d1 == d)[:, :1], tf.int32)
            # this_expr = sp.diff(
            #     self.kernel_expr, self.x_syms[0], int(d), self.x_syms[1], int(d)
            # )
            # this_func = sp.lambdify(
            #     (self.x_syms[0], self.x_syms[1], *self.param_syms),
            #     this_expr,
            #     modules="tensorflow",
            # )
            this_func = self._lambda_kernel(int(d), int(d))

            k_list.append(
                this_func(
                    tf.gather_nd(x1, this_inds),
                    tf.gather_nd(x1, this_inds),
                    *[getattr(self, s.name) for s in self.param_syms],
                )
            )
            inds_list.append(this_inds)

        k_list = tf.dynamic_stitch(inds_list, k_list)
        k_diag = tf.reshape(k_list, (x1.shape[0],))
        return k_diag

    @gcached(prop=False)
    def _lambda_kernel(self, d1, d2):
        expr = sp.diff(self.kernel_expr, self.x_syms[0], d1, self.x_syms[1], d2)

        return sp.lambdify(
            (self.x_syms[0], self.x_syms[1], *self.param_syms),
            expr,
            modules="tensorflow",
        )

    def _split_x_into_locs_and_deriv_info(self, x):
        """Splits input into actual observable input and derivative labels"""
        locs = x[:, : self.obs_dims]
        grad_info = x[:, -self.obs_dims :]
        return locs, grad_info


# A custom GPFlow likelihood with heteroscedastic Gaussian noise
# Comes from GPFlow tutorial on this subject
class HeteroscedasticGaussian(gpflow.likelihoods.Likelihood):
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in
        # the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations
        # and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is
        # not actually needed.
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.
    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError


class combined_loss:
    """Convenience function for training all output dimension in parallel with sum of losses."""

    def __init__(self, loss_list):
        self.loss_list = loss_list

    def __call__(self):
        return tf.reduce_sum([loss() for loss in self.loss_list])


# Now can construct a model class inheriting from StateCollection


class GPRModel:
    """
    perform gaussian process regression

    Parameters
    ----------
    data : stack.GPRData
        data object to analyze
    kernel_expr : sympy expression
    kernel_params : dict
    """

    def __init__(self, data, kernel_expr, kernel_params={}):
        self.data = data
        self.kernel_expr = kernel_expr
        self.kernel_params = kernel_params

    @gcached(prop=False)
    def kern(self, out_dim):
        return [
            DerivativeKernel(
                self.kernel_expr,
                1,  # for now, obs_dims is always 1 while figure out math
                kernel_params=self.kernel_params,
            )
            for _ in range(out_dim)
        ]

    @gcached(prop=False)
    def het_gauss(self, out_dim):
        return [HeteroscedasticGaussian() for _ in range(out_dim)]

    @gcached(prop=False)
    def gp_params(self, order):
        x, ys = self.data.array_data(order=order)
        out_dim = len(ys)

        kernels = self.kern(out_dim)
        likelihoods = self.het_gauss(out_dim)

        # Not sure about adding data, but can reuse this to train some more...
        gp = [
            gpflow.models.VGP(
                (x, y),
                kernel=kernel,
                likelihood=likelihood,
                num_latent_gps=1,
            )
            for y, kernel, likelihood in zip(ys, kernels, likelihoods)
        ]
        # To train all models over all dimension in parallel, create sum over losses
        tot_loss = combined_loss([g.training_loss for g in gp])

        # Make some parameters fixed
        variational_params = []
        trainable_params = []
        for g in gp:
            gpflow.set_trainable(g.q_mu, False)
            gpflow.set_trainable(g.q_sqrt, False)
            variational_params.append((g.q_mu, g.q_sqrt))
            trainable_params.append(g.trainable_variables)
        natgrad = gpflow.optimizers.NaturalGradient(gamma=1.0)
        adam = tf.optimizers.Adam(
            learning_rate=0.5
        )  # Can be VERY aggressive with learning

        return {
            "gp": gp,
            "tot_loss": tot_loss,
            "variational_params": variational_params,
            "trainable_params": trainable_params,
            "natgrad": natgrad,
            "adam": adam,
        }

    def train(self, order=None, opt_steps=100, **kws):
        if order is None:
            order = self.data.order

        params = self.gp_params(order=order)

        natgrad = params["natgrad"]
        adam = params["adam"]

        tot_loss = params["tot_loss"]
        variational_params = params["variational_params"]
        trainable_params = params["trainable_params"]

        # Run optimization
        for _ in range(ci_niter(opt_steps)):
            # Training is extremely slow for vector observables with large dimension
            # Seems to mainly be because natgrad requires matrix inversion
            natgrad.minimize(tot_loss, variational_params)
            # Even running loop, as below, does not see to speed things up, though...
            # So not exactly sure why so much slower
            # for i in range(out_dim):
            #    natgrad.minimize(gp[i].training_loss, [variational_params[i]])
            adam.minimize(tot_loss, trainable_params)

        return self

    def predict(self, alpha, order=None, unstack=False, drop_order=True):
        if order is None:
            order = self.data.order

        gp = self.gp_params(order=order)["gp"]
        xindex = self.data.xindexer_from_arrays(**{self.data.alpha_name: alpha})
        x_pred = multiindex_to_array(xindex)

        out = np.array([np.hstack(g.predict_f(x_pred)) for g in gp])
        # out has form: out[ystack, stack, stats_dim]

        # wrap output
        template = self.data.stacked(order=order)
        xstack_dim, ystack_dim, stats_dim = (
            self.data.xstack_dim,
            self.data.ystack_dim,
            self.data.stats_dim,
        )

        coords = {xstack_dim: xindex}
        for name in [ystack_dim, stats_dim]:
            if name in template.indexes:
                coords[name] = template.indexes[name]

        xout = xr.DataArray(
            out, dims=[ystack_dim, xstack_dim, stats_dim], coords=coords
        )

        if unstack:
            xout = xout.unstack(xstack_dim).unstack(ystack_dim)

        if drop_order:
            xout = xout.sel(**{self.data.order_dim: 0})

        return xout


def factory_gprmodel(data, **kws):
    """
    factory function to create GPR model for beta expansion

    Parameters
    ----------
    states : StateCollection of ExtrapModel objects
    **kws : additional keyword arguments to pass to the model

    Returns
    -------
    gprmodel : GPRModel object
    """

    # Define RBF kernel expression and parameters
    var = sp.symbols("var")
    l = sp.symbols("l")  # noqa: E741
    x1 = sp.symbols("x1")
    x2 = sp.symbols("x2")
    rbf_kern_expr = var * sp.exp(-0.5 * (x1 / l - x2 / l) ** 2)

    rbf_params = {
        "var": [1.0, {"transform": gpflow.utilities.positive()}],
        "l": [1.0, {"transform": gpflow.utilities.positive()}],
    }
    return GPRModel(data, rbf_kern_expr, rbf_params, **kws)

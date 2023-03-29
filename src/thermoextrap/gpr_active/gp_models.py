"""
Models for Gaussian process regression (:mod:`~thermoextrap.gpr_active.gp_models`)
----------------------------------------------------------------------------------
"""
from typing import Any, Optional

import gpflow
import numpy as np
import sympy as sp
import tensorflow as tf
from gpflow import logdensities
from scipy import optimize


# TODO: Bunch of cleanup here
# First define classes needed for a GPR model
# A general derivative kernel based on a sympy expression
class DerivativeKernel(gpflow.kernels.Kernel):
    """
    Creates a kernel that can be differentiated based on a sympy expression for
    the kernel. Given observations that are tagged with the order of the
    derivative, builds the appropriate kernel. Be warned that your kernel_expr
    will not be checked to make sure it is positive definite, stationary, etc.

    There are rules for kernel_expr and kernel_params that guarantee
    consistency. First, the variable names supplied as keys to kernel_params
    should match the symbol names in kernel_expr. Symbol names for the inputs
    should be 'x1' and 'x2' (ignoring case). We could accept anything as long as
    2 symbols are left over after those in kernel_params, but these two rules
    will guarantee that nothing breaks.

    Currently, everything is only intended to work with 1D observables.

    Parameters
    ----------
    kernel_expr : Expr
        Expression for the kernel that can be differentiated - must have at
        least 2 symbols (symbol names should be 'x1' and 'x2', case insensitive,
        if have only 2)
    obs_dims : int
        Number of dimensions for observable input (input should be twice this
        with obs_dims values then obs_dims derivative labels each row)
    kernel_params : mapping
        A dictionary of kernel parameters that can be optimized by tensorflow
        (key should be name, then references list with value then another dict
        with kwargs for gpflow.Parameter, i.e., {'variance', [1.0,
        {'transform':gpflow.utilities.positive()}]} so if you don't want to set
        any kwargs, just pass empty dictionary NOTE THAT THE KEYS MUST MATCH THE
        SYMBOL NAMES IN kernel_expr OTHER THAN 'x1' and 'x2' Default is empty
        dict, so will mine names from kernel_expr and set all parameters to 1.0
    """

    def __init__(
        self, kernel_expr, obs_dims, kernel_params={}, active_dims=None, **kwargs
    ):
        #         if active_dims is not None:
        #             print("active_dims set to: ", active_dims)
        #             print("This is not implemented in this kernel, so setting to 'None'")
        #             active_dims = None
        # Having active_dims should be fine since just slices
        # Will work as long as include derivative info in active_dims

        super().__init__(active_dims=active_dims, **kwargs)

        if kernel_params is None:
            kernel_params = {}

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
        """Whether ARD behavior is active, following gpflow.kernels.Stationary."""
        return self.lengthscales.shape.ndims > 0

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        x1, d1 = self._split_x_into_locs_and_deriv_info(X)
        x2, d2 = self._split_x_into_locs_and_deriv_info(X2)

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
        unique_pairs = tf.raw_ops.UniqueV2(x=deriv_pairs, axis=[0])[0]

        # Loop over unique pairs, tracking indices and kernel values for pairs
        k_list = []
        inds_list = []
        for pair in unique_pairs:
            # Get the right indices
            this_inds = tf.cast(
                tf.where(tf.reduce_all(deriv_pairs == pair, axis=1))[:, :1], tf.int32
            )
            # Use sympy to obtain right derivative
            this_expr = sp.diff(
                self.kernel_expr,
                self.x_syms[0],
                int(pair[0]),
                self.x_syms[1],
                int(pair[1]),
            )
            # Get lambdified function compatible with tensorflow
            this_func = sp.lambdify(
                (self.x_syms[0], self.x_syms[1], *self.param_syms),
                this_expr,
                modules="tensorflow",
            )
            # Plug in our values for the derivative kernel
            k_list.append(
                this_func(
                    tf.gather_nd(expand_x1, this_inds),
                    tf.gather_nd(expand_x2, this_inds),
                    *[getattr(self, s.name) for s in self.param_syms],
                )
            )
            # Also keep track of indices so can dynamically stitch back together
            inds_list.append(this_inds)

        # Stitch back together
        k_list = tf.dynamic_stitch(inds_list, k_list)

        # Reshape to the correct output - will only really work for 1D, I think
        k_mat = tf.reshape(k_list, (x1.shape[0], x2.shape[0]))
        return k_mat

    def K_diag(self, X):
        # Same as for K but don't need every combination, just every x with itself
        x1, d1 = self._split_x_into_locs_and_deriv_info(X)
        unique_d1 = tf.unique(tf.reshape(d1, (-1,)))[0]
        k_list = []
        inds_list = []
        for d in unique_d1:
            this_inds = tf.cast(tf.where(d1 == d)[:, :1], tf.int32)
            this_expr = sp.diff(
                self.kernel_expr, self.x_syms[0], int(d), self.x_syms[1], int(d)
            )
            this_func = sp.lambdify(
                (self.x_syms[0], self.x_syms[1], *self.param_syms),
                this_expr,
                modules="tensorflow",
            )
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

    def _split_x_into_locs_and_deriv_info(self, x):
        """Splits input into actual observable input and derivative labels."""
        locs = x[:, : self.obs_dims]
        grad_info = x[:, -self.obs_dims :]
        return locs, grad_info


class HetGaussianNoiseGP(gpflow.likelihoods.ScalarLikelihood):
    """
    EXPERIMENTAL! NOT INTENDED FOR USE, BUT USEFUL FOR FUTURE WORK!

    Intended to model the noise associated with a GPR model using another GP contained
    within the likelihood. In other words, the likelihood, which usually describes the
    distribution for the added noise, is based on a GP that predicts the noise based on
    a specific input location, allowing for heteroscedastic noise modeling. Typically,
    you will want to actually model the logarithm of the noise variance as a function of
    the input, but this likelihood is more general than that.

    Specifically, the GP over noise is self.noise_GP, and is a standard gpflow.models.GPR
    model with a kernel specified by noise_kernel. If not provided, the default kernel
    used is a Matern52 with separate lengthscales over the different input dimensions.
    """

    def __init__(self, data, noise_kernel=None, **kwargs):
        super().__init__(**kwargs)
        X_data, Y_data = data
        if noise_kernel is not None:
            self.noise_gp = gpflow.models.GPR(data=data, kernel=noise_kernel)
        else:
            self.noise_gp = gpflow.models.GPR(
                data=data,
                kernel=gpflow.kernels.Matern52(lengthscales=np.ones(X_data.shape[1])),
            )

    def _scalar_log_prob(self, F, Y):
        return logdensities.gaussian(
            Y[:, :1], F[:, :1], F[:, 1:]
        ) + logdensities.gaussian(
            tf.math.log(Y[:, 1:]),
            tf.math.log(F[:, 1:]),
            self.noise_gp.likelihood.variance,
        )

    def _conditional_mean(self, F):
        return tf.identity(F[:, :1])

    def _conditional_variance(self, F):
        return tf.identity(F[:, 1:])

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu[:, :1]), Fvar[:, :1] + Fmu[:, 1:]

    def _predict_log_density(self, Fmu, Fvar, Y):
        external_logdens = tf.reduce_sum(
            logdensities.gaussian(Y[:, :1], Fmu[:, :1], Fvar[:, :1] + Fmu[:, 1:]),
            axis=-1,
        )
        latent_logdens = tf.reduce_sum(
            logdensities.gaussian(
                tf.math.log(Y[:, 1:]), tf.math.log(Fmu[:, 1:]), Fvar[:, 1:]
            ),
            axis=-1,
        )
        return external_logdens + latent_logdens

    def _variational_expectations(self, Fmu, Fvar, Y):
        external_likelihood = tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(Fmu[:, 1:])
            - 0.5 * ((Y[:, :1] - Fmu[:, :1]) ** 2 + Fvar[:, :1]) / Fmu[:, 1:],
            axis=-1,
        )
        latent_likelihood = tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.noise_gp.likelihood.variance)
            - 0.5
            * ((Y[:, 1:] - tf.math.log(Fmu[:, 1:])) ** 2 + Fvar[:, 1:])
            / self.noise_gp.likelihood.variance,
            axis=-1,
        )
        return external_likelihood + latent_likelihood


class FullyHeteroscedasticGPR(
    gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin
):
    """
    EXPERIMENTAL! NOT INTENDED FOR USE, BUT USEFUL FOR FUTURE WORK!

    Implements a fully heteroscedastic GPR model in which the noise is modeled
    with another Gaussian Process. To accomplish this, the likelihood is set to
    contain a simple GPR model that predicts the logarithm of the noise based on
    noise estimates passed into the model. The full likelihood involves that of
    both the outer heteroscedastic GPR using the predicted noise values and the
    GP on the noise, as proposed by Binois, et al. 2018.  However, since we do
    not want to model the "full N" data (i.e., all of the outputs for each sim
    configuration), but instead just the means from each simulation (guaranteed
    to be Gaussian by the CLT), we really follow the protocol of Ankenman et al.,
    2010 but allow noise in the GP over noise so that smoothing is applied. And,
    as mentioned above, both likelihoods are combined, not fit separately, as in
    Binois, et al. 2018.

    The input X data just has to match whatever kernel function is used.
    For the input Y data, there must be three columns: (1) the values to model,
    (2) the variance associated with each value, and (3) the number of sim frames
    or configurations used to calculate the provided value and variance.
    """

    def __init__(
        self,
        data: gpflow.models.model.RegressionData,
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_kernel: Optional[gpflow.kernels.Kernel] = None,
    ):
        X_data, Y_data = data
        # This is really a conditional likelihood given the output of self.noise_gp
        likelihood = HetGaussianNoiseGP(
            data=(X_data, tf.math.log(Y_data[:, 1:2] * Y_data[:, -1:])),
            noise_kernel=noise_kernel,
        )
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.data = gpflow.models.util.data_input_to_tensor(data)
        # For new predictions of noise, clearly depends on number of samples
        # As consertative estimate, use smallest number of samples from training
        self.min_samps = np.min(Y_data[:, -1])

    def predict_noise(self, x):
        log_noise, log_noise_var = self.likelihood.noise_gp.predict_f(x)
        noise = tf.math.exp(log_noise)
        return noise, log_noise_var

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        X, Y = self.data
        n = Y[
            :, -1
        ]  # Last entry is number of samples (configs) contributing to Y estimate
        Y = Y[
            :, :1
        ]  # Only take values, not uncertainty estimates, which are handled by likelihood
        K = self.kernel(X)
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.squeeze(self.predict_noise(X)[0])  # Gets predicted noise
        s_diag /= n
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        log_prob = gpflow.logdensities.multivariate_normal(Y, m, L)
        # Add this log probability to that of Gaussian process on noise, as in Binois 2018
        output = (
            tf.reduce_sum(log_prob) + self.likelihood.noise_gp.log_marginal_likelihood()
        )
        return output

    def predict_f(
        self,
        Xnew: gpflow.models.training_mixins.InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> gpflow.models.model.MeanAndVariance:
        """See :meth:`gpflow.models.GPModel.predict_f` for further details."""
        X_data, Y_data = self.data
        n = Y_data[:, -1]
        Y_data = Y_data[:, :1]
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)
        k_diag = tf.linalg.diag_part(kmm)
        s_diag = tf.squeeze(self.predict_noise(X_data)[0])
        s_diag /= n
        kmm_plus_s = tf.linalg.set_diag(kmm, k_diag + s_diag)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )
        f_mean = f_mean_zero + self.mean_function(Xnew)

        return f_mean, f_var

    def predict_y(
        self,
        Xnew: gpflow.models.training_mixins.InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> gpflow.models.model.MeanAndVariance:
        """See :meth:`gpflow.models.GPModel.predict_y` for further details."""
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        noise_mean, noise_var = self.predict_noise(Xnew)
        noise_mean /= self.min_samps
        out_mean = tf.concat([f_mean, noise_mean], axis=1)
        out_var = tf.concat([f_var, noise_var], axis=1)
        return self.likelihood.predict_mean_and_var(out_mean, out_var)

    def predict_log_density(
        self,
        data: gpflow.models.training_mixins.RegressionData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )
        X, Y = data
        f_mean, f_var = self.predict_f(
            X, full_cov=full_cov, full_output_cov=full_output_cov
        )
        noise_mean, noise_var = self.predict_noise(X)
        out_mean = tf.concat([f_mean, noise_mean], axis=1)
        out_var = tf.concat([f_var, noise_var], axis=1)
        return self.likelihood.predict_log_density(out_mean, out_var, Y)


class HetGaussianSimple(gpflow.likelihoods.ScalarLikelihood):
    """
    NOTE MAINTAINED, MAY BE OUT OF DATE AND NOT COMPATIBLE.

    Heteroscedastic Gaussian likelihood with variance provided and no modeling of noise
    variance. Note that the noise variance can be provided as a matrix or a 1D array.
    If a 1D array, it is assumed that the off-diagonal elements of the noise covariance
    matrix are all zeros, otherwise the noise covariance is used. For diagonal elements,
    it would make sense to also provide this information as an additional column in the
    target outputs, Y. However, this is not possible for a provided covariance matrix,
    when some of the noise values may be correlated as for derivatives at the same input
    location, X, measured from the same simulation. Just be careful to make sure shapes of
    Y and F (predicted GP mean values) match shape of provided covariance matrix - if matrix
    is NxN, each of Y and F should be N.
    """

    def __init__(
        self,
        cov,
        init_scale=1.0,
        **kwargs: Any,
    ) -> None:
        """
        :param cov: The covariance matrix (or its diagonal) for the noise.
        :param kwargs: Keyword arguments forwarded to :class:`gpflow.likelihoods.ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if cov.shape == 1:
            self.cov = np.diag(cov)
        else:
            self.cov = cov

        # Can precompute Cholesky decomposition
        self.Lcov = tf.linalg.cholesky(self.cov)

        # Won't learn full model on noise, but can still allow scaling of it to be learned
        # Imagine adding parameter to indicate "trust" of given noise and scale it
        # So just add parameter to train that scales noise
        self.scale_noise = gpflow.Parameter(
            init_scale, transform=gpflow.utilities.positive()
        )

    def build_scaled_cov_mat(self):
        """Creates scaled covariance matrix using noise scale parameters."""
        return self.scale_noise * self.cov

    def _scalar_log_prob(
        self, F: gpflow.base.TensorType, Y: gpflow.base.TensorType
    ) -> tf.Tensor:
        return gpflow.logdensities.multivariate_normal(
            Y, F, tf.math.sqrt(self.scale_noise) * self.Lcov
        )

    def _conditional_mean(
        self, F: gpflow.base.TensorType
    ) -> tf.Tensor:  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F: gpflow.base.TensorType) -> tf.Tensor:
        # Returns full covariance for INPUT Y data
        # May not fit with expected behavior, so could consider making "Not Implemented"
        return self.scale_noise * tf.identity(self.cov)

    def _predict_mean_and_var(
        self, Fmu: gpflow.base.TensorType, Fvar: gpflow.base.TensorType
    ) -> gpflow.models.model.MeanAndVariance:
        # From what I can tell, use this in predict_y, which will not be implemented either
        # Can't predict noise variance at NEW points, so no way to add noise to Fvar
        raise NotImplementedError(
            "Predicting noise at new points is not possible for this likelihood (would require prediction of full covariance between derivative orders at new points)."
        )

    def _predict_log_density(
        self,
        Fmu: gpflow.base.TensorType,
        Fvar: gpflow.base.TensorType,
        Y: gpflow.base.TensorType,
    ) -> tf.Tensor:
        # Again, relates to predictions at new points, which we are not doing
        # Can't predict noise variance at NEW points, so no way to add noise to Fvar
        raise NotImplementedError(
            "Predicting noise at new points is not possible for this likelihood (would require prediction of full covariance between derivative orders at new points)."
        )

    def _variational_expectations(
        self,
        Fmu: gpflow.base.TensorType,
        Fvar: gpflow.base.TensorType,
        Y: gpflow.base.TensorType,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Variational expectations is not implemented for this likelihood."
        )


def multioutput_multivariate_normal(x, mu, L) -> tf.Tensor:
    """
    Follows gpflow.logdensities.multivariate_normal exactly, but changes reducing sums so
    that multiple outputs with DIFFERENT covariance matrices can be taken into account.
    This still assumes that data in different columns of x are independent, but allows for
    a different Cholesky decomposition for each column or dimension. In the code for GPflow,
    everything would work if supplied x.T[..., None] was supplied with an L with leading
    batch dimension of the same dimensionality as the last dimension of x, EXCEPT that the
    last tf.reduce_sum over the diagonal part of L would sum over all independent matrices,
    which we do not want. This could all be accomplished with a loop over dimensions and
    separate applications of multivariate_normal, but hopefully this parallelizes.

    Parameters
    ----------
    x : array
        Shape `N x D` where here `N` is the number of input locations and `D` is
        the dimensionality
    mu : array
        Shape `N x D`, or broadcastable to NxD. mean values
    L : array
        Shape `DxNxN` Cholesky decomposition of `D` independent covariance
        matrices

    Returns
    -------
    p : array
        Shape of length `D`. Vector of log probabilities for each dimension
        (summed over input locations) Since covariance matrices independent
        across dimensions but convey covariances across locations, makes sense
        to sum over locations as would for multivariate Gaussian over each
        dimension
    """

    d = tf.expand_dims(tf.transpose(x - mu), -1)
    alpha = tf.linalg.triangular_solve(L, d, lower=True)
    alpha = tf.squeeze(alpha, axis=-1)
    num_locs = tf.cast(tf.shape(d)[1], L.dtype)
    p = -0.5 * tf.reduce_sum(tf.square(alpha), 1)
    p -= 0.5 * num_locs * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), 1)

    shape_constraints = [
        (d, ["D", "N", 1]),
        (L, ["D", "N", "N"]),
        (p, ["D"]),
    ]
    tf.debugging.assert_shapes(
        shape_constraints, message="multioutput_multivariate_normal()"
    )

    return p


class HetGaussianDeriv(gpflow.likelihoods.ScalarLikelihood):
    r"""
    Heteroscedastic Gaussian likelihood with variance provided and no modeling
    of noise variance. Note that the noise variance can be provided as a matrix
    or a 1D array. If a 1D array, it is assumed that the off-diagonal elements
    of the noise covariance matrix are all zeros, otherwise the noise covariance
    is used. For diagonal elements, it would make sense to also provide this
    information as an additional column in the target outputs, Y. However, this
    is not possible for a provided covariance matrix, when some of the noise
    values may be correlated as for derivatives at the same input location, X,
    measured from the same simulation. Just be careful to make sure shapes of Y
    and F (predicted GP mean values) match shape of provided covariance matrix -
    if matrix is NxN, each of Y and F should be N.

    Additionally, takes derivative orders of each input point. This model by
    default will scale noise differently for different derivative orders,
    effectively assuming that uncertainty is likely to be estimated incorrectly
    at some orders and accurately at others.

    Won't learn full model on noise, but can still allow scaling of it to be learned
    Imagine adding parameter to indicate "trust" of given noise and scale it
    So just add parameter to train that scales noise
    For scaling model, effectively model logarithm of each element in covariance matrix

    .. math::
        \ln {\rm cov}_{i,j} = \ln {\rm cov}_{i,j,0} + p (d_i + d_j) + s

    or

    .. math::
        {\rm cov}_{i,j} = {\rm cov}_{i,j,0} \exp[ p (d_i + d_j)] \exp(s)

    We can accomplish this while keeping the scaled covariance matrix positive
    semidefinite by making the scaling matrix diagonal with positive entries
    If we then take S*Cov*S, with S being the diagonal scaling matrix with positive
    entries, the result will be positive semi-definite because S is positive definite
    and Cov is positive semidefinite
    The scaling matrix is given by :math:`exp(s + p*d_i,j)` if :math:`i=j` and 0 otherwise
    While could make parameters s and p unconstrained, default will set ``s=0``, `p>=0``.
    This means that we CANNOT decrease the uncertainty, only increase it
    Further, if we increase the uncertainty, we must do it MORE for higher order
    derivatives
    Rationale is that it's only a really big deal if underestimate uncertainty
    Further, tend to have more numerical issues, bias, etc. in derivatives
    Even if derivatives actually more certain, typically want to focus on
    Fitting the function itself, not the derivatives
    In that case, can set p effectively to zero and will emphasize derivatives more

    Parameters
    ----------
    cov : array
        covariance matrix (or its diagonal) for the uncertainty (noise) in the data
    d_orders : int
        derivative order of the data; should be in same order as columns/rows of
        the covariance matrix
    p : float, default=10.0
        scaling of the covariance matrix dependent on derivative order
    s : float, default=0.0
        scaling of the covariance matrix independent of derivative order
    transform_p : object, optional
        Defaults to ``gpflow.utilities.positive()`` transformation of p during
        training of the GP model; the default is to require it be positive
    transform_s : object, optional
        transformation of s during GP model training
    constrain_p : bool, default=False
        whether or not p should be constrained and not altered during GP model
        training
    constrain_s : bool, default=True
        whether or not to constrain s during GP model training
    **kwargs
        Extra keyword arguments passed to :class:`gpflow.likelihoods.ScalarLikelihood`
    """

    def __init__(
        self,
        cov,
        d_orders,
        p=10.0,  # Sometimes gets stuck if starts small, but no issues if start large
        s=0.0,
        transform_p=gpflow.utilities.positive(),
        transform_s=None,
        constrain_p=False,
        constrain_s=True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if len(cov.shape) == 1:
            self.cov = np.diag(cov)
        else:
            self.cov = cov

        # Need to track derivative orders associated with each covariance entry
        if len(d_orders.shape) > 1:
            d_orders = np.squeeze(d_orders, axis=-1)
        self.d_orders = d_orders

        # Define parameters for power scale
        self.power_scale = gpflow.Parameter(
            p, transform=transform_p, trainable=(not constrain_p)
        )
        self.power_add = gpflow.Parameter(
            s, transform=transform_s, trainable=(not constrain_s)
        )

        # Define stability threshold
        self.stable_var_min = 1.0e-12

    def build_scaled_cov_mat(self):
        """
        Creates scaled covariance matrix using noise scale parameters.

        Returns
        -------
        :class:`tensorflow.Tensor`
            Tensor of the scaled covariance matrix
        """
        # First step is determining scaling based on exponential function
        # Add 1 so even zeroth order can be scaled
        scale = tf.exp(self.power_scale * (self.d_orders + 1) + 0.5 * self.power_add)
        scale = tf.linalg.diag(scale)
        # Multiply both sides of covariance matrix by diagonal scaling matrix
        output = tf.linalg.matmul(tf.linalg.matmul(scale, self.cov), scale)
        # Add jitter along diagonals to enforce minimum for stability
        out_diag = tf.linalg.diag_part(output)
        out_diag += self.stable_var_min
        return tf.linalg.set_diag(output, out_diag)

    def _scalar_log_prob(
        self, F: gpflow.base.TensorType, Y: gpflow.base.TensorType
    ) -> tf.Tensor:
        return multioutput_multivariate_normal(
            Y, F, tf.linalg.cholesky(self.build_scaled_cov_mat())
        )

    def _conditional_mean(
        self, F: gpflow.base.TensorType
    ) -> tf.Tensor:  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F: gpflow.base.TensorType) -> tf.Tensor:
        # Returns full covariance for INPUT Y data
        # May not fit with expected behavior, so could consider making "Not Implemented"
        return self.build_scaled_cov_mat()

    def _predict_mean_and_var(
        self, Fmu: gpflow.base.TensorType, Fvar: gpflow.base.TensorType
    ) -> gpflow.models.model.MeanAndVariance:
        # From what I can tell, use this in predict_y, which will not be implemented either
        # Can't predict noise variance at NEW points, so no way to add noise to Fvar
        raise NotImplementedError(
            "Predicting noise at new points is not possible for this likelihood (would require prediction of full covariance between derivative orders at new points)."
        )

    def _predict_log_density(
        self,
        Fmu: gpflow.base.TensorType,
        Fvar: gpflow.base.TensorType,
        Y: gpflow.base.TensorType,
    ) -> tf.Tensor:
        # Again, relates to predictions at new points, which we are not doing
        # Can't predict noise variance at NEW points, so no way to add noise to Fvar
        raise NotImplementedError(
            "Predicting noise at new points is not possible for this likelihood (would require prediction of full covariance between derivative orders at new points)."
        )

    def _variational_expectations(
        self,
        Fmu: gpflow.base.TensorType,
        Fvar: gpflow.base.TensorType,
        Y: gpflow.base.TensorType,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Variational expectations is not implemented for this likelihood."
        )


class HeteroscedasticGPR_analytical_scale(
    gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin
):
    """
    EXPERIMENTAL! NOT INTENDED FOR USE, BUT MAYBE INTERESTING TO CONSIDER IN FUTURE!

    Implements a GPR model with heteroscedastic input noise, which can be just a vector
    (diagonal noise covariance matrix) or the full noise covariance matrix if noise is
    correlated within some of the input data. The latter is useful for derivatives from
    the same simulation at the same input location. The covariance matrix is expected to
    be the third element of the input data tuple (X, Y, noise_cov).
    """

    def __init__(
        self,
        data: gpflow.models.model.RegressionData,
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        scale_fac: Optional[float] = None,
    ):
        X_data, Y_data, noise_cov = data

        # To make training behave better, can try scaling covariance matrices and data
        # Just remember to scale mean function and predictions throughout
        # Can make difference, but only impacts ease of training, not optimal model behavior
        # So default is to scale by minimum variance, but can set to 1.0
        if scale_fac is None:
            self.scale_fac = np.sqrt(np.min(np.diag(noise_cov)))
        else:
            self.scale_fac = scale_fac
        Y_data = Y_data / self.scale_fac
        noise_cov = noise_cov / (self.scale_fac**2)

        likelihood = HetGaussianSimple(noise_cov)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.data = gpflow.models.util.data_input_to_tensor((X_data, Y_data))

    def calc_scale_v(self, err=None, L=None):
        # Won't learn full model on noise, but can still allow scaling of it to be learned
        # Imagine adding parameter to indicate "trust" of given noise and scale it
        # Function is mainly useful for getting access to scale calculation from outside
        # (i.e., exposes outside of log-likelihood and predict_f calculations)

        X_data, Y_data = self.data

        # Can optionally provide data and Cholesky decomposition L of K + S
        if err is None:
            err = Y_data - (self.mean_function(X_data) / self.scale_fac)

        # Best if Cholesky decomposition of kernel plus noise covariance given...
        if L is None:
            kmm = self.kernel(X_data) / self.scale_fac
            kmm_plus_s = kmm + self.likelihood.build_scaled_cov_mat()
            L = tf.linalg.cholesky(kmm_plus_s)

        num_dims = tf.cast(tf.shape(err)[0], L.dtype)
        alpha = tf.linalg.triangular_solve(L, err, lower=True)
        v = tf.reduce_sum(tf.square(alpha), 0) / num_dims

        return v

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        X, Y = self.data
        K = self.kernel(X) / self.scale_fac
        ks = K + self.likelihood.build_scaled_cov_mat()
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X) / self.scale_fac

        # log_prob = gpflow.logdensities.multivariate_normal(Y, m, L)

        d = Y - m

        # Add in term for 1st order optimal value of scaling of combined covariance matrix
        # In other words, model has v*(K + S) = scale*(kernel_cov + noise_cov)
        # If K and S are known, can identify optimum for v as (1/N)*Y (K + S)^(-1) Y
        # If substitute this into the log likelihood, end up with this term in a logarithm
        # AND cancel part of rest of likelihood
        # Mirroring gpflow code for multivariate normal, but modified as in Binois, et al. 2018
        # Hopefully helps by adding parameter to adjust noise
        # Preserves noise covariance structure and relative noise levels, though
        num_dims = tf.cast(tf.shape(d)[0], L.dtype)
        log_prob = -0.5 * num_dims * tf.math.log(self.calc_scale_v(err=d, L=L))
        log_prob -= 0.5 * num_dims * np.log(2 * np.pi)
        log_prob -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        log_prob -= 0.5 * num_dims

        return tf.reduce_sum(log_prob)

    def predict_f(
        self,
        Xnew: gpflow.models.training_mixins.InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> gpflow.models.model.MeanAndVariance:
        """See :meth:`gpflow.models.GPModel.predict_f` for further details."""

        X_data, Y_data = self.data
        err = Y_data - (self.mean_function(X_data) / self.scale_fac)

        kmm = self.kernel(X_data) / self.scale_fac
        knn = self.kernel(Xnew, full_cov=full_cov) / self.scale_fac
        kmn = self.kernel(X_data, Xnew) / self.scale_fac
        kmm_plus_s = kmm + self.likelihood.build_scaled_cov_mat()

        # conditional = gpflow.conditionals.base_conditional
        # f_mean_zero, f_var = conditional(kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False)
        # Computes Cholesky decomposition in base conditional anyway, so just bypass here
        # Allows to compute scaling factor v on our model of v*(K + S)
        # Requires solving extra equation, but at least avoiding Cholesky multiple times
        # Probably clever way to cache the scaling factor somehow, but not sure how
        L = tf.linalg.cholesky(kmm_plus_s)
        v = self.calc_scale_v(err=err, L=L)
        scaled_L = tf.math.sqrt(v) * L
        conditional = gpflow.conditionals.util.base_conditional_with_lm
        f_mean_zero, f_var = conditional(
            v * kmn, scaled_L, v * knn, err, full_cov=full_cov, white=False
        )

        f_mean = f_mean_zero + (self.mean_function(Xnew) / self.scale_fac)

        f_mean *= self.scale_fac
        f_var *= self.scale_fac**2

        return f_mean, f_var

    def predict_y(
        self,
        Xnew: gpflow.models.training_mixins.InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> gpflow.models.model.MeanAndVariance:
        """See :meth:`gpflow.models.GPModel.predict_y` for further details."""
        raise NotImplementedError(
            "Predicting y would require knowledge of the noise at new data points, which is not modeled here."
        )

    def predict_log_density(
        self,
        data: gpflow.models.training_mixins.RegressionData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Predicting log density at new points requires knowledge of noise at new points, which is not modeled here."
        )


class HeteroscedasticGPR(
    gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin
):
    """
    Implements a GPR model with heteroscedastic input noise, which must be the full noise
    covariance matrix. This is necessary for derivatives from the same simulation at the
    same input location, which will likely be correlated. If the output is multidimensional,
    a separate covariance matrix may be specified for each dimension of the output - if this
    is not the case, the same covariance matrix will be used for all output dimensions. The
    consequence of this structure is that the model is independent across output dimensions,
    which means that, for multidimensional output, a gpflow shared or separate independent
    multioutput kernel should be used to wrap whatever kernel has been specified. If it is
    detected that the kernel does not satisfy this property, the model will attempt to
    appropriately wrap the specified kernel. The covariance matrix is expected to
    be the third element of the input data tuple (`X, Y, noise_cov`). Specific shapes should be ``X.shape == (N, 2)``, ``Y.shape == (N, D)``, ``noise_cov.shape == (N, D, D) or (D, D)``,
    where `N` is the number of input locations and `D` is the input dimensionality. Note that the
    first column of `X` is for the locations and the second is for the derivative order of
    the observation at that location, so only 1D inputs can be handled, though the output
    dimension, `D`, is not restricted.

    Parameters
    ----------
    data : list of tuple
        A list or tuple of the input locations, output data, and noise
        covariance matrix, in that order
    kernel : :class:`DerivativeKernel` object
        The kernel to use; must be DerivativeKernel or compatible subclass
        expecting derivative information provided in extra columns of the input
        locations
    mean_function : callable, optional
        Mean function to be used (probably should be one that
        handles inputs including the derivative order)
    scale_fac : float, default=1.0
        scaling factor on the output data; can apply to each dimension
        separately if an array; helpful to ensure all output dimensions have
        similar variance
    x_scale_fac : float, default=1.0
        scaling factor on input locations; NOT USEFUL AND SOON TO BE DEPRECATED
    likelihood_kwargs, dict, optional
        Dictionary of keyword arguments to pass to the HetGaussianDeriv
        likelihood model used by this GP model
    """

    def __init__(
        self,
        data: gpflow.models.model.RegressionData,
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        scale_fac: Optional[float] = 1.0,
        x_scale_fac: Optional[float] = 1.0,
        likelihood_kwargs: Optional[dict] = {},
    ):
        X_data, Y_data, noise_cov = data
        self.out_dim = Y_data.shape[-1]

        # Scale data by the desired scaling factor - can help equalize variance across outputs
        # Nice to handle scaling inside model rather than outside
        scale_fac = np.array(scale_fac)
        if len(scale_fac.shape) == 0:
            scale_fac = scale_fac * np.ones(self.out_dim)
        self.scale_fac = scale_fac
        Y_data = Y_data / self.scale_fac
        noise_cov = noise_cov / (
            np.expand_dims(
                self.scale_fac, axis=tuple(range(scale_fac.ndim, noise_cov.ndim))
            )
            ** 2
        )

        # Can also include another scaling factor for x data
        # This can help keep the lengthscale parameter for an RBF kernel >1.0
        # To save computational time, modify data now since always used scaled
        self.x_scale_fac = x_scale_fac
        X_data = np.concatenate(
            [X_data[:, :1] * self.x_scale_fac, X_data[:, 1:]], axis=-1
        )
        Y_data = Y_data / (self.x_scale_fac ** X_data[:, 1:])
        noise_cov = noise_cov / self.x_scale_fac ** (
            np.add(*np.meshgrid(X_data[:, 1:], X_data[:, 1:]))
        )

        # To generally allow for multidimensional outputs, need last Y_data and first
        # noise_cov dimensions to match
        if len(noise_cov.shape) == 2:
            noise_cov = np.tile(noise_cov[None, ...], (self.out_dim, 1, 1))

        # Create specific likelihood for this model
        likelihood = HetGaussianDeriv(noise_cov, X_data[:, 1:], **likelihood_kwargs)

        # Check if kernel is multioutput and if not wrap as SharedIndependent
        # If prefer to have different kernels on different outputs, can use SeparateIndependent
        # If need even more flexibility, like correlations between dimensions, can
        # subclass off of MultioutputKernel in gpflow and make custom
        if not issubclass(type(kernel), gpflow.kernels.MultioutputKernel):
            kernel = gpflow.kernels.SharedIndependent(kernel, output_dim=self.out_dim)

        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.data = gpflow.models.util.data_input_to_tensor((X_data, Y_data))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        X, Y = self.data

        K = self.kernel(X, full_cov=True, full_output_cov=False)
        ks = K + self.likelihood.build_scaled_cov_mat()
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X) / self.scale_fac

        log_prob = multioutput_multivariate_normal(Y, m, L)

        return tf.reduce_sum(log_prob)

    def predict_f(
        self,
        Xnew: gpflow.models.training_mixins.InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> gpflow.models.model.MeanAndVariance:
        """See :meth:`gpflow.models.GPModel.predict_f` for further details."""
        X_data, Y_data = self.data

        # Account for scaling in x for new inputs
        Xnew = tf.concat([Xnew[:, :1] * self.x_scale_fac, Xnew[:, 1:]], -1)

        err = Y_data - (self.mean_function(X_data) / self.scale_fac)

        # With MultiOutput kernels in GPflow, default full_cov and full_output_cov behavior
        # is different from base_kernel, which requires more explicit specifications
        # Following IndependentPosteriorMultiOutput but with custom likelihood covariance
        kmm = self.kernel(X_data, full_cov=True, full_output_cov=False)
        knn = self.kernel(Xnew, full_cov=full_cov, full_output_cov=False)
        kmn = self.kernel(X_data, Xnew, full_cov=True, full_output_cov=False)
        kmm_plus_s = kmm + self.likelihood.build_scaled_cov_mat()

        # To generally handle multioutput data, use appropriate conditional
        # Means also need to tile kernel (not kmm_plus_s, though, since __init__ checks noise)
        # Note that tiling the kernels assumes independence across output dimensions AND
        # that the kernel is shared across all dimensions - independent processes with shared
        # parameters
        # knn = tf.expand_dims(knn, 0)
        # knn = tf.tile(knn, (self.out_dim, 1))
        # kmn = tf.expand_dims(kmn, 0)
        # kmn = tf.tile(kmn, (self.out_dim, 1, 1))
        # But only need to do above if not using GPflow built-in multioutput kernel
        # In that case, just need to transpose knn for inexplicable reason
        # (only if full_cov is False, though)
        if not full_cov:
            knn = tf.transpose(knn)
        conditional = (
            gpflow.conditionals.util.separate_independent_conditional_implementation
        )
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )
        f_var = gpflow.conditionals.util.expand_independent_outputs(
            f_var, full_cov, full_output_cov
        )
        f_mean = f_mean_zero + (self.mean_function(Xnew) / self.scale_fac)

        # Again account for scaling in x for output
        f_mean = f_mean * (self.x_scale_fac ** Xnew[:, 1:])
        # Will be either scaling a vector, or a vector of full covariance matrices
        # Depends on full_cov value
        if not full_cov:
            f_var = f_var * (self.x_scale_fac ** (2 * Xnew[:, 1:]))
        else:
            f_var = f_var * self.x_scale_fac ** (
                np.add(*np.meshgrid(Xnew[:, 1:], Xnew[:, 1:]))
            )

        f_mean *= self.scale_fac
        # Need to appropriately reshape scale factor based on full_cov
        # If full_cov==True, f_var is (D, M, M), otherwise, it's (M, D)
        var_scale_fac = np.reshape(
            self.scale_fac**2, (-1,) + (1,) * (len(f_var.shape) - 1)
        )
        if not full_cov:
            var_scale_fac = var_scale_fac.T
        f_var *= var_scale_fac

        return f_mean, f_var

    def predict_y(
        self,
        Xnew: gpflow.models.training_mixins.InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> gpflow.models.model.MeanAndVariance:
        """See :meth:`gpflow.models.GPModel.predict_y` for further details."""
        raise NotImplementedError(
            "Predicting y would require knowledge of the noise at new data points, which is not modeled here."
        )

    def predict_log_density(
        self,
        data: gpflow.models.training_mixins.RegressionData,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        raise NotImplementedError(
            "Predicting log density at new points requires knowledge of noise at new points, which is not modeled here."
        )


class ConstantMeanWithDerivs(gpflow.mean_functions.MeanFunction):
    """
    Constant mean function that takes derivative-augmented X as input.
    Only applies mean function constant to zeroth order derivatives.
    Because added constant, adding mean function does not change variance or derivatives.

    Parameters
    ----------
    y_data : array-like
        The data for which the mean should be taken
    """

    def __init__(self, y_data) -> None:
        super().__init__()
        c = np.average(y_data, axis=0)
        self.c = c  # gpflow.Parameter(c, trainable=False)
        self.dim = y_data.shape[1]

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        filled_mean = tf.ones([tf.shape(X)[0], self.dim], dtype=X.dtype) * self.c
        filled_zeros = tf.zeros([tf.shape(X)[0], self.dim], dtype=X.dtype)
        return tf.where((X[:, -1:] == 0), filled_mean, filled_zeros)


class LinearWithDerivs(gpflow.mean_functions.MeanFunction):
    """
    Linear mean function that can be applied to derivative data - in other words,
    the 0th order derivative is fit with a linear fit, so the 1st derivative also
    has to be modified (by a constant that is the slope). Currently handles y of
    multiple dimensions, but scalar output only (so fits hyperplane). Columns of
    y_data should be different dimensions while rows are observations.

    Parameters
    ----------
    x_data : array-like
        input locations of data points
    y_data : array-like
        output data to learn linear function for based on input locations
    """

    def __init__(self, x_data, y_data) -> None:
        super().__init__()
        # First define "center" of data to fit around
        mean_x = np.average(x_data, axis=0)
        mean_y = np.average(y_data, axis=0)
        # And compute slope
        x_mat = x_data - mean_x
        y_mat = y_data - mean_y
        slope = np.linalg.inv(x_mat.T @ x_mat) @ (x_mat.T @ y_mat)
        # And intercept
        b = mean_y - slope * mean_x
        self.slope = slope  # gpflow.Parameter(slope, trainable=False)
        self.b = b  # gpflow.Parameter(b, trainable=False)
        self.dim = y_data.shape[1]

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        # Fill in mean function for 0th order for all X
        filled_mean_0 = self.slope * X[:, :1] + self.b
        # Fill in mean function for 1st order for all X
        filled_mean_1 = tf.ones([tf.shape(X)[0], self.dim], dtype=X.dtype) * self.slope
        filled_zeros = tf.zeros([tf.shape(X)[0], self.dim], dtype=X.dtype)
        # Set conditions to fill in mean values for just 0th and 1st derivatives
        output_0 = tf.where((X[:, -1:] == 0), filled_mean_0, filled_zeros)
        output_1 = tf.where((X[:, -1:] == 1), filled_mean_1, filled_zeros)
        # Return sum so that has mean values for only 0th and 1st derivatives and rest 0
        return output_0 + output_1


class SympyMeanFunc(gpflow.mean_functions.MeanFunction):
    """
    Mean function based on sympy expression. This way, can take derivatives up
    to any order. In the provided expression, the input variable should be 'x'
    or 'X', otherwise this will not work. For consistency with other mean
    functions, only fit based on zero-order data, rather than fitting during
    training of full GP model. params is an optional dictionary specifying
    starting parameter values.

    Parameters
    ----------
    expr : Expr
        Representing the functional form of the mean function.
    x_data : array-like
        the input locations of the data
    y_data : array-like
        the output values of the data to fit the mean function to
    params : dict, optional
        dictionary specifying starting parameter values for the mean function;
        in other words, these values will be substituted into the sympy
        expression to start with
    """

    def __init__(self, expr, x_data, y_data, params=None):
        super().__init__()
        self.expr = expr

        expr_syms = tuple(expr.free_symbols)
        self.param_syms = []
        for s in expr_syms:
            if s.name.casefold() == "x":
                self.x_sym = s
            else:
                self.param_syms.append(s)

        # Make sure that parameters here match those in params, if it's provided
        if bool(params):
            if (
                list([s.name for s in self.param_syms]).sort()
                != list(params.keys()).sort()
            ):
                raise ValueError("Symbol names in expr must match keys in " + "params!")
            # If they are the same, obtain parameter values from params dictionary
            # Need to set as gpflow Parameter objects so can optimize over them
            for key, val in params.items():
                setattr(self, key, float(val))

        # If kernel_params is not provided, set everything to 1.0 by default
        else:
            for s in self.param_syms:
                setattr(self, s.name, float(1.0))

        # Collect only zeroth-order data for training mean function
        zero_bool = x_data[:, 1] == 0
        x_zero = x_data[zero_bool, 0]
        y_zero = y_data[zero_bool, 0]

        # Create function at zeroth order
        mean_func = sp.lambdify(
            (self.x_sym, *self.param_syms), self.expr, modules="numpy"
        )
        # And also wrap derivatives w.r.t. parameters for Jacobian
        deriv_funcs = []
        for p_sym in self.param_syms:
            this_jac = sp.diff(self.expr, p_sym, 1)
            deriv_funcs.append(
                sp.lambdify((self.x_sym, *self.param_syms), this_jac, modules="numpy")
            )

        # Create loss function
        def loss_func(params):
            return np.sum((mean_func(x_zero, *params) - y_zero) ** 2)

        # And create Jacobian function
        def jac_func(params):
            prefac = 2.0 * (mean_func(x_zero, *params) - y_zero)
            jac = [np.sum(prefac * deriv(x_zero, *params)) for deriv in deriv_funcs]
            return np.array(jac)

        # Perform optimization with scipy
        opt = optimize.minimize(
            loss_func,
            np.array([getattr(self, s.name) for s in self.param_syms]),
            method="L-BFGS-B",
            jac=jac_func,
        )
        print(opt)

        # Set parameters based on optimization
        for i, s in enumerate(self.param_syms):
            setattr(self, s.name, opt.x[i])

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        """Closely follows K_diag from DerivativeKernel."""
        x_vals = X[:, :1]
        d_vals = X[:, 1:]
        unique_d = tf.unique(tf.reshape(d_vals, (-1,)))[0]
        f_list = []
        inds_list = []
        for d in unique_d:
            this_inds = tf.cast(tf.where(d_vals == d)[:, :1], tf.int32)
            this_expr = sp.diff(self.expr, self.x_sym, int(d))
            this_func = sp.lambdify(
                (self.x_sym, *self.param_syms),
                this_expr,
                modules="tensorflow",
            )
            f_list.append(
                this_func(
                    tf.gather_nd(x_vals, this_inds),
                    *[getattr(self, s.name) for s in self.param_syms],
                )
            )
            inds_list.append(this_inds)

        # NOTE: never used, so commented out.  Something wrong here?
        # k_list = tf.dynamic_stitch(inds_list, f_list)
        out = tf.reshape(f_list, (x_vals.shape[0], 1))  # Really, just for 1D functions
        return out

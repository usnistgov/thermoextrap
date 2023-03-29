# Written by Jacob I. Monroe, NIST employee

"""
Tests for GP models with derivatives and active learning based on those models.
"""

import gpflow
import numpy as np
import pytest

from thermoextrap.gpr_active import sine_active
from thermoextrap.gpr_active.active_utils import make_rbf_expr, train_GPR
from thermoextrap.gpr_active.gp_models import (
    ConstantMeanWithDerivs,
    DerivativeKernel,
    HeteroscedasticGPR,
    HetGaussianDeriv,
    LinearWithDerivs,
    SympyMeanFunc,
    multioutput_multivariate_normal,
)


# For derivative kernel, test if reproduces derivatives for known functional form and values
# Work with RBF and test up to second order derivatives
# Will build covariance matrix with derivative orders indexing, i.e., cov[0, 1] = dk/dx2
class RBF_covs:
    def __init__(self, var, l):
        self.var = var
        self.l = l

    def d(self, x1, x2):
        return x1 / self.l - x2 / self.l

    # k(x1, x2) = var * exp(-0.5*(x1/l - x2/l)**2)
    def cov00(self, x1, x2):
        return self.var * np.exp(-0.5 * self.d(x1, x2) ** 2)

    # dk/dx2 = var * exp(-0.5*(x1/l - x2/l)**2) * ((1/l)*(x1/l - x2/l)) = k(x1, x2) * ((1/l)*(x1/l - x2/l))
    def cov01(self, x1, x2):
        return self.cov00(x1, x2) * ((1 / self.l) * self.d(x1, x2))

    # dk/dx1 = -dk/dx2
    def cov10(self, x1, x2):
        return -self.cov01(x1, x2)

    # d^2k/dx1*dx2 = -dk/dx2 * ((1/l)*(x1/l - x2/l)) + k(x1, x2) * (1/l^2)
    def cov11(self, x1, x2):
        return -self.cov01(x1, x2) * ((1 / self.l) * self.d(x1, x2)) + self.cov00(
            x1, x2
        ) * (1 / self.l**2)

    # d^2k/dx1^2 = -dk/dx1 * ((1/l)*(x1/l - x2/l)) - k(x1, x2) * (1/l^2) = -d^2k/dx1*dx2
    def cov02(self, x1, x2):
        return -self.cov11(x1, x2)

    # d^2k/dx2^2 = dk/dx2 * ((1/l)*(x1/l - x2/l)) - k(x1, x2) * (1/l^2) = -d^2k/dx1*dx2
    def cov20(self, x1, x2):
        return -self.cov11(x1, x2)

    # d^3k/dx2^2*dx1 = d^2k/dx2*dx1 * ((1/l)*(x1/l - x2/l)) - 2*dk/dx1 * (1/l^2)
    def cov12(self, x1, x2):
        return self.cov11(x1, x2) * ((1 / self.l) * self.d(x1, x2)) + 2 * self.cov01(
            x1, x2
        ) * (1 / self.l**2)

    # d^3k/dx1^2*dx2 = -d^2k/dx1*dx2 * ((1/l)*(x1/l - x2/l)) + 2*dk/dx1 * (1/l^2) = -d^3k/dx2^2*dx1
    def cov21(self, x1, x2):
        return -self.cov12(x1, x2)

    # d^4k/dx1^2*dx2^2 = d^3k/dx1^2*dx2 * ((1/l)*(x1/l - x2/l)) + d^2k/dx1*dx2 * (1/l^2) - 2*d^2k/dx1^2 * (1/l^2)
    def cov22(self, x1, x2):
        return -self.cov12(x1, x2) * ((1 / self.l) * self.d(x1, x2)) + 3 * self.cov11(
            x1, x2
        ) * (1 / self.l**2)

    def __call__(self, x1, x2):
        cov_mat = [
            [self.cov00, self.cov01, self.cov02],
            [self.cov10, self.cov11, self.cov12],
            [self.cov20, self.cov21, self.cov22],
        ]

        out = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                out[i, j] = cov_mat[i][j](x1, x2)

        return out


def test_deriv_kernel_manual():
    # Create manually implemented RBF class to use as reference
    rbf_check = RBF_covs(1.0, 2.0)

    # Create a derivative kernel to check
    kern_expr, kern_params = make_rbf_expr()
    # Overwrite transforms on kernel parameters
    kern_params = {"var": [1.0, {}], "l": [2.0, {}]}
    deriv_kern = DerivativeKernel(kern_expr, 1, kernel_params=kern_params)

    # Want to work with a few points to check...
    # (0, 0), (1, 1), (1, 0), (0, 1), and (-1, 0)
    for x_pair in [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]:
        ref = rbf_check(x_pair[0], x_pair[1])
        this_x1 = np.vstack([x_pair[0] * np.ones(3), np.arange(3)]).T
        this_x2 = np.vstack([x_pair[1] * np.ones(3), np.arange(3)]).T
        to_check = deriv_kern.K(this_x1, this_x2).numpy()
        np.testing.assert_allclose(ref, to_check)

    # Next test with multiple points provided at once
    x_check = np.block(
        [
            [0.0 * np.ones((3, 1)), np.arange(3)[:, None]],
            [1.0 * np.ones((3, 1)), np.arange(3)[:, None]],
        ]
    )
    to_check = deriv_kern.K(x_check, x_check).numpy()
    ref = np.block(
        [
            [rbf_check(0.0, 0.0), rbf_check(0.0, 1.0)],
            [rbf_check(1.0, 0.0), rbf_check(1.0, 1.0)],
        ]
    )
    np.testing.assert_allclose(ref, to_check)


def test_deriv_kernel_self():
    # Now test for self-consistency within the derivative kernel
    kern_expr, kern_params = make_rbf_expr()
    kern_params = {"var": [1.0, {}], "l": [2.0, {}]}
    deriv_kern = DerivativeKernel(kern_expr, 1, kernel_params=kern_params)

    # Create test input
    x_check = np.block(
        [
            [0.0 * np.ones((3, 1)), np.arange(3)[:, None]],
            [1.0 * np.ones((3, 1)), np.arange(3)[:, None]],
            [-1.0 * np.ones((3, 1)), np.arange(3)[:, None]],
        ]
    )

    # Test that splits x data correctly (told it that observation dims is 1)
    check_locs, check_dorder = deriv_kern._split_x_into_locs_and_deriv_info(x_check)
    np.testing.assert_allclose(check_locs, x_check[:, :1])
    np.testing.assert_allclose(check_dorder, x_check[:, 1:])

    # Generate full output to check against various equivalent scenarios
    output = deriv_kern.K(x_check, x_check).numpy()

    # Check that K_diag function same as diagonal of K
    output_diag = deriv_kern.K_diag(x_check).numpy()
    np.testing.assert_allclose(output_diag, np.diag(output))

    # Check that passing single x is same as passing twice
    output_single = deriv_kern.K(x_check)
    np.testing.assert_allclose(output_single, output)

    # Make sure that if re-order x inputs get same thing with different order
    x_check_reorder = np.roll(x_check, 3, axis=0)
    output_reorder = deriv_kern.K(x_check_reorder, x_check_reorder).numpy()
    output_reorder = np.roll(np.roll(output_reorder, -3, axis=0), -3, axis=1)
    np.testing.assert_allclose(output_reorder, output)

    # And if switch around derivatives and points
    x_check_dorder = x_check_reorder = np.roll(x_check, 1, axis=0)
    output_dorder = deriv_kern.K(x_check_dorder, x_check_dorder).numpy()
    output_dorder = np.roll(np.roll(output_dorder, -1, axis=0), -1, axis=1)
    np.testing.assert_allclose(output_dorder, output)


# Testing mean functions is straightforward
# First define some data to use for testing
def mean_funcs_check_data():
    # Define points to check at
    x_check = np.block(
        [
            [0.0 * np.ones((3, 1)), np.arange(3)[:, None]],
            [0.5 * np.ones((3, 1)), np.arange(3)[:, None]],
            [1.0 * np.ones((3, 1)), np.arange(3)[:, None]],
        ]
    )

    # Will use y=x^2, so just have that at zero order
    # Then for first order derivative, it's just 2*x
    # Then just 2
    y_check_sq = np.zeros((x_check.shape[0], 1))
    for i in range(3):
        this_inds = np.where(x_check[:, 1] == i)[0]
        this_vals = x_check[this_inds, 0] ** (2 - i)
        if i > 0:
            this_vals *= 2.0
        y_check_sq[this_inds, 0] = this_vals

    # But to check multidimensional data, also provide just linear function of y
    # (as another column)
    y_check_lin = np.zeros((x_check.shape[0], 1))
    y_check_lin[::3, 0] = x_check[::3, 0]
    y_check_lin[1::3, 0] = 1.0

    y_check = np.hstack([y_check_lin, y_check_sq])

    return x_check, y_check


# For constant, just make sure that zeroth order derivatives have constant and all others are zero
def test_constant_mean_func():
    x_check, y_check = mean_funcs_check_data()

    # Generate constant mean function to test
    m_func = ConstantMeanWithDerivs(y_check[::3])

    # Make sure mean is computed correctly
    ref_avg = np.average(y_check[::3, :], axis=0)
    np.testing.assert_allclose(m_func.c, ref_avg)

    # And when call make sure get constant at order zero and zero otherwise
    check_out = m_func(x_check).numpy()
    zero_bool = x_check[:, 1] == 0
    nonzero_bool = x_check[:, 1] != 0
    np.testing.assert_array_equal(
        check_out[zero_bool], ref_avg * np.ones_like(check_out[zero_bool])
    )
    np.testing.assert_array_equal(
        check_out[nonzero_bool], np.zeros_like(check_out[nonzero_bool])
    )


# For linear, just make sure it returns correct linear fit for provided test case
def test_linear_mean_func():
    x_check, y_check = mean_funcs_check_data()

    # Generate linear mean function to test
    m_func = LinearWithDerivs(x_check[::3, :1], y_check[::3])

    # Will check over whole range at each order
    x_full_range = np.linspace(np.min(x_check), np.max(x_check), 100)

    # For checking against quadratic data (second column of y_check), compare to scipy
    from scipy.stats import linregress

    sp_regress = linregress(x_check[::3, 0], y_check[::3, 1])
    sp_regress_check = sp_regress.slope * x_full_range + sp_regress.intercept

    # Loop over orders
    for i in range(3):
        this_x = np.vstack([x_full_range, i * np.ones_like(x_full_range)]).T
        this_output = m_func(this_x)
        if i == 0:
            # For linear function input, fit should be exact (so for first column of y_check)
            np.testing.assert_allclose(this_output[:, 0], x_full_range)
            np.testing.assert_allclose(this_output[:, 1], sp_regress_check)
        elif i == 1:
            np.testing.assert_array_equal(
                this_output, m_func.slope * np.ones_like(this_output)
            )
        else:
            np.testing.assert_array_equal(this_output, np.zeros_like(this_output))


# Testing sympy mean function is more difficult - mainly testing sympy stuff
def test_sympy_mean_func():
    import sympy as sp

    # Will work with logistic because relevant derivatives can be computed in terms of 0 order
    x = sp.symbols("x")
    m = sp.symbols("m")
    b = sp.symbols("b")
    sig_expr = 1.0 / (1.0 + sp.exp(-m * (x + b)))
    params = {
        "m": 1.0,
        "b": 0.0,
    }

    # Create some x data
    x_vals = np.linspace(-10.0, 10.0, 10)
    x_check = [np.vstack([x_vals, d_o * np.ones_like(x_vals)]).T for d_o in range(3)]
    x_check = np.vstack(x_check)

    # And some y data
    m_check = 0.26
    b_check = 3.4
    y_vals = 1.0 / (1.0 + np.exp(-m_check * (x_vals + b_check)))
    # Will not use higher order info, so just set to zeros
    y_check = np.zeros((x_check.shape[0], 1))
    y_check[: len(x_vals), 0] = y_vals

    # Create our mean function to test
    check_sym = SympyMeanFunc(sig_expr, x_check, y_check, params=params)

    # Check that expression matches input
    assert sp.simplify(check_sym.expr - sig_expr) == 0
    # Check that found optimal parameters
    np.testing.assert_allclose(check_sym.m, m_check, rtol=1e-06)
    np.testing.assert_allclose(check_sym.b, b_check, rtol=1e-06)
    # And check values and derivatives
    output = check_sym(x_check)
    np.testing.assert_allclose(
        output[:10].numpy(),
        sp.lambdify(x, sig_expr.subs(((m, m_check), (b, b_check))))(x_check[:10, :1]),
        rtol=1e-04,
    )
    np.testing.assert_allclose(
        output[10:20].numpy(), check_sym.m * (output[:10] - output[:10] ** 2).numpy()
    )
    np.testing.assert_allclose(
        output[20:].numpy(),
        (check_sym.m**2)
        * (output[:10] - 3.0 * output[:10] ** 2 + 2.0 * output[:10] ** 3),
    )


# Need to test all components of the GP model
# Start with testing the multioutput_multivariate_normal function
def test_multiout_multivar_normal():
    # Idea of this function is to parallelize assessment of multivariate Gaussians in tensorflow
    # First need valid covariance matrices
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    cov2 = np.array([[0.5, 0.0], [0.0, 2.0]])
    cov3 = np.array([[0.5, 0.0], [-2.5, 3.5]])
    cov3 = cov3.T @ cov3
    cov_mats = np.array([cov1, cov2, cov3])

    # Generate Cholesky decomposition, which is required input for
    cov_L = np.array([np.linalg.cholesky(c) for c in cov_mats])

    # And need means and points to compute log probability at
    means = np.array([[0.0, 0.5, -1.0], [1.0, 1.0, 1.0]])
    points = np.array([[0.5, 1.0, 1.0], [1.0, 0.5, -1.0]])

    # Generate output to check
    check_out = multioutput_multivariate_normal(points, means, cov_L).numpy()

    # Will compare to a loop with scipy's implementation of log probabilities for Gaussians
    from scipy.stats import multivariate_normal

    for i in range(cov_mats.shape[0]):
        this_dist = multivariate_normal(mean=means[:, i], cov=cov_mats[i, ...])
        ref_logp = this_dist.logpdf(points[:, i])
        np.testing.assert_allclose(ref_logp, check_out[i])


def test_GP_likelihood():
    # Check that handles input correctly
    cov_id = np.eye(3)
    d_orders = np.arange(3.0)
    check = HetGaussianDeriv(cov_id, d_orders)
    check_flat = HetGaussianDeriv(np.diag(cov_id), d_orders)
    np.testing.assert_allclose(check.cov, check_flat.cov)

    # Including if input is 1D
    check_1d = HetGaussianDeriv(cov_id[:1, :1], d_orders[:1])
    check_1d_flat = HetGaussianDeriv(cov_id[0, :1], d_orders[:1])
    np.testing.assert_allclose(check_1d.cov, check_1d_flat.cov)

    # Check that applies scaling appropriately for different scenarios above
    np.testing.assert_allclose(
        check.build_scaled_cov_mat().numpy(), check_flat.build_scaled_cov_mat().numpy()
    )
    np.testing.assert_allclose(
        check_1d.build_scaled_cov_mat().numpy(),
        check_1d_flat.build_scaled_cov_mat().numpy(),
    )

    # Now check specifics of building covariance matrix
    # Check using alternative calculation
    def log_cov_model(cov0, p, s, d_o):
        d_o = d_o + 1
        return np.log(cov0) + p * np.add(*np.meshgrid(d_o, d_o)) + s

    # Default has p=10.0, s=0.0
    cov = np.array([[0.5, 0.0, 0.0], [-1.5, 2.5, 0.0], [3.5, 4.5, 5.5]])
    cov = cov.T @ cov
    check = HetGaussianDeriv(cov, d_orders)
    np.testing.assert_allclose(
        log_cov_model(cov, 10.0, 0.0, d_orders),
        np.log(check.build_scaled_cov_mat().numpy()),
    )
    # Also do for new model with p and s changed
    check_newps = HetGaussianDeriv(cov, d_orders, p=2.0, s=-2.0)
    np.testing.assert_allclose(
        log_cov_model(cov, 2.0, -2.0, d_orders),
        np.log(check_newps.build_scaled_cov_mat().numpy()),
    )


# Testing full GP model with derivative information
# This test method is quite large and slow
# Can consider turning into a test class with methods inside
# Would allow for most comparisons to a "base" model created in __init__
# But would make testing more modular and specific
# Manually parsing and running all methods in class is a pain for testing, though
@pytest.mark.slow
def test_GP():
    # First create data we can use
    rng = np.random.default_rng(42)
    x_data, y_data, y_var_data = sine_active.make_data(
        np.linspace(-np.pi, np.pi, 5), noise=0.01, max_order=2, rng=rng
    )
    cov_data = np.diag(np.squeeze(y_var_data))[
        None, ...
    ]  # Give first dimension of 1 since 1D output

    # Next specify parameters for GP model, with kernel produced by a function
    # Want to make sure no parameters shared across models we test
    def make_kern():
        kern_expr, kern_params = make_rbf_expr()
        kern = DerivativeKernel(kern_expr, 1, kernel_params=kern_params)
        return kern

    like_kwargs = {"p": 1.0, "s": -2.0}

    # Create GP model with just the 1D data
    check_1d = HeteroscedasticGPR(
        (x_data, y_data, cov_data), kernel=make_kern(), likelihood_kwargs=like_kwargs
    )

    # Check to make sure likelihood handled correctly
    ref_like = HetGaussianDeriv(cov_data, x_data[:, 1], **like_kwargs)
    np.testing.assert_allclose(ref_like.cov, check_1d.likelihood.cov)
    np.testing.assert_allclose(
        ref_like.build_scaled_cov_mat(), check_1d.likelihood.build_scaled_cov_mat()
    )

    # Since working with 1D data, scaling should not change the trained model
    # Also should not change the kernel lengthscale
    # And variance should change according to scale factor squared
    # Or for x scaling, changes lengthscale proportionally, but not variance
    check_scale = HeteroscedasticGPR(
        (x_data, y_data, cov_data),
        kernel=make_kern(),
        scale_fac=2.0,
    )
    check_xscale = HeteroscedasticGPR(
        (x_data, y_data, cov_data),
        kernel=make_kern(),
        x_scale_fac=2.0,
    )
    check_base = HeteroscedasticGPR(
        (x_data, y_data, cov_data),
        kernel=make_kern(),
    )
    train_GPR(check_base)
    train_GPR(check_scale)
    train_GPR(check_xscale)

    x_test = np.linspace(-np.pi, np.pi, 100)[:, None]
    x_test = np.block([[x_test, np.zeros_like(x_test)], [x_test, np.ones_like(x_test)]])

    pred_base = check_base.predict_f(x_test)
    pred_scale = check_base.predict_f(x_test)
    pred_xscale = check_base.predict_f(x_test)

    # Comparing predictions
    np.testing.assert_allclose(pred_base[0], pred_scale[0])
    np.testing.assert_allclose(pred_base[1], pred_scale[1])
    np.testing.assert_allclose(pred_base[0], pred_xscale[0])
    np.testing.assert_allclose(pred_base[1], pred_xscale[1])

    # Comparing parameters
    np.testing.assert_allclose(
        check_base.kernel.kernel.l.numpy(),
        check_scale.kernel.kernel.l.numpy(),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        check_base.kernel.kernel.var.numpy(),
        check_scale.kernel.kernel.var.numpy() * (check_scale.scale_fac**2),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        check_base.kernel.kernel.l.numpy(),
        check_xscale.kernel.kernel.l.numpy() / (check_xscale.x_scale_fac),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        check_base.kernel.kernel.var.numpy(),
        check_xscale.kernel.kernel.var.numpy(),
        rtol=1e-03,
    )

    # Define points for testing model prediction and compare to reference
    # This will test for correct behavior across operating systems and software versions
    x_val = np.array([-3.0, 0.5, 3.0])[:, None]
    x_val = np.block([[x_val, np.zeros_like(x_val)], [x_val, np.ones_like(x_val)]])

    pred_val_base = check_base.predict_f(x_val)
    ref_pred_means = np.array(
        [
            [-0.12116922],
            [0.54235242],
            [0.02270773],
            [-1.12612318],
            [0.87582276],
            [-1.10413224],
        ]
    )
    ref_pred_vars = np.array(
        [
            [0.00460669],
            [0.00413819],
            [0.01032148],
            [0.00841806],
            [0.00238915],
            [0.01527506],
        ]
    )
    np.testing.assert_allclose(pred_val_base[0].numpy(), ref_pred_means, rtol=1e-03)
    np.testing.assert_allclose(pred_val_base[1].numpy(), ref_pred_vars, rtol=1e-03)

    # Test handling of multiple outputs (extra y dimensions)
    x_shift, y_shift, y_var_shift = sine_active.make_data(
        np.linspace(-np.pi, np.pi, 5),
        phase_shift=np.pi / 3.0,
        noise=0.01,
        max_order=2,
        rng=rng,
    )
    cov_shift = np.diag(np.squeeze(y_var_shift))[None, ...]
    check_multiD = HeteroscedasticGPR(
        (
            x_data,
            np.hstack([y_data, y_shift]),
            np.concatenate([cov_data, cov_shift], axis=0),
        ),
        kernel=make_kern(),
    )

    # Make sure wrapping works into SharedIndependent kernel works correctly
    assert isinstance(check_multiD.kernel, gpflow.kernels.SharedIndependent)

    # Train
    train_GPR(check_multiD)

    # Produce output and compare to below - will depend on optimization procedure
    # But better than nothing?
    pred_val_multiD = check_multiD.predict_f(x_val)
    ref_multiD_means = np.array(
        [
            [-0.12177132, -0.8197274],
            [0.54288101, 0.97107731],
            [0.02439859, -0.74359773],
            [-1.11681105, -0.42663647],
            [0.87410287, -0.01394911],
            [-1.0930425, -0.48817702],
        ]
    )
    ref_multiD_vars = np.array(
        [
            [0.00461916, 0.00477108],
            [0.00430657, 0.00396219],
            [0.01033009, 0.00807202],
            [0.00865764, 0.00895764],
            [0.00256747, 0.00244668],
            [0.01557182, 0.01329149],
        ]
    )
    np.testing.assert_allclose(pred_val_multiD[0].numpy(), ref_multiD_means, rtol=1e-03)
    np.testing.assert_allclose(pred_val_multiD[1].numpy(), ref_multiD_vars, rtol=1e-03)

    # Make sure multiple outputs also work with separate independent kernels
    check_sepInd = HeteroscedasticGPR(
        (
            x_data,
            np.hstack([y_data, y_shift]),
            np.concatenate([cov_data, cov_shift], axis=0),
        ),
        kernel=gpflow.kernels.SeparateIndependent([make_kern(), make_kern()]),
    )

    # With independent kernels, can compare first dimension to model trained on 1D data only
    train_GPR(check_sepInd)
    pred_sepInd = check_sepInd.predict_f(x_test)
    np.testing.assert_allclose(pred_base[0], pred_sepInd[0][:, :1], rtol=1e-03)
    np.testing.assert_allclose(pred_base[1], pred_sepInd[1][:, :1], rtol=1e-03)
    np.testing.assert_allclose(
        check_base.kernel.kernel.l.numpy(),
        check_sepInd.kernel.kernels[0].l.numpy(),
        rtol=1e-03,
    )
    np.testing.assert_allclose(
        check_base.kernel.kernel.var.numpy(),
        check_sepInd.kernel.kernels[0].var.numpy(),
        rtol=1e-03,
    )

    # Test output with full_cov (specifically variance, second output)
    # Best done with multidimensional output data
    pred_FF = check_sepInd.predict_f(x_data, full_cov=False)[1]  # Default
    pred_TF = check_sepInd.predict_f(x_data, full_cov=True)[1]

    # Assert correct shapes
    assert pred_FF.shape == (x_data.shape[0], 2)
    assert pred_TF.shape == (2, x_data.shape[0], x_data.shape[0])

    # And assert that diagonal of full_cov=True matches full_cov=False
    np.testing.assert_allclose(
        pred_FF.numpy(), np.vstack([np.diag(v) for v in pred_TF]).T
    )

    # Check proper handling of mean functions
    # Will assume mean function itself correct, so just checking how model uses it
    # Easy enough to pass in constant mean function and make sure model output very similar
    # Does change predictions, just shouldn't change by much
    # To work, need to make sure no scaling happens to covariance matrix
    mean_func = ConstantMeanWithDerivs(np.ones((3, 1)))
    check_meanf = HeteroscedasticGPR(
        (x_data, y_data, cov_data),
        kernel=make_kern(),
        mean_function=mean_func,
        likelihood_kwargs={"p": 0.0, "transform_p": None},
    )

    check_meanf.kernel.kernel.l.assign(1e-06)
    np.testing.assert_allclose(y_data, check_meanf.predict_f(x_data)[0], atol=1e-01)

    train_GPR(check_meanf)
    pred_meanf = check_meanf.predict_f(x_test)

    # Can compare predictions and lengthscale parameter, but not kernel variance parameter
    np.testing.assert_allclose(pred_base[0], pred_meanf[0], atol=1e-01)
    np.testing.assert_allclose(pred_base[1], pred_meanf[1], atol=1e-01)
    np.testing.assert_allclose(
        check_base.kernel.kernel.l.numpy(),
        check_meanf.kernel.kernel.l.numpy(),
        atol=2e-01,
    )

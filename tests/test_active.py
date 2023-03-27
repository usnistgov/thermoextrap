# Written by Jacob I. Monroe, NIST employee

"""
Tests for active learning based on those GP models with derivatives.
"""

import gpflow
import numpy as np
import pytest
import sympy as sp
from scipy import linalg

from thermoextrap import idealgas
from thermoextrap.gpr_active import active_utils, gp_models, ig_active

# For now, not testing DataWrapper or SimWrapper objects
# These will likely change in future iterations and be highly dependent on individual problems


# First test some simple utility functions, starting with sympy expressions for RBF
def test_rbf_expr():
    l_sym = sp.symbols("l", real=True)
    var_sym = sp.symbols("var", real=True)
    x1_sym = sp.symbols("x1", real=True)
    x2_sym = sp.symbols("x2", real=True)

    check_expr, check_params = active_utils.make_rbf_expr()
    assert l_sym in check_expr.free_symbols
    assert var_sym in check_expr.free_symbols
    assert x1_sym in check_expr.free_symbols
    assert x2_sym in check_expr.free_symbols
    assert "l" in check_params.keys()
    assert "var" in check_params.keys()

    def rbf(var, l, x1, x2):
        return var * np.exp(-0.5 * (x1 / l - x2 / l) ** 2)

    check_var = np.array([0.0, 1.0, 2.0, -1.0])
    check_l = np.array([1.0, 2.0])
    check_x = np.array([0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    for v in check_var:
        for l in check_l:
            for x1 in check_x:
                for x2 in check_x:
                    out_expr = check_expr.subs(
                        [(var_sym, v), (l_sym, l), (x1_sym, x1), (x2_sym, x2)]
                    )
                    assert out_expr == rbf(v, l, x1, x2)


# Next test function for building GP inputs from thermoextrap ExtrapModel objects
# (representing thermodynamic states)
def test_make_GP_input():
    beta = 5.6
    state = ig_active.extrap_IG(beta)

    # Test without logarithm on x
    check_x, check_y, check_cov = active_utils.input_GP_from_state(state)
    # Check shapes
    assert check_x.shape == (state.order + 1, 2)
    assert check_y.shape == (state.order + 1, 1)
    assert check_cov.shape == (1, state.order + 1, state.order + 1)
    # Check derivatives calculated
    for k in range(3):
        np.testing.assert_allclose(
            check_y[k], idealgas.dbeta_xave(k)(beta, 1.0), rtol=(10**k) * 1e-02
        )

    # Test with logarithm on x
    # Note that using base 10 logarithm
    # Using sympy to compare because have analytical expression for ideal gas
    def dxave_dlogbeta(k):
        logbeta_sym = sp.symbols("lnbeta_sym")
        log_expr = idealgas.xave_sym.subs([(idealgas.beta_sym, 10 ** (logbeta_sym))])
        deriv = sp.diff(log_expr, logbeta_sym, k)
        return sp.lambdify([logbeta_sym, idealgas.vol_sym], deriv, "numpy")

    check_x_log, check_y_log, check_cov_log = active_utils.input_GP_from_state(
        state, log_scale=True
    )
    np.testing.assert_equal(np.log10(check_x[:, 0]), check_x_log[:, 0])
    assert check_x_log.shape == (state.order + 1, 2)
    assert check_y_log.shape == (state.order + 1, 1)
    assert check_cov_log.shape == (1, state.order + 1, state.order + 1)
    for k in range(3):
        np.testing.assert_allclose(
            check_y_log[k],
            dxave_dlogbeta(k)(np.log10(beta), 1.0),
            rtol=(10**k) * 1e-02,
        )

    # Test multiple output dimensions
    def dbeta_xsqave(k):
        beta_sym = sp.symbols("beta_sym")
        xsqave_sym = (
            (2 / (beta_sym**2))
            - (2 / (beta_sym * (sp.exp(beta_sym) - 1)))
            - (1 / (sp.exp(beta_sym) - 1))
        )
        deriv = sp.diff(xsqave_sym, beta_sym, k)
        return sp.lambdify(
            [
                beta_sym,
            ],
            deriv,
            "numpy",
        )

    state_mult = ig_active.multiOutput_extrap_IG(beta)
    check_x_mult, check_y_mult, check_cov_mult = active_utils.input_GP_from_state(
        state_mult
    )
    assert check_x_mult.shape == (state.order + 1, 2)
    assert check_y_mult.shape == (state.order + 1, 2)
    assert check_cov_mult.shape == (2, state.order + 1, state.order + 1)
    for k in range(3):
        np.testing.assert_allclose(
            check_y_mult[k, 0],
            idealgas.dbeta_xave(k)(beta, 1.0),
            rtol=(10**k) * 1e-02,
        )
        np.testing.assert_allclose(
            check_y_mult[k, 1], dbeta_xsqave(k)(beta), rtol=(10**k) * 1e-02
        )


# Next want to test creation of a GP model
def test_base_GP_creation():
    # Need data to work with
    raw_x_data = []
    raw_y_data = []
    raw_cov_data = []
    for beta in [1.0, 5.6, 9.0]:
        s = ig_active.extrap_IG(beta)
        this_x, this_y, this_cov = active_utils.input_GP_from_state(s)
        raw_x_data.append(this_x)
        raw_y_data.append(this_y)
        raw_cov_data.append(this_cov)

    # Mainly testing options, so testing decision structure
    # Test using 2 data points
    x_data = np.vstack([raw_x_data[0], raw_x_data[-1]])
    y_data = np.vstack([raw_y_data[0], raw_y_data[-1]])
    cov_data = linalg.block_diag(*[raw_cov_data[0][0, ...], raw_cov_data[-1][0, ...]])
    data_input_2p = (x_data, y_data, cov_data)

    order = int(np.max(x_data[:, 1]))

    check_gp = active_utils.create_base_GP_model(
        data_input_2p, d_order_ref=0, shared_kernel=True
    )
    # Should have constant mean function
    assert isinstance(check_gp.mean_function, gp_models.ConstantMeanWithDerivs)
    # And should scale by the std of the y inputs - with 2 data points, that's just the distance to the mean
    # Also taking zeroth order data because should be 0 order for d_order_ref
    np.testing.assert_allclose(
        check_gp.scale_fac, np.abs(y_data[0, :] - np.mean(y_data[:: (order + 1), :]))
    )

    # Test using 3 data points
    data_input_3p = (
        np.vstack(raw_x_data),
        np.vstack(raw_y_data),
        linalg.block_diag(*[dat[0, ...] for dat in raw_cov_data]),
    )
    check_gp_3p = active_utils.create_base_GP_model(
        data_input_3p, d_order_ref=0, shared_kernel=True
    )
    # Should be same as for 2 points but use linear mean function
    assert isinstance(check_gp_3p.mean_function, gp_models.LinearWithDerivs)
    # And check that scaling factor is std of zeroth-order data
    np.testing.assert_allclose(
        check_gp_3p.scale_fac,
        np.std(
            np.vstack(raw_y_data)[:: (order + 1), :]
            - check_gp_3p.mean_function(np.vstack(raw_x_data)[:: (order + 1), :]),
            axis=0,
        ),
    )

    # Or with 1 data point
    data_input_1p = (
        np.vstack([raw_x_data[0]]),
        np.vstack([raw_y_data[0]]),
        linalg.block_diag(*[raw_cov_data[0][0, ...]]),
    )
    check_gp_1p = active_utils.create_base_GP_model(
        data_input_1p, d_order_ref=0, shared_kernel=True
    )
    assert isinstance(check_gp_1p.mean_function, gp_models.ConstantMeanWithDerivs)
    # And scaling factor should be 1
    np.testing.assert_allclose(check_gp_1p.scale_fac, 1.0)

    # Test if d_order_ref is 1, not 0
    check_gp_d1 = active_utils.create_base_GP_model(
        data_input_2p, d_order_ref=1, shared_kernel=True
    )
    assert isinstance(check_gp_d1.mean_function, gp_models.ConstantMeanWithDerivs)
    # And should use 1st derivative to determine scaling instead of zeroth order
    np.testing.assert_allclose(
        check_gp_d1.scale_fac,
        np.abs(y_data[1, :] - np.mean(y_data[1 :: (order + 1), :])),
    )

    # Check properties of kernel
    # Make sure using shared kernel if shared_kernel=True
    assert isinstance(check_gp.kernel, gpflow.kernels.SharedIndependent)
    # Should be an RBF kernel (for default inputs, but just check here)
    # (note default kernel is SharedIndependent, so referencing kernel.kernel)
    assert (
        sp.simplify(
            check_gp.kernel.kernel.kernel_expr - active_utils.make_rbf_expr()[0]
        )
        == 0
    )

    # If shared_kernel is False, check to make sure uses SeparateIndependent
    check_gp_sep = active_utils.create_base_GP_model(
        data_input_2p, d_order_ref=0, shared_kernel=False
    )
    assert isinstance(check_gp_sep.kernel, gpflow.kernels.SeparateIndependent)
    assert (
        sp.simplify(
            check_gp_sep.kernel.kernels[0].kernel_expr - active_utils.make_rbf_expr()[0]
        )
        == 0
    )
    for k in check_gp_sep.kernel.kernels[1:]:
        for j, t_par in enumerate(k.trainable_parameters):
            assert check_gp_sep.kernel.kernels[0].trainable_parameters[j] != t_par

    # Test if can pass in full kernel
    # Note instantiation of kernel so is object, not class
    # And note that checking to make sure gets passed through to HeteroscedasticGPR
    # and wrapped as SharedIndependent, even though shared_kernel is False
    # Warning should be printed, so should come up with some way to check for that...
    k_rbf = active_utils.RBFDerivKernel()
    # Changing parameter values to check if passed faithfully
    k_rbf.l.assign(0.5)
    k_rbf.var.assign(5.0)
    check_kernel = active_utils.create_base_GP_model(
        data_input_1p, d_order_ref=0, shared_kernel=False, kernel=k_rbf
    )
    assert isinstance(check_kernel.kernel, gpflow.kernels.SharedIndependent)
    assert check_kernel.kernel.kernel == k_rbf
    assert check_kernel.kernel.kernel.l.numpy() == 0.5
    assert check_kernel.kernel.kernel.var.numpy() == 5.0


# Simple test for checking training of GP model
# Test is a bit slow, though
@pytest.mark.slow
def test_train_GP():
    # Will compare training results to a reference
    # Need data to work with
    raw_x_data = []
    raw_y_data = []
    raw_cov_data = []
    for beta in [1.0, 5.6, 9.0]:
        s = ig_active.extrap_IG(beta)
        this_x, this_y, this_cov = active_utils.input_GP_from_state(s)
        raw_x_data.append(this_x)
        raw_y_data.append(this_y)
        raw_cov_data.append(this_cov)

    # Mainly testing options, so testing decision structure
    # Test using 2 data points
    x_data = np.vstack(raw_x_data)
    y_data = np.vstack(raw_y_data)
    cov_data = linalg.block_diag(*[dat[0, ...] for dat in raw_cov_data])

    gp = active_utils.create_base_GP_model((x_data, y_data, cov_data))
    output = active_utils.train_GPR(gp, record_loss=False)
    assert output is None

    gp = active_utils.create_base_GP_model((x_data, y_data, cov_data))
    output = active_utils.train_GPR(gp, record_loss=True)
    assert output is not None

    ref_params = [4.9320208698871015, 17.07650174925236, 0.0, 4.3460319697777543e-16]
    for p, rp in zip(gp.parameters, ref_params):
        np.testing.assert_allclose(p.numpy(), rp, rtol=1e-01, atol=1e-12)

    # Also test training from a different starting point
    # Not checking for specific behavior here, just seeing if runs
    output = active_utils.train_GPR(
        gp, record_loss=False, start_params=[ref_params[ind] for ind in [0, 1, 3]]
    )


# Simple test for creating a GP model from list of states
# Also a bit slow
@pytest.mark.slow
def test_create_GP_from_states():
    # Need data to work with
    states = []
    for beta in [1.0, 5.6, 9.0]:
        states.append(ig_active.extrap_IG(beta))

    n_gp_points = np.sum([s.order + 1 for s in states])

    gp = active_utils.create_GPR(states)
    assert gp.data[0].shape == (n_gp_points, 2)
    assert gp.data[1].shape == (n_gp_points, 1)
    assert gp.likelihood.cov.shape == (gp.data[1].shape[1], n_gp_points, n_gp_points)

    # Also test with multidimensional data
    states_mult = []
    for beta in [1.0, 5.6, 9.0]:
        states_mult.append(ig_active.multiOutput_extrap_IG(beta))
    gp_mult = active_utils.create_GPR(states_mult)
    assert gp_mult.data[0].shape == (n_gp_points, 2)
    assert gp_mult.data[1].shape == (n_gp_points, 2)
    assert gp_mult.likelihood.cov.shape == (
        gp_mult.data[1].shape[1],
        n_gp_points,
        n_gp_points,
    )


# Testing update and stopping function classes
@pytest.mark.slow
def test_update_stop_ABC():
    # Need data to work with
    states = []
    beta_list = [1.0, 5.6, 9.0]
    for beta in beta_list:
        states.append(ig_active.extrap_IG(beta))
    # And trained GP
    gp = active_utils.create_GPR(states)

    # Start by checking with default inputs
    check_default = active_utils.UpdateStopABC()

    # Check creation of alpha grid
    out_default_grid, out_default_sel = check_default.create_alpha_grid(beta_list)
    np.testing.assert_allclose(
        out_default_grid, np.linspace(beta_list[0], beta_list[-1], 1000)
    )
    np.testing.assert_equal(out_default_grid, out_default_sel)

    # With log scale and avoiding repeats
    check_logrepeats = active_utils.UpdateStopABC(log_scale=True, avoid_repeats=True)
    out_logrepeats_grid, out_logrepeats_sel = check_logrepeats.create_alpha_grid(
        beta_list
    )
    np.testing.assert_equal(
        out_logrepeats_grid,
        np.linspace(np.log10(beta_list[0]), np.log10(beta_list[-1]), 1000),
    )
    assert not np.array_equal(out_logrepeats_grid, out_logrepeats_sel)
    # Make sure randomization stays within grid
    assert np.all(
        abs(out_logrepeats_sel - out_logrepeats_grid[1:-1])
        <= (out_logrepeats_grid[1] - out_logrepeats_grid[0])
    )

    # Check transformation for default - should just return same thing
    x = np.arange(10.0)
    y = x**3
    out_notrans_y, out_notrans_std, out_notrans_conf = check_default.transform_func(
        x, y, 1.0
    )
    np.testing.assert_equal(out_notrans_y, y)
    np.testing.assert_allclose(out_notrans_std, 1.0)
    np.testing.assert_allclose(out_notrans_conf[0], y - 2.0 * 1.0)
    np.testing.assert_allclose(out_notrans_conf[1], y + 2.0 * 1.0)

    # And if provide fancier transformation
    def fancy_transform(x, y, y_var):
        # Transformation itself being checked, resulting uncertainty and confidence
        # interval are nonsense
        y_std = np.sqrt(y_var)
        out = x * (y ** (1 / 3))
        conf_int = [x * (y - 2.0 * y_std), x * (y + 2.0 * y_std)]
        return out, y_std, conf_int

    check_trans = active_utils.UpdateStopABC(transform_func=fancy_transform)
    out_trans_y = check_trans.transform_func(x, y, 1.0)[0]
    np.testing.assert_allclose(out_trans_y, x**2)

    # Check to make sure GP used properly for default case (just runs the function)
    (
        out_default_mu,
        out_default_std,
        out_default_conf,
    ) = check_default.get_transformed_GP_output(gp, out_default_grid)


# For update classes, all have different update criteria
# Rather than check all of these (hard for random...)
# just check to make sure satisfy correct input/output structure
@pytest.mark.slow
def test_update_classes():
    # Need data to work with
    states = []
    beta_list = [1.0, 5.6, 9.0]
    for beta in beta_list:
        states.append(ig_active.extrap_IG(beta))
    # And trained GP
    gp = active_utils.create_GPR(states)

    # Start with base, which should throw an error
    check_base = active_utils.UpdateFuncBase()
    # Should be able to call, which just wraps do_update()
    np.testing.assert_raises(NotImplementedError, check_base, gp, beta_list)
    np.testing.assert_raises(NotImplementedError, check_base.do_update, gp, beta_list)

    # Random update function
    check_rand = active_utils.UpdateRandom()
    out_rand = check_rand(gp, beta_list)
    assert len(out_rand) == 3
    assert isinstance(out_rand[0], float)

    # Space filling update function (furthest point)
    check_space = active_utils.UpdateSpaceFill()
    out_space = check_space(gp, beta_list)
    assert len(out_space) == 3
    assert isinstance(out_space[0], float)

    # ALM (max variance) update function
    check_ALM = active_utils.UpdateALMbrute()
    out_ALM = check_ALM(gp, beta_list)
    assert len(out_ALM) == 3
    assert isinstance(out_ALM[0], float)

    # Adaptive integration updates
    # (furthest point satisfying error tolerance)
    check_adapt = active_utils.UpdateAdaptiveIntegrate(tol=0.005)
    out_adapt = check_adapt(gp, beta_list)
    assert len(out_adapt) == 3
    assert isinstance(out_adapt[0], float)


@pytest.mark.slow
def test_update_classes_multioutput():
    # Need data to work with
    states = []
    beta_list = [1.0, 5.6, 9.0]
    for beta in beta_list:
        states.append(ig_active.multiOutput_extrap_IG(beta))
    # And trained GP
    gp = active_utils.create_GPR(states)

    # Random update function
    check_rand = active_utils.UpdateRandom()
    out_rand = check_rand(gp, beta_list)
    assert len(out_rand) == 3
    assert isinstance(out_rand[0], float)
    assert out_rand[1].shape[0] == 2

    # Space filling update function (furthest point)
    check_space = active_utils.UpdateSpaceFill()
    out_space = check_space(gp, beta_list)
    assert len(out_space) == 3
    assert isinstance(out_space[0], float)
    assert out_space[1].shape[0] == 2

    # ALM (max variance) update function
    check_ALM = active_utils.UpdateALMbrute()
    out_ALM = check_ALM(gp, beta_list)
    assert len(out_ALM) == 3
    assert isinstance(out_ALM[0], float)
    assert out_ALM[1].shape[0] == 2

    # Adaptive integration updates
    # (furthest point satisfying error tolerance)
    check_adapt = active_utils.UpdateAdaptiveIntegrate()
    out_adapt = check_adapt(gp, beta_list)
    assert len(out_adapt) == 3
    assert isinstance(out_adapt[0], float)
    assert out_adapt[1].shape[0] == 2


# Same for metric classes
# just check mechanics for taking inputs and generic features of outputs
@pytest.mark.slow
def test_metrics():
    # Need to create inputs to work with
    # Expects "history" which is a list of array-likes
    # where index zero is GP means over time and index one is vars
    x = np.arange(-10.0, 10.0, 1.0)
    hist = [np.vstack([x, x**3]), np.vstack([0.01 * x**2, 0.01 * x**2])]

    # Base class
    tol = 1e-03
    check_base = active_utils.MetricBase("Base", tol)
    assert check_base.tol == tol
    # Use to test _check_history function for all inheriting classes
    np.testing.assert_raises(ValueError, check_base._check_history, None)
    np.testing.assert_raises(ValueError, check_base._check_history, x)
    assert check_base._check_history(hist) is None
    # And generic call and calc_metric both work
    np.testing.assert_raises(NotImplementedError, check_base, hist, x, None)
    np.testing.assert_raises(NotImplementedError, check_base.calc_metric, hist, x, None)

    # Other classes
    check_maxvar = active_utils.MaxVar(tol)
    assert isinstance(check_maxvar(hist, x, None), float)

    check_avgvar = active_utils.AvgVar(tol)
    assert isinstance(check_avgvar(hist, x, None), float)

    check_maxrelvar = active_utils.MaxRelVar(tol)
    assert isinstance(check_maxrelvar(hist, x, None), float)

    check_avgrelvar = active_utils.AvgRelVar(tol)
    assert isinstance(check_avgrelvar(hist, x, None), float)

    check_msd = active_utils.MSD(tol)
    assert isinstance(check_msd(hist, x, None), float)

    check_maxabsreldev = active_utils.MaxAbsRelDeviation(tol)
    assert isinstance(check_maxabsreldev(hist, x, None), float)

    check_avgabsreldev = active_utils.AvgAbsRelDeviation(tol)
    assert isinstance(check_avgabsreldev(hist, x, None), float)

    # For metric ensuring maximum iterations, do check criteria since important and easy (never reaches tol)
    check_maxiter = active_utils.MaxIter()
    assert isinstance(check_maxiter(hist, x, None), float)
    assert check_maxiter(hist, x, None) > check_maxiter.tol

    # For special ErrorStability class, actually need a GP model
    states = []
    beta_list = [1.0, 2.3, 5.6, 9.0]
    for beta in beta_list:
        states.append(ig_active.extrap_IG(beta))
    # And trained GP
    gp_2 = active_utils.create_GPR([states[0], states[-1]])
    gp_3 = active_utils.create_GPR([states[0], states[2], states[-1]])
    gp_4 = active_utils.create_GPR(states)

    check_errorstability = active_utils.ErrorStability(0.05)
    # Check that returns 1 for just 2 points
    assert check_errorstability(hist, x, gp_2) == 1
    # Check that sets normalization once provide more points
    assert check_errorstability.r1 is None
    out_errorstability = check_errorstability(hist, x, gp_3)
    assert isinstance(check_errorstability.r1, float)
    assert isinstance(out_errorstability, float)
    assert out_errorstability == 1.0
    # Check with 4 points
    out_errorstability_4 = check_errorstability(hist, x, gp_4)
    assert isinstance(out_errorstability, float)
    assert out_errorstability_4 < out_errorstability

    # Also need GP for relative quantities compared to variance of GP data
    check_maxrelglobalvar = active_utils.MaxRelGlobalVar(tol)
    assert isinstance(check_maxrelglobalvar(hist, x, gp_4), float)

    check_maxabsrelglobaldev = active_utils.MaxAbsRelGlobalDeviation(tol)
    assert isinstance(check_maxabsrelglobaldev(hist, x, gp_4), float)


# Test class for implementing stopping criteria
@pytest.mark.slow
def test_stop_criteria():
    # Need data to work with
    states = []
    beta_list = [1.0, 5.6, 9.0]
    for beta in beta_list:
        states.append(ig_active.extrap_IG(beta))
    # And trained GP
    gp_prev = active_utils.create_GPR([states[0], states[-1]])
    gp = active_utils.create_GPR(states)

    tol = 1e-03
    m_var = active_utils.MaxVar(tol)
    m_msd = active_utils.MSD(tol)
    m_es = active_utils.ErrorStability(tol)

    # With one metric
    check_stop_1m = active_utils.StopCriteria([m_var])
    # Test call for first round without history
    assert check_stop_1m.history is None
    out_1m_bool, out_1m_info = check_stop_1m(gp_prev, beta_list)
    assert check_stop_1m.history is not None
    assert len(check_stop_1m.history) == 2
    assert check_stop_1m.history[0].shape == (1, 1000, 1)
    assert check_stop_1m.history[1].shape == (1, 1000, 1)
    assert out_1m_bool.dtype == bool
    assert isinstance(out_1m_info, dict)
    assert len(out_1m_info.keys()) == 2  # Also have key for the tolerance
    # Call again and check if updated correctly
    out_1m_bool, out_1m_info = check_stop_1m(gp, beta_list)
    assert len(check_stop_1m.history) == 2
    assert check_stop_1m.history[0].shape == (2, 1000, 1)
    assert check_stop_1m.history[1].shape == (2, 1000, 1)
    assert out_1m_bool.dtype == bool
    assert isinstance(out_1m_info, dict)
    assert len(out_1m_info.keys()) == 2

    # With multiple metrics
    check_stop_2m = active_utils.StopCriteria([m_var, m_msd])
    assert check_stop_2m.history is None
    out_2m_bool, out_2m_info = check_stop_2m(gp_prev, beta_list)
    assert check_stop_2m.history is not None
    assert len(check_stop_2m.history) == 2
    assert check_stop_2m.history[0].shape == (1, 1000, 1)
    assert check_stop_2m.history[1].shape == (1, 1000, 1)
    assert out_2m_bool.dtype == bool
    assert isinstance(out_2m_info, dict)
    assert len(out_2m_info.keys()) == 4
    out_2m_bool, out_2m_info = check_stop_2m(gp, beta_list)
    assert len(check_stop_2m.history) == 2
    assert check_stop_2m.history[0].shape == (2, 1000, 1)
    assert check_stop_2m.history[1].shape == (2, 1000, 1)
    assert out_2m_bool.dtype == bool
    assert isinstance(out_2m_info, dict)
    assert len(out_2m_info.keys()) == 4

    # Check that correctly manipulates ErrorStability metric attributes
    def check_transform(x, y, y_var):
        y_std = x * np.sqrt(y_var)
        out = x * y
        conf = [x * (y - 2.0 * y_std), x * (y + 2.0 * y_std)]
        return out, y_std, conf

    check_stop_withES = active_utils.StopCriteria(
        [m_var, m_msd, m_es],
        d_order_pred=1,
        transform_func=check_transform,
        log_scale=True,
    )
    assert m_es.d_order_pred == 1
    assert m_es.transform_func == check_stop_withES.transform_func
    assert m_es.log_scale is True


# Test full active learning routine on ideal gas
# Need way to make this fully reproducible...
# Specifically, need to be able to set random seed for all parts of process
# Includes bootstrapping of data in thermoextrap/cmomy
# Just need to carefully check where calling np.random throughout all code
import io
from contextlib import redirect_stdout


@pytest.mark.slow
def test_active_learning():
    # Starting beta values
    init_states = [1.0, 9.6]
    sims = ig_active.SimulateIG()
    updates = active_utils.UpdateALMbrute()
    metrics = [
        active_utils.MaxVar(1e-03),
        active_utils.MaxRelVar(1e-02),
        active_utils.MSD(1.0),
        active_utils.MaxAbsRelDeviation(1e-02),
        active_utils.MaxIter(),
    ]

    with io.StringIO() as buf, redirect_stdout(buf):
        # Once where should reach maximum iterations
        stops = active_utils.StopCriteria(metrics)
        active_utils.active_learning(
            init_states,
            sims,
            updates,
            stop_criteria=stops,
            max_iter=4,
        )
        output = buf.getvalue()
    assert "Reached maximum iterations" in output

    with io.StringIO() as buf, redirect_stdout(buf):
        # And once where expect to stop early
        stops_early = active_utils.StopCriteria(metrics[:-1])
        active_utils.active_learning(
            init_states,
            sims,
            updates,
            stop_criteria=stops_early,
            max_iter=4,
        )
        output = buf.getvalue()
    assert "Stopping criteria satisfied" in output

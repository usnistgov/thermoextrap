import numpy as np
import pytest
import xarray as xr

import thermoextrap
import thermoextrap.xtrapy.data as xtrapy_data
import thermoextrap.xtrapy.idealgas as idealgas
import thermoextrap.xtrapy.models as xtrapy_models
import thermoextrap.xtrapy.xpan_beta as xpan_beta
from thermoextrap.xtrapy.cached_decorators import gcached


@pytest.mark.slow
def test_xpan_beta_derivs_slow(fixture):
    a = np.array(fixture.derivs_list)
    s = xpan_beta.factory_derivatives(xalpha=False, central=False)
    b = s.derivs(fixture.rdata, norm=False)
    np.testing.assert_allclose(a, b)

    s = xpan_beta.factory_derivatives(xalpha=False, central=True)
    b = s.derivs(fixture.cdata, norm=False)
    np.testing.assert_allclose(a, b)


def test_xpan_beta_derivs(fixture):
    s = xpan_beta.factory_derivatives(xalpha=False, central=False)
    b = s.derivs(fixture.rdata, norm=False)
    fixture.xr_test(b, s.derivs(fixture.xrdata, norm=False))
    fixture.xr_test(b, s.derivs(fixture.xrdata_val, norm=False))

    # central
    s = xpan_beta.factory_derivatives(xalpha=False, central=True)
    b = s.derivs(fixture.cdata, norm=False)
    fixture.xr_test(b, s.derivs(fixture.xdata, norm=False))
    fixture.xr_test(b, s.derivs(fixture.xdata_val, norm=False))


@pytest.mark.slow
def test_extrapmodel_slow(fixture):
    betas = [0.3, 0.4]
    em = fixture.em
    xem = xpan_beta.factory_extrapmodel(beta=fixture.beta0, data=fixture.rdata)
    np.testing.assert_allclose(em.predict(betas, order=3), xem.predict(betas, order=3))


@pytest.mark.slow
def test_extrapmodel_resample_slow(fixture):
    betas = [0.3, 0.4]
    em = fixture.em
    xem = xpan_beta.factory_extrapmodel(beta=fixture.beta0, data=fixture.rdata)

    a = em.bootstrap(betas, n=10, order=3)
    b = xem.resample(nrep=10).predict(betas).std("rep")

    np.testing.assert_allclose(a, b, atol=0.1, rtol=0.1)


def test_extrapmodel(fixture):
    betas = [0.3, 0.4]
    xem0 = xpan_beta.factory_extrapmodel(beta=fixture.beta0, data=fixture.rdata)

    for data in [
        fixture.cdata,
        fixture.xdata,
        fixture.xrdata,
        fixture.xdata_val,
        fixture.xrdata_val,
    ]:
        xem1 = xpan_beta.factory_extrapmodel(beta=fixture.beta0, data=fixture.cdata)
        fixture.xr_test(xem0.predict(betas, order=3), xem1.predict(betas, order=3))


def test_extrapmodel_ig():
    ref_beta = 5.0
    max_order = 3
    test_betas = np.array([4.9, 5.1])
    ref_vol = 1.0

    # Get ideal gas data to compare to analytical results
    # Should consider moving ideal-gas generated data to conftest.py and use as type of fixture
    xdata, udata = idealgas.generate_data((100_000, 1), ref_beta, ref_vol)
    xdata = xr.DataArray(xdata, dims=["rec"])
    udata = xr.DataArray(udata, dims=["rec"])
    dat = xpan_beta.DataCentralMomentsVals.from_vals(
        order=max_order, xv=xdata, uv=udata, central=True
    )

    # Create extrapolation model to test against analytical
    ex = xpan_beta.factory_extrapmodel(ref_beta, dat, xalpha=False)
    # Will need estimate of uncertainty for test data so can check if within that bound
    # So resample
    ex_res = ex.resample(nrep=100)

    # Loop over orders and compare based on uncertainty
    for o in range(max_order + 1):
        # Get the exact values we're shooting for
        true_extrap, true_derivs = idealgas.x_beta_extrap(
            o, ref_beta, test_betas, ref_vol
        )
        # Get the derivatives up to this order
        test_derivs = ex.derivs(order=o, norm=False).values
        # And extrapolations
        test_extrap = ex.predict(test_betas, order=o).values
        test_derivs_err = ex_res.derivs(order=o, norm=False).std("rep").values
        test_extrap_err = ex_res.predict(test_betas, order=o).std("rep").values
        # Redefine as confidence interval, and just for the highest derivative order
        test_derivs_err = 2.0 * test_derivs_err[-1]
        test_extrap_err = 2.0 * np.max(
            test_extrap_err
        )  # Taking max because can't be array
        # Just checking to make sure within 1 std
        # Not the best check... tried p-values...
        # But bootstrapped std decreases faster with N than absolute error from analytical
        # Must be good way to do this, but not sure what it is
        # As long as well-control random number seed and number of samples, should work
        print("Order %i" % o)
        print(true_derivs[-1], test_derivs[-1])
        print(true_extrap, test_extrap)
        np.testing.assert_allclose(
            true_derivs[-1], test_derivs[-1], rtol=0.0, atol=test_derivs_err * 5
        )
        np.testing.assert_allclose(
            true_extrap, test_extrap, rtol=0.0, atol=test_extrap_err
        )


def test_extrapmodel_resample(fixture):
    betas = [0.3, 0.4]

    ndat = len(fixture.u)
    nrep = 10
    idx = np.random.choice(ndat, (nrep, ndat), replace=True)

    xem0 = xpan_beta.factory_extrapmodel(beta=fixture.beta0, data=fixture.rdata)
    a = xem0.resample(indices=idx).predict(betas, order=3)

    for data in [
        fixture.cdata,
        fixture.xdata,
        fixture.xrdata,
        fixture.xdata_val,
        fixture.xrdata_val,
    ]:
        xem1 = xpan_beta.factory_extrapmodel(beta=fixture.beta0, data=fixture.cdata)
        b = xem1.resample(indices=idx).predict(betas, order=3)
        fixture.xr_test(a, b)


def test_perturbmodel(fixture):
    beta0 = 0.5

    betas = [0.3, 0.7]
    pm = thermoextrap.PerturbModel(beta0, xData=fixture.x, uData=fixture.u)

    xpm = xpan_beta.factory_perturbmodel(beta0, uv=fixture.u, xv=fixture.x)

    np.testing.assert_allclose(pm.predict(betas), xpm.predict(betas))


@pytest.mark.slow
def test_extrapmodel_weighted_slow(fixture):

    beta0 = [0.05, 0.5]

    X = np.array((fixture.x, fixture.xb))
    U = np.array((fixture.u, fixture.ub))

    emw = thermoextrap.ExtrapWeightedModel(fixture.order, beta0, xData=X, uData=U)

    betas = [0.3, 0.4]

    # stateB
    xem0 = xpan_beta.factory_extrapmodel(beta=beta0[0], data=fixture.rdata)
    xem1 = xpan_beta.factory_extrapmodel(
        beta=beta0[1],
        data=xpan_beta.factory_data(
            uv=fixture.ub, xv=fixture.xb, order=fixture.order, central=False
        ),
    )

    xemw = xtrapy_models.ExtrapWeightedModel([xem0, xem1])

    np.testing.assert_allclose(
        emw.predict(betas, order=3), xemw.predict(betas, order=3)
    )


def test_extrapmodel_weighted(fixture):

    beta0 = [0.05, 0.5]
    betas = [0.3, 0.4]

    # stateB
    xem0 = xpan_beta.factory_extrapmodel(beta=beta0[0], data=fixture.rdata)
    xem1 = xpan_beta.factory_extrapmodel(
        beta=beta0[1],
        data=xpan_beta.factory_data(
            uv=fixture.ub, xv=fixture.xb, order=fixture.order, central=False
        ),
    )

    xemw = xtrapy_models.ExtrapWeightedModel([xem0, xem1])

    a = xemw.predict(betas, order=3)

    # central
    xem0 = xpan_beta.factory_extrapmodel(beta=beta0[0], data=fixture.cdata)
    xem1 = xpan_beta.factory_extrapmodel(
        beta=beta0[1],
        data=xpan_beta.factory_data(
            uv=fixture.ub,
            xv=fixture.xb,
            order=fixture.order,
            central=True,
        ),
    )

    xemw1 = xtrapy_models.ExtrapWeightedModel([xem0, xem1])
    b = xemw1.predict(betas, order=3)
    fixture.xr_test(a, b)

    # xdata
    xem0 = xpan_beta.factory_extrapmodel(beta=beta0[0], data=fixture.xdata)
    xem1 = xpan_beta.factory_extrapmodel(
        beta=beta0[1],
        data=xpan_beta.DataCentralMoments.from_vals(
            uv=fixture.ub,
            xv=fixture.xb,
            order=fixture.order,
            central=True,
            dims=["val"],
        ),
    )

    xemw1 = xtrapy_models.ExtrapWeightedModel([xem0, xem1])
    b = xemw1.predict(betas, order=3)
    fixture.xr_test(a, b)


def test_extrapmodel_weighted_multi(fixture):

    beta0 = [0.05, 0.2, 1.0]
    betas = [0.3, 0.4, 0.6, 0.7]

    xems_r = [
        xpan_beta.factory_extrapmodel(
            beta=beta,
            data=xpan_beta.factory_data(
                xv=np.random.rand(*fixture.x.shape),
                uv=np.random.rand(*fixture.u.shape),
                central=False,
                order=fixture.order,
            ),
        )
        for beta in beta0
    ]

    xems_c = [
        xpan_beta.factory_extrapmodel(
            beta=xem.alpha0,
            data=xpan_beta.factory_data(
                order=fixture.order, uv=xem.data.uv, xv=xem.data.xv, central=True
            ),
        )
        for xem in xems_r
    ]

    xems_x = [
        xpan_beta.factory_extrapmodel(
            beta=xem.alpha0,
            data=xpan_beta.DataCentralMomentsVals.from_vals(
                order=fixture.order, uv=xem.data.uv, xv=xem.data.xv, central=True
            ),
        )
        for xem in xems_r
    ]

    # test picking models
    # xemw_r should pick the two models closest to beta value
    xemw_a = xtrapy_models.ExtrapWeightedModel([xems_r[0], xems_r[1]])
    xemw_b = xtrapy_models.ExtrapWeightedModel([xems_r[1], xems_r[2]])
    xemw_r = xtrapy_models.ExtrapWeightedModel(xems_r)

    vals = [0.2, 0.4]
    fixture.xr_test(xemw_a.predict(vals), xemw_r.predict(vals, method="nearest"))

    vals = [0.4, 0.8]
    fixture.xr_test(xemw_b.predict(vals), xemw_r.predict(vals, method="between"))

    # other data models
    xemw_c = xtrapy_models.ExtrapWeightedModel(xems_c)
    xemw_x = xtrapy_models.ExtrapWeightedModel(xems_x)

    fixture.xr_test(xemw_r.predict(betas, order=3), xemw_c.predict(betas, order=3))
    fixture.xr_test(xemw_r.predict(betas, order=3), xemw_x.predict(betas, order=3))

    # resample
    nrep = 20
    indices = []
    for xem in xems_c:
        ndat = xem.data.uv.shape[0]
        indices.append(np.random.choice(ndat, (nrep, ndat), True))

    a = xemw_c.resample(indices=indices)
    b = xemw_x.resample(indices=indices)
    fixture.xr_test(a.predict(betas), b.predict(betas))


@pytest.mark.slow
def test_interpmodel_slow(fixture):

    beta0 = [0.05, 0.5, 1.0]

    X = np.array([np.random.rand(*fixture.x.shape) for _ in beta0])
    U = np.array([np.random.rand(*fixture.u.shape) for _ in beta0])
    emi = thermoextrap.InterpModel(fixture.order, beta0, xData=X, uData=U)

    betas = [0.3, 0.4, 0.6, 0.7]

    # stateB
    xems = [
        xpan_beta.factory_extrapmodel(
            beta=beta,
            data=xpan_beta.factory_data(uv=u, xv=x, order=fixture.order, central=False),
        )
        for beta, u, x in zip(beta0, U, X)
    ]

    xemi = xtrapy_models.InterpModel(xems)

    np.testing.assert_allclose(emi.predict(betas), xemi.predict(betas))


def test_interpmodel(fixture):

    beta0 = [0.05, 0.5, 1.0]
    betas = [0.3, 0.4, 0.6, 0.7]

    xems_r = [
        xpan_beta.factory_extrapmodel(
            beta=beta,
            data=xpan_beta.factory_data(
                xv=np.random.rand(*fixture.x.shape),
                uv=np.random.rand(*fixture.u.shape),
                central=False,
                order=fixture.order,
            ),
        )
        for beta in beta0
    ]

    xems_c = [
        xpan_beta.factory_extrapmodel(
            beta=xem.alpha0,
            data=xpan_beta.factory_data(
                order=fixture.order, uv=xem.data.uv, xv=xem.data.xv, central=True
            ),
        )
        for xem in xems_r
    ]

    xems_x = [
        xpan_beta.factory_extrapmodel(
            beta=xem.alpha0,
            data=xpan_beta.DataCentralMomentsVals.from_vals(
                order=fixture.order, uv=xem.data.uv, xv=xem.data.xv, central=True
            ),
        )
        for xem in xems_r
    ]

    xemi_r = xtrapy_models.InterpModel(xems_r)
    xemi_c = xtrapy_models.InterpModel(xems_c)
    xemi_x = xtrapy_models.InterpModel(xems_x)

    fixture.xr_test(xemi_r.predict(betas, order=3), xemi_c.predict(betas, order=3))
    fixture.xr_test(xemi_r.predict(betas, order=3), xemi_x.predict(betas, order=3))

    # resample
    nrep = 20
    indices = []
    for xem in xems_c:
        ndat = xem.data.uv.shape[0]
        indices.append(np.random.choice(ndat, (nrep, ndat), True))

    a = xemi_c.resample(indices=indices)
    b = xemi_x.resample(indices=indices)
    fixture.xr_test(a.predict(betas), b.predict(betas))


def test_interpmodelpiecewise(fixture):

    beta0 = [0.05, 0.2, 1.0]
    betas = [0.3, 0.4, 0.6, 0.7]

    xems_r = [
        xpan_beta.factory_extrapmodel(
            beta=beta,
            data=xpan_beta.factory_data(
                xv=np.random.rand(*fixture.x.shape),
                uv=np.random.rand(*fixture.u.shape),
                central=False,
                order=fixture.order,
            ),
        )
        for beta in beta0
    ]

    # test picking models
    # xemw_r should pick the two models closest to beta value
    xemw_a = xtrapy_models.InterpModel([xems_r[0], xems_r[1]])
    xemw_b = xtrapy_models.InterpModel([xems_r[1], xems_r[2]])
    xemw_r = xtrapy_models.InterpModelPiecewise(xems_r)

    vals = [0.2, 0.4]
    fixture.xr_test(xemw_a.predict(vals), xemw_r.predict(vals, method="nearest"))

    vals = [0.4, 0.8]
    fixture.xr_test(xemw_b.predict(vals), xemw_r.predict(vals, method="between"))


def test_interpmodel_polynomial():
    # Test 1st, 2nd, and 3rd order polynomials at [-1.0, 1.0] points
    xdat2 = xr.DataArray([0.5, 1.5], dims=["rec"])

    for i in range(3):
        xdat1 = ((-1.0) ** (i + 1)) * xdat2
        udat1 = (i + 1) * xr.DataArray([-2.0, 2.0], dims=["rec"])
        udat2 = (i + 1) * xr.DataArray([2.0, -2.0], dims=["rec"])
        print(udat1, udat2)
        dat1 = xpan_beta.DataCentralMomentsVals.from_vals(
            order=1, xv=xdat1, uv=udat1, central=True
        )
        dat2 = xpan_beta.DataCentralMomentsVals.from_vals(
            order=1, xv=xdat2, uv=udat2, central=True
        )

        ex1 = xpan_beta.factory_extrapmodel(-1.0, dat1, xalpha=False)
        ex2 = xpan_beta.factory_extrapmodel(1.0, dat2, xalpha=False)
        interp = xtrapy_models.InterpModel([ex1, ex2])
        check_array = np.zeros(4)
        check_array[i + 1] = 1.0
        np.testing.assert_array_equal(interp.coefs().values, check_array)


def test_mbar(fixture):

    beta0 = [0.05, 0.5]

    X = np.array((fixture.x, fixture.xb))
    U = np.array((fixture.u, fixture.ub))

    emi = thermoextrap.MBARModel(fixture.order, beta0, xData=X, uData=U)

    betas = [0.3, 0.4]

    # stateB
    xem0 = xpan_beta.factory_extrapmodel(beta=beta0[0], data=fixture.rdata)
    xem1 = xpan_beta.factory_extrapmodel(
        beta=beta0[1],
        data=xpan_beta.factory_data(
            uv=fixture.ub, xv=fixture.xb, order=fixture.order, central=False
        ),
    )

    xemi = xtrapy_models.MBARModel([xem0, xem1])

    # NOTE: in old class, can't specify order after train
    # so just use the max order here
    np.testing.assert_allclose(emi.predict(betas), xemi.predict(betas))


from sympy import bell

# Test log
from thermoextrap.utilities import buildAvgFuncs


class LogAvgExtrapModel(thermoextrap.ExtrapModel):
    def calcDerivVals(self, refB, x, U):

        if x.shape[0] != U.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy array (%i) do not match!"
                % (x.shape[0], U.shape[0])
            )
            return

        avgUfunc, avgXUfunc = thermoextrap.buildAvgFuncs(x, U, self.maxOrder)
        derivVals = np.zeros((self.maxOrder + 1, x.shape[1]))
        for o in range(self.maxOrder + 1):
            if o == 0:
                derivVals[o] = -np.log(avgXUfunc(0))
                continue
            for k in range(1, o + 1):
                # Get the derivatives of the average quantity
                thisDiffs = np.array(
                    [self.derivF[l](avgUfunc, avgXUfunc) for l in range(1, o - k + 2)]
                )
                # Loop to apply the chain rule to each element of the observable array
                for l in range(x.shape[1]):
                    derivVals[o, l] += (
                        np.math.factorial(k - 1)
                        * ((-1 / avgXUfunc(0)[l]) ** k)
                        * bell(o, k, thisDiffs[:, l])
                    )

        return derivVals


@pytest.mark.slow
def test_extrapmodel_minuslog_slow(fixture):
    beta0 = 0.5
    betas = [0.2, 0.3]
    u, x, order = fixture.u, fixture.x, fixture.order

    em = LogAvgExtrapModel(
        maxOrder=order,
        refB=beta0,
        xData=x,
        uData=u,
    )
    a = em.params

    # test coefs
    xem = xpan_beta.factory_extrapmodel(beta0, fixture.rdata, post_func="minus_log")
    b = xem.derivatives.derivs(xem.data, norm=False, minus_log=False)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(em.predict(betas), xem.predict(betas))

    # or passing minus_log to predict
    xem = xpan_beta.factory_extrapmodel(beta0, fixture.rdata, post_func=None)
    b = xem.derivatives.derivs(xem.data, norm=False, minus_log=True)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(em.predict(betas), xem.predict(betas, minus_log=True))


def test_extrapmodel_minuslog_slow(fixture):
    beta0 = 0.5
    betas = [0.2, 0.3]
    u, x, order = fixture.u, fixture.x, fixture.order

    xem0 = xpan_beta.factory_extrapmodel(beta0, fixture.rdata, post_func="minus_log")

    for data in [
        fixture.cdata,
        fixture.xdata,
        fixture.xrdata,
        fixture.xdata_val,
        fixture.xrdata_val,
    ]:
        xem1 = xpan_beta.factory_extrapmodel(
            beta=fixture.beta0, data=fixture.cdata, post_func="minus_log"
        )
        fixture.xr_test(xem0.predict(betas, order=3), xem1.predict(betas, order=3))


def test_extrapmodel_minuslog_ig():
    ref_beta = 5.0
    max_order = 3
    test_betas = np.array([4.9, 5.1])
    ref_vol = 1.0

    # Get ideal gas data to compare to analytical results
    xdata, udata = idealgas.generate_data((100_000, 1), ref_beta, ref_vol)
    xdata = xr.DataArray(xdata, dims=["rec"])
    udata = xr.DataArray(udata, dims=["rec"])
    dat = xpan_beta.DataCentralMomentsVals.from_vals(
        order=max_order, xv=xdata, uv=udata, central=True
    )

    # Create extrapolation model to test against analytical
    ex = xpan_beta.factory_extrapmodel(
        ref_beta, dat, xalpha=False, post_func="minus_log"
    )
    # Will need estimate of uncertainty for test data so can check if within that bound
    # So resample
    ex_res = ex.resample(nrep=100)

    true_derivs = np.zeros(max_order + 1)
    # Loop over orders and compare based on uncertainty
    for o in range(max_order + 1):
        # Get the exact values we're shooting for
        true_extrap, true_derivs = idealgas.x_beta_extrap_minuslog(
            o, ref_beta, test_betas, ref_vol
        )
        # Get the derivatives up to this order
        test_derivs = ex.derivs(order=o, norm=False).values
        # And extrapolations
        test_extrap = ex.predict(test_betas, order=o).values
        test_derivs_err = ex_res.derivs(order=o, norm=False).std("rep").values
        test_extrap_err = ex_res.predict(test_betas, order=o).std("rep").values
        # Redefine as confidence interval, and just for the highest derivative order
        test_derivs_err = 2.0 * test_derivs_err[-1]
        test_extrap_err = 2.0 * np.max(test_extrap_err)
        # Just checking to make sure within 1 std
        # Not the best check... tried p-values...
        # But bootstrapped std decreases faster with N than absolute error from analytical
        # Must be good way to do this, but not sure what it is
        # As long as well-control random number seed and number of samples, should work
        np.testing.assert_allclose(
            true_derivs[-1], test_derivs[-1], rtol=0.0, atol=test_derivs_err
        )
        np.testing.assert_allclose(
            true_extrap, test_extrap, rtol=0.0, atol=test_extrap_err
        )


# depend on alpha/betas
# Need to import from utilities
from thermoextrap.utilities import buildAvgFuncsDependent, symDerivAvgXdependent


class ExtrapModelDependent(thermoextrap.ExtrapModel):
    """Class to hold information about an extrapolation that is dependent on the extrapolation variable."""

    # Calculates symbolic derivatives up to maximum order given data
    # Returns list of functions that can be used to evaluate derivatives for specific data
    def calcDerivFuncs(self):
        derivs = []
        for o in range(self.maxOrder + 1):
            derivs.append(
                symDerivAvgXdependent(o)
            )  # Only changing this line to get dependent information
        return derivs

    # And given data, calculate numerical values of derivatives up to maximum order
    # Will be very helpful when generalize to different extrapolation techniques
    # (and interpolation)
    def calcDerivVals(self, refB, x, U):
        """Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives.
        """
        if x.shape[0] != U.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy array (%i) don't match!"
                % (x.shape[0], U.shape[0])
            )
            return

        avgUfunc, avgXUfunc = buildAvgFuncsDependent(
            x, U, self.maxOrder
        )  # Change this line to use dependent function
        derivVals = np.zeros(
            (self.maxOrder + 1, x.shape[2])
        )  # And change this line because x data is of different shape
        for o in range(self.maxOrder + 1):
            derivVals[o] = self.derivF[o](avgUfunc, avgXUfunc)
        return derivVals


@pytest.mark.slow
def test_extrapmodel_alphadep_slow(fixture):

    beta0 = 0.5
    betas = [0.2, 0.7]

    # need new data
    # x[rec, deriv, val]
    n, nv = fixture.x.shape
    order = fixture.order
    x = np.random.rand(n, fixture.order + 1, nv) + fixture.xoff
    u = fixture.u

    em = ExtrapModelDependent(order, beta0, xData=x, uData=u)

    # by passign a derivative name, we are
    xem = xpan_beta.factory_extrapmodel(
        beta0,
        xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )

    np.testing.assert_allclose(em.predict(betas), xem.predict(betas))


def test_extrapmodel_alphadep(fixture):

    beta0 = 0.5
    betas = [0.2, 0.7]

    # need new data
    # x[rec, deriv, val]
    n, nv = fixture.x.shape
    order = fixture.order
    x = np.random.rand(n, fixture.order + 1, nv) + fixture.xoff
    u = fixture.u

    # by passign a derivative name, we are
    xem0 = xpan_beta.factory_extrapmodel(
        beta0,
        xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )

    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas), xem1.predict(betas))

    # for central, only test up to third order
    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=True, deriv_dim="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas, order=3), xem1.predict(betas, order=3))


def test_extrapmodel_alphadep_ig():
    ref_beta = 5.0
    max_order = 3
    test_betas = np.array([4.9, 5.1])
    ref_vol = 1.0

    # Get ideal gas data to compare to analytical results
    xdata, udata = idealgas.generate_data((100_000, 1), ref_beta, ref_vol)
    xdata = xr.DataArray(xdata, dims=["rec"])
    xdata = (
        xr.concat([xdata * ref_beta, xdata], dim="deriv")
        .assign_coords(deriv=lambda x: np.arange(x.sizes["deriv"]))
        .reindex(deriv=np.arange(max_order + 1))
        .fillna(0.0)
    )
    udata = xr.DataArray(udata, dims=["rec"])
    dat = xpan_beta.DataCentralMomentsVals.from_vals(
        order=max_order, xv=xdata, uv=udata, deriv_dim="deriv", central=True
    )

    # Create extrapolation model to test against analytical
    ex = xpan_beta.factory_extrapmodel(ref_beta, dat, xalpha=True)
    # Will need estimate of uncertainty for test data so can check if within that bound
    # So resample
    ex_res = ex.resample(nrep=100)

    # Loop over orders and compare based on uncertainty
    for o in range(max_order + 1):
        # Get the exact values we're shooting for
        true_extrap, true_derivs = idealgas.x_beta_extrap_depend(
            o, ref_beta, test_betas, ref_vol
        )
        # Get the derivatives up to this order
        test_derivs = ex.derivs(order=o, norm=False).values
        # And extrapolations
        test_extrap = ex.predict(test_betas, order=o).values
        test_derivs_err = ex_res.derivs(order=o, norm=False).std("rep").values
        test_extrap_err = ex_res.predict(test_betas, order=o).std("rep").values
        # Redefine as confidence interval, and just for the highest derivative order
        test_derivs_err = 2.0 * test_derivs_err[-1]
        test_extrap_err = 2.0 * np.max(test_extrap_err)
        # Just checking to make sure within 1 std
        # Not the best check... tried p-values...
        # But bootstrapped std decreases faster with N than absolute error from analytical
        # Must be good way to do this, but not sure what it is
        # As long as well-control random number seed and number of samples, should work
        np.testing.assert_allclose(
            true_derivs[-1], test_derivs[-1], rtol=0.0, atol=test_derivs_err * 2
        )
        np.testing.assert_allclose(
            true_extrap, test_extrap, rtol=0.0, atol=test_extrap_err * 2
        )


# minus_log, alphadep


from thermoextrap.utilities import buildAvgFuncsDependent


class LogAvgExtrapModelDependent(ExtrapModelDependent):
    """Class to hold information about an extrapolation that is dependent on the extrapolation variable and
    involves the negative logarithm of an average.
    """

    def calcDerivVals(self, refB, x, U):
        """Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives.
        """
        if x.shape[0] != U.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy array (%i) don't match!"
                % (x.shape[0], U.shape[0])
            )
            return

        avgUfunc, avgXUfunc = buildAvgFuncsDependent(
            x, U, self.maxOrder
        )  # Change this line to use dependent function
        derivVals = np.zeros(
            (self.maxOrder + 1, x.shape[2])
        )  # And change this line because x data is of different shape
        for o in range(self.maxOrder + 1):
            if o == 0:
                derivVals[o] = -np.log(
                    avgXUfunc(0, 0)
                )  # First index is derivative of function, next is power on U
                continue
            for k in range(1, o + 1):
                # Get the derivatives of the average quantity

                thisDiffs = np.array(
                    [self.derivF[l](avgUfunc, avgXUfunc) for l in range(1, o - k + 2)]
                )
                # Loop to apply the chain rule to each element of the observable array
                for l in range(x.shape[2]):
                    derivVals[o, l] += (
                        np.math.factorial(k - 1)
                        * ((-1 / avgXUfunc(0, 0)[l]) ** k)
                        * bell(o, k, thisDiffs[:, l])
                    )

        return derivVals


@pytest.mark.slow
def test_extrapmodel_alphadep_minuslog_slow(fixture):

    beta0 = 0.5
    betas = [0.2, 0.7]

    # need new data
    # x[rec, deriv, val]
    n, nv = fixture.x.shape
    order = fixture.order
    x = np.random.rand(n, fixture.order + 1, nv) + fixture.xoff
    u = fixture.u

    em = LogAvgExtrapModelDependent(order, beta0, xData=x, uData=u)

    # test coefs
    a = em.params

    #                                              use false here because defined above
    # by passign a derivative name, we are
    # set minus_log here, so expressions will have log part in them
    xem = xpan_beta.factory_extrapmodel(
        beta0,
        post_func="minus_log",
        data=xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )
    b = xem.derivatives.derivs(xem.data, minus_log=False, norm=False)
    np.testing.assert_allclose(a, b)

    # test prediction
    np.testing.assert_allclose(em.predict(betas), xem.predict(betas))

    # alternatively, define without minus_log, then call with minus_log later
    xem = xpan_beta.factory_extrapmodel(
        beta0,
        post_func=None,
        data=xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )
    b = xem.derivatives.derivs(xem.data, minus_log=True, norm=False)
    np.testing.assert_allclose(a, b)
    # test prediction
    np.testing.assert_allclose(em.predict(betas), xem.predict(betas, minus_log=True))


def test_extrapmodel_alphadep_minuslog(fixture):

    beta0 = 0.5
    betas = [0.2, 0.7]

    # need new data
    # x[rec, deriv, val]
    n, nv = fixture.x.shape
    order = fixture.order
    x = np.random.rand(n, fixture.order + 1, nv) + fixture.xoff
    u = fixture.u

    # by passign a derivative name, we are
    xem0 = xpan_beta.factory_extrapmodel(
        beta0,
        post_func="minus_log",
        data=xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )

    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        post_func=None,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=False, deriv_dim="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas), xem1.predict(betas, minus_log=True))

    # for central, only test up to third order
    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        post_func=None,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=True, deriv_dim="deriv"
        ),
    )

    fixture.xr_test(
        xem0.predict(betas, order=3), xem1.predict(betas, order=3, minus_log=True)
    )


def test_extrapmodel_alphadep_minuslog_ig():
    ref_beta = 5.0
    max_order = 3
    test_betas = np.array([4.9, 5.1])
    ref_vol = 1.0

    # Get ideal gas data to compare to analytical results
    xdata, udata = idealgas.generate_data((100_000, 1), ref_beta, ref_vol)
    xdata = xr.DataArray(xdata, dims=["rec"])
    xdata = (
        xr.concat([xdata * ref_beta, xdata], dim="deriv")
        .assign_coords(deriv=lambda x: np.arange(x.sizes["deriv"]))
        .reindex(deriv=np.arange(max_order + 1))
        .fillna(0.0)
    )
    udata = xr.DataArray(udata, dims=["rec"])
    dat = xpan_beta.DataCentralMomentsVals.from_vals(
        order=max_order, xv=xdata, uv=udata, deriv_dim="deriv", central=True
    )

    # Create extrapolation model to test against analytical
    ex = xpan_beta.factory_extrapmodel(
        ref_beta, dat, xalpha=True, post_func="minus_log"
    )
    # Will need estimate of uncertainty for test data so can check if within that bound
    # So resample
    ex_res = ex.resample(nrep=100)

    # Loop over orders and compare based on uncertainty
    for o in range(max_order + 1):
        # Get the exact values we're shooting for
        true_extrap, true_derivs = idealgas.x_beta_extrap_depend_minuslog(
            o, ref_beta, test_betas, ref_vol
        )
        # Get the derivatives up to this order
        test_derivs = ex.derivs(order=o, norm=False).values
        # And extrapolations
        test_extrap = ex.predict(test_betas, order=o).values
        test_derivs_err = ex_res.derivs(order=o, norm=False).std("rep").values
        test_extrap_err = ex_res.predict(test_betas, order=o).std("rep").values
        # Redefine as confidence interval, and just for the highest derivative order
        test_derivs_err = 2.0 * test_derivs_err[-1]
        test_extrap_err = 2.0 * np.max(test_extrap_err)
        # Just checking to make sure within 1 std
        # Not the best check... tried p-values...
        # But bootstrapped std decreases faster with N than absolute error from analytical
        # Must be good way to do this, but not sure what it is
        # As long as well-control random number seed and number of samples, should work
        np.testing.assert_allclose(
            true_derivs[-1], test_derivs[-1], rtol=0.0, atol=test_derivs_err * 2
        )
        np.testing.assert_allclose(
            true_extrap, test_extrap, rtol=0.0, atol=test_extrap_err * 2
        )

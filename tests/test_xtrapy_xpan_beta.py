import numpy as np
import pytest
import xarray as xr

import thermoextrap
import thermoextrap.xtrapy.core as xtrapy_core
import thermoextrap.xtrapy.data as xtrapy_data
import thermoextrap.xtrapy.xpan_beta as xpan_beta
from thermoextrap.xtrapy.cached_decorators import gcached


@pytest.mark.slow
def test_xpan_beta_coefs_slow(fixture):
    a = np.array(fixture.coefs_list)
    s = xpan_beta.factory_coefs(xalpha=False, central=False)
    b = s.xcoefs(fixture.rdata, norm=False)
    np.testing.assert_allclose(a, b)

    s = xpan_beta.factory_coefs(xalpha=False, central=True)
    b = s.xcoefs(fixture.cdata, norm=False)
    np.testing.assert_allclose(a, b)


def test_xpan_beta_coefs(fixture):
    s = xpan_beta.factory_coefs(xalpha=False, central=False)
    b = s.xcoefs(fixture.rdata, norm=False)
    fixture.xr_test(b, s.xcoefs(fixture.xrdata, norm=False))
    fixture.xr_test(b, s.xcoefs(fixture.xrdata_val, norm=False))

    # central
    s = xpan_beta.factory_coefs(xalpha=False, central=True)
    b = s.xcoefs(fixture.cdata, norm=False)
    fixture.xr_test(b, s.xcoefs(fixture.xdata, norm=False))
    fixture.xr_test(b, s.xcoefs(fixture.xdata_val, norm=False))


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

    xemw = xtrapy_core.ExtrapWeightedModel([xem0, xem1])

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

    xemw = xtrapy_core.ExtrapWeightedModel([xem0, xem1])

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

    xemw1 = xtrapy_core.ExtrapWeightedModel([xem0, xem1])
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

    xemw1 = xtrapy_core.ExtrapWeightedModel([xem0, xem1])
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
    xemw_a = xtrapy_core.ExtrapWeightedModel([xems_r[0], xems_r[1]])
    xemw_b = xtrapy_core.ExtrapWeightedModel([xems_r[1], xems_r[2]])
    xemw_r = xtrapy_core.ExtrapWeightedModel(xems_r)

    vals = [0.2, 0.4]
    fixture.xr_test(xemw_a.predict(vals), xemw_r.predict(vals, method="nearest"))

    vals = [0.4, 0.8]
    fixture.xr_test(xemw_b.predict(vals), xemw_r.predict(vals, method="between"))

    # other data models
    xemw_c = xtrapy_core.ExtrapWeightedModel(xems_c)
    xemw_x = xtrapy_core.ExtrapWeightedModel(xems_x)

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

    xemi = xtrapy_core.InterpModel(xems)

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

    xemi_r = xtrapy_core.InterpModel(xems_r)
    xemi_c = xtrapy_core.InterpModel(xems_c)
    xemi_x = xtrapy_core.InterpModel(xems_x)

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

    xemi = xtrapy_core.MBARModel([xem0, xem1])

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

    xem = xpan_beta.factory_extrapmodel(beta0, fixture.rdata, minus_log=True)

    # test coefs
    a = em.params
    b = xem.coefs.xcoefs(xem.data, norm=False, minus_log=True)
    np.testing.assert_allclose(a, b)

    np.testing.assert_allclose(em.predict(betas), xem.predict(betas))


def test_extrapmodel_minuslog_slow(fixture):
    beta0 = 0.5
    betas = [0.2, 0.3]
    u, x, order = fixture.u, fixture.x, fixture.order

    xem0 = xpan_beta.factory_extrapmodel(beta0, fixture.rdata, minus_log=True)

    for data in [
        fixture.cdata,
        fixture.xdata,
        fixture.xrdata,
        fixture.xdata_val,
        fixture.xrdata_val,
    ]:
        xem1 = xpan_beta.factory_extrapmodel(
            beta=fixture.beta0, data=fixture.cdata, minus_log=True
        )
        fixture.xr_test(xem0.predict(betas, order=3), xem1.predict(betas, order=3))


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
        xpan_beta.factory_data(uv=u, xv=x, order=order, central=False, deriv="deriv"),
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
        xpan_beta.factory_data(uv=u, xv=x, order=order, central=False, deriv="deriv"),
    )

    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=False, deriv="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas), xem1.predict(betas))

    # for central, only test up to third order
    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=True, deriv="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas, order=3), xem1.predict(betas, order=3))


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

    # by passign a derivative name, we are
    xem = xpan_beta.factory_extrapmodel(
        beta0,
        minus_log=True,
        data=xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv="deriv"
        ),
    )

    # test coefs
    a = em.params
    b = xem.coefs.xcoefs(xem.data, minus_log=True, norm=False)
    np.testing.assert_allclose(a, b)

    # test prediction
    np.testing.assert_allclose(em.predict(betas), xem.predict(betas))


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
        minus_log=True,
        data=xpan_beta.factory_data(
            uv=u, xv=x, order=order, central=False, deriv="deriv"
        ),
    )

    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        minus_log=True,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=False, deriv="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas), xem1.predict(betas))

    # for central, only test up to third order
    xem1 = xpan_beta.factory_extrapmodel(
        beta0,
        minus_log=True,
        data=xpan_beta.DataCentralMomentsVals.from_vals(
            uv=u, xv=x, order=order, central=True, deriv="deriv"
        ),
    )

    fixture.xr_test(xem0.predict(betas, order=3), xem1.predict(betas, order=3))

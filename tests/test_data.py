import cmomy
import numpy as np

import thermoextrap as xtrap


def test_rdata(fixture) -> None:
    """Testing new interface against old interface"""
    ufunc, xufunc = fixture.u_xu_funcs

    # ufunc = lambda x: float(_ufunc(x))
    # xufunc = lambda x: _xufunc.avgdict[x]

    # a = fixture.rdata.xu
    # b = [xufunc(i) for i in range(fixture.order + 1)]

    # print('hello a', a)
    # print('hello b', b)
    # print('hello a1', type(a[0]))
    # print('hello b1', type(b[0]))

    # raise AssertionError

    u_order = (fixture.rdata.umom_dim, ...)
    x_order = (
        (fixture.rdata.umom_dim, fixture.rdata.deriv_dim, ...)
        if fixture.rdata.deriv_dim is not None
        else u_order
    )

    np.testing.assert_allclose(
        fixture.rdata.u.transpose(*u_order),
        [ufunc(i) for i in range(fixture.order + 1)],
    )
    np.testing.assert_allclose(
        fixture.rdata.xu.transpose(*x_order),
        [xufunc(i) for i in range(fixture.order + 1)],
    )


def test_xdata(fixture) -> None:
    fixture.xr_test_raw(fixture.xdata)
    fixture.xr_test_central(fixture.xdata)


def test_xdata_val(fixture) -> None:
    fixture.xr_test_raw(fixture.xdata_val)
    fixture.xr_test_central(fixture.xdata_val)


def test_xdata_from_ave_raw(fixture) -> None:
    a = fixture.rdata

    # ## base on raw arrays
    # b = xtrap.DataCentralMoments.from_ave_raw(
    #     u=a.u.values, xu=a.xu.values, weight=len(a.uv), axis=0, dims=["val"]
    # )

    # fixture.xr_test_raw(b)

    # base on xarray
    b = xtrap.DataCentralMoments.from_ave_raw(
        u=a.u,
        xu=a.xu,
        weight=len(a.uv),
    )
    fixture.xr_test_raw(b)


def test_xdata_from_ave_central(fixture) -> None:
    a = fixture.cdata

    # base on raw values
    b = xtrap.DataCentralMoments.from_ave_central(
        du=a.du.values,
        dxdu=a.dxdu.values,
        xave=a.xave.values,
        uave=fixture.rdata.u.values[1],
        weight=len(a.uv),
        axis=-1,
        dims=["val"],
    )

    fixture.xr_test_central(b)

    # base on xarray
    b = xtrap.DataCentralMoments.from_ave_central(
        du=a.du, dxdu=a.dxdu, xave=a.xave, uave=fixture.rdata.u[1], weight=len(a.uv)
    )

    fixture.xr_test_central(b)


def test_resample(fixture) -> None:
    nrep = 10
    ndat = fixture.x.shape[0]

    rng = cmomy.random.default_rng()

    idx = rng.choice(ndat, (nrep, ndat), replace=True)

    sampler = cmomy.factory_sampler(indices=idx)

    b = fixture.xdata_val.resample(sampler=sampler)

    # raw
    a = fixture.rdata.resample(sampler=sampler)
    fixture.xr_test_raw(a=a, b=b)

    # central
    a = fixture.cdata.resample(sampler=sampler)
    fixture.xr_test_central(a=a, b=b)

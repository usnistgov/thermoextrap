import numpy as np

import thermoextrap as xtrap


def test_rdata(fixture):
    """testing new interface against old interface"""
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

    np.testing.assert_allclose(
        fixture.rdata.u, [ufunc(i) for i in range(fixture.order + 1)]
    )
    np.testing.assert_allclose(
        fixture.rdata.xu, [xufunc(i) for i in range(fixture.order + 1)]
    )


def test_xdata(fixture):
    fixture.xr_test_raw(fixture.xdata)
    fixture.xr_test_central(fixture.xdata)


def test_xdata_val(fixture):
    fixture.xr_test_raw(fixture.xdata_val)
    fixture.xr_test_central(fixture.xdata_val)


def test_xdata_from_ave_raw(fixture):
    a = fixture.rdata

    # base on raw arrays
    b = xtrap.DataCentralMoments.from_ave_raw(
        u=a.u.values, xu=a.xu.values, w=len(a.uv), axis=0, dims=["val"]
    )

    fixture.xr_test_raw(b)

    # base on xarray
    b = xtrap.DataCentralMoments.from_ave_raw(
        u=a.u,
        xu=a.xu,
        w=len(a.uv),
    )
    fixture.xr_test_raw(b)


def test_xdata_from_ave_central(fixture):
    a = fixture.cdata

    # base on raw values
    b = xtrap.DataCentralMoments.from_ave_central(
        du=a.du.values,
        dxdu=a.dxdu.values,
        xave=a.xave.values,
        uave=fixture.rdata.u.values[1],
        w=len(a.uv),
        axis=0,
        dims=["val"],
    )

    fixture.xr_test_central(b)

    # base on xarray
    b = xtrap.DataCentralMoments.from_ave_central(
        du=a.du, dxdu=a.dxdu, xave=a.xave, uave=fixture.rdata.u[1], w=len(a.uv)
    )

    fixture.xr_test_central(b)


def test_resample(fixture):
    nrep = 10
    ndat = fixture.x.shape[0]

    idx = np.random.choice(ndat, (nrep, ndat), replace=True)

    b = fixture.xdata_val.resample(indices=idx)

    # raw
    a = fixture.rdata.resample(indices=idx)
    fixture.xr_test_raw(a=a, b=b)

    # central
    a = fixture.cdata.resample(indices=idx)
    fixture.xr_test_central(a=a, b=b)

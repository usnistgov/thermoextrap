"""Some simple tests for factory methods of xCentral"""
import numpy as np
import pytest
import xarray as xr

import cmomy

my_fixture = lambda **kws: pytest.fixture(scope="module", **kws)


@my_fixture(params=[3, (3, 3)])
def mom(request):
    return request.param


@my_fixture()
def mom_tuple(mom):
    if isinstance(mom, int):
        return (mom,)
    else:
        return mom


@my_fixture(params=[(10,), (10, 5, 6), (10, 5, 6, 7)])
def shape(request):
    return request.param


@my_fixture()
def axis(shape):
    return np.random.randint(0, len(shape))


@my_fixture()
def xy(shape, mom_tuple):

    x = np.random.rand(*shape)

    if len(mom_tuple) == 2:
        y = np.random.rand(*shape)
        return (x, y)
    else:
        return x


@my_fixture()
def dc(xy, mom, axis):
    return cmomy.CentralMoments.from_vals(xy, mom=mom, w=None, axis=axis)


@my_fixture()
def dcx(dc):
    return dc.to_xcentralmoments()


def test_CS(dc, dcx):
    np.testing.assert_allclose(dc, dcx)


def test_from_data(dc, dcx):

    mom_ndim = dc.mom_ndim

    t = cmomy.CentralMoments.from_data(dc.data, mom_ndim=mom_ndim)

    dims = [f"hello_{i}" for i in range(len(dc.data.shape))]
    o1 = cmomy.xCentralMoments.from_data(dc.data, dims=dims, mom_ndim=mom_ndim)

    np.testing.assert_allclose(t, o1)

    # create from xarray?
    o2 = cmomy.xCentralMoments.from_data(
        dcx.values.rename(dict(zip(dcx.dims, dims))), mom_ndim=mom_ndim
    )
    xr.testing.assert_allclose(o1.values, o2.values)


def test_from_datas(dc, dcx):

    mom_ndim = dc.mom_ndim

    for axis in range(dc.val_ndim):
        t = cmomy.CentralMoments.from_datas(dc.data, axis=axis, mom_ndim=mom_ndim)

        dims = dcx.dims[:axis] + dcx.dims[axis + 1 :]

        o1 = cmomy.xCentralMoments.from_datas(
            dc.data, axis=axis, mom_ndim=mom_ndim, dims=dims
        )

        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        o2 = cmomy.xCentralMoments.from_datas(dcx.values, dim=dim, mom_ndim=mom_ndim)

        xr.testing.assert_allclose(o1.values, o2.values)


def test_from_raw(dc, dcx):

    mom_ndim = dc.mom_ndim

    t = cmomy.CentralMoments.from_raw(dc.to_raw(), mom_ndim=mom_ndim)

    o1 = cmomy.xCentralMoments.from_raw(dc.to_raw(), mom_ndim=mom_ndim)

    np.testing.assert_allclose(t, o1)

    o2 = cmomy.xCentralMoments.from_raw(dcx.to_raw(), mom_ndim=mom_ndim)

    xr.testing.assert_allclose(o1.values, o2.values)


def test_from_raws(dc, dcx):
    mom_ndim = dc.mom_ndim

    for axis in range(dc.val_ndim):

        # first test from raws
        raws = dc.to_raw()
        t = cmomy.CentralMoments.from_raws(raws, axis=axis, mom_ndim=mom_ndim)
        r = dc.reduce(axis=axis)

        np.testing.assert_allclose(t.values, r.values)

        # test xCentral
        o1 = cmomy.xCentralMoments.from_raws(raws, axis=axis, mom_ndim=mom_ndim)

        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        o2 = cmomy.xCentralMoments.from_raws(dcx.to_raw(), dim=dim, mom_ndim=mom_ndim)


def test_from_vals(xy, shape, mom):
    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    if isinstance(xy, tuple):
        xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)
    else:
        xy_xr = xr.DataArray(xy, dims=dims)

    for axis in range(len(shape)):

        t = cmomy.xCentralMoments.from_vals(xy, axis=axis, mom=mom)

        # dims of output
        o1 = cmomy.xCentralMoments.from_vals(
            xy, axis=axis, mom=mom, dims=dims[:axis] + dims[axis + 1 :]
        )
        np.testing.assert_allclose(t, o1)

        o2 = cmomy.xCentralMoments.from_vals(xy_xr, dim=dims[axis], mom=mom)

        xr.testing.assert_allclose(o1.values, o2.values)


def test_from_resample_vals(xy, shape, mom):

    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    if isinstance(xy, tuple):
        xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)
    else:
        xy_xr = xr.DataArray(xy, dims=dims)

    for axis in range(len(shape)):

        t, freq = cmomy.xCentralMoments.from_resample_vals(
            xy, nrep=10, full_output=True, axis=axis, mom=mom
        )

        # dims of output
        o1 = cmomy.xCentralMoments.from_resample_vals(
            xy, axis=axis, mom=mom, freq=freq, dims=dims[:axis] + dims[axis + 1 :]
        )
        np.testing.assert_allclose(t, o1)

        o2 = cmomy.xCentralMoments.from_resample_vals(
            xy_xr, dim=dims[axis], mom=mom, freq=freq
        )

        xr.testing.assert_allclose(o1.values, o2.values)


def test_resample_and_reduce(dc, dcx):

    for axis in range(dc.val_ndim):

        t, freq = dc.resample_and_reduce(nrep=10, full_output=True, axis=axis)

        o = dcx.resample_and_reduce(freq=freq, dim=dcx.dims[axis])

        np.testing.assert_allclose(t.data, o.data)

        assert o.val_dims == ("rep",) + dcx.val_dims[:axis] + dcx.val_dims[axis + 1 :]

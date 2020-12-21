import random

import numpy as np
import xarray as xr

import cmomy.xcentral as xcentral

# specific xcentral stuff


def xtest(a, b):
    xr.testing.assert_allclose(a, b.transpose(*a.dims))


def test_fix_test(other):
    np.testing.assert_allclose(other.data_test_xr, other.data_fix)


def test_s(other):
    xtest(other.data_test_xr, other.s_xr.values)


def scramble_xr(x):
    if isinstance(x, tuple):
        return tuple(scramble_xr(_) for _ in x)

    elif isinstance(x, xr.DataArray):
        order = list(x.dims)
        random.shuffle(order)
        return x.transpose(*order)
    else:
        return x


def test_create(other):
    t = xcentral.xCentralMoments.zeros(mom=other.mom, val_shape=other.val_shape)

    # from array
    t.push_vals(other.x, w=other.w, axis=other.axis, broadcast=other.broadcast)
    xtest(other.data_test_xr, t.values)

    # from xarray
    t.zero()
    t.push_vals(
        x=scramble_xr(other.x_xr),
        w=scramble_xr(other.w_xr),
        axis="rec",
        broadcast=other.broadcast,
    )
    xtest(other.data_test_xr, t.values)


def test_from_vals(other):
    t = xcentral.xCentralMoments.from_vals(
        x=other.x, w=other.w, mom=other.mom, axis=other.axis, broadcast=other.broadcast
    )
    xtest(other.data_test_xr, t.values)

    t = xcentral.xCentralMoments.from_vals(
        x=scramble_xr(other.x_xr),
        w=scramble_xr(other.w_xr),
        axis="rec",
        mom=other.mom,
        broadcast=other.broadcast,
    )
    xtest(other.data_test_xr, t.values)


def test_push_val(other):
    if other.axis == 0 and other.style == "total":
        if other.s._mom_ndim == 1:
            print("do_push_val")
            t = other.s_xr.zeros_like()
            for ww, xx in zip(other.w, other.x):
                t.push_val(x=xx, w=ww, broadcast=other.broadcast)
            xtest(other.data_test_xr, t.values)

            t.zero()
            for ww, xx in zip(other.w_xr, other.x_xr):
                t.push_val(
                    x=scramble_xr(xx), w=scramble_xr(ww), broadcast=other.broadcast
                )
            xtest(other.data_test_xr, t.values)


def test_push_vals_mult(other):
    t = other.s_xr.zeros_like()
    for ww, xx in zip(other.W, other.X):
        t.push_vals(x=xx, w=ww, axis=other.axis, broadcast=other.broadcast)
    xtest(other.data_test_xr, t.values)

    t.zero()
    for ww, xx in zip(other.W_xr, other.X_xr):
        t.push_vals(
            x=scramble_xr(xx), w=scramble_xr(ww), axis="rec", broadcast=other.broadcast
        )
    xtest(other.data_test_xr, t.values)


def test_combine(other):
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t.push_data(scramble_xr(s.values))
    xtest(other.data_test_xr, t.values)


def test_from_datas(other):

    datas = xr.concat([s.values for s in other.S_xr], dim="rec")
    datas = scramble_xr(datas).transpose(*(...,) + other.s_xr.mom_dims)
    t = other.cls_xr.from_datas(datas, mom=other.mom, axis="rec")
    xtest(other.data_test_xr, t.values)


def test_push_datas(other):
    datas = xr.concat([s.values for s in other.S_xr], dim="rec")

    datas = scramble_xr(datas).transpose(*(...,) + other.s_xr.mom_dims)

    t = other.s_xr.zeros_like()
    t.push_datas(datas, axis="rec")
    xtest(other.data_test_xr, t.values)


# def test_push_stat(other):
#     if other.s._mom_ndim == 1:

#         t = other.s_xr.zeros_like()
#         for s in other.S_xr:
#             t.push_stat(s.mean(), v=s.values[..., 2:], w=s.weight())
#         xtest(other.data_test_xr, t.values)


# def test_from_stat(other):
#     if other.s._mom_ndim == 1:
#         t = other.cls.from_stat(
#             a=other.s.mean(),
#             v=other.s.values[..., 2:],
#             w=other.s.weight(),
#             mom=other.mom,
#         )
#         other.test_values(t.values)


# def test_from_stats(other):
#     if other.s._mom_ndim == 1:
#         t = other.s.zeros_like()
#         t.push_stats(
#             a=np.array([s.mean() for s in other.S]),
#             v=np.array([s.values[..., 2:] for s in other.S]),
#             w=np.array([s.weight() for s in other.S]),
#             axis=0,
#         )
#         other.test_values(t.values)


def test_add(other):
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t = t + s
    xtest(other.data_test_xr, t.values)


def test_sum(other):
    t = sum(other.S_xr, other.s_xr.zeros_like())
    xtest(other.data_test_xr, t.values)


def test_iadd(other):
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t += s
    xtest(other.data_test_xr, t.values)


def test_sub(other):
    t = other.s_xr - sum(other.S_xr[1:], other.s_xr.zeros_like())
    xtest(t.values, other.S_xr[0].values)


def test_isub(other):
    t = other.s_xr.copy()
    for s in other.S_xr[1:]:
        t -= s
    xtest(t.values, other.S_xr[0].values)


def test_mult(other):
    s = other.s_xr

    xtest((s * 2).values, (s + s).values)

    t = s.copy()
    t *= 2
    xtest(t.values, (s + s).values)


def test_resample_and_reduce(other):

    ndim = len(other.val_shape)

    if ndim > 0:

        for axis in range(ndim):

            ndat = other.val_shape[axis]
            nrep = 10

            idx = np.random.choice(ndat, (nrep, ndat), replace=True)

            t0 = other.s.resample_and_reduce(indices=idx, axis=axis)

            dim = "dim_{}".format(axis)
            t1 = other.s_xr.resample_and_reduce(indices=idx, axis=dim, rep_dim="hello")

            np.testing.assert_allclose(t0.data, t1.data)

            # check dims
            dims = list(other.s_xr.values.dims)
            dims.pop(axis)
            dims = tuple(["hello"] + dims)
            assert t1.values.dims == dims

            # resample
            tr = other.s.resample(idx, axis=axis)

            # note: tx may be in different order than tr
            tx = other.s_xr.isel(**{dim: xr.DataArray(idx, dims=["hello", dim])})

            np.testing.assert_allclose(tr.data, tx.transpose("hello", dim, ...).data)

            # # check dims
            # assert tx.dims == ('hello', ) + other.s_xr.values.dims

            # reduce
            tx = tx.reduce(dim)
            xtest(t1.values, tx.values)

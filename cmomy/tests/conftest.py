import numpy as np
import pytest
import xarray as xr

import cmomy.central as central
import cmomy.xcentral as xcentral
from cmomy.cached_decorators import gcached


def _get_cmom(w, x, moments, axis=0, last=True):

    if w is None:
        w = np.array(1.0)

    if w.ndim == 1 and w.ndim != x.ndim and len(w) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        w = w.reshape(*shape)

    if w.shape != x.shape:
        w = np.broadcast_to(w, x.shape)

    wsum_keep = w.sum(axis, keepdims=True)
    wsum_keep_inv = 1.0 / wsum_keep

    wsum = w.sum(axis)
    wsum_inv = 1.0 / wsum

    # get moments
    xave = (w * x).sum(axis, keepdims=True) * wsum_keep_inv
    dx = x - xave

    xmean = (w * x).sum(axis) * wsum_inv
    weight = wsum
    data = [weight, xmean]

    for n in range(2, moments + 1):
        y = (w * dx ** n).sum(axis) * wsum_inv
        data.append(y)

    data = np.array(data)
    if last:
        data = np.moveaxis(data, 0, -1)
    return data


def _get_comom(w, x, y, moments, axis=0, broadcast=True):

    if w is None:
        w = np.array(1.0)

    if w.ndim == 1 and w.ndim != x.ndim and len(w) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        w = w.reshape(*shape)

    if w.shape != x.shape:
        w = np.broadcast_to(w, x.shape)

    if y.ndim != x.ndim and y.ndim == 1 and len(y) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        y = y.reshape(*shape)

    if broadcast and y.shape != x.shape:
        y = np.broadcast_to(y, x.shape)

    assert w.shape == x.shape
    assert y.shape == x.shape

    shape = list(x.shape)
    shape.pop(axis)
    shape = tuple(shape) + tuple(x + 1 for x in moments)

    out = np.zeros(shape)
    wsum = w.sum(axis)
    wsum_inv = 1.0 / wsum

    wsum_keep = w.sum(axis, keepdims=True)
    wsum_keep_inv = 1.0 / wsum_keep

    xave = (w * x).sum(axis, keepdims=True) * wsum_keep_inv
    dx = x - xave

    yave = (w * y).sum(axis, keepdims=True) * wsum_keep_inv
    dy = y - yave

    for i in range(moments[0] + 1):
        for j in range(moments[1] + 1):
            if i == 0 and j == 0:
                val = wsum

            elif i + j == 1:
                val = (w * x ** i * y ** j).sum(axis) * wsum_inv
            else:
                val = (w * dx ** i * dy ** j).sum(axis) * wsum_inv

            out[..., i, j] = val

    return out


class Data(object):
    """wrapper around stuff for generic testing"""

    # _count = 0

    def __init__(self, shape, axis, style, mom, nsplit=3):
        print(
            f"shape:{shape}, axis:{axis}, style:{style}, mom:{mom}, nsplit:{nsplit}",
            end=" ",
        )
        # self.__class__._count += 1

        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.axis = axis
        self.style = style
        self.mom = mom
        self.nsplit = nsplit

    @gcached()
    def cov(self):
        if isinstance(self.mom, int):
            return False
        if isinstance(self.mom, tuple) and len(self.mom) == 2:
            return True

    @property
    def mom_ndim(self):
        if self.cov:
            return 2
        else:
            return 1

    @property
    def broadcast(self):
        return self.style == "broadcast"

    @property
    def cls(self):
        return central.CentralMoments

    @gcached()
    def val_shape(self):
        val_shape = list(self.shape)
        val_shape.pop(self.axis)
        return tuple(val_shape)

    def _get_data(self, style=None):
        if style is None or style == "total":
            return np.random.rand(*self.shape)
        elif style == "broadcast":
            return np.random.rand(self.shape[self.axis])
        else:
            raise ValueError("bad style")

    @gcached()
    def xdata(self):
        return self._get_data()

    @gcached()
    def ydata(self):
        return self._get_data(style=self.style)

    @gcached()
    def w(self):
        if self.style is None:
            return None
        #            return np.array(1.0)
        else:
            return self._get_data(style=self.style)
        return self._get_weight()

    @gcached()
    def x(self):
        if self.cov:
            return (self.xdata, self.ydata)
        else:
            return self.xdata

    @gcached()
    def split_data(self):
        v = self.xdata.shape[self.axis] // self.nsplit
        splits = [v * i for i in range(1, self.nsplit)]
        X = np.split(self.xdata, splits, axis=self.axis)

        if self.style == "total":
            W = np.split(self.w, splits, axis=self.axis)
        elif self.style == "broadcast":
            W = np.split(self.w, splits)
        else:
            W = [self.w for xx in X]

        if self.cov:
            if self.style == "broadcast":
                Y = np.split(self.ydata, splits)
            else:
                Y = np.split(self.ydata, splits, axis=self.axis)

            # pack X, Y
            X = list(zip(X, Y))

        return W, X

    @property
    def W(self):
        return self.split_data[0]

    @property
    def X(self):
        return self.split_data[1]

    @gcached()
    def data_fix(self):
        if self.cov:
            return _get_comom(
                w=self.w,
                x=self.x[0],
                y=self.x[1],
                moments=self.mom,
                axis=self.axis,
                broadcast=self.broadcast,
            )
        else:
            return _get_cmom(
                w=self.w, x=self.x, moments=self.mom, axis=self.axis, last=True
            )

    @gcached()
    def data_test(self):
        return central.central_moments(
            x=self.x,
            mom=self.mom,
            w=self.w,
            axis=self.axis,
            last=True,
            broadcast=self.broadcast,
        )

    @gcached()
    def s(self):
        s = self.cls.zeros(val_shape=self.val_shape, mom=self.mom)
        s.push_vals(x=self.x, w=self.w, axis=self.axis, broadcast=self.broadcast)
        return s

    @gcached()
    def S(self):
        return [
            self.cls.from_vals(
                x=xx, w=ww, axis=self.axis, mom=self.mom, broadcast=self.broadcast
            )
            for ww, xx in zip(self.W, self.X)
        ]

    @property
    def values(self):
        return self.data_test

    def unpack(self, *args):
        out = tuple(getattr(self, x) for x in args)
        if len(out) == 1:
            out = out[0]
        return out

    def test_values(self, x):
        np.testing.assert_allclose(self.values, x)

    @property
    def raw(self):
        if self.style == "total":
            if not self.cov:
                raw = np.array(
                    [
                        np.average(self.x ** i, weights=self.w, axis=self.axis)
                        for i in range(self.mom + 1)
                    ]
                )
                raw[0, ...] = self.w.sum(self.axis)

                raw = np.moveaxis(raw, 0, -1)

            else:

                raw = np.zeros_like(self.data_test)
                for i in range(self.mom[0] + 1):
                    for j in range(self.mom[1] + 1):
                        raw[..., i, j] = np.average(
                            self.x[0] ** i * self.x[1] ** j,
                            weights=self.w,
                            axis=self.axis,
                        )

                raw[..., 0, 0] = self.w.sum(self.axis)

        else:
            raw = None
        return raw

    @gcached()
    def indices(self):
        ndat = self.xdata.shape[self.axis]
        nrep = 10
        return np.random.choice(ndat, (nrep, ndat), replace=True)

    @gcached()
    def freq(self):
        return central.randsamp_freq(indices=self.indices)

    @gcached()
    def xdata_resamp(self):
        xdata = self.xdata

        if self.axis != 0:
            xdata = np.moveaxis(xdata, self.axis, 0)

        return np.take(xdata, self.indices, axis=0)

    @gcached()
    def ydata_resamp(self):
        ydata = self.ydata

        if self.style == "broadcast":
            return np.take(ydata, self.indices, axis=0)
        else:
            if self.axis != 0:
                ydata = np.moveaxis(ydata, self.axis, 0)
            return np.take(ydata, self.indices, axis=0)

    @property
    def x_resamp(self):
        if self.cov:
            return (self.xdata_resamp, self.ydata_resamp)
        else:
            return self.xdata_resamp

    @gcached()
    def w_resamp(self):
        w = self.w

        if self.style is None:
            return w
        elif self.style == "broadcast":
            return np.take(w, self.indices, axis=0)
        else:
            if self.axis != 0:
                w = np.moveaxis(w, self.axis, 0)
            return np.take(w, self.indices, axis=0)

    @gcached()
    def data_test_resamp(self):
        return central.central_moments(
            x=self.x_resamp,
            mom=self.mom,
            w=self.w_resamp,
            axis=1,
            broadcast=self.broadcast,
        )

    # xcentral specific stuff
    @property
    def cls_xr(self):
        return xcentral.xCentralMoments

    @gcached()
    def s_xr(self):
        return self.cls_xr.from_vals(
            x=self.x, w=self.w, axis=self.axis, mom=self.mom, broadcast=self.broadcast
        )

    @gcached()
    def xdata_xr(self):
        dims = [f"dim_{i}" for i in range(len(self.shape) - 1)]
        dims.insert(self.axis, "rec")
        return xr.DataArray(self.xdata, dims=dims)

    @gcached()
    def ydata_xr(self):
        if self.style is None or self.style == "total":
            dims = self.xdata_xr.dims
        else:
            dims = "rec"

        return xr.DataArray(self.ydata, dims=dims)

    @gcached()
    def w_xr(self):
        if self.style is None:
            return None
        elif self.style == "broadcast":
            dims = "rec"
        else:
            dims = self.xdata_xr.dims

        return xr.DataArray(self.w, dims=dims)

    @property
    def x_xr(self):
        if self.cov:
            return (self.xdata_xr, self.ydata_xr)
        else:
            return self.xdata_xr

    @gcached()
    def data_test_xr(self):
        return xcentral.xcentral_moments(
            x=self.x_xr, mom=self.mom, axis="rec", w=self.w_xr, broadcast=self.broadcast
        )

    @gcached()
    def W_xr(self):
        if isinstance(self.w_xr, xr.DataArray):
            dims = self.w_xr.dims
            return [xr.DataArray(_, dims=dims) for _ in self.W]
        else:
            return self.W

    @gcached()
    def X_xr(self):
        xdims = self.xdata_xr.dims

        if self.cov:
            ydims = self.ydata_xr.dims

            return [
                (xr.DataArray(x, dims=xdims), xr.DataArray(y, dims=ydims))
                for x, y in self.X
            ]
        else:
            return [xr.DataArray(x, dims=xdims) for x in self.X]

    @gcached()
    def S_xr(self):
        return [
            self.cls_xr.from_vals(
                x=x, w=w, axis=self.axis, mom=self.mom, broadcast=self.broadcast
            )
            for w, x, in zip(self.W, self.X)
        ]


# Fixutre
# def get_params():
#     for shape, axis in [(20, 0), ((20, 2, 3), 0), ((2, 20, 3), 1), ((2, 3, 20), 2)]:
#         for style in [None, "total", "broadcast"]:
#             for mom in [4, (3, 3)]:
#                 yield Data(shape, axis, style, mom)


# @pytest.fixture(params=get_params(), scope="module")
# def other(request):
#     return request.param
def get_params():
    for shape, axis in [(20, 0), ((20, 2, 3), 0), ((2, 20, 3), 1), ((2, 3, 20), 2)]:
        for style in [None, "total", "broadcast"]:
            for mom in [4, (3, 3)]:
                yield shape, axis, style, mom


@pytest.fixture(params=get_params(), scope="module")
def other(request):
    return Data(*request.param)

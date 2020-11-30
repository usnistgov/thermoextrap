
import numpy as np
import pytest

from cmomy.cached_decorators import gcached
import cmomy.central as central


def _get_cmom(w, x, moments, axis=0, last=True):
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
    """wrapper around stuff for generic testing

    """

    def __init__(self, shape, axis, style, mom, nsplit=3):
        print("creating data", shape, axis, style, mom, nsplit)
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

    @gcached()
    def cls(self):
        if self.cov:
            return central.StatsAccumCov
        else:
            return central.StatsAccum

    @gcached()
    def shape_val(self):
        shape_val = list(self.shape)
        shape_val.pop(self.axis)
        return tuple(shape_val)

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
            return np.array(1.0)
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

    @property
    def broadcast(self):
        return self.style == "broadcast"

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
            x=self.x, mom=self.mom, w=self.w, axis=self.axis, last=True, broadcast=self.broadcast)

        # if self.cov:
        #     return central.central_comoments(
        #         x=self.x[0],
        #         y=self.x[1],
        #         mom=self.mom,
        #         w=self.w,
        #         axis=self.axis,
        #         last=True,
        #         broadcast=self.broadcast,
        #     )
        # else:
        #     return central.central_moments(
        #         x=self.x, mom=self.mom, w=self.w, axis=self.axis, last=True
        #     )

    @gcached()
    def s(self):
        s = self.cls.zeros(shape=self.shape_val, mom=self.mom)
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

    # @gcached()
    # def sfix(self):
    #     return self.cls.from_vals(
    #         x=self.x, w=self.w, axis=self.axis, mom=self.mom, broadcast=self.broadcast
    #     )

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
        if self.style == 'total':
            if not self.cov:
                raw = np.array([np.average(self.x**i, weights=self.w, axis=self.axis) for i in range(self.mom + 1)])
                raw[0, ...] = self.w.sum(self.axis)

                raw = np.moveaxis(raw, 0, -1)

            else:

                raw = np.zeros_like(self.data_test)
                for i in range(self.mom[0] + 1):
                    for j in range(self.mom[1] + 1):
                        raw[..., i, j] = np.average(self.x[0]**i * self.x[1]**j, weights=self.w, axis=self.axis)

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
    def xr_data(self):
        xdata = self.xdata

        if self.axis != 0:
            xdata = np.moveaxis(xdata, self.axis, 0)

        return np.take(
            xdata,
            self.indices,
            axis=0
        )

    @gcached()
    def yr_data(self):
        ydata = self.ydata

        if self.style == 'broadcast':
            return np.take(ydata, self.indices, axis=0)
        else:
            if self.axis != 0:
                ydata = np.moveaxis(ydata, self.axis, 0)
            return np.take(
                ydata,
                self.indices,
                axis=0
            )

    @property
    def xr(self):
        if self.cov:
            return (self.xr_data, self.yr_data)
        else:
            return self.xr_data


    @gcached()
    def wr(self):
        w = self.w

        if self.style is None:
            return w
        elif self.style == 'broadcast':
            return np.take(w, self.indices, axis=0)
        else:
            if self.axis != 0:
                w = np.moveaxis(w, self.axis, 0)
            return np.take(
                w,
                self.indices,
                axis=0
            )


    @gcached()
    def datar_test(self):
        return central.central_moments(
            x=self.xr, mom=self.mom, w=self.wr, axis=1, broadcast=self.broadcast
        )

        # if self.cov:
        #     return central.central_comoments(
        #         x=self.xr[0],
        #         y=self.xr[1],
        #         mom=self.mom,
        #         w=self.wr,
        #         axis=1,
        #         broadcast=self.broadcast
        #     )
        # else:
        #     return central.central_moments(
        #         x=self.xr,
        #         mom=self.mom,
        #         w=self.wr,
        #         axis=1,
        #     )







# Fixutre

def get_params():
    i = -1
    for shape, axis in [(20, 0), ((20, 2, 3), 0), ((2, 20, 3), 1), ((2, 3, 20), 2)]:
        for style in [None, "total", "broadcast"]:
            for mom in [4, (3, 3)]:
                # i += 1
                # if i != 12:
                #     continue
                yield Data(shape, axis, style, mom)
#                params.append(Data(shape, axis, style, mom))


@pytest.fixture(params=get_params(), scope="module")
def other(request):
    return request.param

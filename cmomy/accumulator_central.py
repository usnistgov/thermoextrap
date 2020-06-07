from __future__ import absolute_import

import numpy as np
from numba import njit
from scipy.special import binom

from .cached_decorators import gcached


def _my_broadcast(x, shape):
    x = np.array(x)
    if x.shape != shape:
        x = np.broadcast(x, shape)
    return x

def _shape_insert_axis(shape, axis, new_size):
    """
    given shape, get new shape with size put in position axis
    """
    if axis < 0:
        axis += len(shape) + 1
        print(len(shape), axis)

    shape = list(shape)
    shape.insert(axis, new_size)
    return tuple(shape)

def _multi_broadcast(x, shape, axis):
    # figure out
    x = np.array(x)

    # if array, and 1d with size same as shape[axis]
    # broadcast from here
    if (x.ndim == 1 and
        x.ndim != len(shape) and
        len(x) == shape[axis]):
        # reshape for broadcasting
        reshape = [1] * (len(shape) - 1)
        reshape = _shape_insert_axis(reshape, axis, -1)
        x = x.reshape(*reshape)

    if x.shape != shape:
        x = np.broadcast_to(x, shape)
    if axis != 0:
        x = np.rollaxis(x, axis, 0)
    return x

def central_moments(x, moments, weights=None, axis=0, out=None):
    """
    calculate central moments along axis
    """

    if weights is None:
        weights = np.ones_like(x)

    weights = np.array(weights)
    x = np.array(x)

    if weights.shape != x.shape:
        # if 1d try to broadcast
        if (weights.ndim == 1 and
            weights.ndim != x.ndim and
            len(weights) == x.shape[axis]):
            shape = [1] * x.ndim
            shape[axis] = -1
            weights = weights.reshape(*shape)

        # try to broadcast
        weights = np.broadcast_to(weights, x.shape)

    if axis < 0:
        axis += x.ndim
    if axis != 0:
        x = np.rollaxis(x, axis, 0)
        weights = np.rollaxis(weights, axis, 0)


    shape = (moments+1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=x.dtype)
    assert out.shape == shape


    wsum = weights.sum(axis=0)
    wsum_inv = 1.0 / wsum
    xave = np.einsum('r...,r...->...', weights, x) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, moments+1).reshape(*shape)

    dx = (x[None,...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum('r..., mr...->m...', weights, dx) * wsum_inv
    return out


def central_comoments(x0, x1, moments, weights=None, axis=0, out=None):
    """
    calculate central co-moments (covariance, etc) along axis
    """

    if weights is None:
        weights = np.ones_like(x0)

    if isinstance(moments, int):
        moments = (moments,) * 2

    moments = tuple(moments)
    assert len(moments) == 2


    x0 = np.array(x0)
    def _broadcast(w):
        w = np.array(w)
        if w.shape != x0.shape:
            # if 1d try to broadcast
            if (w.ndim == 1 and
                w.ndim != x0.ndim and
                len(w) == x0.shape[axis]):
                shape = [1] * x0.ndim
                shape[axis] = -1
                w = w.reshape(*shape)
            # try to broadcast
            w = np.broadcast_to(w, x0.shape)
        return w

    weights = _broadcast(weights)
    x1 = _broadcast(x1)

    if axis < 0:
        axis += x.ndim
    if axis != 0:
        x0 = np.rollaxis(x0, axis, 0)
        x1 = np.rollaxis(x1, axis, 0)
        weights = np.rollaxis(weights, axis, 0)


    shape = tuple(x+1 for x in moments) + x0.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=x0.dtype)
    assert out.shape == shape

    wsum = weights.sum(axis=0)
    wsum_inv = 1.0 / wsum

    x0ave = np.einsum('r...,r...->...', weights, x0) * wsum_inv
    x1ave = np.einsum('r...,r...->...', weights, x1) * wsum_inv

    shape = (-1,) + (1,) * (x0.ndim)
    p0 = np.arange(0, moments[0]+1).reshape(*shape)
    p1 = np.arange(0, moments[1]+1).reshape(*shape)

    dx0 = (x0[None, ...] - x0ave) ** p0
    dx1 = (x1[None, ...] - x1ave) ** p1

    #return weights, dx0, dx1

    #data = np.empty(tuple(x+1 for x in moments) + x0.shape[1:], dtype=x0.dtype)
    out[...] = np.einsum('r...,ir...,jr...->ij...', weights, dx0, dx1) * wsum_inv

    out[0,0, ...] = wsum
    out[1,0, ...] = x0ave
    out[0,1, ...] = x1ave

    return out


def myjit(func):
    """
    "my" jit function
    uses option inline='always', fastmath=True
    """
    return njit(inline="always", fastmath=True)(func)


def _factory_numba_binomial(order):
    irange = np.arange(order + 1)
    bfac = np.array([binom(i, irange) for i in irange])
    @myjit
    def numba_binom(n, k):
        return bfac[n, k]
    return numba_binom
_bfac = _factory_numba_binomial(10)

@myjit
def _push_val(data, w, x):

    if w == 0.0:
        return

    order = data.shape[0] - 1

    data[0] += w
    alpha = w / data[0]
    one_alpha = 1.0 - alpha

    delta = x - data[1]
    incr = delta * alpha

    data[1] += incr
    if order == 1:
        return

    for a in range(order, 2, -1):
        # c > 1
        tmp = 0.0
        delta_b = 1.0
        alpha_b = 1.0
        minus_b = 1.0
        one_alpha_b = 1.0
        for b in range(0, a - 1):
            c = a - b
            tmp += (
                _bfac(a, b)
                * delta_b
                * (minus_b * alpha_b * one_alpha * data[c])
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        # c == 0
        # b = a
        tmp += (
            delta * alpha * one_alpha * delta_b *
            (- minus_b * alpha_b + one_alpha_b)
        )

        data[a] = tmp

    data[2] = one_alpha * (data[2] + delta * incr)


@myjit
def _push_vals(data, W, X):
    ns = X.shape[0]
    for s in range(ns):
        _push_val(data, W[s], X[s])


@myjit
def _push_stat(data, w, a, v):
    """
    w : weight
    a : average
    v[i] : <dx**(i+2)>

    scale : parameter to rescale the weight
    """
    if w == 0:
        return

    order = data.shape[0] - 1

    data[0] += w

    alpha = w / data[0]
    one_alpha = 1.0 - alpha
    delta = a - data[1]
    incr = delta * alpha

    data[1] += incr

    if order == 1:
        return

    for a1 in range(order, 2, -1):
        tmp = 0.0
        delta_b = 1.0
        alpha_b = 1.0
        minus_b = 1.0
        one_alpha_b = 1.0
        for b in range(0, a1 - 1):
            c = a1 - b
            tmp += (
                _bfac(a1, b)
                * delta_b
                * (
                    minus_b * alpha_b * one_alpha * data[c]
                    + one_alpha_b * alpha * v[c - 2]
                )
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0


        c = 0
        b = a1 - c
        tmp += (
            delta * alpha * one_alpha * delta_b *
            (-minus_b * alpha_b + one_alpha_b)
        )
        data[a1] = tmp

    data[2] = v[0] * alpha + one_alpha * (data[2] + delta * incr)


@myjit
def _push_stats(data, W, A, V):
    ns = A.shape[0]
    for s in range(ns):
        _push_stat(data, W[s], A[s], V[s, ...])

@myjit
def _push_data(data, data_in):
    _push_stat(data, data_in[0], data_in[1], data_in[2:])


@myjit
def _push_datas(data, data_in):
    ns = data_in.shape[0]
    for s in range(ns):
        _push_stat(data, data_in[s, 0], data_in[s, 1], data_in[s, 2:])

@myjit
def _push_data_scale(data, data_in, scale):
    _push_stat(data, data_in[0] * scale, data_in[1], data_in[2:])

@myjit
def _push_datas_scale(data, data_in, scale):
    ns = data_in.shape[0]
    for s in range(ns):
        _push_stat(data,
                   data_in[s, 0] * scale[s, 0],
                   data_in[s, 1],
                   data_in[s, 2:])

# Vector
# Note: decided to change order from [value, moment] to [moment, value]
# typically as fast as previous, but makes things a bit cleaner.
@myjit
def _push_val_vec(data, w, x):
    nv = data.shape[1]
    for k in range(nv):
        _push_val(data[:, k], w[k], x[k])

@myjit
def _push_vals_vec(data, W, X):
    ns = X.shape[0]
    nv = data.shape[1]
    for s in range(ns):
        for k in range(nv):
            _push_val(data[:, k], W[s, k], X[s, k])


@myjit
def _push_stat_vec(data, w, a, v):
    nv = data.shape[1]
    for k in range(nv):
        _push_stat(data[:, k], w[k], a[k], v[:, k])


@myjit
def _push_stats_vec(data, W, A, V):
    # V[sample, moment-2, value]
    ns = A.shape[0]
    nv = data.shape[1]
    for s in range(ns):
        for k in range(nv):
            _push_stat(data[:, k], W[s, k], A[s, k], V[s, :, k])


@myjit
def _push_data_vec(data, data_in):
    nv = data.shape[1]
    for k in range(nv):
        _push_data(data[:, k], data_in[:, k])


@myjit
def _push_datas_vec(data, Data_in):
    ns = Data_in.shape[0]
    nv = data.shape[1]
    for s in range(ns):
        for k in range(nv):
            _push_data(data[:, k], Data_in[s, :, k])

@myjit
def _push_data_scale_vec(data, data_in, scale):
    nv = data.shape[1]
    for k in range(nv):
        _push_data(data[:, k], data_in[:, k], scale[k])


@myjit
def _push_datas_scale_vec(data, Data_in, scale):
    ns = Data_in.shape[0]
    nv = data.shape[1]
    for s in range(ns):
        for k in range(nv):
            _push_data(data[:, k], Data_in[s, :, k], scale[s, k])



class StatsAccumBase(object):
    """
    Base class for moments accumulation
    """
    def __init__(self, moments, shape=None, dtype=np.float):
        """
        Parameters
        ----------
        moments : int or tuple
        shape : tuple, optional
        dtype : data type
        """

        if isinstance(moments, int):
            moments = (moments,)

        self.moments = moments

        if shape is None:
            shape = ()
        self.shape = shape
        self.dtype = dtype

        self._init_subclass()

        self._data = np.zeros(self._data_shape, dtype=self.dtype)
        self._data_flat = self._data.reshape(self._data_flat_shape)


    @gcached()
    def _moments_shape(self):
        """tuple of shape for moments"""
        return tuple(x+1 for x in self.moments)

    @gcached()
    def _moments_shape_var(self):
        """shape of variance"""
        return tuple(x-1 for x in self.moments)

    @gcached()
    def _shape_flat(self):
        if self.shape is ():
            return ()
        else:
            return (np.prod(self.shape),)

    @gcached()
    def _shape_var(self):
        return self._moments_shape_var + self.shape

    @gcached()
    def _shape_flat_var(self):
        return self._moments_shape_var + self._shape_flat

    @gcached()
    def _data_shape(self):
        """shape of data"""
        return self._moments_shape + self.shape

    @gcached()
    def _data_flat_shape(self):
        """shape of conformed data"""
        return self._moments_shape + self._shape_flat


    def __setstate__(self, state):
        self.__dict__ = state
        # make sure datar points to data
        self._data_flat = self._data.reshape(self._data_flat_shape)


    @property
    def values(self):
        return self._data

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return len(self.shape)

    @gcached()
    def _unit_weight(self):
        return np.ones(self.shape)

    def _reshape_flat(self, x, nrec=None):
        if nrec is None:
            x = x.reshape(self._shape_flat)
        else:
            x = x.reshape(*((nrec,) + self._shape_flat))

        if x.ndim == 0:
            x = x[()]
        return x

    def check_weight(self, w):
        if w is None:
            w = self._unit_weight
        else:
            w = _my_broadcast(w, self.shape)
        return self._reshape_flat(w)

    def check_weights(self, w, nrec, axis=0):
        if w is None:
            w = self._unit_weight
            w = w.reshape(
                _shape_insert_axis(w.shape, axis, 1))

        shape = _shape_insert_axis(self.shape, axis, nrec)
        w = _multi_broadcast(w, shape, axis)
        return self._reshape_flat(w, nrec)


    def check_val(self, x, broadcast=False):
        x = np.array(x)
        if broadcast:
            x = np.broadcast_to(x.shape)

        assert x.shape == self.shape
        return self._reshape_flat(x)

    def check_vals(self, x, axis=0, broadcast=False):
        x = np.array(x)

        if broadcast:
            shape = _shape_insert_axis(self.shape, axis, x.shape[axis])
            x = _multi_broadcast(x, shape, axis)
        else:
            if axis !=0:
                x = np.rollaxis(x, axis, 0)

        assert x.shape[1:] == self.shape
        return self._reshape_flat(x, x.shape[0])

    def check_ave(self, a):
        return self.check_val(a)

    def check_aves(self, a, axis=0):
        return self.check_vals(a)

    def check_var(self, v):
        v = np.array(v)
        assert v.shape == self._shape_var
        return v.reshape(self._shape_flat_var)

    def check_vars(self, v, axis=0):
        v = np.array(v)
        if axis != 0:
            v = np.rollaxis(v, axis, 0)
        assert v.shape[1:] == self._shape_var
        return v.reshape(
            v.shape[:1] + self._shape_flat_var)

    def check_data(self, data):
        data = np.array(data)
        assert data.shape == self._data_shape
        return data.reshape(self._data_flat_shape)

    def check_datas(self, datas, axis=0):
        datas = np.array(datas)
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)
        assert datas.shape[1:] == self._data_shape
        return datas.reshape(
            datas.shape[:1] + self._data_flat_shape)

    def zero(self):
        self._data.fill(0.0)

    def zeros_like(self):
        """create zero object like self"""
        return self.__class__(moments=self.moments, shape=self.shape, dtype=self.dtype)


    def zeros_like(self):
        """create zero object like self"""
        return self.__class__(shape=self.shape, dtype=self.dtype, moments=self.moments)

    def copy(self):
        new = self.__class__(shape=self.shape, dtype=self.dtype, moments=self.moments)
        new._data[...] = self._data[...]
        return new

    @gcached()
    def _weight_index(self):
        index = [0] * len(self.moments)
        if self.ndim > 0:
            index += [...]
        return tuple(index)

    @gcached(prop=False)
    def _single_index(self, val):
        dims = len(self.moments)
        if dims ==0:
            index = (val,)
        else:
            index = []
            for i in range(dims):
                indexer = [0] * dims
                indexer[i] = val
                index.append(indexer)
        return index


    def weight(self):
        return self._data[self._weight_index]

    def mean(self):
        return self._data[self._single_index(1)]

    def var(self):
        return self._data[self._single_index(2)]

    def std(self):
        return np.sqrt(self.var(mom=2))


    def _check_other(self, b):
        assert type(self) == type(b)
        assert self.shape == b.shape
        assert self.moments == b.moments

    def __iadd__(self, b):
        self._check_other(b)
        self.push_data(b.data)
        return self

    def __add__(self, b):
        self._check_other(b)
        new = self.copy()
        new.push_data(b.data)
        return new

    def __isub__(self, b):
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())

        data = b.data.copy()
        data[self._weight_index] *= -1
        self.push_data(data)

        return self

    def __sub__(self, b):
        assert type(self) == type(b)
        assert np.all(self.weight() > b.weight())

        new = b.copy()
        new._data[self._weight_index] *= -1
        new.push_data(self.data)
        return new

    def __mul__(self, scale):
        """
        new object with weights scaled by scale
        """
        scale = float(scale)
        new = self.copy()
        new._data[self._weight_index] *= scale
        return new


    def __imul__(self, scale):
        scale = float(scale)
        self._data[self._weight_index] *= scale
        return self

    def push_data(self, data):
        data = self.check_data(data)
        self._push_data(self._data_flat, data)

    def push_datas(self, datas, axis=0):
        datas = self.check_datas(datas, axis)
        self._push_datas(self._data_flat, datas)



class _StatsAccum(StatsAccumBase):
    def push_val(self, x, w=None):
        xr = self.check_val(x)
        wr = self.check_weight(w)
        self._push_val(self._data_flat, wr, xr)

    def push_vals(self, x, w=None, axis=0):
        xr = self.check_vals(x, axis)
        wr = self.check_weights(w, xr.shape[0], axis)
        self._push_vals(self._data_flat, wr, xr)

    def push_stat(self, a, v=0.0, w=None):
        ar = self.check_ave(a)
        vr = self.check_var(v)
        wr = self.check_weight(w)
        self._push_stat(self._data_flat, wr, ar, vr)

    def push_stats(self, a, v=0.0, w=None, axis=0):
        ar = self.check_aves(a, axis)
        vr = self.check_vars(v, axis)
        wr = self.check_weights(w, ar.shape[0], axis)
        self._push_stats(self._data_flat, wr, ar, vr)


    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_vals(cls, x, w=None, axis=0, dtype=None, shape=None, moments=2):
        # get shape
        if shape is None:
            shape = list(x.shape)
            shape.pop(axis)
            shape = tuple(shape)

        if dtype is None:
            dtype = x.dtype
        new = cls(shape=shape, dtype=dtype, moments=moments)
        new.push_vals(x, axis=axis, w=w)
        return new


    @classmethod
    def from_data(cls, data, shape=None, moments=None):
        if moments is None:
            moments = data.shape[0] - 1
        assert data.shape[0] == moments + 1

        if shape is None:
            shape = data.shape[1:]
        new = cls(shape=shape, dtype=data.dtype, moments=moments)

        datar = new.check_data(data)
        new._data_flat[...] = datar
        return new

    @classmethod
    def from_datas(cls, datas, shape=None, axis=0, moments=None):
        """
        Data should have shape

        [:, moment, ...] (axis=0)

        [moment, axis, ...] (axis=1)

        [moment, ..., axis, ...] (axis=n)
        """

        datas = np.array(datas)
        if axis < 0:
            axis += datas.ndim
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)

        if moments is None:
            moments = datas.shape[1] - 1

        assert datas.shape[1] == moments + 1

        if shape is None:
            shape = datas.shape[2:]

        new = cls(shape=shape, dtype=datas.dtype, moments=moments)
        new.push_datas(datas=datas, axis=0)
        return new


    @classmethod
    def from_stat(cls, a, v=0.0, w=None, shape=None, moments=2):
        """
        object from single weight, average, variance/covariance
        """


        if shape is None:
            shape = a.shape
        new = cls(shape=shape, dtype=a.dtype, moments=moments)
        new.push_stat(w=w, a=a, v=v)
        return new

    @classmethod
    def from_stats(cls, a, v=0.0, w=None, axis=0, shape=None, moments=2):
        """
        object from several weights, averages, variances/covarainces along axis
        """

        # get shape
        if shape is None:
            shape = list(A.shape)
            shape.pop(axis)
            shape = tuple(shape)

        new = cls(shape=shape, dtype=A.dtype, moments=moments)
        new.push_stats(a=a, v=v, w=w, axis=axis)
        return new



class _VecMixin(object):
    def reduce(self, axis=0):
        """
        create new object reduced along axis
        """
        ndim = len(self.shape)
        if axis < 0:
            axis += ndim
        assert axis >= 0 and axis <= ndim

        shape = list(self.shape)
        shape.pop(axis)
        shape = tuple(shape)

        Data = self.data
        if Data.ndim == len(self.moments) +1:
            assert axis == 0
            Data = Data[..., None]

        # offset axis because first dim is for moments
        axis += len(self.moments)

        new = self.__class__.from_datas(Data, axis=axis, moments=self.moments)
        return new

    # def to_array(self, axis=0):
    #     if axis < 0:
    #         axis += self.data.ndim

    #     data = self.data
    #     if axis != 0:
    #         data = np.rollaxis(data, axis, 0)

    #     if data.ndim == 2:
    #         # expand
    #         data = data[:, None, :]

    #     # data[rec, moment, ...]
    #     shape = data.shape[2:]
    #     return StatsArray.from_datas(Data=data, child=self.__class__, shape=shape)




class StatsAccumVec(_StatsAccum, _VecMixin):
    def _init_subclass(self):
        self._push_val = _push_val_vec
        self._push_vals = _push_vals_vec
        self._push_stat = _push_stat_vec
        self._push_stats = _push_stats_vec

        self._push_data = _push_data_vec
        self._push_datas = _push_datas_vec



class StatsAccum(_StatsAccum):
    def _init_subclass(self):
        self._push_val = _push_val
        self._push_vals = _push_vals
        self._push_stat = _push_stat
        self._push_stats = _push_stats

        self._push_data = _push_data
        self._push_datas = _push_datas


######################################################################
# Covariance "stuff"
######################################################################

@myjit
def _push_val_cov(data, w, x0, x1):

    if w == 0.0:
        return

    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    data[0, 0] += w
    alpha = w / data[0, 0]
    one_alpha = 1.0 - alpha

    delta0 = x0 - data[1, 0]
    delta1 = x1 - data[0, 1]

    incr0 = delta0 * alpha
    incr1 = delta1 * alpha

    data[1, 0] += incr0
    data[0, 1] += incr1

    a0_min = max(0, 2 - order1)
    for a0 in range(order0, a0_min-1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(0, a0+1):
                c0 = a0 - b0
                f0 = _bfac(a0, b0)

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(0, a1+1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += delta0_b0 * delta1_b1  * (
                            minus_bb * alpha_bb * one_alpha 
                            + one_alpha_bb * alpha)
                    elif cs != 1:
                        tmp += (
                            f0 * _bfac(a1, b1)
                            * delta0_b0 * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha * data[c0,c1])
                        )
                    delta1_b1 *= delta1
                    alpha_bb *= alpha
                    one_alpha_bb *= one_alpha
                    minus_bb *= -1

                delta0_b0 *= delta0
                alpha_b0 *= alpha
                minus_b0 *= -1
                one_alpha_b0 *= one_alpha

            # This is slower
            # tmp = 0.0
            # for b0 in range(0, a0+1):
            #     c0 = a0 - b0
            #     if b0 == 0:
            #         f0 = 1.0
            #         delta0_b0 = 1.0
            #         alpha_b0 = 1.0
            #         minus_b0 = 1.0
            #         one_alpha_b0 = 1.0
            #     else:
            #         f0 = _bfac(a0, b0)
            #         delta0_b0 *= delta0
            #         alpha_b0 *= alpha
            #         minus_b0 *= -1
            #         one_alpha_b0 *= one_alpha

            #     for b1 in range(0, a1+1):
            #         c1 = a1 - b1

            #         if b1 == 0:
            #             delta1_b1 = 1.0
            #             alpha_bb = alpha_b0
            #             minus_bb = minus_b0
            #             one_alpha_bb = one_alpha_b0
            #         else:
            #             delta1_b1 *= delta1
            #             alpha_bb *= alpha
            #             one_alpha_bb *= one_alpha
            #             minus_bb *= -1

            #         cs = c0 + c1
            #         if cs == 0:
            #             tmp += delta0_b0 * delta1_b1  * (
            #                 minus_bb * alpha_bb * one_alpha 
            #                 + one_alpha_bb * alpha)
            #         elif cs != 1:
            #             tmp += (
            #                 f0 * _bfac(a1, b1)
            #                 * delta0_b0 * delta1_b1
            #                 * (minus_bb * alpha_bb * one_alpha * data[c0,c1])
            #             )
            data[a0, a1] = tmp


@myjit
def _push_vals_cov(data, W, X1, X2):
    ns = X1.shape[0]
    for s in range(ns):
        _push_val_cov(data, W[s], X1[s], X2[s])


@myjit
def _push_data_scale_cov(data, data_in, scale):

    w = data_in[0, 0] * scale
    if w == 0.0:
        return

    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    data[0, 0] += w
    alpha = w / data[0, 0]
    one_alpha = 1.0 - alpha

    delta0 = data_in[1, 0] - data[1, 0]
    delta1 = data_in[0, 1] - data[0, 1]

    incr0 = delta0 * alpha
    incr1 = delta1 * alpha

    data[1, 0] += incr0
    data[0, 1] += incr1

    a0_min = max(0, 2 - order1)
    for a0 in range(order0, a0_min-1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            # Alternative
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(0, a0+1):
                c0 = a0 - b0
                f0 = _bfac(a0, b0)

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(0, a1+1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += delta0_b0 * delta1_b1  * (
                            minus_bb * alpha_bb * one_alpha
                            + one_alpha_bb * alpha)
                    elif cs != 1:
                        tmp += (
                            f0 * _bfac(a1, b1)
                            * delta0_b0 * delta1_b1
                            * (
                                minus_bb * alpha_bb * one_alpha * data[c0,c1]
                                + one_alpha_bb * alpha * data_in[c0, c1]
                            )
                        )
                    delta1_b1 *= delta1
                    alpha_bb *= alpha
                    one_alpha_bb *= one_alpha
                    minus_bb *= -1

                delta0_b0 *= delta0
                alpha_b0 *= alpha
                minus_b0 *= -1
                one_alpha_b0 *= one_alpha

            data[a0, a1] = tmp


@myjit
def _push_data_cov(data, data_in):
    _push_data_scale_cov(data, data_in, 1.0)

@myjit
def _push_datas_cov(data, datas):
    ns = datas.shape[0]
    for s in range(ns):
        _push_data_scale_cov(data, datas[s], 1.0)

def _push_datas_scale_cov(data, datas, scale):
    ns = datas.shape[0]
    for s in range(ns):
        _push_data_scale_cov(data, datas[s], scale[s])



# Vector
@myjit
def _push_val_cov_vec(data, w, x0, x1):
    nv = data.shape[-1]
    for k in range(nv):
        _push_val_cov(data[..., k], w[k], x0[k], x1[k])


@myjit
def _push_vals_cov_vec(data, W, X0, X1):
    nv = data.shape[-1]
    ns = X0.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_val_cov(data[...,k], W[s, k], X0[s, k], X1[s, k])

@myjit
def _push_data_cov_vec(data, data_in):
    nv = data.shape[-1]
    for k in range(nv):
        _push_data_scale_cov(data[...,k], data_in[..., k], 1.0)

@myjit
def _push_datas_cov_vec(data, Datas):
    nv = data.shape[-1]
    ns = Datas.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_data_scale_cov(data[..., k],
                                 Datas[s, ..., k],
                                 1.0)

@myjit
def _push_data_scale_cov_vec(data, data_in, scale):
    nv = data.shape[-1]
    for k in range(nv):
        _push_data_scale_cov(data[...,k], data_in[..., k],
                             scale[k])

@myjit
def _push_datas_scale_cov_vec(data, Datas, scale):
    nv = data.shape[-1]
    ns = Datas.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_data_scale_cov(data[..., k],
                                 Datas[s, ..., k],
                                 scale[s, k])

def _central_cov_from_vals(data, W, X0, X1):
    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    wsum = W.sum(axis=0)
    wsum_inv = 1.0 / wsum

    x0ave = (W * X0).sum(axis=0) * wsum_inv
    x1ave = (W * X1).sum(axis=0) * wsum_inv


    dx0 = X0 - x0ave
    dx1 = X1 - x1ave

    p0 = np.arange(0, order0+1)
    p1 = np.arange(0, order1+1)

    dx0 = dx0[:, None, ...] ** p0
    dx1 = dx1[:, None, ...] ** p1

    # data[0] = wsum
    # data[1] = xave

    data[...] = np.einsum('r,ri...,rj...->ij...', W, dx0, dx1) * wsum_inv
    data[0, 0] = wsum
    data[1, 0] = x0ave
    data[0, 1] = x1ave



class _StatsAccumCov(StatsAccumBase):
    def __init__(self, moments, shape=None, dtype=np.float):
        if isinstance(moments, int):
            moments = (moments,) * 2
        moments = tuple(moments)
        assert len(moments) == 2
        super(_StatsAccumCov, self).__init__(moments=moments, shape=shape, dtype=dtype)



    def push_val(self, x0, x1, w=None, broadcast=False):
        x0 = self.check_val(x0)
        x1 = self.check_val(x1, broadcast=broadcast)
        w = self.check_weight(w)
        self._push_val(self._data_flat, w, x0, x1)


    def push_vals(self, x0, x1, w=None, axis=0, broadcast=False):
        x0 = self.check_vals(x0, axis)
        x1 = self.check_vals(x1, axis, broadcast)
        w = self.check_weights(w, x0.shape[0], axis)
        self._push_vals(self._data_flat, w, x0, x1)

    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_vals(cls, x0, x1, w=None, axis=0, dtype=None, shape=None,
                  broadcast=False,
                  moments=2):

        # get shape
        if shape is None:
            shape = list(x0.shape)
            shape.pop(axis)
            shape = tuple(shape)
        if dtype is None:
            dtype = x0.dtype

        new = cls(shape=shape, dtype=dtype, moments=moments)
        new.push_vals(x0=x0, x1=x1, axis=axis, w=w, broadcast=broadcast)
        return new


    @classmethod
    def from_data(cls, data, shape=None, moments=None):

        if moments is None:
            moments = tuple(x - 1 for x in data.shape[:2])

        if isinstance(moments, int):
            moments = (moments,) * 2


        assert data.shape[:2] == tuple(x+1 for x in moments)

        if shape is None:
            shape = data.shape[2:]

        new = cls(shape=shape, dtype=data.dtype, moments=moments)
        datar = new.check_data(data)
        new._data_flat[...] = datar
        return new

    @classmethod
    def from_datas(cls, datas, shape=None, axis=0, moments=None):
        """
        Data should have shape

        [:, moment, ...] (axis=0)

        [moment, axis, ...] (axis=1)

        [moment, ..., axis, ...] (axis=n)
        """

        datas = np.array(datas)
        if axis < 0:
            axis += datas.ndim
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)

        if moments is None:
            moments = tuple(x-1 for x in datas.shape[1:3])

        if isinstance(moments, int):
            moments = (moments,) * 2

        assert datas.shape[1:3] == tuple(x + 1 for x in moments)

        if shape is None:
            shape = datas.shape[3:]

        new = cls(shape=shape, dtype=datas.dtype, moments=moments)
        new.push_datas(datas=datas, axis=0)
        return new




class StatsAccumCov(_StatsAccumCov):
    def _init_subclass(self):
        self._push_val = _push_val_cov
        self._push_vals = _push_vals_cov
        self._push_data = _push_data_cov
        self._push_datas = _push_datas_cov


class StatsAccumCovVec(_StatsAccumCov, _VecMixin):
    def _init_subclass(self):
        self._push_val = _push_val_cov_vec
        self._push_vals = _push_vals_cov_vec
        self._push_data = _push_data_cov_vec
        self._push_datas = _push_datas_cov_vec








# class _StatsAccum(object):
#     def __init__(self, shape, dtype=np.float, nmom=2):

#         self.nmom = nmom
#         self._nmom_shape = (self.nmom + 1,)
#         self._nmom_var_shape = (self.nmom - 1,)

#         self._shape = shape
#         self._shape_var = self._nmom_var_shape + self._shape
#         self._dtype = dtype

#         self._init_subclass()

#         self._data = np.empty(self._nmom_shape + self._shape, dtype=self._dtype)

#         if getattr(self, "_shape_r", None) is None:
#             if self.shape is ():
#                 self._shape_r = ()
#             else:
#                 self._shape_r = (np.prod(self.shape),)

#         if getattr(self, "_shape_var_r", None) is None:
#             self._shape_var_r = self._nmom_var_shape + self._shape_r

#         self._datar = self._data.reshape(self._nmom_shape + self._shape_r)
#         self.zero()


#     # when unpickling, make sure self._datar points to same
#     # underlieing array as self._data
#     def __setstate__(self, state):
#         self.__dict__ = state
#         # make sure datar points to data
#         self._datar = self._data.reshape(self._shape_var_r + self._nmom_shape)

#     def _init_subclass(self):
#         """any special subclass stuff here"""
#         pass

#     @property
#     def data(self):
#         return self._data

#     @property
#     def shape(self):
#         """shape of input values"""
#         return self._shape

#     @property
#     def shape_var(self):
#         return self._shape_var

#     @property
#     def ndim(self):
#         return len(self.shape)

#     @property
#     def ndim_var(self):
#         return len(self.shape_var)

#     @property
#     def dtype(self):
#         return self._dtype

#     @property
#     def _unit_weight(self):
#         if not hasattr(self, "_unit_weight_val"):
#             self._unit_weight_val = np.ones(self.shape)
#         return self._unit_weight_val

#     def _check_weight(self, w):
#         if w is None:
#             w = self._unit_weight
#         if np.shape(w) != self.shape:
#             w = np.broadcast_to(w, self.shape)
#         return np.reshape(w, self._shape_r)

#     def _check_weights(self, W, X, axis=0):
#         shape = list(self.shape)
#         shape.insert(axis, X.shape[axis])

#         if W is None:
#             W = self._unit_weight

#         if W.ndim == 1 and W.ndim != X.ndim and len(W) == X.shape[axis]:
#             # assume equal weight for each record
#             rshape = [1] * X.ndim
#             rshape[axis] = -1
#             W = W.reshape(*rshape)


#         if np.shape(W) != shape:
#             W = np.broadcast_to(W, shape)
#         if axis != 0:
#             W = np.rollaxis(W, axis, 0)
#         shaper = W.shape[:1] + self._shape_r
#         return np.reshape(W, shaper)

#     def _check_val(self, x):
#         assert np.shape(x) == self.shape
#         return np.reshape(x, self._shape_r)

#     def _check_vals(self, X, axis=0):
#         if axis != 0:
#             X = np.rollaxis(X, axis, 0)
#         assert np.shape(X)[1:] == self.shape
#         shaper = X.shape[:1] + self._shape_r
#         return np.reshape(X, shaper)

#     def _check_ave(self, a):
#         if a.shape != self.shape:
#             a = np.broadcast_to(a, self.shape)
#         return np.reshape(a, self._shape_r)

#     def _check_aves(self, A, axis=0):
#         if axis != 0:
#             A = np.rollaxis(A, axis, 0)
#         assert np.shape(A)[1:] == self.shape, "{}, {}".format(np.shape(A), self.shape)
#         # if np.shape(A)[1:] != self.shape_var:
#         #     new_shape = (A.shape[0], ) + self.shape_var
#         #     A = np.broadcast_to(A, new_shape)
#         shaper = A.shape[:1] + self._shape_r
#         return np.reshape(A, shaper)

#     def _check_var(self, v):
#         if np.shape(v) != self.shape_var:
#             v = np.broadcast_to(v, self.shape_var)
#         return np.reshape(v, self._shape_var_r)

#     def _check_vars(self, V, X, axis=0):
#         shape = list(self.shape_var)
#         shape.insert(axis, X.shape[axis])
#         if np.shape(V) != shape:
#             V = np.broadcast_to(V, shape)
#         if axis != 0:
#             V = np.rollaxis(V, axis, 0)
#         shaper = V.shape[:1] + self._shape_var_r
#         return np.reshape(V, shaper)

#     def _check_data(self, data):
#         shape = self._nmom_shape + self.shape
#         if np.shape(data) != shape:
#             raise ValueError("data must of have shape {}".format(shape))
#         return np.reshape(data, self._datar.shape)

#     def _check_datas(self, datas, axis=0):
#         shape = self._data.shape
#         if axis != 0:
#             datas = np.rollaxis(datas, axis, 0)
#         if np.shape(datas)[1:] != shape:
#             raise ValueError(
#                 "bad shape {} != {}, axis={}".format(datas.shape, shape, axis)
#             )
#         shaper = datas.shape[:1] + self._datar.shape
#         return np.reshape(datas, shaper)

#     def zero(self):
#         self._data.fill(0.0)

#     def zeros_like(self):
#         """create zero object like self"""
#         return self.__class__(shape=self.shape, dtype=self.dtype, nmom=self.nmom)

#     def copy(self):
#         new = self.__class__(shape=self.shape, dtype=self.dtype, nmom=self.nmom)
#         new._data[...] = self._data[...]
#         return new

#     def push_val(self, x, w=None):
#         xr = self._check_val(x)
#         wr = self._check_weight(w)
#         self._push_val(self._datar, wr, xr)

#     def push_vals(self, X, W=None, axis=0):
#         Xr = self._check_vals(X, axis)
#         Wr = self._check_weights(W, X, axis)
#         self._push_vals(self._datar, Wr, Xr)

#     def push_stat(self, a, v=0.0, w=None):
#         ar = self._check_ave(a)
#         vr = self._check_var(v)
#         wr = self._check_weight(w)
#         self._push_stat(self._datar, wr, ar, vr)

#     def push_stats(self, A, V=0.0, W=None, axis=0):
#         Ar = self._check_aves(A, axis)
#         Vr = self._check_vars(V, A, axis)
#         Wr = self._check_weights(W, A, axis)
#         self._push_stats(self._datar, Wr, Ar, Vr)

#     def push_stat_data(self, data):
#         data = self._check_data(data)
#         self._push_data(self._datar, data)

#     def push_stats_data(self, Data, axis=0):
#         Data = self._check_datas(Data, axis)
#         self._push_datas(self._datar, Data)

#     def _check_other(self, b):
#         assert type(self) == type(b)
#         assert self.shape == b.shape

#     def __iadd__(self, b):
#         self._check_other(b)
#         self.push_stat(w=b.weight(), a=b.mean(), v=b.cmom())
#         return self

#     def __add__(self, b):
#         self._check_other(b)
#         new = self.copy()
#         new.push_stat(w=b.weight(), a=b.mean(), v=b.cmom())
#         return new

#     def __isub__(self, b):
#         self._check_other(b)
#         assert np.all(self.weight() >= b.weight())
#         self.push_stat(w=-b.weight(), a=b.mean(), v=b.cmom())
#         return self

#     def __sub__(self, b):
#         assert type(self) == type(b)
#         assert np.all(self.weight() > b.weight())
#         new = self.copy()
#         new.push_stat(w=-b.weight(), a=b.mean(), v=b.cmom())
#         return new

#     def __mul__(self, scale):
#         """
#         new object with weights scaled by scale
#         """
#         scale = float(scale)
#         new = self.copy()
#         new._data[0] = new._data[0] * scale
#         return new


#     def __imul__(self, scale):
#         scale = float(scale)
#         self._data[0] = self._data[0] * scale
#         return self

#     def weight(self):
#         return self._data[0]

#     def mean(self):
#         return self._data[1]

#     def var(self, mom=2):
#         if mom is None:
#             mom = slice(2, None)
#         out = self._data[mom]

#     def cmom(self):
#         return self._data[2:]

#     def std(self):
#         return np.sqrt(self._data[2])

#     # --------------------------------------------------
#     # constructors
#     # --------------------------------------------------
#     @classmethod
#     def from_stat(cls, a=None, v=0.0, w=None, data=None, shape=None, nmom=2):
#         """
#         object from single weight, average, variance/covariance
#         """

#         if data is not None:
#             w = data[0]
#             a = data[1]
#             v = data[2:]
#         else:
#             assert a is not None

#         if shape is None:
#             shape = a.shape
#         new = cls(shape=shape, dtype=a.dtype, nmom=nmom)
#         new.push_stat(w=w, a=a, v=v)
#         return new

#     @classmethod
#     def from_stats(cls, A=None, V=0.0, W=None, Data=None, axis=0, shape=None, nmom=2):
#         """
#         object from several weights, averages, variances/covarainces along axis
#         """

#         if Data is not None:
#             return cls.from_datas(Data, shape=shape, axis=axis, nmom=nmom)
#         else:
#             assert A is not None
#             # get shape
#             if shape is None:
#                 shape = list(A.shape)
#                 shape.pop(axis)
#                 shape = tuple(shape)

#             new = cls(shape=shape, dtype=A.dtype, nmom=nmom)
#             new.push_stats(W=W, A=A, V=V, axis=axis)
#             return new

#     @classmethod
#     def from_data(cls, data, shape=None, nmom=None):
#         if nmom is None:
#             nmom = data.shape[0] - 1
#         assert data.shape[0] == nmom + 1

#         if shape is None:
#             shape = data.shape[1:]
#         new = cls(shape=shape, dtype=data.dtype, nmom=nmom)

#         # new.push_stat_data(data=data)
#         # below is much faster
#         datar = new._check_data(data)
#         new._datar[...] = datar
#         return new

#     @classmethod
#     def from_datas(cls, Data, shape=None, axis=0, nmom=None):
#         """
#         Data should have shape

#         [:, moment, ...] (axis=0)

#         [moment, axis, ...] (axis=1)

#         [moment, ..., axis, ...] (axis=n)
#         """

#         Data = np.array(Data)
#         if axis < 0:
#             axis += Data.ndim

#         if axis != 0:
#             Data = np.rollaxis(Data, axis, 0)

#         if nmom is None:
#             nmom = Data.shape[1] - 1

#         assert Data.shape[1] == nmom + 1

#         if shape is None:
#             shape = Data.shape[2:]

#         new = cls(shape=shape, dtype=Data.dtype, nmom=nmom)
#         new.push_stats_data(Data=Data, axis=0)
#         return new

#     @classmethod
#     def from_vals(cls, X, W=None, axis=0, dtype=None, shape=None, nmom=2):

#         # get shape
#         if shape is None:
#             shape = list(X.shape)
#             shape.pop(axis)
#             shape = tuple(shape)

#         if dtype is None:
#             dtype = X.dtype
#         new = cls(shape=shape, dtype=dtype, nmom=nmom)
#         new.push_vals(X, axis=axis, W=W)
#         return new

#     def reduce(self, axis=0):
#         """
#         create new object reduced along axis
#         """
#         ndim = len(self.shape)
#         if axis < 0:
#             axis += ndim
#         assert axis >= 0 and axis <= ndim

#         shape = list(self.shape)
#         shape.pop(axis)
#         shape = tuple(shape)

#         Data = self.data
#         if Data.ndim == 2:
#             assert axis == 0
#             Data = Data[..., None]

#         # offset axis because first dim is for moments
#         axis += 1

#         new = self.__class__.from_datas(Data, axis=axis, nmom=self.nmom)
#         return new


# class StatsAccumVec(_StatsAccum):
#     def _init_subclass(self):
#         self._push_val = _push_val_vec
#         self._push_vals = _push_vals_vec
#         self._push_stat = _push_stat_vec
#         self._push_stats = _push_stats_vec

#         self._push_data = _push_data_vec
#         self._push_datas = _push_datas_vec

#     def to_array(self, axis=0):
#         if axis < 0:
#             axis += self.data.ndim

#         data = self.data
#         if axis != 0:
#             data = np.rollaxis(data, axis, 0)

#         if data.ndim == 2:
#             # expand
#             data = data[:, None, :]

#         # data[rec, moment, ...]
#         shape = data.shape[2:]
#         return StatsArray.from_datas(Data=data, child=self.__class__, shape=shape)


# class StatsAccum(_StatsAccum):
#     def __init__(self, shape=(), dtype=np.float, nmom=2):
#         super(StatsAccum, self).__init__(shape=(), dtype=dtype, nmom=nmom)

#     def _init_subclass(self):
#         self._push_val = _push_val
#         self._push_vals = _push_vals
#         self._push_stat = _push_stat
#         self._push_stats = _push_stats

#         self._push_data = _push_data
#         self._push_datas = _push_datas











# This doens't work because need data[n, 0], data[0, n]
# @myjit
# def _push_stat_cov(data, w, a, v):
#     # data[0, 0] = weight
#     # data[1, 0] = ave0
#     # data[0, 1] = ave1
#     # data[1:,1:] = var

#     if w == 0.0:
#         return

#     order0 = data.shape[0] - 1
#     order1 = data.shape[1] - 1

#     data[0, 0] += w
#     alpha = w / data[0, 0]
#     one_alpha = 1.0 - alpha

#     delta0 = a[0] - data[1, 0]
#     delta1 = a[1] - data[0, 1]

#     incr0 = delta0 * alpha
#     incr1 = delta1 * alpha

#     data[1, 0] += incr0
#     data[0, 1] += incr1

#     a0_min = max(0, 2 - order1)
#     for a0 in range(order0, a0_min-1, -1):
#         # if a0 + order1 < 2:
#         #     continue

#         a1_min = max(0, 2 - a0)
#         for a1 in range(order1, a1_min - 1, -1):
#             tmp = 0.0
#             delta0_b0 = 1.0
#             alpha_b0 = 1.0
#             minus_b0 = 1.0
#             one_alpha_b0 = 1.0
#             for b0 in range(0, a0+1):
#                 c0 = a0 - b0
#                 f0 = _bfac(a0, b0)

#                 delta1_b1 = 1.0
#                 alpha_bb = alpha_b0
#                 minus_bb = minus_b0
#                 one_alpha_bb = one_alpha_b0
#                 for b1 in range(0, a1+1):
#                     c1 = a1 - b1
#                     cs = c0 + c1
#                     if cs == 0:
#                         tmp += delta0_b0 * delta1_b1  * (
#                             minus_bb * alpha_bb * one_alpha
#                             + one_alpha_bb * alpha)
#                     elif cs != 1:
#                         tmp += (
#                             f0 * _bfac(a1, b1)
#                             * delta0_b0 * delta1_b1
#                             * (
#                                 minus_bb * alpha_bb * one_alpha * data[c0,c1]
#                                 + one_alpha_bb * alpha * v[c0-1, c1-1]
#                             )
#                         )
#                     delta1_b1 *= delta1
#                     alpha_bb *= alpha
#                     one_alpha_bb *= one_alpha
#                     minus_bb *= -1

#                 delta0_b0 *= delta0
#                 alpha_b0 *= alpha
#                 minus_b0 *= -1
#                 one_alpha_b0 *= one_alpha

#             data[a0, a1] = tmp


# @myjit
# def _push_stats_cov(data, W, A, V):
#     ns = A.shape[0]
#     for s in range(ns):
#         _push_stat_cov(data, W[s], A[s], V[s])


# class _StatsAccumCov(object):
#     def __init__(self, shape, dtype=np.float, nmom=(2, 2)):
#         self.nmom = nmom
#         self._nmom_shape = tuple(x + 1 for x in nmom)
#         self._nmom_var_shape = tuple(x - 1 for x in nmom)

#         self._shape = shape
#         self._shape_var = self._nmom_var_shape + self._shape
#         self._dtype = dtype

#         self._init_subclass()

#         self._data = np.empty(self._nmom_shape + self._shape, dtype=self._dtype)

#         if getattr(self, "_shape_r", None) is None:
#             if self.shape is ():
#                 self._shape_r = ()
#             else:
#                 self._shape_r = (np.prod(self.shape),)

#         if getattr(self, "_shape_var_r", None) is None:
#             self._shape_var_r = self._nmom_var_shape + self._shape_r
#         self._datar = self._data.reshape(self._nmom_shape + self._shape_r)
#         self.zero()

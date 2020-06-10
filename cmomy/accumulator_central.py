from __future__ import absolute_import
from functools import lru_cache

import numpy as np
from numba import njit, prange
from scipy.special import binom

from .cached_decorators import cached_clear, gcached



###############################################################################
# central moments/comoments routines
###############################################################################

def central_moments(x, moments, weights=None, axis=0, last=True, out=None):
    """
    calculate central moments along axis

    Parameters
    ----------
    x : array-like
        input data
    moments : int
        number of moments to calculate
    weights : array-like, optional
        if passed, should be able to broadcast to `x`.  An exception is if
        weights is a 1d array with len(weights) == x.shape[axis].  In this case, weights
        will be reshaped and broadcast against x
    axis : int, default=0
        axis to reduce along
    last : bool, default=True
        if True, put moments as last dimension.
        Otherwise, moments will be in first dimension
    out : array
        if present, use this for output data
        Needs to have shape of either (moments,) + shape or shape + (moments,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape shape + (moments,) or (moments,) + shape depending on value of `last`,
        where `shape` is the shape of `x` with axis removed, i.e., shape=x.shape[:axis] + x.shape[axis+1:].
        Assuming `last is True`, output[...,0] is the total weight (or count), output[...,1] is the mean value of x,
        output[...,n] with n>1 is the nth central moment
    """
    x = np.array(x)
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.array(weights)

    if weights.shape != x.shape:
        # if 1d try to broadcast
        if (
            weights.ndim == 1
            and weights.ndim != x.ndim
            and len(weights) == x.shape[axis]
        ):
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

    shape = (moments + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=x.dtype)
    else:
        if out.shape != shape:
            # try rolling
            out = np.moveaxis(out, -1, 0)
        assert out.shape == shape

    wsum = weights.sum(axis=0)
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", weights, x) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, moments + 1).reshape(*shape)

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", weights, dx) * wsum_inv

    if last:
        out = np.moveaxis(out, 0, -1)

    return out


def central_comoments(x0, x1, moments, weights=None, axis=0, last=True, out=None):
    """
    calculate central co-moments (covariance, etc) along axis
    """
    x0 = np.array(x0)
    if weights is None:
        weights = np.ones_like(x0)


    if isinstance(moments, int):
        moments = (moments,) * 2

    moments = tuple(moments)
    assert len(moments) == 2



    def _broadcast(w):
        w = np.array(w)
        if w.shape != x0.shape:
            # if 1d try to broadcast
            if w.ndim == 1 and w.ndim != x0.ndim and len(w) == x0.shape[axis]:
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

    shape = tuple(x + 1 for x in moments) + x0.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=x0.dtype)
    else:
        if out.shape != shape:
            # try moving axis
            out = np.moveaxis(out, [-2, -1], [0, 1])
        assert out.shape == shape

    wsum = weights.sum(axis=0)
    wsum_inv = 1.0 / wsum

    x0ave = np.einsum("r...,r...->...", weights, x0) * wsum_inv
    x1ave = np.einsum("r...,r...->...", weights, x1) * wsum_inv

    shape = (-1,) + (1,) * (x0.ndim)
    p0 = np.arange(0, moments[0] + 1).reshape(*shape)
    p1 = np.arange(0, moments[1] + 1).reshape(*shape)

    dx0 = (x0[None, ...] - x0ave) ** p0
    dx1 = (x1[None, ...] - x1ave) ** p1

    # return weights, dx0, dx1

    # data = np.empty(tuple(x+1 for x in moments) + x0.shape[1:], dtype=x0.dtype)
    out[...] = np.einsum("r...,ir...,jr...->ij...", weights, dx0, dx1) * wsum_inv

    out[0, 0, ...] = wsum
    out[1, 0, ...] = x0ave
    out[0, 1, ...] = x1ave

    if last:
        out = np.moveaxis(out, [0, 1], [-2, -1])
    return out









###############################################################################
# Moments
###############################################################################

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
            tmp += _bfac(a, b) * delta_b * (minus_b * alpha_b * one_alpha * data[c])
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        # c == 0
        # b = a
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)

        data[a] = tmp

    data[2] = one_alpha * (data[2] + delta * incr)


@myjit
def _push_vals(data, W, X):
    ns = X.shape[0]
    for s in range(ns):
        _push_val(data, W[s], X[s])

@myjit
def _push_vals_scale(data, W, X, scale):
    ns = X.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        _push_val(data, W[s] * f, X[s])



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
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)
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
        f = scale[s]
        if f == 0:
            continue
        _push_stat(data, data_in[s, 0] * f, data_in[s, 1], data_in[s, 2:])


# Vector
@myjit
def _push_val_vec(data, w, x):
    nv = data.shape[0]
    for k in range(nv):
        _push_val(data[k, :], w[k], x[k])


@myjit
def _push_vals_vec(data, W, X):
    ns = X.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_val(data[k, :], W[s, k], X[s, k])


@myjit
def _push_stat_vec(data, w, a, v):
    nv = data.shape[0]
    for k in range(nv):
        _push_stat(data[k, :], w[k], a[k], v[k, :])


@myjit
def _push_stats_vec(data, W, A, V):
    # V[sample, moment-2, value]
    ns = A.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_stat(data[k, :], W[s, k], A[s, k], V[s, k, :])


@myjit
def _push_data_vec(data, data_in):
    nv = data.shape[0]
    for k in range(nv):
        _push_data(data[k, :], data_in[k, :])


@myjit
def _push_datas_vec(data, Data_in):
    ns = Data_in.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_data(data[k, :], Data_in[s, k, :])

@myjit
def _push_vals_scale_vec(data, W, X, scale):
    ns = X.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_val(data[k,:], W[s, k] * f, X[s, k])


@myjit
def _push_data_scale_vec(data, data_in, scale):
    nv = data.shape[0]
    if scale != 0:
        for k in range(nv):
            _push_data_scale(data[i, :], data_in[i, :], scale)


@myjit
def _push_datas_scale_vec(data, Data_in, scale):
    ns = Data_in.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_data_scale(data[k, :], Data_in[s, k, :], f)




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
    for a0 in range(order0, a0_min - 1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(0, a0 + 1):
                c0 = a0 - b0
                f0 = _bfac(a0, b0)

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(0, a1 + 1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += (
                            delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha + one_alpha_bb * alpha)
                        )
                    elif cs != 1:
                        tmp += (
                            f0
                            * _bfac(a1, b1)
                            * delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha * data[c0, c1])
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
def _push_vals_cov(data, W, X1, X2):
    ns = X1.shape[0]
    for s in range(ns):
        _push_val_cov(data, W[s], X1[s], X2[s])

@myjit
def _push_vals_scale_cov(data, W, X1, X2, scale):
    ns = X1.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        _push_val_cov(data, W[s] * f, X1[s], X2[s])




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
    for a0 in range(order0, a0_min - 1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            # Alternative
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(0, a0 + 1):
                c0 = a0 - b0
                f0 = _bfac(a0, b0)

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(0, a1 + 1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += (
                            delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha + one_alpha_bb * alpha)
                        )
                    elif cs != 1:
                        tmp += (
                            f0
                            * _bfac(a1, b1)
                            * delta0_b0
                            * delta1_b1
                            * (
                                minus_bb * alpha_bb * one_alpha * data[c0, c1]
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
        f = scale[s]
        if f == 0:
            continue
        _push_data_scale_cov(data, datas[s], f)


# Vector
@myjit
def _push_val_cov_vec(data, w, x0, x1):
    nv = data.shape[0]
    for k in range(nv):
        _push_val_cov(data[k, ...], w[k], x0[k], x1[k])


@myjit
def _push_vals_cov_vec(data, W, X0, X1):
    nv = data.shape[0]
    ns = X0.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_val_cov(data[k, ...], W[s, k], X0[s, k], X1[s, k])

@myjit
def _push_vals_scale_cov_vec(data, W, X0, X1, scale):
    nv = data.shape[0]
    ns = X0.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_val_cov(data[k, ...], W[s, k] * f, X0[s, k], X1[s, k])


@myjit
def _push_data_cov_vec(data, data_in):
    nv = data.shape[0]
    for k in range(nv):
        _push_data_scale_cov(data[k, ...], data_in[k, ...], 1.0)


@myjit
def _push_datas_cov_vec(data, Datas):
    nv = data.shape[0]
    ns = Datas.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_data_scale_cov(data[k, ...], Datas[s, k, ...], 1.0)


@myjit
def _push_data_scale_cov_vec(data, data_in, scale):
    nv = data.shape[0]
    if scale > 0:
        for k in range(nv):
            _push_data_scale_cov(data[k, ...], data_in[k, ...], scale)


@myjit
def _push_datas_scale_cov_vec(data, Datas, scale):
    nv = data.shape[0]
    ns = Datas.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_data_scale_cov(data[k, ...], Datas[s, k, ...], f)


###############################################################################
# utilities
###############################################################################

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


def _axis_expand_broadcast(x, shape, axis, expand=True, roll=True, broadcast=True):
    """
    broadcast x to shape.  If x is 1d, and shape is n-d, but len(x) is same
    as shape[axis], broadcast x across all dimensions
    """

    x = np.array(x)

    # if array, and 1d with size same as shape[axis]
    # broadcast from here
    if expand:
        if x.ndim == 1 and x.ndim != len(shape) and len(x) == shape[axis]:
            # reshape for broadcasting
            reshape = [1] * (len(shape) - 1)
            reshape = _shape_insert_axis(reshape, axis, -1)
            x = x.reshape(*reshape)

    if broadcast and x.shape != shape:
        x = np.broadcast_to(x, shape)
    if roll and axis != 0:
        x = np.rollaxis(x, axis, 0)
    return x


###############################################################################
# raw moments
###############################################################################

@myjit
def _central_to_raw_moments(central, raw):
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        c = central[v]

        w = c[0]
        ave = c[1]

        r = raw[v]
        r[0] = w
        r[1] = ave

        for n in range(2, order+1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(0, n-1):
                tmp += c[n-i] * ave_i * _bfac(n, i)
                ave_i *= ave

            # last two
            # <dx> = 0 so skip i = n-1
            # i = n
            tmp += ave_i * ave
            r[n] = tmp


def central_to_raw_moments(central, axis=-1):
    """
    convert central moments to raw moments
    """

    central = np.array(central)
    raw = np.zeros_like(central)

    if axis != -1:
        central = np.moveaxis(central, axis, -1)
        raw = np.moveaxis(raw, axis, -1)


    shape = central.shape[:-1]
    if shape == ():
        reshape = (1,) + central.shape[-1:]
    else:
        reshape = (np.prod(shape),) + central.shape[-1:]

    central_r = central.reshape(reshape)
    raw_r = raw.reshape(reshape)

    _central_to_raw_moments(central_r, raw_r)

    return raw



#@myjit
def _raw_to_central_moments(raw, central):
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        w = r[0]
        ave = r[1]

        c[0] = w
        c[1] = ave

        for n in range(2, order+1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(0, n-1):
                tmp += r[n-i] * ave_i * _bfac(n, i)
                ave_i *= -ave

            # last two
            # right now, ave_i = (-ave)**(n-1)
            # i = n-1
            # ave * ave_i * n
            # i = n
            # 1 * (-ave) * ave_i 
            tmp += ave * ave_i * (n - 1)
            c[n] = tmp


def raw_to_central_moments(raw, axis=-1):
    """
    convert central moments to raw moments
    """
    raw = np.array(raw)
    central = np.zeros_like(raw)
    if axis != -1:
        raw = np.moveaxis(raw, axis, -1)
        central = np.moveaxis(central, axis, -1)

    shape = central.shape[:-1]
    if shape == ():
        reshape = (1,) + central.shape[-1:]
    else:
        reshape = (np.prod(shape),) + central.shape[-1:]

    central_r = central.reshape(reshape)
    raw_r = raw.reshape(reshape)

    _raw_to_central_moments(raw_r, central_r)

    return central



# comoments
@myjit
def _central_to_raw_comoments(central, raw):
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        w = c[0, 0]
        ave0 = c[1, 0]
        ave1 = c[0, 1]


        for n in range(0, order0+1):
            for m in range(0, order1+1):
                nm = n + m
                if nm <= 1:
                    r[n, m] = c[n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n+1):
                        ave_j = 1.0
                        for j in range(m+1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp +=  ave_i * ave_j
                            elif nm_ij == 1:
                                # <dx**0 * dy**1> = 0
                                pass
                            else:
                                tmp += c[n-i, m-j] * ave_i * ave_j * _bfac(n, i) * _bfac(m, j)
                            ave_j *= ave1
                        ave_i *= ave0
                    r[n, m] = tmp

def central_to_raw_comoments(central, axis=[-2, -1]):
    """
    convert central moments to raw moments
    """

    central = np.array(central)
    raw = np.zeros_like(central)

    axis = tuple(axis)
    assert len(axis) == 2

    if axis != (-2, -1):
        central = np.moveaxis(central, axis, (-2, -1))
        raw = np.moveaxis(raw, axis, (-2, -1))


    shape = central.shape[:-2]
    if shape == ():
        reshape = (1,) + central.shape[-2:]
    else:
        reshape = (np.prod(shape),) + central.shape[-2:]

    central_r = central.reshape(reshape)
    raw_r = raw.reshape(reshape)

    _central_to_raw_comoments(central_r, raw_r)

    return raw


@myjit
def _raw_to_central_comoments(raw, central):
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        w = r[0, 0]
        ave0 = r[1, 0]
        ave1 = r[0, 1]

        for n in range(0, order0+1):
            for m in range(0, order1+1):
                nm = n + m
                if nm <= 1:
                    c[n, m] = r[n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n+1):
                        ave_j = 1.0
                        for j in range(m+1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp +=  ave_i * ave_j
                            else:
                                tmp += r[n-i, m-j] * ave_i * ave_j * _bfac(n, i) * _bfac(m, j)
                            ave_j *= -ave1
                        ave_i *= -ave0
                    c[n, m] = tmp

def raw_to_central_comoments(raw, axis=[-2, -1]):
    """
    convert central moments to raw moments

    """

    raw = np.array(raw)
    central = np.zeros_like(raw)

    axis = tuple(axis)
    assert len(axis) == 2

    if axis != (-2, -1):
        central = np.moveaxis(central, axis, (-2, -1))
        raw = np.moveaxis(raw, axis, (-2, -1))

    shape = central.shape[:-2]
    if shape == ():
        reshape = (1,) + central.shape[-2:]
    else:
        reshape = (np.prod(shape),) + central.shape[-2:]

    central_r = central.reshape(reshape)
    raw_r = raw.reshape(reshape)

    _raw_to_central_comoments(raw_r, central_r)

    return central





###############################################################################
# resampling
###############################################################################

@njit
def _randsamp_freq_out(freq):
    nrep = freq.shape[0]
    ndat = freq.shape[1]
    for i in range(nrep):
        for j in range(ndat):
            index = np.random.randint(0, ndat)
            freq[i, index] += 1


@njit
def _randsamp_freq_index(index, freq):
    assert freq.shape == index.shape
    nrep, ndat = freq.shape
    for r in range(nrep):
        for d in range(ndat):
            idx = index[r, d]
            freq[r, idx] += 1


def randsamp_freq(size, nrep, index=None, transpose=False):
    """
    produce a random sample for bootstrapping

    Parameters
    ----------
    size : int
        data dimension size
    nrep : int
        number of replicates
    index : array-like, optional
        if passed, build frequency table based on this sampling
    transpose: bool
        see output


    Returns
    -------
    output : frequency table
        if not transpose: output.shape == (nrep, size)
        if tranpose, output.shae = (size, nrep)
    """

    freq = np.zeros((nrep, size), dtype=np.int64)
    if index is None:
        _randsamp_freq_out(freq)

    else:
        assert index.shape == (nrep, size)
        _randsamp_freq_index(index, freq)

    if transpose:
        freq = freq.T
    return freq




@lru_cache(10)
def _factory_resample(push_datas_scale, fastmath=True, parallel=False):
    @njit(fastmath=fastmath, parallel=parallel)
    def resample(data, freq, out):
        nrep = freq.shape[0]
        for irep in prange(nrep):
            push_datas_scale(out[irep, ...], data, freq[irep, ...])

    return resample


def resample_data(data, freq, moments, fastmath=True, parallel=False, out=None):
    """
    resample data according to frequency table
    """
    nrep, ndat = freq.shape
    assert len(data) == ndat

    if isinstance(moments, int):
        moments = (moments,)
    shape = data.shape[1 : -len(moments)]

    moments_shape = tuple(x + 1 for x in moments)
    assert data.shape[-len(moments) :] == moments_shape

    # reshape data
    out_shape = (nrep,) + data.shape[1:]
    if out is None:
        out = np.empty(out_shape, dtype=data.dtype)
    assert out.shape == out_shape

    # data_reshape = (ndat, nmeta) + moments_shape
    # out_reshape = (nrep, nmeta) + moments_shape
    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)

    data_reshape = (ndat,) + meta_reshape + moments_shape
    out_reshape = (nrep,) + meta_reshape + moments_shape

    datar = data.reshape(data_reshape)
    outr = out.reshape(out_reshape)

    if len(moments) == 1:
        if shape == ():
            pusher = _push_datas_scale
        else:
            pusher = _push_datas_scale_vec

    else:
        if shape == ():
            pusher = _push_datas_scale_cov
        else:
            pusher = _push_datas_scale_cov_vec

    resample = _factory_resample(pusher, fastmath=fastmath, parallel=parallel)

    outr[...] = 0.0

    resample(datar, freq, outr)
    return out


@lru_cache(10)
def _factory_resample_vals(push_vals_scale, fastmath=True, parallel=False):
    @njit(fastmath=fastmath, parallel=parallel)
    def resample(W, X, freq, out):
        nrep = freq.shape[0]
        for irep in prange(nrep):
            push_vals_scale(out[irep, ...], W, X, freq[irep, ...])

    return resample



@lru_cache(10)
def _factory_resample_vals_cov(push_vals_scale, fastmath=True, parallel=False):
    @njit(fastmath=fastmath, parallel=parallel)
    def resample(W, X, X1, freq, out):
        nrep = freq.shape[0]
        for irep in prange(nrep):
            push_vals_scale(out[irep, ...], W, X, X1, freq[irep, ...])

    return resample



def resample_vals(x, freq, moments, x1=None, axis=0, weights=None,
                  fastmath=True, parallel=False, out=None):
    """
    resample data according to frequency table
    """
    nrep, ndat = freq.shape
    x = np.array(x)
    assert x.shape[axis] == ndat

    # are we doing covariance?
    cov = x1 is not None

    if isinstance(moments, int):
        if not cov:
            moments = (moments,)
        else:
            moments = (moments,) * 2
    moments_shape = tuple(x + 1 for x in moments)

    if cov:
        assert len(moments) == 2
    else:
        assert len(moments) == 1

    # make sure things are right shape
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = _axis_expand_broadcast(weights, x.shape, axis, roll=False)

    if cov:
        x1 = _axis_expand_broadcast(x1, x.shape, axis, roll=False)


    if axis != 0:
        x = np.rollaxis(x, axis, 0)
        weights = np.rollaxis(x, axis, 0)
        if cov:
            x1 = np.rollaxis(x1, axis, 0)


    # output shapes
    shape = x.shape[1:]
    out_shape = (nrep,) + shape + moments_shape
    if out is None:
        out = np.empty(out_shape, dtype=x.dtype)
    assert out.shape == out_shape

    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)

    data_reshape = (ndat,) + meta_reshape
    out_reshape = (nrep,) + meta_reshape + moments_shape


    # reshape
    xr = x.reshape(data_reshape)
    wr = weights.reshape(data_reshape)
    outr = out.reshape(out_reshape)
    if cov:
        x1r = x1.reshape(data_reshape)


    # select push function
    outr[...] = 0.0

    if len(moments) == 1:
        if shape == ():
            pusher = _push_vals_scale
        else:
            pusher = _push_vals_scale_vec

        resample = _factory_resample_vals(pusher, fastmath=fastmath, parallel=parallel)

        resample(wr, xr, freq, out)
    else:
        if shape == ():
            pusher = _push_vals_scale_cov
        else:
            pusher = _push_vals_scale_cov_vec

        resample = _factory_resample_vals_cov(pusher, fastmath=fastmath, parallel=parallel)
        resample(wr, xr, x1r, freq, out)



    return out


###############################################################################
# Classes
###############################################################################

class StatsAccumBase(object):
    """
    Base class for moments accumulation
    """

    def __init__(self, moments, shape=None, dtype=None, data=None):
        """
        Parameters
        ----------
        moments : int or tuple
        shape : tuple, optional
        dtype : data type
        """

        if isinstance(moments, int):
            moments = (moments,)

        for m in moments:
            assert m > 0

        self.moments = moments

        if dtype is None:
            dtype = np.float

        if shape is None:
            shape = ()
        self.shape = shape
        self.dtype = dtype

        self._init_subclass()

        if data is None:
            data = np.zeros(self._data_shape, dtype=self.dtype)
        else:
            if data.shape != self._data_shape:
                raise ValueError(f'passed data shape {data.shape} must equal {self._data_shape}')

        self._data = data
        self._data_flat = self._data.reshape(self._data_flat_shape)

    def _init_subclass(self):
        pass

    def _moments_ndim(self):
        """number of dimensions for moments"""
        return len(self.moments)

    @gcached()
    def _moments_shape(self):
        """tuple of shape for moments"""
        return tuple(x + 1 for x in self.moments)

    @gcached()
    def _moments_shape_var(self):
        """shape of variance"""
        return tuple(x - 1 for x in self.moments)

    @gcached()
    def _shape_flat(self):
        if self.shape is ():
            return ()
        else:
            return (np.prod(self.shape),)

    @gcached()
    def _shape_var(self):
        return self.shape + self._moments_shape_var

    @gcached()
    def _shape_flat_var(self):
        return self._shape_flat + self._moments_shape_var

    @gcached()
    def _data_shape(self):
        """shape of data"""
        return self.shape + self._moments_shape

    @gcached()
    def _data_flat_shape(self):
        """shape of conformed data"""
        return self._shape_flat + self._moments_shape

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
        return np.ones(self.shape, dtype=self.dtype)

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
            w = w.reshape(_shape_insert_axis(w.shape, axis, 1))

        shape = _shape_insert_axis(self.shape, axis, nrec)
        w = _axis_expand_broadcast(w, shape, axis)
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
            x = _axis_expand_broadcast(x, shape, axis)
        else:
            if axis != 0:
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
        return v.reshape(v.shape[:1] + self._shape_flat_var)

    def check_data(self, data):
        data = np.array(data)
        assert data.shape == self._data_shape
        return data.reshape(self._data_flat_shape)

    def check_datas(self, datas, axis=0):
        datas = np.array(datas)
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)
        assert datas.shape[1:] == self._data_shape
        return datas.reshape(datas.shape[:1] + self._data_flat_shape)

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
            index = [...] + index
        return tuple(index)

    @gcached(prop=False)
    def _single_index(self, val):
        # index with things like
        # data[1,0 ,...]
        # data[0,1 ,...]

        # so build a total with indexer like
        # data[indexer]
        # with
        # index = ([1,0],[0,1],...)

        dims = len(self.moments)

        if dims == 1:
            index = [val]
        else:
            # this is a bit more complicated
            index = [[0] * dims for _ in range(dims)]
            for i in range(dims):
                index[i][i] = val

        if self.ndim > 0:
            index = [...] + index

        return tuple(index)

    def weight(self):
        return self._data[self._weight_index]

    def mean(self):
        return self._data[self._single_index(1)]

    def var(self):
        return self._data[self._single_index(2)]

    def std(self):
        return np.sqrt(self.var(mom=2))

    def cmom(self):
        return self._data

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


    @classmethod
    def from_data(cls, data, moments, shape=None, dtype=None):

        assert isinstance(moments, tuple)

        if shape is None:
            shape = data.shape[:-len(moments)]

        assert data.shape == shape + tuple(x+1 for x in moments)

        if dtype is None:
            dtype = data.dtype

        new = cls(shape=shape, dtype=dtype, moments=moments)
        new._data_flat[...] = new.check_data(data)
        return new

    @classmethod
    def from_datas(cls, datas, moments, shape=None, axis=0, dtype=None):
        """
        Data should have shape

        [:, moment, ...] (axis=0)

        [moment, axis, ...] (axis=1)

        [moment, ..., axis, ...] (axis=n)
        """

        assert isinstance(moments, tuple)

        datas = np.array(datas)
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)
        if shape is None:
            shape = datas.shape[1:-len(moments)]
        assert datas.shape[1:] == shape + tuple(x+1 for x in moments)

        if dtype is None:
            dtype = datas.dtype

        new = cls(shape=shape, dtype=dtype, moments=moments)
        new.push_datas(datas=datas, axis=0)
        return new


    def resample_and_reduce(self, freq, axis=0, fastmath=True, parallel=False):
        nrep, ndat = freq.shape

        data = self._data
        if axis != 0:
            data = np.rollaxis(data, axis, 0)
        assert data.shape[0] == ndat

        moments = self.moments
        meta_shape = data.shape[1:-len(moments)]

        new_shape = (nrep,) + meta_shape
        new = self.__class__(shape=new_shape, moments=moments, dtype=self.dtype)


        # reshape for calculation
        out = new._data
        if meta_shape == ():
            datar = data
            outr = out

        else:
            meta_reshape = (np.prod(meta_shape),)
            datar = data.reshape((ndat,) + meta_reshape + self._moments_shape)
            outr = out.reshape((nrep,) + meta_reshape + self._moments_shape)

        resampler = _factory_resample(self._push_datas_scale,
                                      fastmath=fastmath, parallel=parallel)

        # Don't need to re-zero, since just made the thing
        # outr[...] = 0.0
        resampler(datar, freq, outr)
        return new



class StatsAccum(StatsAccumBase):
    def _init_subclass(self):
        if self.shape == ():
            self._push_val = _push_val
            self._push_vals = _push_vals
            self._push_stat = _push_stat
            self._push_stats = _push_stats

            self._push_data = _push_data
            self._push_datas = _push_datas

            self._push_vals_scale = _push_vals_scale
            self._push_datas_scale = _push_datas_scale

        else:
            self._push_val = _push_val_vec
            self._push_vals = _push_vals_vec
            self._push_stat = _push_stat_vec
            self._push_stats = _push_stats_vec

            self._push_data = _push_data_vec
            self._push_datas = _push_datas_vec

            self._push_vals_scale = _push_vals_scale_vec
            self._push_datas_scale = _push_datas_scale_vec


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
    def from_vals(cls, x, w=None, moments=2, axis=0, dtype=None, shape=None):
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
    def from_data(cls, data, moments=None, shape=None, dtype=None):
        if moments is None:
            moments = data.shape[-1] - 1
        if isinstance(moments, int):
            moments = (moments,)
        return super(StatsAccum, cls).from_data(data=data, moments=moments,
                                                shape=shape, dtype=dtype)



    @classmethod
    def from_datas(cls, datas, moments=None, shape=None, axis=0, dtype=None):
        """
        Data should have shape
        [...,axis, ...moments] (axis!=-1)
        [...,moments, axis] (axis==-1)
        """

        if axis != 0:
            datas = np.array(datas)
            datas = np.rollaxis(datas, axis, 0)
        if moments is None:
            moments = data.shape[-1] - 1
        if isinstance(moments, int):
            moments = (moments,)
        return super(StatsAccum, cls).from_datas(datas=datas, moments=moments,
                                                shape=shape, dtype=dtype,
                                                axis=0)


    @classmethod
    def from_stat(cls, a, v=0.0, w=None, shape=None, moments=2, dtype=None):
        """
        object from single weight, average, variance/covariance
        """
        if shape is None:
            shape = a.shape
        new = cls(shape=shape, moments=moments, dtype=dtype)
        new.push_stat(w=w, a=a, v=v)
        return new

    @classmethod
    def from_stats(cls, a, v=0.0, w=None, axis=0, shape=None, moments=2, dtype=None):
        """
        object from several weights, averages, variances/covarainces along axis
        """

        # get shape
        if shape is None:
            shape = list(A.shape)
            shape.pop(axis)
            shape = tuple(shape)

        if dtype is None:
            a = np.array(a)
            dtype = a.dtype

        new = cls(shape=shape, dtype=A.dtype, moments=moments)
        new.push_stats(a=a, v=v, w=w, axis=axis)
        return new

    @classmethod
    def from_resample_vals(cls, x, freq, w=None, axis=0, dtype=None, shape=None, moments=2, **kwargs):

        nrep, ndat = freq.shape
        x = np.array(x)
        assert x.shape[axis] == ndat

        # weights
        if w is None:
            w = np.ones_like(x)
        else:
            w = _axis_expand_broadcast(w, x.shape, axis, roll=False)

        # moments
        if isinstance(moments, int):
            moments = (moments,)
        if dtype is None:
            dtype = x.dtype

        # shape
        if axis != 0:
            x = np.rollaxis(x, axis, 0)
            w = np.rollaxis(w, axis, 0)

        shape = x.shape[1:]
        new = cls(shape=(nrep,) + shape,
                  moments=moments,
                  dtype=dtype)

        # resample
        resampler = _factory_resample_vals(new._push_vals_scale, **kwargs)

        out = new._data

        if shape == ():
            xr = x
            wr = w
            outr = out
        else:
            meta_shape = (np.prod(shape),)
            reshape = (ndat,) + meta_shape
            xr = x.reshape(reshape)
            wr = w.reshape(reshape)

            # reshape out
            reshape_out = (nrep,) + meta_shape + new._moments_shape
            outr = out.reshape(reshape_out)

        resampler(wr, xr, freq, outr)
        return new


    def reduce(self, axis=0):
        """
        create new object reduced along axis
        """
        assert len(self.shape) > 0
        new = self.__class__.from_datas(self.data, axis=axis, moments=self.moments)
        return new



class StatsAccumCov(StatsAccumBase):
    def __init__(self, moments, shape=None, dtype=None):
        if isinstance(moments, int):
            moments = (moments,) * 2
        moments = tuple(moments)
        assert len(moments) == 2

        super(StatsAccumCov, self).__init__(moments=moments, shape=shape, dtype=dtype)

    def _init_subclass(self):
        if self.shape == ():
            self._push_val = _push_val_cov
            self._push_vals = _push_vals_cov
            self._push_data = _push_data_cov
            self._push_datas = _push_datas_cov
        else:
            self._push_val = _push_val_cov_vec
            self._push_vals = _push_vals_cov_vec
            self._push_data = _push_data_cov_vec
            self._push_datas = _push_datas_cov_vec


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
    def from_vals(
        cls, x0, x1, w=None, axis=0, shape=None, broadcast=False, moments=2, dtype=None
    ):

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
    def from_data(cls, data, moments=None, shape=None, dtype=None):

        if moments is None:
            moments = tuple(x - 1 for x in data.shape[-2:])
        if isinstance(moments, int):
            moments = (moments,) * 2

        return super(StatsAccumCov, cls).from_data(data=data, moments=moments,
                                                   shape=shape, dtype=dtype)

    @classmethod
    def from_datas(cls, datas, moments=None, shape=None, axis=0, dtype=None):
        """
        Data should have shape

        [...,axis, ...moments] (axis!=-1)
        [...,moments, axis] (axis==-1)
        """

        if axis != 0:
            datas = np.array(datas)
            datas = np.rollaxis(datas, axis, 0)

        if moments is None:
            moments = tuple(x - 1 for x in datas.shape[-2:])
        if isinstance(moments, int):
            moments = (moments,) * 2
        return super(StatsAccumCov, cls).from_datas(datas=datas, moments=moments,
                                                shape=shape, dtype=dtype,
                                                axis=0)




def weighted_var(x, w, axis=None, axis_sum=None, unbiased=True, **kwargs):
    """
    return the weighted variance over x with weight w

    v = sum(w)**2/(sum(w)**2 - sum(w**2)) * sum(w * (x-mu)**2 )

    Parameters
    ----------
    x : array
        values to consider

    w : array
        weights 

    axis : axis to average over

    axis_sum : axis to sum over for  w,w**2

    unbiased : bool (default True)
    if True, then apply unbiased norm (like ddof=1)
    else, apply biased norm (like ddof=0)


    **kwargs : arguments to np.average

    Returns
    -------
    Ave : weighted average
        shape x with `axis` removed

    Var : weighted variance 
        shape x with `axis` removed
    """

    if axis_sum is None:
        axis_sum = axis

    m1 = np.average(x, weights=w, axis=axis, **kwargs)
    m2 = np.average((x - m1) ** 2, weights=w, axis=axis, **kwargs)

    if unbiased:
        w1 = w.sum(axis=axis_sum)
        w2 = (w * w).sum(axis=axis_sum)
        m2 *= w1 * w1 / (w1 * w1 - w2)
    return m1, m2


class StatsArray(object):
    """
    Collection of Accumulator objects
    """

    def __init__(self, moments, shape=None, dtype=np.float, child=None):
        """
        moments : int or tuple
            moments to consider
        shape : tuple, optional
            shape of data
        dtype : numpy dtype
        child : Accumulator object, optional
            if not specified, choose child class based on moments.
            If moments is a scalar or length 1 tuple, child is StatsAccum object.
            If moments is a length 2 tuple, child is a StatsAccumCov object
        """

        if isinstance(moments, int):
            moments = (moments,)

        assert isinstance(moments, tuple)

        if child is None:
            if len(moments) == 1:
                child = StatsAccum
            else:
                child = StatsAccumCov

        self._child = child
        self._accum = child(shape=shape, moments=moments, dtype=dtype)
        self.zero()

    @property
    def accum(self):
        return self._accum

    @property
    def moments(self):
        return self._accum.moments

    @property
    def dtype(self):
        return self._accum.dtype

    @property
    def values(self):
        return self._values

    @values.setter
    @cached_clear()
    def values(self, values):
        if not isinstance(values, list):
            raise ValueError("trying to set list to non-list value")
        self._values = values

    @gcached()
    def data(self):
        return np.array(self._values)

    def new_like(self):
        return self.__class__(
            shape=self.accum.shape,
            child=self._child,
            dtype=self.dtype,
            moments=self.accum.moments,
        )

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        new = self.new_like()

        try:
            y = self._values[idx]
        except:
            y = list(self.data[idx])
        if not isinstance(y, list):
            y = [y]

        new._values = y
        return new

    def to_stats(self, indices=None):
        data = self.data
        if indices is None:
            new = self._child.from_data(data, moments=self.moments, dtype=self.dtype)

        else:
            shape = indices.shape + data.shape[1 : -len(self.moments)]
            new = self._child(shape=shape, moments=self.moments, dtype=self.dtype)
            np.take(data, indices, axis=0, out=new._data)
        return new

    def resample(self, indices, axis=0):
        """
        axis = axis of indices to average over
        """
        data = self.data.take(indices, axis=0)
        return self._child.from_datas(data, moments=self.moments, axis=axis)

    def resample_and_reduce(self, freq, **kwargs):
        """
        for bootstrapping
        """
        data = self.data
        data_new = resample_data(data, freq, moments=self.moments, **kwargs)
        return self.__class__.from_datas(
            data_new, shape=self._accum.shape, child=self._child, moments=self.moments
        )

    def zero(self):
        self.values = []
        self.accum.zero()

    @cached_clear()
    def append(self, data):
        self._values.append(data)

    @cached_clear()
    def push_stat(self, a, v=0.0, w=1.0):
        s = self._child.from_stat(a=a, v=v, w=w)
        self._values.append(s.data)

    @cached_clear()
    def push_stats(self, a, v=None, w=None):
        if v is None:
            v = np.zeros_like(a)
        if w is None:
            w = np.ones_like(a)
        for (ww, aa, vv) in zip(w, a, v):
            self.push_stat(a=aa, v=vv, w=ww)

    @cached_clear()
    def push_data(self, data):
        assert data.shape == self._accum._data_shape
        self._values.append(data)

    @cached_clear()
    def push_datas(self, datas, axis=0):
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)

        assert datas.shape[1:] == self._accum._data_shape
        for data in datas:
            self._values.append(data)

    @gcached()
    def _weight_index(self):
        return (slice(None),) + self._accum._weight_index

    @gcached(prop=False)
    def _single_index(self, val):
        return (slice(None),) + self._accum._single_index(val)

    def weight(self):
        return self.data[self._weight_index]

    def mean(self):
        return self.data[self._single_index(1)]

    def var(self):
        return self.data[self._single_index(2)]

    @property
    def data_last(self):
        return self._values[-1]

    def mean_last(self):
        return self.data_last[self.accum._single_index(1)]

    def var_last(self):
        return self.data_last[self.accum._single_index(2)]

    def std_last(self):
        return np.sqrt(self.var_last())

    def weight_last(self):
        return self.data_last[self.accum._weight_index]

    def get_stat(self, stat_name="mean", *args, **kwargs):
        return getattr(self, stat_name)(*args, **kwargs)

    @classmethod
    def from_datas(cls, datas, moments, axis=0, shape=None, child=None, dtype=np.float):
        if isinstance(moments, int):
            moments = (moments,)

        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)

        if shape is None:
            shape = datas.shape[1 : -len(moments)]

        new = cls(child=child, shape=shape, moments=moments, dtype=dtype)

        new.values = list(datas)
        return new

    @classmethod
    def from_accum(cls, accum, axis=None):
        """
        create StatsArray from StatsAccum object

        if accum object is a scalar object, or a vector object with no specified axis,
        then create a StatsArray object with accum as the sole elements

        if accum object is a vector object with a specified axis, create a StatsArray object
        with elements along this axis
        """

        ndim = len(accum.moments)
        if ndim == 0:
            axis = None

        if axis is None:
            # create single object
            new = StatsArray(
                moments=accum.moments,
                shape=accum.shape,
                child=type(accum),
                dtype=accum.dtype,
            )

        else:
            new = StatsArray.from_datas(
                accum.data,
                moments=accum.moments,
                axis=axis,
                shape=None,
                child=type(accum),
                dtype=accum.dtype,
            )

        return new

    @gcached()
    def cumdata(self):
        cumdata = np.zeros((len(self),) + self.accum._data_shape)
        self._accum.zero()
        for i, data in enumerate(self.values):
            self._accum.push_data(data)
            cumdata[i, ...] = self._accum.data
        return cumdata

    def cummean(self):
        return self.cumdata[self._single_index(1)]

    def cumvar(self):
        return self.cumdata[self._single_index(2)]

    def cumstd(self):
        return np.sqrt(self.cumvar())

    def cumweight(self):
        return self.cumdata[self._weight_index]

    @property
    def cumdata_last(self):
        return self.cumdata[-1, ...]

    def cummean_last(self):
        return self.cumdata_last[self.accum._single_index(1)]

    def cumvar_last(self):
        return self.cumdata_last[self.accum._single_index(2)]

    def cumstd_last(self):
        return np.sqrt(self.cumvar_last())

    def cumweight_last(self):
        return self.cumdata_last[self.accum._weight_index]

    @gcached()
    def stats_list(self):
        """
        list of stats objects
        """
        return [
            self._child.from_data(
                data=data,
                shape=self.accum.shape,
                moments=self.moments,
                dtype=self.dtype,
            )
            for data in self.values
        ]

    def block(self, block_size=None):
        """
        create a new stats array object from block averaging this one
        """
        new = self.new_like()
        new.values = self.blockdata(block_size)
        return new

    @gcached(prop=False)
    def blockdata(self, block_size):
        blockdata = []

        n = len(self)
        if block_size is None:
            block_size = n
        if block_size > n:
            block_size = n

        for lb in range(0, len(self), block_size):
            ub = lb + block_size
            if ub > n:
                break
            self._accum.zero()
            datas = self.data[lb:ub, ...]
            self._accum.push_datas(datas)
            blockdata.append(self._accum.data.copy())
        return blockdata

    def blockweight(self, block_size=None):
        return self.blockdata(block_size)[self._weight_index]

    def blockmean(self, block_size=None):
        return self.blockdata(block_size)[self._single_index(1)]

    def blockvar(self, block_size=None):
        return self.blockdata(block_size)[self._single_index(2)]

    def val_SEM(self, x, weighted, unbiased, norm):
        """
        find the standard error in the mean (SEM) of a value

        Parameters
        ----------
        x : array
            array (self.mean(), etc) to consider

        weighted : bool
            if True, use `weighted_var`
            if False, use `np.var`

        unbiased : bool
            if True, use unbiased stats (e.g., ddof=1 for np.var)
            if False, use biased stats (e.g., ddof=0 for np.var)

        norm : bool
            if True, scale var by x.shape[0], i.e., number of samples

        Returns
        -------
        sem : standard error in mean 
        """
        if weighted:
            v = weighted_var(x, w=self.weight(), axis=0, unbiased=unbiased)[-1]
        else:
            if unbiased:
                ddof = 1
            else:
                ddof = 0

            v = np.var(x, ddof=ddof, axis=0)
        if norm:
            v = v / x.shape[0]

        return np.sqrt(v)

    def mean_SEM(self, weighted=True, unbiased=True, norm=True):
        """self.val_SEM with x=self.mean()"""
        return self.val_SEM(self.mean(), weighted, unbiased, norm)

    def __repr__(self):
        return "nsample: {}".format(len(self))

    def to_xarray(
        self,
        rec_dim="rec",
        meta_dims=None,
        mom_dims=None,
        rec_coords=None,
        meta_coords=None,
        mom_coords=None,
        **kwargs
    ):
        import xarray as xr

        if meta_dims is None:
            meta_dims = ["dim_{}".format(i) for i in range(len(self.accum.shape))]
        else:
            meta_dims = list(meta_dims)
        assert len(meta_dims) == len(self.accum.shape)

        if mom_dims is None:
            mom_dims = ["mom_{}".format(i) for i in range(len(self.moments))]

        if isinstance(mom_dims, str):
            mom_dims = [mom_dims]

        assert len(mom_dims) == len(self.accum.moments)

        dims = [rec_dim] + meta_dims + mom_dims

        coords = {}
        coords.update(rec_coords or {})
        coords.update(meta_coords or {})
        coords.update(mom_coords or {})
        return xr.DataArray(self.data, dims=dims, coords=coords, **kwargs)

    @classmethod
    def from_xarray(
        cls,
        data,
        rec_dim="rec",
        meta_dims=None,
        mom_dims=None,
        shape=None,
        moments=None,
        child=None,
        dtype=None,
    ):
        import xarray as xr

        if mom_dims is None:
            # try to infer moment dimensions
            mom_dims = []
            for k in sorted(data.dims):
                if "mom_" in k:
                    mom_dims.append(k)

        if isinstance(mom_dims, str):
            mom_dims = [mom_dims]

        if moments is None:
            # infer moments
            moments = []
            for k in mom_dims:
                moments.append(len(data[k]) - 1)
            moments = tuple(moments)

        assert len(moments) == len(mom_dims)

        order = [rec_dim]
        if meta_dims is not None:
            if isinstance(meta_dims, str):
                meta_dims = [meta_dims]
            assert data.ndim == 1 + len(mom_dims) + len(meta_dims)
            order += meta_dims
        else:
            order += [...]

        order += mom_dims

        data = data.transpose(*order)

        return cls.from_datas(
            datas=data, moments=moments, axis=0, shape=shape, child=child, dtype=dtype
        )



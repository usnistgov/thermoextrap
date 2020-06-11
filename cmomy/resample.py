"""
Routines to perform resampling
"""
from __future__ import absolute_import
from functools import lru_cache

import numpy as np
from numba import njit, prange


from .utils import _axis_expand_broadcast
from .pushers import factory_pusher_datas_scale, factory_pusher_vals_scale




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
def _randsamp_freq_indices(indices, freq):
    assert freq.shape == indices.shape
    nrep, ndat = freq.shape
    for r in range(nrep):
        for d in range(ndat):
            idx = indices[r, d]
            freq[r, idx] += 1


def randsamp_freq(nrep=None, size=None, indices=None, transpose=False):
    """
    produce a random sample for bootstrapping

    Parameters
    ----------
    size : int
        data dimension size
    nrep : int
        number of replicates
    indices : array-like, optional
        if passed, build frequency table based on this sampling.
        shape = (nrep, ndat)
    transpose: bool
        see output


    Returns
    -------
    output : frequency table
        if not transpose: output.shape == (nrep, size)
        if tranpose, output.shae = (size, nrep)
    """
    if indices is not None:
        indices = np.array(indices)
        if nrep is None:
            nrep = indices.shape[0]
        if size is None:
            size = indices.shape[1]
        if indices.shape != (nrep, size):
            raise ValueError('passed indices shape {indices.shape} doesn not match {(nrep, size)}')

    elif nrep is None or size is None:
        raise ValueError('must specify nrep and size or indices')

    freq = np.zeros((nrep, size), dtype=np.int64)
    if indices is None:
        _randsamp_freq_out(freq)
    else:
        _randsamp_freq_indices(indices, freq)

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


def resample_data(data, freq, moments, fastmath=True, axis=0, parallel=False, pusher=None, out=None):
    """
    resample data according to frequency table

    Parameters
    ----------
    data : array-like
        central moments array to be resampled
    freq : array-like
        frequency array with shape (nrep, data.shape[axis])
    moments : int or array-like
        if int or length 1, then data contains central moments.
        if length is 2, then data contains central comoments
    axis : int, default=0
        axis to reduce along
    pusher : callable, optiona
        jitted function to perform scaled reduction
    out : optional output

    Returns
    -------
    output : array
        output shape is `(nrep,) + shape + moments`, where shape is
        the shape of data less axis, and moments is the shape of the resulting moments.
    """

    if isinstance(moments, int):
        moments = (moments,)

    # check inputs
    data = np.array(data)
    freq = np.array(freq)

    nrep, ndat = freq.shape
    ndim = data.ndim - len(moments)
    if axis < 0:
        axis += ndim
    assert 0 <= axis < ndim

    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    shape = data.shape[1 : -len(moments)]
    moments_shape = tuple(x + 1 for x in moments)

    assert data.shape == (ndat,) + shape + moments_shape

    # output
    out_shape = (nrep,) + data.shape[1:]
    if out is None:
        out = np.empty(out_shape, dtype=data.dtype)
    assert out.shape == out_shape

    # resahpe
    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)

    data_reshape = (ndat,) + meta_reshape + moments_shape
    out_reshape = (nrep,) + meta_reshape + moments_shape

    datar = data.reshape(data_reshape)
    outr = out.reshape(out_reshape)

    # get resampler
    if pusher is None:
        pusher = factory_pusher_datas_scale(cov=len(moments) > 1,
                                            vec=len(shape) > 0)

        # if len(moments) == 1:
        #     if shape == ():
        #         pusher = _push_datas_scale
        #     else:
        #         pusher = _push_datas_scale_vec

        # else:
        #     if shape == ():
        #         pusher = _push_datas_scale_cov
        #     else:
        #         pusher = _push_datas_scale_cov_vec

    resample = _factory_resample(pusher, fastmath=fastmath, parallel=parallel)

    # resample
    outr.fill(0.0)
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



def resample_vals(x, freq, moments, x1=None, axis=0, weights=None, broadcast=False,
                  fastmath=True, parallel=False, pusher=None, out=None):
    """
    resample data according to frequency table
    """

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

    # check input data
    freq = np.array(freq)
    x = np.array(x)
    nrep, ndat = freq.shape

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = _axis_expand_broadcast(weights, x.shape, axis, roll=False)

    if cov:
        x1 = _axis_expand_broadcast(x1, x.shape, axis, roll=False, broadcast=broadcast)

    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        weights = np.moveaxis(weights, axis, 0)
        if cov:
            x1 = np.moveaxis(x1, axis, 0)

    assert len(x) == ndat

    # output
    shape = x.shape[1:]
    out_shape = (nrep,) + shape + moments_shape
    if out is None:
        out = np.empty(out_shape, dtype=x.dtype)
    assert out.shape == out_shape

    # reshape
    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)
    data_reshape = (ndat,) + meta_reshape
    out_reshape = (nrep,) + meta_reshape + moments_shape

    xr = x.reshape(data_reshape)
    wr = weights.reshape(data_reshape)
    outr = out.reshape(out_reshape)
    if cov:
        x1r = x1.reshape(data_reshape)


    # select push function
    if pusher is None:
        pusher = factory_pusher_vals_scale(cov=cov,
                                           vec=len(shape) > 0)
        # if len(moments) == 1:
        #     if shape == ():
        #         pusher = _push_vals_scale
        #     else:
        #         pusher = _push_vals_scale_vec

        # else:
        #     if shape == ():
        #         pusher = _push_vals_scale_cov
        #     else:
        #         pusher = _push_vals_scale_cov_vec

    # sample
    outr[...] = 0.0
    if cov:
        resample = _factory_resample_vals_cov(pusher, fastmath=fastmath, parallel=parallel)
        resample(wr, xr, x1r, freq, outr)

    else:
        resample = _factory_resample_vals(pusher, fastmath=fastmath, parallel=parallel)
        resample(wr, xr, freq, outr)


    return out

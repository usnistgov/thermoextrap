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


def randsamp_freq(nrep=None, size=None, indices=None, transpose=False, freq=None, check=False):
    """
    produce a random sample for bootstrapping

    Parameters
    ----------
    size : int, optional
        data dimension size
    freq : array-like, shape=(nrep, size), optional
        if passed, use this frequency array.
        overides size
    indices : array-like, shape=(nrep, size), optional
        if passed and `freq` is `None`, construct frequency
        array from this indices array

    nrep : int, optional
        if `freq` and `indices` are `None`, construct
        sample with this number of repititions
    indices : array-like, optional
        if passed, build frequency table based on this sampling.
        shape = (nrep, ndat)
    freq : array-like, optional
        if passed, use this frequency array
    transpose : bool
        see output
    check : bool, default=False
        if `check` is `True`, then check `freq` and `indices` against `size` and `nrep`

    Returns
    -------
    output : frequency table
        if not transpose: output.shape == (nrep, size)
        if tranpose, output.shae = (size, nrep)

    """

    def _array_check(x, name=''):
        x = np.asarray(x, dtype=np.int64)
        if check:
            if nrep is not None:
                if x.shape[0] != nrep:
                    raise ValueError('{} has wrong nrep'.format(name))

            assert size is not None
            if x.shape[1] != size:
                raise ValueError('{} has wrong size'.format(name))
        return x



    if freq is not None:
        freq = _array_check(freq, 'freq')

    elif indices is not None:
        indices = _array_check(indices, 'indices')
        freq = np.zeros(indices.shape, dtype=np.int64)
        _randsamp_freq_indices(indices, freq)

    elif nrep is not None and size is not None:
        freq = np.zeros((nrep, size), dtype=np.int64)
        _randsamp_freq_out(freq)

    else:
        raise ValueError('must specify freq, indices, or nrep and size')


    # if indices is not None:
    #     indices = np.asarray(indices, dtype=np.int64)
    #     if nrep is None:
    #         nrep = indices.shape[0]
    #     if size is None:
    #         size = indices.shape[1]
    #     if indices.shape != (nrep, size):
    #         raise ValueError('passed indices shape {indices.shape} doesn not match {(nrep, size)}')

    # elif nrep is None or size is None:
    #     raise ValueError('must specify nrep and size or indices')

    # freq = np.zeros((nrep, size), dtype=np.int64)
    # if indices is None:
    #     _randsamp_freq_out(freq)
    # else:
    #     _randsamp_freq_indices(indices, freq)

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


def resample_data(data, freq, mom, axis=0,
                  dtype=None, order=None,
                  fastmath=True, parallel=False,
                  pusher=None, out=None):
    """
    resample data according to frequency table

    Parameters
    ----------
    data : array-like
        central mom array to be resampled
    freq : array-like
        frequency array with shape (nrep, data.shape[axis])
    mom : int or array-like
        if int or length 1, then data contains central mom.
        if length is 2, then data contains central comom
    axis : int, default=0
        axis to reduce along
    pusher : callable, optiona
        jitted function to perform scaled reduction
    fastmath, parallel : bool
        options for jitting pusher
    dtype, order : options to np.asarray
    out : optional output

    Returns
    -------
    output : array
        output shape is `(nrep,) + shape + mom`, where shape is
        the shape of data less axis, and mom is the shape of the resulting mom.
    """

    if isinstance(mom, int):
        mom = (mom,)

    # check inputs
    data = np.asarray(data, dtype=dtype, order=order)
    freq = np.asarray(freq, dtype=np.int64, order=order)

    if dtype is None:
        dtype = data.dtype

    nrep, ndat = freq.shape
    ndim = data.ndim - len(mom)
    if axis < 0:
        axis += ndim
    assert 0 <= axis < ndim

    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    shape = data.shape[1 : -len(mom)]
    mom_shape = tuple(x + 1 for x in mom)

    assert data.shape == (ndat,) + shape + mom_shape

    # output
    out_shape = (nrep,) + data.shape[1:]
    if out is None:
        out = np.empty(out_shape, dtype=dtype)
    assert out.shape == out_shape

    # resahpe
    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)

    data_reshape = (ndat,) + meta_reshape + mom_shape
    out_reshape = (nrep,) + meta_reshape + mom_shape

    datar = data.reshape(data_reshape)
    outr = out.reshape(out_reshape)

    # get resampler
    if pusher is None:
        pusher = factory_pusher_datas_scale(cov=len(mom) > 1,
                                            vec=len(shape) > 0)

        # if len(mom) == 1:
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
    def resample(W, X, Y, freq, out):
        nrep = freq.shape[0]
        for irep in prange(nrep):
            push_vals_scale(out[irep, ...], W, X, Y, freq[irep, ...])
    return resample



def resample_vals(x, freq, mom,
                  axis=0, w=None,
                  ndim_mom=None,
                  broadcast=False,
                  dtype=None, order=None,
                  fastmath=True, parallel=False,
                  pusher=None, out=None):
    """
    resample data according to frequency table
    """


    if isinstance(mom, int):
        mom = (mom,) * 1
    assert isinstance(mom, tuple)

    if ndim_mom is None:
        ndim_mom = len(mom)
    assert len(mom) == ndim_mom
    mom_shape = tuple(x + 1 for x in mom)

    if ndim_mom == 1:
        y = None
    elif ndim_mom == 2:
        x, y = x
    else:
        raise ValueError('only ndim_mom <= 2 supported')

    cov = y is not None

    # check input data
    freq = np.asarray(freq, dtype=np.int64)
    nrep, ndat = freq.shape

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype
    if w is None:
        w = np.ones_like(x)
    else:
        w = _axis_expand_broadcast(w, x.shape, axis,
                                   roll=False,
                                   dtype=dtype, order=order)

    if cov:
        y = _axis_expand_broadcast(y, x.shape, axis, roll=False,
                                   broadcast=broadcast,
                                   dtype=dtype, order=order)

    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)
        if cov:
            y = np.moveaxis(y, axis, 0)

    assert len(x) == ndat

    # output
    shape = x.shape[1:]
    out_shape = (nrep,) + shape + mom_shape
    if out is None:
        out = np.empty(out_shape, dtype=dtype)
    assert out.shape == out_shape

    # reshape
    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)
    data_reshape = (ndat,) + meta_reshape
    out_reshape = (nrep,) + meta_reshape + mom_shape

    xr = x.reshape(data_reshape)
    wr = w.reshape(data_reshape)
    outr = out.reshape(out_reshape)
    if cov:
        yr = y.reshape(data_reshape)


    # select push function
    if pusher is None:
        pusher = factory_pusher_vals_scale(cov=cov,
                                           vec=len(shape) > 0)
    # sample
    outr[...] = 0.0
    if cov:
        resample = _factory_resample_vals_cov(pusher, fastmath=fastmath, parallel=parallel)
        resample(wr, xr, yr, freq, outr)

    else:
        resample = _factory_resample_vals(pusher, fastmath=fastmath, parallel=parallel)
        resample(wr, xr, freq, outr)

    return out

"""Routine to perform resampling."""
from __future__ import annotations

from typing import Hashable, Literal, Sequence, Tuple, cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from ._resample import factory_resample_data, factory_resample_vals
from ._typing import ArrayOrder, Moments, XvalStrict
from .utils import _axis_expand_broadcast, myjit

###############################################################################
# resampling
###############################################################################


@myjit
def numba_random_seed(seed):
    """Set the random seed for numba functions"""
    np.random.seed(seed)


@myjit
def _randsamp_freq_out(freq):
    nrep = freq.shape[0]
    ndat = freq.shape[1]
    for i in range(nrep):
        for j in range(ndat):
            index = np.random.randint(0, ndat)
            freq[i, index] += 1


@myjit
def _randsamp_freq_indices(indices, freq):
    assert freq.shape == indices.shape
    nrep, ndat = freq.shape
    for r in range(nrep):
        for d in range(ndat):
            idx = indices[r, d]
            freq[r, idx] += 1


def freq_to_indices(freq):
    """Convert a frequency array to indices array."""

    indices_all = []

    for f in freq:

        indices = np.concatenate([np.repeat(i, count) for i, count in enumerate(f)])

        indices_all.append(indices)

    return np.array(indices_all)


def indices_to_freq(indices):
    """Convert indices to frequency array."""
    freq = np.zeros_like(indices)
    _randsamp_freq_indices(indices, freq)

    return freq


def randsamp_freq(
    nrep: int | None = None,
    size: int | None = None,
    indices: ArrayLike | None = None,
    transpose: bool = False,
    freq: ArrayLike | None = None,
    check: bool = False,
) -> np.ndarray:
    """Produce a random sample for bootstrapping.

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

    def _array_check(x: ArrayLike, name="") -> np.ndarray:
        x = np.asarray(x, dtype=np.int64)
        if check:
            if nrep is not None:
                if x.shape[0] != nrep:
                    raise ValueError("{} has wrong nrep".format(name))

            assert size is not None
            if x.shape[1] != size:
                raise ValueError("{} has wrong size".format(name))
        return x

    if freq is not None:
        freq = _array_check(freq, "freq")

    elif indices is not None:
        indices = _array_check(indices, "indices")
        freq = np.zeros(indices.shape, dtype=np.int64)
        _randsamp_freq_indices(indices, freq)

    elif nrep is not None and size is not None:
        freq = np.zeros((nrep, size), dtype=np.int64)
        _randsamp_freq_out(freq)

    else:
        raise ValueError("must specify freq, indices, or nrep and size")

    if transpose:
        freq = freq.T
    return cast(np.ndarray, freq)


def resample_data(
    data: ArrayLike,
    freq: ArrayLike,
    mom: Moments,
    axis: int = 0,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Resample data according to frequency table.

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
    parallel : bool
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

    shape: Tuple[int, ...] = data.shape[1 : -len(mom)]
    mom_shape = tuple(x + 1 for x in mom)

    assert data.shape == (ndat,) + shape + mom_shape

    # output
    out_shape = (nrep,) + data.shape[1:]
    if out is None:
        out = np.empty(out_shape, dtype=dtype)
    else:
        assert out.shape == out_shape
        # make sure out is in correct order
        out = np.asarray(out, dtype=dtype, order="C")

    meta_reshape: Tuple[()] | Tuple[int]
    if shape == ():
        meta_reshape = ()
    else:
        meta_reshape = (np.prod(shape),)

    data_reshape = (ndat,) + meta_reshape + mom_shape
    out_reshape = (nrep,) + meta_reshape + mom_shape

    datar = data.reshape(data_reshape)
    outr = out.reshape(out_reshape)

    resample = factory_resample_data(
        cov=len(mom) > 1, vec=len(shape) > 0, parallel=parallel
    )

    outr.fill(0.0)
    resample(datar, freq, outr)

    return outr.reshape(out.shape)


def resample_vals(
    x: XvalStrict,
    freq: np.ndarray,
    mom: Moments,
    axis: int = 0,
    w: np.ndarray | None = None,
    mom_ndim: int | None = None,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Resample data according to frequency table."""

    if isinstance(mom, int):
        mom = (mom,) * 1
    assert isinstance(mom, tuple)

    if mom_ndim is None:
        mom_ndim = len(mom)
    assert len(mom) == mom_ndim
    mom_shape = tuple(x + 1 for x in mom)

    if mom_ndim == 1:
        y = None
    elif mom_ndim == 2:
        x, y = x
    else:
        raise ValueError("only mom_ndim <= 2 supported")

    # check input data
    freq = np.asarray(freq, dtype=np.int64)
    nrep, ndat = freq.shape

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype
    if w is None:
        w = np.ones_like(x)
    else:
        w = _axis_expand_broadcast(
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    if y is not None:
        y = _axis_expand_broadcast(
            y, x.shape, axis, roll=False, broadcast=broadcast, dtype=dtype, order=order
        )

    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)
        if y is not None:
            y = np.moveaxis(y, axis, 0)

    assert len(x) == ndat

    # output
    shape = x.shape[1:]
    out_shape = (nrep,) + shape + mom_shape
    if out is None:
        out = np.empty(out_shape, dtype=dtype)
    else:
        assert out.shape == out_shape
        out = np.asarray(out, dtype=dtype, order="C")

    # reshape

    if shape == ():
        meta_reshape = cast(Tuple[int, ...], ())
    else:
        meta_reshape = cast(Tuple[int, ...], (np.prod(shape),))
    data_reshape = (ndat,) + meta_reshape
    out_reshape = (nrep,) + meta_reshape + mom_shape

    xr = x.reshape(data_reshape)
    wr = w.reshape(data_reshape)
    outr = out.reshape(out_reshape)
    # if cov:
    #     yr = y.reshape(data_reshape)

    resample = factory_resample_vals(
        cov=y is not None, vec=len(shape) > 0, parallel=parallel
    )
    outr.fill(0.0)
    if y is not None:
        yr = y.reshape(data_reshape)
        resample(wr, xr, yr, freq, outr)
    else:
        resample(wr, xr, freq, outr)

    return outr.reshape(out.shape)


def bootstrap_confidence_interval(
    distribution: np.ndarray,
    stats_val: np.ndarray | Literal["percentile", "mean", "median"] = "mean",
    axis: int = 0,
    alpha: float = 0.05,
    style: Literal[None, "delta", "pm"] = None,
    **kws,
) -> np.ndarray:
    """Calculate the error bounds.

    Parameters
    ----------
    distribution : array-like
        distribution of values to consider
    stats_val : array-like, {None, 'mean','median'}
        * array: perform pivotal error bounds (correct) with this as `value`.
        * percentile: percentiles, with value as median
        * mean: pivotal error bounds with mean as value
        * median: pivotal error bounds with median as value
    axis : int, default=0
        axis to analyze along
    alpha : float
        alpha value for confidence interval.
        Percent confidence = `100 * (1 - alpha)`
    kws : dict
        extra arguments to `numpy.percentile`
    style : {None, 'delta', 'pm'}
        controls style of output

    Returns
    -------
    out : array
        fist dimension will be statistics.  Other dimensions
        have shape of input less axis reduced over.
        Depending on `style` first dimension will be
        (note val is either stats_val or median):

    * None: [val, low, high]
    * delta:  [val, val-low, high - val]
    * pm : [val, (high - low) / 2]
    """

    if stats_val is None:
        p_low = 100 * (alpha / 2.0)
        p_mid = 50
        p_high = 100 - p_low
        val, low, high = np.percentile(
            a=distribution, q=[p_mid, p_low, p_high], axis=axis, **kws
        )

    else:
        if isinstance(stats_val, str):
            if stats_val == "mean":
                sv = np.mean(distribution, axis=axis)
            elif stats_val == "median":
                sv = np.median(distribution, axis=axis)
            else:
                raise ValueError("stats val should be None, mean, median, or an array")

        else:
            sv = stats_val

        q_high = 100 * (alpha / 2.0)
        q_low = 100 - q_high
        val = sv
        low = 2 * sv - np.percentile(a=distribution, q=q_low, axis=axis, **kws)
        high = 2 * sv - np.percentile(a=distribution, q=q_high, axis=axis, **kws)

    if style is None:
        out = np.array([val, low, high])
    elif style == "delta":
        out = np.array([val, val - low, high - val])
    elif style == "pm":
        out = np.array([val, (high - low) / 2.0])
    return out


def xbootstrap_confidence_interval(
    x: xr.DataArray,
    stats_val: np.ndarray | Literal["percentile", "mean", "median"] = "mean",
    axis: int = 0,
    dim: Hashable | None = None,
    alpha: float = 0.05,
    style: Literal[None, "delta", "pm"] = None,
    bootstrap_dim: str = "bootstrap",
    bootstrap_coords: Sequence | None = None,
    **kws,
):
    """Bootstrap xarray object.

    Parameters
    ----------
    dim : str
        if passed, use reduce along this dimension
    bootstrap_dim : str, default='bootstrap'
        name of new dimension.  If `bootstrap_dim` conflicts, then
        `new_name = dim + new_name`
    bootstrap_coords : array-like or str
        coords of new dimension.
        If `None`, use default names
        If string, use this for the 'values' name
    """

    if dim is not None:
        axis = x.get_axis_num(dim)  # type: ignore
    else:
        dim = x.dims[axis]  # type: ignore

    template = x.isel(indexers={dim: 0})

    if bootstrap_dim is None:
        bootstrap_dim = "bootstrap"

    if bootstrap_dim in template.dims:
        bootstrap_dim = "{}_{}".format(dim, bootstrap_dim)
    dims = (bootstrap_dim,) + template.dims

    if bootstrap_coords is None:
        if isinstance(stats_val, str):
            bootstrap_coords = stats_val
        else:
            bootstrap_coords = "stats_val"

    if isinstance(bootstrap_coords, str):
        if style is None:
            bootstrap_coords = [bootstrap_coords, "low", "high"]
        elif style == "delta":
            bootstrap_coords = [bootstrap_coords, "err_low", "err_high"]
        elif style == "pm":
            bootstrap_coords = [bootstrap_coords, "err"]

    if not isinstance(stats_val, str):
        stats_val = np.array(stats_val)

    out = bootstrap_confidence_interval(
        x.values, stats_val=stats_val, axis=axis, alpha=alpha, style=style, **kws
    )

    out_xr = xr.DataArray(
        out,
        dims=dims,
        coords=template.coords,
        attrs=template.attrs,
        name=template.name,
        # indexes=template.indexes,
    )
    if bootstrap_coords is not None:
        out_xr.coords[bootstrap_dim] = bootstrap_coords
    return out_xr

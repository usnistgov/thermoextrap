"""
A set of routines to stack data for gpflow analysis
"""

import numpy as np
import pandas as pd
import xarray as xr


def stack_dataarray(
    da, xdims, ydims=None, xname="xstack", yname="ystack", vdim=None, policy="infer"
):
    """
    Given an xarray.DataArray, stack for gpflow analysis

    Parameters
    ----------
    da : xarray.DataArray
        input DataArray
    xdims : str or sequence of str
        dimensions stacked under `xname`
    ydims : str or sequance of str, optional
        dimensions stacked under `yname`.
        Defaults to all dimenions not specified by `xdims` or `vdim`.
    vdim : str or tuple of strings, optional
        Use this to indicate a dimension that contains variables (mean and variance).
        This dimenension is moved to the last position.
    policy : {'infer', 'raise'}
        policy if coordinates not available

    Returns
    -------
    da_stacked : xarray.DataArray
        stacked DataArray
    """
    dims = da.dims
    for name in [xname, yname]:
        if name in dims:
            raise ValueError("{} conficts with existing {}".format(xname, dims))

    if isinstance(xdims, str):
        xdims = (xdims,)

    stacker = {xname: xdims}
    if isinstance(ydims, str):
        ydims = (ydims,)
    elif ydims is None:
        # could use set, but would rather keep order
        ydims = [x for x in dims if x not in xdims]
        if vdim is not None:
            ydims.remove(vdim)
        ydims = tuple(ydims)

    if len(ydims) > 0:
        stacker[yname] = ydims

    if policy == "raise":
        for dim in xdims:
            if dim not in da.coords:
                raise ValueError("da.coords[{}] not set".format(dim))

    out = da.stack(**stacker)

    if vdim is not None:
        if isinstance(vdim, str):
            vdim = (vdim,)
        out = out.transpose(..., *vdim)

    return out


def wrap_like_dataarray(x, da):
    """
    wrap an array x with properties of da
    """
    return xr.DataArray(
        x,
        dims=da.dims,
        coords=da.coords,
        indexes=da.indexes,
        attrs=da.attrs,
        name=da.name,
    )


def multiindex_to_array(idx):
    """
    turn xarray multiindex to numpy arrayj
    """
    return np.array(list(idx.values))


def to_mean_var(da, dim, concat_dim=None, concat_kws=None, **kws):
    """
    for a dataarray apply mean/variance along a dimension

    Parameters
    ----------
    da : DataArray
        DataArray to be analyzed
    dim : str
        dimension to reduce along
    kws : dict
        Reduction arguments to `xarray.DataArray.mean` and `xarray.DataArray.var`
    concat_dim : str or DataArray or pandas.Index, optional
        dimension parameter to `xarray.concat`.
        Defaults to creating a dimension `variable` with names `['mean', 'var']`
    concat_kws : dict
        key-word arguments to `xarray.concat`
    """
    if concat_kws is None:
        concat_kws = {}

    if concat_dim is None:
        concat_dim = pd.Index(["mean", "var"], name="variable")

    return xr.concat(
        (da.mean(dim, **kws), da.var(dim, **kws)), dim=concat_dim, **concat_kws
    )


def states_xcoefs_concat(states, dim=None, **kws):
    """
    concatanate [s.xcoefs(norm=False) for s in states]

    Parameters
    ----------
    states : StateCollection
        states to consider
    dim : str or DataArray or pandas.Index, optional
        dimension to concat along.
        Defaults to `pandas.Index(states.alpha0, name=states.alpha_name)`
    kws : dict
        extra arguments to xarray.concat

    Returns
    -------
    out : DataArray
    """
    if dim is None:
        dim = pd.Index(states.alpha0, name=states.alpha_name)

    return xr.concat((s.xcoefs(norm=False) for s in states), dim=dim, **kws)

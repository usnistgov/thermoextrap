"""Utilities for working with :mod:`xarray`."""
import numpy as np
import xarray as xr


###############################################################################
# Structure(s) to handle data
###############################################################################
def _check_xr(x, dims, strict=True, name=None):
    if isinstance(x, xr.Dataset):
        # don't do anything to datasets
        pass

    elif not isinstance(x, xr.DataArray):
        x = np.array(x)
        if isinstance(dims, dict):
            dims = dims[x.ndim]
        x = xr.DataArray(x, dims=dims, name=name)
    elif strict:
        if isinstance(dims, dict):
            dims = dims[x.ndim]
        for d in dims:
            if d not in x.dims:
                raise ValueError(f"{d} not in dims")
    return x


def xrwrap_uv(uv, dims=None, rec_dim="rec", rep_dim="rep", name="u", stict=True):
    """
    Wrap uv (energy values) array.

    assumes uv[rec_dim], or uv[rep_dim, rec_dim] where rec_dim is recorded (or time) and rep_dim is replicate
    """
    if dims is None:
        dims = {1: [rec_dim], 2: [rep_dim, rec_dim]}
    return _check_xr(uv, dims, strict=stict, name=name)


def xrwrap_xv(
    xv,
    dims=None,
    rec_dim="rec",
    rep_dim="rep",
    deriv_dim=None,
    val_dims="val",
    name="x",
    strict=None,
):
    """
    Wraps xv (x values) array.

    if deriv_dim is None, assumes xv[rec_dim], xv[rec_dim, vals], xv[rep_dim, rec_dim, val_dims]
    if deriv_dim is not None, assumes xv[rec_dim, deriv_dim], xv[rec_dim,deriv_dim, val_dims], xv[rep_dim,rec_dim,deriv_dim,val_dims]
    """

    if isinstance(val_dims, str):
        val_dims = [val_dims]
    elif not isinstance(val_dims, list):
        val_dims = list(val_dims)

    if deriv_dim is None:
        if strict is None:
            strict = False
        if dims is None:
            rec_val = [rec_dim] + val_dims
            rep_val = [rep_dim, rec_dim] + val_dims

            dims = {
                1: [rec_dim],
                len(rec_val): [rec_dim] + val_dims,
                len(rep_val): [rep_dim, rec_dim] + val_dims,
            }

    else:
        if strict is None:
            strict = False
        if dims is None:
            rec_val = [rec_dim, deriv_dim] + val_dims
            rep_val = [rep_dim, rec_dim, deriv_dim] + val_dims
            dims = {
                2: [rec_dim, deriv_dim],
                len(rec_val): [rec_dim, deriv_dim] + val_dims,
                len(rep_val): [rep_dim, rec_dim, deriv_dim] + [val_dims],
            }
    return _check_xr(xv, dims=dims, strict=strict, name=name)


def xrwrap_alpha(alpha, dims=None, stict=False, name="alpha"):
    """Wrap alpha values."""
    if isinstance(alpha, xr.DataArray):
        pass
    else:
        alpha = np.array(alpha)
        if dims is None:
            dims = name

        if alpha.ndim == 0:
            alpha = xr.DataArray(alpha, coords={dims: alpha}, name=name)
        elif alpha.ndim == 1:
            alpha = xr.DataArray(alpha, dims=dims, coords={dims: alpha}, name=name)
        else:
            alpha = xr.DataArray(alpha, dims=dims, name=name)
    return alpha

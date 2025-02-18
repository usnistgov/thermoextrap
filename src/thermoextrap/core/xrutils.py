"""Utilities for working with :mod:`xarray`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from cmomy.core.validate import (
    is_dataarray,
    is_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from cmomy.core.typing import DataT
    from numpy.typing import ArrayLike

    from thermoextrap.core.typing import MultDims, SingleDim
    from thermoextrap.core.typing_compat import TypeAlias

    DimsMapping: TypeAlias = Sequence[Hashable] | Mapping[int, MultDims]


###############################################################################
# Structure(s) to handle data
###############################################################################
def _check_xr(
    x: ArrayLike | DataT,
    dims: DimsMapping,
    name: str | None = None,
    strict: bool = False,
) -> xr.DataArray | DataT:
    if is_dataset(x):
        return x

    if is_dataarray(x):
        if strict:
            if isinstance(dims, Mapping):
                dims = dims[x.ndim]
            for d in dims:
                if d not in x.dims:
                    msg = f"{d} not in dims"
                    raise ValueError(msg)
        return x

    x = np.asarray(x)
    if isinstance(dims, dict):
        dims = dims[x.ndim]
    return xr.DataArray(x, dims=dims, name=name)


def xrwrap_uv(
    uv: ArrayLike | xr.DataArray,
    dims: DimsMapping | None = None,
    rec_dim: SingleDim = "rec",
    rep_dim: SingleDim = "rep",
    name: str | None = "u",
    strict: bool = True,
) -> xr.DataArray:
    """
    Wrap uv (energy values) array.

    assumes uv[rec_dim], or uv[rep_dim, rec_dim] where rec_dim is recorded (or time) and rep_dim is replicate
    """
    if dims is None:
        dims = {1: [rec_dim], 2: [rep_dim, rec_dim]}
    return _check_xr(uv, dims, name=name, strict=strict)


def xrwrap_xv(
    xv: ArrayLike | DataT,
    dims: DimsMapping | None = None,
    rec_dim: SingleDim = "rec",
    rep_dim: SingleDim = "rep",
    deriv_dim: SingleDim | None = None,
    val_dims: MultDims = "val",
    name: str | None = "x",
    strict: bool | None = None,
) -> xr.DataArray | DataT:
    """
    Wraps xv (x values) array.

    if deriv_dim is None, assumes xv[rec_dim], xv[rec_dim, vals], xv[rep_dim, rec_dim, val_dims]
    if deriv_dim is not None, assumes xv[rec_dim, deriv_dim], xv[rec_dim,deriv_dim, val_dims], xv[rep_dim,rec_dim,deriv_dim,val_dims]
    """
    if isinstance(val_dims, str):
        val_dims = [val_dims]
    elif not isinstance(val_dims, list):
        val_dims = list(val_dims)

    if strict is None:
        strict = False

    if deriv_dim is None:
        if dims is None:
            rec_val = [rec_dim, *val_dims]
            rep_val = [rep_dim, rec_dim, *val_dims]

            dims = {
                1: [rec_dim],
                len(rec_val): [rec_dim, *val_dims],
                len(rep_val): [rep_dim, rec_dim, *val_dims],
            }

    elif dims is None:
        rec_val = [rec_dim, deriv_dim, *val_dims]
        rep_val = [rep_dim, rec_dim, deriv_dim, *val_dims]
        dims = {
            2: [rec_dim, deriv_dim],
            len(rec_val): [rec_dim, deriv_dim, *val_dims],
            len(rep_val): [rep_dim, rec_dim, deriv_dim, val_dims],
        }
    return _check_xr(xv, dims=dims, name=name, strict=strict)


def xrwrap_alpha(
    alpha: ArrayLike | xr.DataArray,
    dims: MultDims | None = None,
    name: str = "alpha",
) -> xr.DataArray:
    """Wrap alpha values."""
    if is_dataarray(alpha):
        return alpha

    alpha = np.array(alpha)
    if dims is None:
        dims = name

    if alpha.ndim == 0:
        return xr.DataArray(alpha, coords={dims: alpha}, name=name)

    if alpha.ndim == 1:
        return xr.DataArray(alpha, dims=dims, coords={dims: alpha}, name=name)

    return xr.DataArray(alpha, dims=dims, name=name)

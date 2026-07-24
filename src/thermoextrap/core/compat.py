"""Compatibility code."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable
    from typing import Any

    from .typing import DataT
    from .typing_compat import EllipsisType


@overload
def xr_dot(
    x: DataT,
    y: xr.DataArray,
    *arrays: xr.DataArray,
    dim: Hashable | Iterable[Hashable] | EllipsisType | None = ...,
    **kwargs: Any,
) -> DataT: ...
@overload
def xr_dot(
    x: xr.DataArray,
    y: DataT,
    *arrays: xr.DataArray,
    dim: Hashable | Iterable[Hashable] | EllipsisType | None = ...,
    **kwargs: Any,
) -> DataT: ...


def xr_dot(
    x: xr.DataArray | xr.Dataset,
    y: xr.DataArray | xr.Dataset,
    *arrays: xr.DataArray,
    dim: Hashable | Iterable[Hashable] | EllipsisType | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Interface to xarray.dot.

    Remove a deprecation warning for older xarray versions.
    xarray deprecated `dims` keyword.  Use `dim` instead.
    """
    if isinstance(x, xr.Dataset):
        assert isinstance(y, xr.DataArray)  # ruff:ignore[assert]
        return x.map(xr_dot, args=arrays, dim=dim, **kwargs)  # pyright: ignore[reportUnknownMemberType]

    if isinstance(y, xr.Dataset):
        return x.map(lambda _x: xr_dot(_x, y, *arrays, dim=dim, **kwargs))  # type: ignore[no-any-return]  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]

    try:
        return xr.dot(x, y, *arrays, dim=dim, **kwargs)  # type: ignore[arg-type,unused-ignore,no-any-return]  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    except TypeError:
        # TODO(wpk): pretty sure can get rid of this.
        return xr.dot(x, y, *arrays, dims=dim, **kwargs)  # type: ignore[arg-type,unused-ignore,no-any-return]  # pyright: ignore[reportUnknownMemberType]

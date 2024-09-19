"""Typing utils."""

from __future__ import annotations

from typing import Union

import xarray as xr

from .typing_compat import TypeAlias, TypeVar

DataT = TypeVar("DataT", xr.DataArray, xr.Dataset)
DataT_ = TypeVar("DataT_", xr.DataArray, xr.Dataset)

XArrayObj: TypeAlias = Union[xr.DataArray, xr.Dataset]

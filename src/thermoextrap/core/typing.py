"""Typing utils."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Union

import xarray as xr

from .typing_compat import TypeAlias, TypeVar

DataT = TypeVar("DataT", xr.DataArray, xr.Dataset)
DataT_ = TypeVar("DataT_", xr.DataArray, xr.Dataset)

XArrayObj: TypeAlias = Union[xr.DataArray, xr.Dataset]


MetaKws: TypeAlias = Mapping[str, Any]

SingleDim: TypeAlias = Hashable
MultDims: TypeAlias = str | Sequence[Hashable]

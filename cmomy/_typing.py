"""Useful typing stuff."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .central import CentralMoments


T_MOM = Union[int, Tuple[int], Tuple[int, int]]
T_XVAL_LIKE = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]
T_XVAL_STRICT = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

ASARRAY_ORDER = Union[Literal["C", "F", "A", "K"], None]

T_CENTRALMOMENTS = TypeVar("T_CENTRALMOMENTS", bound="CentralMoments")

"""Validation routines"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cmomy.core.validate import is_xarray

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import attrs
    from numpy.typing import ArrayLike, NDArray

    from .typing import SupportsModel


# * Validate
# func(val) -> val
def validate_positive_integer(value: Any, name: str = "value") -> int:
    """Validate that value is a positive integer"""
    if isinstance(value, int):
        if value < 0:
            msg = f"{name}={value} must be a positive integer."
            raise ValueError(msg)
        return value

    msg = f"{name} must be an integer."
    raise TypeError(msg)


def validate_alpha(
    alpha: ArrayLike, states: Iterable[SupportsModel[Any]] | None
) -> NDArray[Any]:
    """
    Validate alpha to array with optional bounds check

    If pass ``states``, check that ``alpha`` values bounded by min/max
    ``states[k].alpha0``.
    """
    alpha = np.asarray(alpha)
    if states is not None:
        alpha0 = [x.alpha0 for x in states]
        lb, ub = min(alpha0), max(alpha0)
        if np.any((alpha < lb) | (ub < alpha)):
            msg = f"{alpha} outside of bounds [{lb}, {ub}]"
            raise ValueError(msg)

    return alpha


# * Validators
# Used by attrs
def validator_dims(self: Any, attribute: attrs.Attribute[Any], dims: Any) -> None:  # ruff:ignore[unused-function-argument]
    """Attrs validator for dimensions"""
    for d in dims:
        if d not in self.data.dims:
            msg = f"{d} not in data.dimensions {self.data.dims}"
            raise ValueError(msg)


def validator_xarray_typevar(
    self: Any,  # ruff:ignore[unused-function-argument]
    attribute: attrs.Attribute[Any],  # ruff:ignore[unused-function-argument]
    x: Any,
) -> None:
    """Attrs validator for xarray"""
    if not is_xarray(x):
        msg = f"Must pass xarray object.  Passed {type(x)}"
        raise TypeError(msg)

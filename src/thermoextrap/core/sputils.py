"""Utilities for sympy."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

# fix an issue with typing
from ._imports import sympy as sp

if TYPE_CHECKING:
    import sympy.core.symbol
    import sympy.tensor.indexed


@lru_cache(100)
def get_default_symbol(*args) -> tuple[sympy.core.symbol.Symbol, ...]:
    """Helper to get sympy symbols."""
    return sp.symbols(",".join(args))


@lru_cache(100)
def get_default_indexed(*args) -> list[sympy.tensor.indexed.IndexedBase]:
    """Helper to get sympy IndexBase objects."""
    out = [sp.IndexedBase(key) for key in args]
    if len(out) == 1:
        out = out[0]
    return out

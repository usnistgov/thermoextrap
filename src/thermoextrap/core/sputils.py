"""Utilities for sympy."""
from functools import lru_cache

import sympy as sp


@lru_cache(100)
def get_default_symbol(*args):
    return sp.symbols(",".join(args))


@lru_cache(100)
def get_default_indexed(*args):
    out = [sp.IndexedBase(key) for key in args]
    if len(out) == 1:
        out = out[0]
    return out

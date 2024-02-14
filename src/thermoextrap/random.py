"""Default numpy.random.Generator."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_DATA: dict[str, np.random.Generator | np.random.RandomState] = {}


@lru_cache
def _use_random_state() -> bool:
    import os

    return os.getenv("THERMOEXTRAP_TEST", "false").lower() in {"1", "t", "true"}


def default_rng(
    seed: int | np.random.Generator | np.random.RandomState | None = None,
) -> np.random.Generator | np.random.RandomState:
    """
    Get default random number generator.

    Parameters
    ----------
    seed: int, optional
        If specified, set the internal seed to this value.

    Returns
    -------
    :class:`numpy.random.Generator`
        If called with `seed=None` (default), return the previously created rng (if already created).
        This means you can call `default_rng(seed=...)` and subsequent calls of form `default_rng()`
        or `default_rng(None)` will continue rng sequence from first call with `seed=...`.  If
        New call with `seed` set will create a new rng sequence.
    """
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        return seed

    if seed is None:
        if "rng" not in _DATA:
            if _use_random_state():
                _DATA["rng"] = np.random.RandomState()
            else:
                _DATA["rng"] = np.random.default_rng()

    elif _use_random_state():
        _DATA["rng"] = np.random.RandomState(seed=seed)
    else:
        _DATA["rng"] = np.random.default_rng(seed=seed)

    return _DATA["rng"]


def validate_rng(
    rng: np.random.Generator | np.random.RandomState | None, seed: int | None = None
) -> np.random.Generator | np.random.RandomState:
    """
    Validate rng.

    Parameters
    ----------
    rng : :class:`numpy.random.Generator`, optional
        If pass a rng, then use it.  Otherwise, use `default_rng(seed)`
    seed : int, optional
        Seed to use if call :func:`default_rng`

    Returns
    -------
    :class:`numpy.random.Generator`
    """
    if rng is None:
        return default_rng(seed=seed)

    if not isinstance(rng, (np.random.Generator, np.random.RandomState)):
        msg = f"{type(rng)=} must be NoneType or numpy.random.Generator"
        raise TypeError(msg)

    return rng

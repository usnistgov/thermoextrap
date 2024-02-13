"""Default numpy.random.Generator."""

from __future__ import annotations

import numpy as np

_DATA: dict[str, np.random.Generator] = {}


def default_rng(seed: int | None = None) -> np.random.Generator:
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
    if seed is None:
        if "rng" not in _DATA:
            _DATA["rng"] = np.random.default_rng()

    else:
        _DATA["rng"] = np.random.default_rng(seed=seed)

    return _DATA["rng"]

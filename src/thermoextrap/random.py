"""Default numpy.random.Generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Sequence

    SEED_TYPES = Optional[
        int,
        Sequence[int],
        np.random.SeedSequence,
        np.random.BitGenerator,
        np.random.Generator,
    ]


_DATA: dict[str, np.random.Generator] = {}


def set_internal_rng(rng: np.random.Generator) -> None:
    """
    Set the internal random number :class:`~numpy.random.Generator`.

    The function :func:`default_rng` will call `rng` if called with a new seed
    (or when called the first time). However, if want to override the internal
    rng, you can use this function.

    Parameters
    ----------
    force: bool, default=False
        If false, only set internal rng if not already set
    If true, set internal rng

    """
    _DATA["rng"] = rng


def get_internal_rng() -> np.random.Generator:
    """Get the internal random number :class:`~numpy.random.Generator`."""
    if "rng" not in _DATA:
        msg = "must set rng."
        raise ValueError(msg)
    return _DATA["rng"]


def _missing_internal_rng() -> bool:
    return "rng" not in _DATA


def default_rng(seed: SEED_TYPES = None) -> np.random.Generator:
    """
    Get default random number generator.

    Parameters
    ----------
    seed: int, sequence of int, :class:`~numpy.random.SeedSequence`, :class:`~numpy.random.BitGenerator`, Generator, optional
        If specified, set the internal seed to this value. If pass in a
        :class:`numpy.random.Generator`, return that object.

    Returns
    -------
    Generator
        If called with `seed=None` (default), return the previously created rng
        (if already created). This means you can call `default_rng(seed=...)`
        and subsequent calls of form `default_rng()` or `default_rng(None)`
        will continue rng sequence from first call with `seed=...`. If New call
        with `seed` set will create a new rng sequence. Note that if you pass a
        :class:`~numpy.random.Generator` for seed, that object will be
        returned, but in this case, the internal generator will not be altered.


    """
    if isinstance(seed, np.random.Generator):
        return seed

    if seed is None:
        if _missing_internal_rng():
            set_internal_rng(np.random.default_rng())

    else:
        set_internal_rng(np.random.default_rng(seed=seed))

    return get_internal_rng()


def validate_rng(
    rng: np.random.Generator | None, seed: int | None = None
) -> np.random.Generator:
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

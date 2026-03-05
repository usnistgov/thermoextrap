"""Typing compatibility."""

# pyright: reportUnreachable=false
from __future__ import annotations

import sys
from types import EllipsisType
from typing import Concatenate, TypeAlias, TypeGuard

if sys.version_info >= (3, 11):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, Self, Unpack
else:
    from typing_extensions import NotRequired, Required, Self, Unpack


if sys.version_info >= (3, 13):
    from typing import TypeIs, TypeVar
else:  # pragma: no cover
    from typing_extensions import TypeIs, TypeVar


__all__ = [
    "Concatenate",
    "EllipsisType",
    "NotRequired",
    "Required",
    "Self",
    "TypeAlias",
    "TypeGuard",
    "TypeIs",
    "TypeVar",
    "TypedDict",
    "Unpack",
    "override",
]

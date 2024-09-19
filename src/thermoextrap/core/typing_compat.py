"""Typing compatibility."""

import sys
from typing import Any

if sys.version_info < (3, 10):
    EllipsisType = Any
    from typing_extensions import TypeAlias, TypeGuard
else:
    from types import EllipsisType
    from typing import TypeAlias, TypeGuard


if sys.version_info < (3, 11):
    from typing_extensions import Required, Self, Unpack
else:
    from typing import Required, Self, Unpack


if sys.version_info < (3, 13):
    from typing_extensions import TypeIs, TypeVar
else:  # pragma: no cover
    from typing import TypeIs, TypeVar


__all__ = [
    "EllipsisType",
    "Required",
    "Self",
    "TypeAlias",
    "TypeGuard",
    "TypeIs",
    "TypeVar",
    "Unpack",
]

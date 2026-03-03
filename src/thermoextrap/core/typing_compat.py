"""Typing compatibility."""
# pyright: reportUnreachable=false

import sys

if sys.version_info >= (3, 10):
    from types import EllipsisType
    from typing import Concatenate, TypeAlias, TypeGuard
else:
    from typing import TYPE_CHECKING

    from typing_extensions import Concatenate, TypeAlias, TypeGuard

    if TYPE_CHECKING:
        import builtins

    EllipsisType: TypeAlias = "builtins.ellipsis"


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
    "Unpack",
    "override",
]

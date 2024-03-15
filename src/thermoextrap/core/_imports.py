from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # some weird stuff happens with sympy and pyright
    # this should stop those errors:
    import importlib

    sympy = importlib.import_module("sympy")
else:
    import sympy


def has_pymbar():
    from importlib.util import find_spec

    return find_spec("pymbar") is not None


__all__ = ["has_pymbar", "sympy"]

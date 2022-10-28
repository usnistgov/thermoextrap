from __future__ import annotations

import os
from collections.abc import Mapping
from textwrap import dedent
from typing import Any, Callable, TypeVar

try:
    # Default is DOC_SUB is True
    DOC_SUB = os.getenv("THERMOEXTRAP_DOC_SUB", "True").lower() not in (
        "0",
        "f",
        "false",
    )
except KeyError:
    DOC_SUB = True


# Taken directly from pandas.utils._decorators
# See https://github.com/pandas-dev/pandas/blob/main/pandas/util/_decorators.py
# would just use the pandas version, but since it's a private
# module, feel it's better to have static version here.


# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def doc(*docstrings: str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if hasattr(docstring, "_docstring_components"):
                # error: Item "str" of "Union[str, Callable[..., Any]]" has no attribute
                # "_docstring_components"
                # error: Item "function" of "Union[str, Callable[..., Any]]" has no
                # attribute "_docstring_components"
                docstring_components.extend(
                    docstring._docstring_components  # type: ignore[union-attr]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # formatting templates and concatenating docstring
        decorated.__doc__ = "".join(
            [
                component.format(**params)
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in docstring_components
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator


# Factory method to create docfiller
def docfiller(*args, **kwargs):
    """To fill common docs.

    Taken from pandas.utils._decorators

    """

    if DOC_SUB:
        return doc(*args, **kwargs)
    else:

        def decorated(func):
            return func

        return decorated


def _tuple_keys(d):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            key = [(k,) + t for t in _tuple_keys(v)]
        else:
            key = [(k,)]

        out.extend(key)
    return out


class AttributeDict(Mapping):
    __slots__ = "entries"

    def __init__(self, entries=None):
        if entries is None:
            entries = {}
        self.entries = entries

    def __getitem__(self, key):
        return self.entries[key]

    def __setitem__(self, key, value):
        self.entries[key] = value

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def items(self):
        yield from self.entries.items()

    def __getattr__(self, attr):
        if attr in self.entries:
            out = self.entries[attr]
            if isinstance(out, dict):
                out = type(self)(out)
            return out
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                # If Python is run with -OO, it will strip docstrings and our lookup
                # from self.entries will fail. We check for __debug__, which is actually
                # set to False by -O (it is True for normal execution).
                # But we only want to see an error when building the docs;
                # not something users should see, so this slight inconsistency is fine.
                if __debug__:
                    raise err
                else:
                    pass

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.entries)})"

    # def as_dotted_dict(self):
    #     keys =

    @classmethod
    def from_dict(cls, d):
        """
        Recursively apply
        """
        out = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                v = cls.from_dict(v)
            out[k] = v
        return out


def prepare_shared_docs(**shared_docs):
    """
    Dedent and apply AttributeDict
    """
    out = AttributeDict()
    for k, v in shared_docs.items():
        if isinstance(v, dict):
            v = prepare_shared_docs(v)
        out[k] = dedent(v).strip()
    return out

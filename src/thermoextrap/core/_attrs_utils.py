from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import Any

    from thermoextrap.core.typing import MultDims
    from thermoextrap.core.typing_compat import Self


def attrs_clear_cache(self, attribute, value):
    """Clear out _cache if setting value."""
    self._cache = {}
    return value


def convert_mapping_or_none_to_dict(kws: Mapping[str, Any] | None) -> dict[str, Any]:
    if kws is None:
        return {}
    return dict(kws)


def optional_converter(converter):
    """Create a converter which can pass through None."""

    def wrapped(value):
        if value in {None, attrs.NOTHING}:
            return value
        return converter(value)

    return wrapped


def field_formatter(fmt: str = "{:.5g}"):
    """Formatter for attrs field."""

    @optional_converter
    def wrapped(value):
        return fmt.format(value)

    return wrapped


def field_array_formatter(threshold: int = 3, **kws: Any):
    """Formatter for numpy array field."""

    @optional_converter
    def wrapped(value):
        with np.printoptions(threshold=threshold, **kws):
            return str(value)

    return wrapped


# def kw_only_field(kw_only: bool = True, **kws: Any):
#     return attrs.field(kw_only=kw_only, **kws)


def convert_dims_to_tuple(dims: MultDims | None) -> tuple[Hashable, ...]:
    if dims is None:
        return ()
    return (dims,) if isinstance(dims, str) else tuple(dims)


# def cache_field(
#     init: bool = False,
#     repr: bool = False,
#     factory: Callable[[], Any] = dict,
#     **kws: Any,
# ):
#     return attrs.field(init=False, repr=False, factory=dict, **kws)


# def validate_dims_in_object(self, attrs, dims):
#     for d in dims:
#         if d not in self.data.dims:
#             raise ValueError(f"{d} not in data dimensions {self.data.dims}")


@attrs.define
class MyAttrsMixin:
    """Baseclass for adding some sugar to attrs.derived classes."""

    def asdict(self) -> dict[str, Any]:
        """Convert object to dictionary."""
        return attrs.asdict(self, filter=self._get_smart_filter())

    def new_like(self, **kws: Any) -> Self:
        """
        Create a new object with optional parameters.

        Parameters
        ----------
        **kws
            `attribute`, `value` pairs.
        """
        return attrs.evolve(self, **kws)

    def assign(self, **kws: Any) -> Self:
        """Alias to :meth:`new_like`."""
        return self.new_like(**kws)

    def set_params(self, **kws: Any) -> Self:
        """Set parameters of self, and return self (for chaining)."""
        for name in kws:
            if not hasattr(self, name):
                msg = f"{name=} not an attribute."
                raise AttributeError(msg)

        for name, val in kws.items():
            setattr(self, name, val)
        return self

    def _immutable_setattrs(self, **kws: Any) -> None:
        """
        Set attributes of frozen attrs class.

        This should only be used to set 'derived' attributes on creation.

        Parameters
        ----------
        **kws :
            `key`, `value` pairs to be set.  `key` should be the name of the attribute to be set,
            and `value` is the value to set `key` to.

        Examples
        --------
        >>> @attrs.frozen
        ... class Derived(MyAttrsMixin):
        ...     _derived = attrs.field(init=False)
        ...
        ...     def __attrs_post_init__(self):
        ...         self._immutable_setattrs(_derived=10)
        >>> x = Derived()
        >>> x._derived
        10
        """
        for key, value in kws.items():
            object.__setattr__(self, key, value)  # noqa: PLC2801

    def _get_smart_filter(
        self,
        include: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        exclude_private: bool = True,
        exclude_no_init: bool = True,
    ) -> Callable[..., Any]:
        """
        Create a filter to include exclude names.

        Parameters
        ----------
        include : sequence of str
            Names to include
        exclude : sequence of str
            names to exclude
        exclude_private : bool, default=True
            If True, exclude any names starting with '_'
        exclude_no_init : bool, default=True
            If True, exclude any fields defined with ``field(init=False)``.

        Notes
        -----
        Precedence is in order
        `include, exclude, exclude_private exclude_no_init`.
        That is, if a name is in include and exclude and is private/no_init,
        it will be included
        """
        fields = attrs.fields(type(self))

        if include is None:
            include = []
        elif isinstance(include, str):
            include = [include]

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]

        includes = []
        for f in fields:
            if f.name in include:
                includes.append(f)

            elif (
                f.name in exclude
                or (exclude_private and f.name.startswith("_"))
                or (exclude_no_init and not f.init)
            ):
                pass

            else:
                includes.append(f)
        return attrs.filters.include(*includes)

"""
Typing aliases (:mod:`thermoextrap.core.typing`)
================================================
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import (
    Callable,  # noqa: F401
    Hashable,  # noqa: F401
    Iterable,  # noqa: F401
    Iterator,
    Mapping,
    Sequence,
    Sized,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    SupportsIndex,
    overload,
    runtime_checkable,
)

import numpy as np
import xarray as xr
from cmomy import IndexSampler  # pylint: disable=no-name-in-module
from numpy.typing import NDArray

from .typing_compat import Self, TypeAlias, TypeVar

if TYPE_CHECKING:
    import tensorflow as tf  # noqa: F401
    from cmomy.resample.typing import SamplerType
    from numpy.typing import ArrayLike
    from sympy.core.expr import (  # pyright: ignore[reportMissingTypeStubs]
        Expr,  # noqa: F401
    )


DataT = TypeVar("DataT", xr.DataArray, xr.Dataset, default=xr.DataArray)
"""Type variable for :class:`~xarray.DataArray` or :class:`~xarray.Dataset`."""

T_co = TypeVar("T_co", covariant=True)
"""Generic covariant type variable."""

_T = TypeVar("_T")

NDArrayOrDataArrayT = TypeVar(
    "NDArrayOrDataArrayT",
    bound="NDArray[Any] | xr.DataArray",
)

# * Alias
XArrayObj: TypeAlias = "xr.DataArray | xr.Dataset"
OptionalKws: TypeAlias = "Mapping[str, _T]| None"
OptionalKwsAny: TypeAlias = "OptionalKws[Any]"
SingleDim: TypeAlias = str
MultDims: TypeAlias = "str | Sequence[Hashable]"
PostFunc: TypeAlias = "str | Callable[[Expr], Expr] | None"
OptionalRng: TypeAlias = "np.random.Generator | None"
NDArrayAny: TypeAlias = NDArray[Any]

TensorType: TypeAlias = "NDArrayAny| tf.Tensor"

# DataDerivArgs: TypeAlias = "tuple[XArrayObj | DataSelector[xr.DataArray] | DataSelector[xr.Dataset], ...]"   # more specific, but not needed.
DataDerivArgs: TypeAlias = "tuple[Any, ...]"


# * Literals
SymDerivNames = Literal[
    "x_ave", "u_ave", "dun_ave", "dxdun_ave", "un_ave", "xun_ave", "lnPi_energy"
]

StackPolicy = Literal["infer", "raise"]
ApplyReduceFuncs: TypeAlias = (
    "str | Callable[..., Any] | Iterable[str | Callable[..., Any]]"
)


# * Protocols
class SupportsIdentityTransform(Protocol):
    """Callable for identity transform."""

    def __call__(
        self, x: NDArrayOrDataArrayT, y: NDArrayOrDataArrayT, y_var: ArrayLike
    ) -> tuple[NDArrayOrDataArrayT, NDArrayOrDataArrayT, list[NDArrayOrDataArrayT]]: ...


@runtime_checkable
class SupportsGetItem(Protocol[T_co]):
    """Protocol for thing that container that supports __getitem__"""

    def __getitem__(self, index: SupportsIndex, /) -> T_co: ...


@runtime_checkable
class SupportsData(Protocol[T_co]):
    """Protocol for Data"""

    # @property
    # def meta(self) -> DataCallbackABC: ...
    # @property
    # def umom_dim(self) -> SingleDim: ...
    @property
    def deriv_dim(self) -> SingleDim | None: ...
    @property
    def x_is_u(self) -> bool: ...
    @property
    def order(self) -> int: ...
    @property
    def central(self) -> bool: ...
    @property
    def xalpha(self) -> bool: ...
    @property
    def deriv_args(self) -> DataDerivArgs: ...

    @abstractmethod
    def resample(self, sampler: SamplerType) -> Self: ...


# TODO(wpk): remove when rework PerturbModel
@runtime_checkable
class SupportsDataXU(Protocol[T_co]):
    """Data protocol for PerturbModel"""

    @property
    def uv(self) -> xr.DataArray: ...
    @property
    def xv(self) -> T_co: ...
    @property
    def rec_dim(self) -> str: ...

    @abstractmethod
    def resample(self, sampler: SamplerType) -> Self: ...


@runtime_checkable
class SupportsModel(Protocol[T_co]):
    """Protocol for single model."""

    @property
    def data(self) -> SupportsData[T_co]: ...
    @property
    def alpha0(self) -> float: ...
    @property
    def order(self) -> int: ...
    @property
    def alpha_name(self) -> str: ...

    # @property
    # def data(self) -> SupportsData[T_co]: ...
    # @property
    # def derivatives(self) -> Derivatives: ...

    # def derivs(self, *args: Any, **kwargs: Any) -> T_co: ...
    # def coefs(self, *args: Any, **kwargs: Any) -> T_co: ...
    @abstractmethod
    def predict(self, alpha: ArrayLike) -> T_co: ...
    @abstractmethod
    def resample(self, sampler: SamplerType) -> Self: ...


@runtime_checkable
class SupportsModelDerivs(SupportsModel[T_co], Protocol[T_co]):
    """Protocol for single model with derivs"""

    @abstractmethod
    def derivs(self, *args: Any, **kwargs: Any) -> T_co: ...


# Type Variables with name `...DataT...` have defaults bound to DataT
SupportsModelDataT = TypeVar(
    "SupportsModelDataT", bound=SupportsModel[XArrayObj], default=SupportsModel[DataT]
)
"""Generic type variable for :class:`SupportsModel`"""

SupportsModelDataT_co = TypeVar(
    "SupportsModelDataT_co",
    bound=SupportsModel[XArrayObj],
    default=SupportsModel[DataT],
    covariant=True,
)
"""Generic type variable for :class:`SupportsModel`"""

SupportsModelDerivsDataT = TypeVar(
    "SupportsModelDerivsDataT",
    bound=SupportsModelDerivs[XArrayObj],
    default=SupportsModelDerivs[DataT],
)
"""Generic type variable for :class:`SupportsModelDerivs`"""

# Special case for DataArray only
SupportsModelDerivsDataArrayT = TypeVar(
    "SupportsModelDerivsDataArrayT",
    bound=SupportsModelDerivs[xr.DataArray],
    default=SupportsModelDerivs[xr.DataArray],
)
"""Generic type variable for :class:`SupportsModelDerivs` with DataArray"""

SupportsModelT_co = TypeVar(
    "SupportsModelT_co",
    bound=SupportsModel[Any],
    covariant=True,
    default=SupportsModel[T_co],
)
"""Generic covariant type variable for :class:`SupportsModel`"""


# TODO(wpk): Create SupportsStateCollection protocol
# M_co = TypeVar("M_co", covariant=True)
@runtime_checkable
class SupportsStateCollection(Protocol[T_co, SupportsModelT_co]):
    """Protocol for State collection"""

    def __init__(
        self, states: Sequence[SupportsModelT_co], *, kws: OptionalKwsAny = None
    ) -> None: ...

    @property
    def states(self) -> Sequence[SupportsModelT_co]: ...
    @property
    def kws(self) -> Mapping[str, Any]: ...

    def new_like(self, **kws: Any) -> Self: ...

    def __len__(self) -> int:
        return len(self.states)

    @overload
    def __getitem__(self, idx: SupportsIndex, /) -> SupportsModelT_co: ...
    @overload
    def __getitem__(self, idx: slice[Any, Any, Any], /) -> Self: ...

    def __getitem__(
        self, idx: SupportsIndex | slice[Any, Any, Any], /
    ) -> SupportsModelT_co | Self:
        if isinstance(idx, slice):
            return self.new_like(states=self.states[idx])
        return self.states[int(idx)]

    def __iter__(self) -> Iterator[SupportsModelT_co]:
        return iter(self.states)

    @property
    def alpha_name(self) -> str:
        return self.states[0].alpha_name

    @property
    def order(self) -> int:
        return min(m.order for m in self.states)

    @property
    def alpha0(self) -> list[float]:
        return [m.alpha0 for m in self]

    def resample(
        self, sampler: SamplerType | Sequence[SamplerType], **kws: Any
    ) -> Self:
        """
        Resample underlying models.

        If pass in a single sampler, use it for all states. For example, to
        resample all states with some ``nrep``, use
        ``.resample(sampler={"nrep": nrep})``. Note that the if you pass a
        single mapping, the mapping will be passed to each state ``resample``
        method, which will in turn create unique sample for each state. To
        specify a different sampler for each state, pass in a sequence of
        sampler.
        """
        if isinstance(
            sampler,
            (np.ndarray, xr.DataArray, xr.Dataset, IndexSampler, Mapping),
        ):
            sampler = [sampler] * len(self)
        elif not isinstance(sampler, Sized) or len(sampler) != len(self):
            msg = f"SamplerType must be a sized object with length {len(self)=}"
            raise ValueError(msg)

        return self.new_like(
            states=tuple(
                state.resample(sampler=sampler, **kws)
                for state, sampler in zip(self.states, sampler, strict=True)
            ),
        )

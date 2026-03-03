# pyright: reportUnknownMemberType=false,reportUnknownVariableType=false,reportMissingTypeStubs=false, reportUnknownArgumentType=false
"""
General extrapolation/interpolation models (:mod:`~thermoextrap.models`)
========================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Sized
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    SupportsIndex,
    cast,
    overload,
)

import attrs
import cmomy
import numpy as np
import pandas as pd
import xarray as xr
from attrs import field
from attrs import validators as attv
from module_utilities import cached

from .core._attrs_utils import (
    MyAttrsMixin,
    convert_mapping_or_none_to_dict,
)
from .core._imports import module_available
from .core._imports import sympy as sp
from .core.compat import xr_dot
from .core.docstrings import DOCFILLER_SHARED
from .core.sputils import (
    get_default_indexed,
    get_default_symbol,
    lambdify_with_defaults,
)
from .core.typing import (
    DataT,
    SupportsData,
    SupportsDataXU,
    SupportsGetItem,
    SupportsModelDataT,
    SupportsModelDataT_co,
    SupportsModelDerivsDataT,
)
from .core.typing_compat import override
from .core.validate import validate_alpha
from .core.xrutils import xrwrap_alpha

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from typing import ClassVar

    from cmomy.resample.typing import SamplerType
    from numpy.typing import ArrayLike, NDArray
    from pymbar.mbar import MBAR
    from sympy.core.expr import Expr
    from sympy.core.numbers import Number
    from sympy.core.symbol import Symbol
    from sympy.tensor.indexed import IndexedBase

    from .core.typing import OptionalKwsAny, PostFunc
    from .core.typing_compat import Self, TypeVar

    _T = TypeVar("_T")

docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap")


# * Utils ---------------------------------------------------------------------
def _validate_supports_getitem(
    instance: Any,  # noqa: ARG001
    attribute: attrs.Attribute[Any],  # noqa: ARG001
    value: Any,
) -> None:
    if not isinstance(value, SupportsGetItem):
        msg = "{attribute.name} must support __getitem__"
        raise TypeError(msg)


# * Structure(s) to deal with analytic derivatives, etc -----------------------
class SymFuncBase(sp.Function):  # type: ignore[misc,name-defined]
    """
    Base class to define a sympy function for user defined derivatives.


    See Also
    --------
    :class:`thermoextrap.models.SymDerivBase`
    """

    nargs: ClassVar[int | tuple[int, ...]]

    # allow override of this simply for typing...
    @classmethod
    def deriv_args(cls) -> tuple[Symbol | IndexedBase, ...]:
        """
        Symbol arguments of function.

        This is used by Data class to create a 'lambdfied' callable function.

        See Also
        --------
        sympy.utilities.lambdify.lambdify
        """
        raise NotImplementedError

    def fdiff(self, argindex: int | Number = 1) -> Expr:
        """Derivative of function.  This will be used by :class:`thermoextrap.models.SymDerivBase`."""
        raise NotImplementedError

    def _doit_args(self, deep: bool = False, **hints: Any) -> None:
        """Perform doit on args"""
        if deep:
            for a in self.args:
                a.doit(deep=deep, **hints)

    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        """Generic ``doit`` method (See :meth:`~sympy.core.basic.Basic.doit`)."""
        raise NotImplementedError

    @classmethod
    def eval(cls, beta: Symbol) -> Any:  # noqa: ARG003
        """
        Evaluate function.

        Other than trivial cases where the function returns a number, Function
        should return None. The heavy lifting is deferred to :meth:`doit`
        """
        return None

    @classmethod
    def tcall(cls, beta: Symbol) -> Expr:
        """
        Typed interface to cls(...)

        This also allows us to call with named keyword arguments, etc.
        """
        raise NotImplementedError


@attrs.define
class _GetItemSympyOperations:
    """
    Class to perform sympy operations on :class:`thermoextrap.models.SymDerivBase`.

    Parameters
    ----------
    funcs : sequence of SymFunction
        Symbolic functions to consider.
    subs : Sequence, optional
        Substitutions.
    subs_final : Sequence, optional
        Final substitutions.
    subs_all : mapping, optional
        Total substitution.
    recursive : bool, default=True
        If True, recursively apply substitutions.
    simplify : bool, default=False
        If True, simplify result.
    expand : bool, default=True
        If True, try to expand result.
    """

    funcs: SupportsGetItem[Expr] = field(validator=_validate_supports_getitem)
    simplify: bool = field(default=False)
    expand: bool = field(default=True)

    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict)

    @cached.meth
    def __getitem__(self, order: SupportsIndex, /) -> Expr:
        func = self.funcs[order]

        func = func.doit()

        if self.simplify:
            func = func.simplify()

        if self.expand:
            func = func.expand()

        return func


def _convert_lambdify_kws(value: OptionalKwsAny) -> dict[str, Any]:
    out: dict[str, Any] = {} if value is None else dict(value)
    out.setdefault("cse", True)
    return out


@attrs.define
class Lambdify:
    """
    Create list-like of python functions from list-like of expressions.

    Parameters
    ----------
    exprs : sequence of symFunction
        array of sympy expressions to ``lambdify``
    args : sequence of Symbol
        array of symbols which will be in args of the resulting function
    lambdify_kws : dict
        Extra arguments to :func:`~sympy.utilities.lambdify.lambdify`.
        Note that by default we set ``cse = True``, ``dummify = True`` by default.

    See Also
    --------
    sympy.utilities.lambdify.lambdify
    """

    exprs: SupportsGetItem[Expr] = field()
    args: Sequence[Symbol | IndexedBase] = field()
    lambdify_kws: dict[str, Any] = field(
        kw_only=True,
        factory=dict[str, "Any"],
        converter=_convert_lambdify_kws,
    )

    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict)

    @cached.meth
    def __getitem__(self, order: SupportsIndex, /) -> Callable[..., Any]:
        return lambdify_with_defaults(self.args, self.exprs[order], **self.lambdify_kws)


@docfiller_shared.decorate
class SymDerivBase:
    """
    Base class for working with recursive derivatives in expansions.

    Parameters
    ----------
    func : symFunction
        Function to differentiate.  This should (most likely) be an instance
        of :class:`thermoextrap.models.SymFuncBase`
    args : sequence of Symbol
        Arguments to func
    {expand}
    {post_func}
    """

    def __init__(
        self,
        func: SymFuncBase,
        args: Sequence[Symbol | IndexedBase] | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
        beta: Symbol | None = None,
    ) -> None:
        if args is None:
            args = func.deriv_args()

        if beta is None:
            beta = get_default_symbol("beta")

        self._func_orig = func
        self._post_func = post_func

        if isinstance(post_func, str):
            if post_func == "minus_log":

                def post_func_log(f: Expr) -> Expr:
                    return -sp.log(f)

                post_func = post_func_log
            elif post_func.startswith("pow_"):
                i = int(post_func.split("_")[-1])

                def post_func_pow(f: Expr) -> Expr:
                    return sp.Pow(f, i)

                post_func = post_func_pow

            else:
                msg = "post_func must be callable or in {minus_log, pow_1, pow_2, ...}"
                raise ValueError(msg)

        self.func: Expr = func if post_func is None else post_func(func)
        self.args = args
        self.expand = expand
        self.beta = beta
        self._cache: dict[str, Any] = {}

    @cached.meth
    def __getitem__(self, order: SupportsIndex, /) -> Expr:
        if order == 0:
            return self.func
        out = self[int(order) - 1].diff(self.beta, 1)
        if self.expand:
            out = out.expand()
        return out

    @cached.meth
    def doit(
        self,
        simplify: bool = False,
        expand: bool = False,
    ) -> _GetItemSympyOperations:
        """Indexer for Derivatives with :meth:`~sympy.core.basic.Basic.doit` applied"""
        return _GetItemSympyOperations(
            funcs=self,
            simplify=simplify,
            expand=expand,
        )

    @cached.meth
    def lambdify(
        self,
        /,
        args: Sequence[Symbol | IndexedBase] | None = None,
        simplify: bool = False,
        expand: bool = False,
        **lambdify_kws: Any,
    ) -> Lambdify:
        """
        Indexer of lambdified functions

        See Also
        --------
        doit
        Lambdify
        """
        return Lambdify(
            self.doit(simplify=simplify, expand=expand),
            args=self.args if args is None else args,
            lambdify_kws=lambdify_kws,
        )


# -log<X>
class SymMinusLog:
    """Class to compute derivatives of Y = -log(<X>)."""

    X, dX = get_default_indexed("X", "dX")

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    @cached.meth
    def __getitem__(self, order: SupportsIndex, /) -> Expr:
        if (order := int(order)) == 0:
            return -sp.log(self.X[0])

        expr: Expr = sp.Number(0)
        for k in range(1, order + 1):
            expr += (
                sp.factorial(k - 1) * (-1 / self.X[0]) ** k * sp.bell(order, k, self.dX)  # pyright: ignore[reportOperatorIssue]
            )
        # subber
        subs: dict[Any, Any] = {self.dX[j]: self.X[j + 1] for j in range(order + 1)}
        return expr.subs(subs).expand().simplify()


@lru_cache(5)
def factory_minus_log() -> Lambdify:
    """Create lambdified version of :class:`SymMinusLog`"""
    s = SymMinusLog()
    return Lambdify(s, (s.X,))


@attrs.define
class Derivatives(MyAttrsMixin):
    """
    Class to wrap functions calculating derivatives to specified order.


    Parameters
    ----------
    funcs : sequence of callable
        ``funcs[i](*args)`` gives the ith derivative
    exprs : sequence of Expr, optional
        expressions corresponding to the `funcs`
        Mostly for debugging purposes.
    """

    #: Sequence of callable functions
    funcs: SupportsGetItem[Callable[..., Any]] = field(
        validator=_validate_supports_getitem
    )
    #: Sequence of sympy expressions, optional
    exprs: SupportsGetItem[Expr] | None = field(
        kw_only=True,
        default=None,
        validator=attv.optional(_validate_supports_getitem),
    )
    #: Arguments
    args: Sequence[Symbol | IndexedBase] | None = field(kw_only=True, default=None)

    @staticmethod
    def _apply_minus_log(X: list[DataT], order: int) -> list[DataT]:
        func = factory_minus_log()
        return [func[i](X) for i in range(order + 1)]

    @overload
    def derivs(
        self,
        data: SupportsData[DataT],
        *,
        order: int | None = ...,
        minus_log: bool = ...,
        order_dim: None,
        concat_kws: OptionalKwsAny = ...,
        norm: bool = ...,
    ) -> list[DataT]: ...
    @overload
    def derivs(
        self,
        data: SupportsData[DataT],
        *,
        order: int | None = ...,
        minus_log: bool = ...,
        order_dim: str = ...,
        concat_kws: OptionalKwsAny = ...,
        norm: bool = ...,
    ) -> DataT: ...

    def derivs(
        self,
        data: SupportsData[DataT],
        *,
        order: int | None = None,
        minus_log: bool = False,
        order_dim: str | None = "order",
        concat_kws: OptionalKwsAny = None,
        norm: bool = False,
    ) -> DataT | list[DataT]:
        """
        Calculate derivatives for orders range(0, order+1).

        Parameters
        ----------
        data : object
            Data object.
            If passed, use `args=data.deriv_args`
        order : int, optional
            If pass `data` and `order` is `None`, then `order=data.order`
            Otherwise, must mass order
        minus_log : bool, default=False
            If `True`, apply transform for `Y = -log(<X>)`
        order_dim : str, default='order'
            If `None`, output will be a list
            If `order_dim` is a string, then apply `xarray.concat` to output
            To yield a single DataArray
        concat_kws : dict, optional
            extra arguments to `xarray.concat`
        norm : bool, default=False
            If true, then normalize derivatives by `1/n!`, where `n` is the order of
            the derivative.  That is, transform derivatives to taylor series coefficients
            See also taylor_series_norm

        Returns
        -------
        output : list of xarray.DataArray
            See above for nature of output
        """
        args = data.deriv_args
        if order is None:
            order = data.order

        out: list[DataT] = [self.funcs[i](*args) for i in range(order + 1)]

        if minus_log:
            out = self._apply_minus_log(X=out, order=order)

        if norm:
            out = [x / math.factorial(i) for i, x in enumerate(out)]

        if order_dim is None:
            return out

        return xr.concat(out, dim=order_dim, **(concat_kws or {}))  # pyright: ignore[reportCallIssue, reportArgumentType]

    @overload
    def coefs(
        self,
        data: SupportsData[DataT],
        *,
        order: int | None = ...,
        minus_log: bool = ...,
        order_dim: None,
        **kwargs: Any,
    ) -> list[DataT]: ...
    @overload
    def coefs(
        self,
        data: SupportsData[DataT],
        *,
        order: int | None = ...,
        minus_log: bool = ...,
        order_dim: str = ...,
        **kwargs: Any,
    ) -> DataT: ...

    def coefs(
        self,
        data: SupportsData[DataT],
        *,
        order: int | None = None,
        minus_log: bool = False,
        order_dim: str | None = "order",
        **kwargs: Any,
    ) -> DataT | list[DataT]:
        """
        Alias to `self.derivs(..., norm=True)`.

        See Also
        --------
        derivs
        """
        return self.derivs(
            data=data,
            order=order,
            minus_log=minus_log,
            order_dim=order_dim,
            norm=True,
            **kwargs,
        )

    @classmethod
    def from_sympy(
        cls, exprs: SupportsGetItem[Expr], args: Sequence[Symbol | IndexedBase]
    ) -> Derivatives:
        """
        Create object from list of sympy functions.

        Parameters
        ----------
        exprs : sequence of symFunction
            sequence of sympy functions.
        args : sequence of Symbol
            Arguments

        Returns
        -------
        output : object
        """
        funcs = Lambdify(exprs, args=args)
        return cls(funcs=funcs, exprs=exprs, args=args)


@lru_cache(10)
def taylor_series_norm(order: int, order_dim: str = "order") -> xr.DataArray:
    """``taylor_series_coefficients = derivs * taylor_series_norm``."""
    out = np.array([1 / math.factorial(i) for i in range(order + 1)])
    return xr.DataArray(out, dims=order_dim)


@attrs.define
class ExtrapModel(MyAttrsMixin, Generic[DataT]):
    """Apply taylor series extrapolation."""

    #: Alpha value data is evaluated at
    _alpha0: float = field(converter=float, alias="alpha0")

    #: Data object
    data: SupportsData[DataT] = field(
        validator=attv.instance_of(SupportsData),  # type: ignore[type-abstract]  # pyright: ignore[reportCallIssue, reportArgumentType]
    )

    #: Derivatives object
    derivatives: Derivatives = field(validator=attv.instance_of(Derivatives))

    #: Maximum order of expansion (defaults to self.data.order)
    _order: int | None = field(default=None, alias="order")

    #: Whether to apply `X <- -log(X)`.
    minus_log: bool = field(
        kw_only=True,
        default=False,
        validator=attv.instance_of(bool),
    )
    #: Name of `alpha`
    alpha_name: str = field(kw_only=True, default="alpha")

    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict)

    @property
    def alpha0(self) -> float:
        return self._alpha0

    @property
    def order(self) -> int:
        if self._order is None:
            return self.data.order
        return self._order

    @cached.meth
    def _derivs(self, order: int, order_dim: str, minus_log: bool) -> DataT:
        return self.derivatives.derivs(
            data=self.data,
            order=order,
            norm=False,
            minus_log=minus_log,
            order_dim=order_dim,
        )

    def derivs(
        self,
        order: int | None = None,
        order_dim: str = "order",
        minus_log: bool | None = None,
        norm: bool = False,
    ) -> DataT:
        if minus_log is None:
            minus_log = self.minus_log
        if order is None:
            order = self.order
        out = self._derivs(order=order, order_dim=order_dim, minus_log=minus_log)
        if norm:
            return out * taylor_series_norm(order, order_dim)
        return out

    def coefs(
        self,
        order: int | None = None,
        order_dim: str = "order",
        minus_log: bool | None = None,
    ) -> DataT:
        return self.derivs(
            order=order, order_dim=order_dim, minus_log=minus_log, norm=True
        )

    def __call__(self, *args: Any, **kwargs: Any) -> DataT:
        return self.predict(*args, **kwargs)

    def predict(
        self,
        alpha: ArrayLike,
        *,
        order: int | None = None,
        order_dim: str = "order",
        cumsum: bool = False,
        no_sum: bool = False,
        minus_log: bool | None = None,
        alpha_name: str | None = None,
        dalpha_coords: str = "dalpha",
        alpha0_coords: str | bool = True,
    ) -> DataT:
        """
        Calculate taylor series at values "alpha".

        Parameters
        ----------
        alpha : float or sequence of DataArray
            Value of `alpha` to evaluate expansion at.
        order : int, optional
            Optional order to perform expansion to.
        order_dim : str, default="order"
            Name of dimension for new order dimension, if created.
        cumsum : bool, default=False
            If True, perform a cumsum on output for all orders.  Otherwise,
            to total sum.
        no_sum : bool, default=False
            If True, do not sum the results.  Useful if manually performing any
            math with series.
        minus_log : bool, default=False
            If True, transform expansion to ``Y = - log(X)``.
        alpha_name : str, optional
            Name to apply to created alpha dimension.
        dalpha_coords : str, default="dalpha"
            Name of coordinate ``dalpha = alpha - alpha0``.
        alpha0_coords : str or bool, default=True
            If True, add ``alpha_name`` + "0" to the coordinates of the results.
            If ``str``, use this as the alpha0 coordinate names.

        Returns
        -------
        output : DataArray or Dataset
        """
        if order is None:
            order = self.order

        if alpha_name is None:
            alpha_name = self.alpha_name

        coefs = self.coefs(order=order, order_dim=order_dim, minus_log=minus_log)

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        dalpha = alpha - self.alpha0
        p = xr.DataArray(np.arange(order + 1), dims=order_dim)
        prefac = dalpha**p

        # TODO(wpk): this should be an option, same for derivs
        coords = {}
        if dalpha_coords:
            coords[dalpha_coords] = dalpha

        if alpha0_coords:
            if not isinstance(alpha0_coords, str):
                alpha0_coords = alpha_name + "0"
            coords[alpha0_coords] = self.alpha0  # type: ignore[assignment]

        # coords = {"dalpha": dalpha, alpha_name + "0": self.alpha0}

        out = (prefac * coefs.sel({order_dim: prefac[order_dim]})).assign_coords(coords)

        if no_sum:
            pass
        elif cumsum:
            out = out.cumsum(order_dim)
        else:
            out = out.sum(order_dim)

        return out

    def resample(self, sampler: SamplerType, **kws: Any) -> Self:
        """Create new object with resampled data."""
        return self.new_like(
            order=self.order,
            alpha0=self.alpha0,
            derivatives=self.derivatives,
            data=self.data.resample(sampler=sampler, **kws),
            minus_log=self.minus_log,
            alpha_name=self.alpha_name,
        )


@attrs.define
class StateCollection(
    MyAttrsMixin,
    Sequence[SupportsModelDataT_co],
    Generic[DataT, SupportsModelDataT_co],
):
    """
    Sequence of models.

    Parameters
    ----------
    states : list
        list of states to consider
        Note that some subclasses require this list to be sorted
    kws : Mapping, optional
        additional key word arguments to keep internally in self.kws
    """

    states: Sequence[SupportsModelDataT_co] = field()
    kws: dict[str, Any] = field(
        kw_only=True,
        factory=dict[str, "Any"],
        converter=convert_mapping_or_none_to_dict,
    )

    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict)

    @override
    def __len__(self) -> int:
        return len(self.states)

    @overload
    def __getitem__(self, idx: SupportsIndex, /) -> SupportsModelDataT_co: ...
    @overload
    def __getitem__(self, idx: slice[Any, Any, Any], /) -> Self: ...

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]  # NOTE: adding slice functionality...
        self, idx: SupportsIndex | slice[Any, Any, Any], /
    ) -> SupportsModelDataT_co | Self:
        if isinstance(idx, slice):
            return type(self)(self.states[idx], kws=self.kws)
        return self.states[int(idx)]

    @property
    def alpha_name(self) -> str:
        return getattr(self[0], "alpha_name", "alpha")

    @property
    def order(self) -> int:
        return min(m.order for m in self)

    @property
    def alpha0(self) -> list[float]:
        return [m.alpha0 for m in self]

    def predict(self, alpha: ArrayLike, **kwargs: Any) -> DataT:  # pylint: disable=no-self-use
        """
        Prediction routine.

        To be implemented by subclasses.
        """
        msg = "To be implemented by subclass"
        raise NotImplementedError(msg)

    def coefs(self, order: int | None = None, **kwargs: Any) -> DataT:  # pylint: disable=no-self-use
        """
        Coefficients.

        To be implemented by subclass
        """
        msg = "To be implemented by subclass"
        raise NotImplementedError(msg)

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
            (np.ndarray, xr.DataArray, xr.Dataset, cmomy.IndexSampler, Mapping),
        ):
            sampler = [sampler] * len(self)
        elif not isinstance(sampler, Sized) or len(sampler) != len(self):
            msg = f"SamplerType must be a sized object with length {len(self)=}"
            raise ValueError(msg)

        return type(self)(
            states=tuple(
                state.resample(sampler=sampler, **kws)
                for state, sampler in zip(self.states, sampler)
            ),
            **self.kws,
        )

    def map(self, func: Callable[..., _T], *args: Any, **kwargs: Any) -> list[_T]:
        """
        Apply a function to elements self.
        ``out = [func(s, *args, **kwargs) for s in self]``.
        """
        return [func(s, *args, **kwargs) for s in self]

    def map_concat(
        self,
        func: Callable[..., DataT],
        *args: Any,
        concat_dim: Any = None,
        concat_kws: OptionalKwsAny = None,
        **kwargs: Any,
    ) -> DataT:
        """
        Apply function and concat output.

        defaults to concat with dim=pd.Index(self.alpha0, name=self.alpha_name)
        """
        out = self.map(func, *args, **kwargs)

        if concat_dim is None:
            concat_dim = pd.Index(self.alpha0, name=self.alpha_name)
        if concat_kws is None:
            concat_kws = {}
        return xr.concat(out, dim=concat_dim, **concat_kws)

    def append(
        self,
        states: Sequence[Any],
        sort: bool = True,
        key: Callable[..., Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Create new object with states appended to self.states.

        Parameters
        ----------
        states : list
            states to append to self.states
        sort : bool, default=True
            if true, sort states by key `alpha0`
        key : callable, optional
            callable function to use as key if ``sort=True``.
            Default is `lambda x: x.alpha0`
            see `sorted` function
        kws : dict
            extra arguments to `sorted`

        Returns
        -------
        out : object
            same type as `self` with new states added to `states` list
        """
        new_states = list(self.states) + list(states)

        if sort:
            if key is None:

                def key_func(x: Any) -> Any:
                    return x.alpha0

                key = key_func
            new_states = sorted(new_states, key=key, **kws)
        return type(self)(new_states, **self.kws)


def xr_weights_minkowski(
    deltas: xr.DataArray, m: int = 20, dim: str = "state"
) -> xr.DataArray:
    """Minkowski weights DataArray."""
    deltas_m = deltas**m
    return 1.0 - deltas_m / deltas_m.sum(dim)


@attrs.define
class PiecewiseStateCollection(StateCollection[DataT, SupportsModelDataT]):
    """Provide methods for Piecewise state collection."""

    def _indices_between_alpha(self, alpha: float) -> NDArray[np.int64]:
        idx = int(np.digitize(alpha, self.alpha0, right=False) - 1)
        if idx < 0:
            idx = 0
        elif idx == len(self) - 1:
            idx = len(self) - 2
        return np.array([idx, idx + 1], dtype=np.int64)

    def _indices_nearest_alpha(self, alpha: float) -> NDArray[np.int64]:
        dalpha = np.abs(np.array(self.alpha0) - alpha)
        # two lowest
        return np.argsort(dalpha)[:2]

    def _indices_alpha(self, alpha: float, method: str | None) -> NDArray[np.int64]:
        if method is None or method == "between":
            return self._indices_between_alpha(alpha)
        if method == "nearest":
            return self._indices_nearest_alpha(alpha)
        msg = f"unknown method {method}"
        raise ValueError(msg)

    def _states_alpha(
        self, alpha: float, method: str | None
    ) -> list[ExtrapModel[DataT]]:
        return [self[i] for i in self._indices_alpha(alpha, method)]


@attrs.define
@docfiller_shared.inherit(StateCollection)
class ExtrapWeightedModel(
    PiecewiseStateCollection[DataT, ExtrapModel[DataT]], Generic[DataT]
):
    """
    Weighted extrapolation model.

    Parameters
    ----------
    states : sequence of ExtrapModel
        Extrap models to consider.
    """

    @override
    def predict(
        self,
        alpha: ArrayLike,
        *,
        order: int | None = None,
        order_dim: str = "order",
        cumsum: bool = False,
        minus_log: bool | None = None,
        alpha_name: str | None = None,
        method: str | None = None,
        bounded: bool = False,
        **kwargs: Any,
    ) -> DataT:
        """
        Parameters
        ----------
        method : {None, 'between', 'nearest'}
            method to select which models are chosen to predict value for given
            value of alpha.

            - None or between: use states such that `state[i].alpha0 <= alpha < states[i+1]`
              if alpha < state[0].alpha0 use first two states
              if alpha > states[-1].alpha0 use last two states
            - nearest: use two states with minimum `abs(state[k].alpha0 - alpha)`

        Notes
        -----
        This requires that `states` are ordered in ascending `alpha0` order
        """
        alpha = validate_alpha(alpha, self.states if bounded else None)

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        if len(self) == 2:
            states = self.states

        else:
            # multiple states
            if alpha.ndim > 0:
                # have multiple alphas
                # recursively call
                return xr.concat(  # pyright: ignore[reportCallIssue]
                    (  # pyright: ignore[reportArgumentType]
                        self.predict(
                            alpha=a,
                            order=order,
                            order_dim=order_dim,
                            cumsum=cumsum,
                            minus_log=minus_log,
                            alpha_name=alpha_name,
                            method=method,
                            **kwargs,
                        )
                        for a in alpha
                    ),
                    dim=alpha_name,
                )
            states = self._states_alpha(
                float(alpha),
                method,
            )

        out = xr.concat(  # pyright: ignore[reportCallIssue]
            (  # pyright: ignore[reportArgumentType]
                m.predict(
                    alpha,
                    order=order,
                    order_dim=order_dim,
                    cumsum=cumsum,
                    minus_log=minus_log,
                    alpha_name=alpha_name,
                    **kwargs,
                )
                for m in states
            ),
            dim="state",
        )

        w = xr_weights_minkowski(np.abs(out.dalpha))
        return (out * w).sum("state") / w.sum("state")


@attrs.define
@docfiller_shared.inherit(StateCollection)
class InterpModel(StateCollection[DataT, SupportsModelDerivsDataT]):
    """Interpolation model."""

    @cached.meth
    def coefs(  # pylint: disable=arguments-differ
        self,
        /,
        order: int | None = None,
        *,
        order_dim: str = "porder",
        minus_log: bool | None = None,
        **kwargs: Any,
    ) -> DataT:
        from scipy.special import factorial as sp_factorial

        if order is None:
            order = self.order

        porder = len(self) * (order + 1) - 1

        # keep track of these to reconstruct index later
        states = []
        orders = []

        # construct mat[porder, porder]
        # by stacking
        mat_ = []
        power = np.arange(porder + 1)
        num = sp_factorial(np.arange(porder + 1))

        for istate, m in enumerate(self.states):
            alpha = m.alpha0
            for j in range(order + 1):
                with np.errstate(divide="ignore"):
                    val = (
                        (alpha ** (power - j))
                        * num
                        / sp_factorial(np.arange(porder + 1) - j)
                    )
                mat_.append(val)
                states.append(istate)
                orders.append(j)

        mat = np.array(mat_)
        mat[np.isinf(mat)] = 0.0

        mat_inv = (
            xr.DataArray(np.linalg.inv(mat), dims=[order_dim, "state_order"])
            .assign_coords(state=("state_order", states))
            .assign_coords(order=("state_order", orders))
            .set_index(state_order=["state", "order"])
            .unstack()
        )

        coefs = cast(
            "DataT",
            xr.concat(  # type: ignore[type-var]  # pyright: ignore[reportCallIssue]
                (  # pyright: ignore[reportArgumentType]
                    m.derivs(order, norm=False, minus_log=minus_log, order_dim="order")
                    for m in self.states
                ),
                dim="state",
            ),
        )
        return xr_dot(mat_inv, coefs)

    @override
    def predict(
        self,
        alpha: ArrayLike,
        *,
        order: int | None = None,
        order_dim: str = "porder",
        minus_log: bool = False,
        alpha_name: str | None = None,
        **kwargs: Any,
    ) -> DataT:
        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        coefs = self.coefs(order=order, order_dim=order_dim, minus_log=minus_log)
        alpha = xrwrap_alpha(alpha, name=alpha_name)

        porder = len(coefs[order_dim]) - 1

        p = xr.DataArray(np.arange(porder + 1), dims=order_dim)
        prefac = alpha**p

        return (prefac * coefs).sum(order_dim)


@docfiller_shared.inherit(StateCollection)
class InterpModelPiecewise(PiecewiseStateCollection[DataT, SupportsModelDerivsDataT]):
    """Apposed to the multiple model InterpModel, perform a piecewise interpolation."""

    # @cached.meth
    # def single_interpmodel(self, state0, state1):
    #     return InterpModel([state0, state1])
    @cached.meth
    def single_interpmodel(
        self, *state_indices: SupportsIndex
    ) -> InterpModel[DataT, SupportsModelDerivsDataT]:
        state0, state1 = (self[i] for i in state_indices)
        return InterpModel([state0, state1])

    @override
    def predict(
        self,
        alpha: ArrayLike,
        *,
        order: int | None = None,
        order_dim: str = "porder",
        minus_log: bool = False,
        alpha_name: str | None = None,
        method: str | None = None,
        bounded: bool = False,
        **kwargs: Any,
    ) -> DataT:
        """
        Parameters
        ----------
        alpha : float or sequence of float

        """
        alpha = validate_alpha(alpha, self.states if bounded else None)

        if alpha_name is None:
            alpha_name = self.alpha_name

        if len(self) == 2:
            # model = self.single_interpmodel(self[0], self[1])
            model = self.single_interpmodel(0, 1)

            return model.predict(
                alpha=alpha,
                order=order,
                order_dim=order_dim,
                minus_log=minus_log,
                alpha_name=alpha_name,
                **kwargs,
            )

        out: list[DataT] = []
        for a in np.atleast_1d(alpha):
            # state0, state1 = self._states_alpha(a, method)
            # model = self.single_interpmodel(state0, state1)
            model = self.single_interpmodel(
                *self._indices_alpha(alpha=a, method=method)
            )

            out.append(
                model.predict(
                    alpha=a,
                    order=order,
                    order_dim=order_dim,
                    minus_log=minus_log,
                    alpha_name=alpha_name,
                    **kwargs,
                )
            )

        return out[0] if len(out) == 1 else xr.concat(out, dim=alpha_name)  # pyright: ignore[reportCallIssue, reportArgumentType]


# TODO(wpk): rework this to only take uv/xv/rec_dim
@attrs.define
class PerturbModel(MyAttrsMixin, Generic[DataT]):
    """Perturbation model."""

    alpha0: float = field(converter=float)
    data: SupportsDataXU[DataT] = field(
        validator=attv.instance_of(SupportsDataXU)  # type: ignore[type-abstract]  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    alpha_name: str = field(
        default="alpha",
    )

    def predict(
        self, alpha: ArrayLike, *, alpha_name: str | None = None, **kwargs: Any
    ) -> DataT:
        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        uv = self.data.uv
        xv = self.data.xv

        alpha0 = self.alpha0

        rec_dim = self.data.rec_dim
        dalpha = alpha - alpha0

        dalpha_uv = (-1.0) * dalpha * uv
        dalpha_uv_diff = dalpha_uv - dalpha_uv.max(rec_dim)
        expvals = cast("xr.DataArray", np.exp(dalpha_uv_diff))

        num = xr_dot(expvals, xv, dim=rec_dim) / len(xv[rec_dim])
        den = expvals.mean(rec_dim)

        return num / den

    def resample(self, sampler: SamplerType, **kws: Any) -> Self:
        return self.__class__(
            alpha0=self.alpha0,
            data=self.data.resample(sampler=sampler, **kws),
            alpha_name=self.alpha_name,
        )


@attrs.define
@docfiller_shared.inherit(StateCollection)
class MBARModel(StateCollection[xr.DataArray, Any]):
    """Sadly, this doesn't work as beautifully."""

    def __attrs_pre_init__(self) -> None:  # pylint: disable=bad-dunder-name,no-self-use
        if not module_available("pymbar"):
            msg = "need pymbar to use this"
            raise ImportError(msg)

    @cached.meth
    def _default_params(
        self, state_dim: str = "state", alpha_name: Hashable = "alpha"
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, MBAR]:
        import pymbar

        # all xvalues:
        xv = xr.concat([m.data.xv for m in self], dim=state_dim)
        uv = xr.concat([m.data.uv for m in self], dim=state_dim)
        alpha0 = xrwrap_alpha([m.alpha0 for m in self], name=alpha_name)

        # make sure uv, xv in correct orde
        rec_dim = self[0].data.rec_dim  # pylint: disable=no-member
        xv = xv.transpose(state_dim, rec_dim, ...)
        uv = uv.transpose(state_dim, rec_dim, ...)

        # alpha[alpha] * uv[state, rec_dim] = out[alpha, state, rec_dim]
        ukn = (alpha0 * uv).values.reshape(len(self), -1)
        n = np.ones(len(self)) * len(xv[rec_dim])
        mbar_obj = pymbar.mbar.MBAR(ukn, n)

        return uv, xv, alpha0, mbar_obj

    @override
    def predict(
        self, alpha: ArrayLike, *, alpha_name: str | None = None, **kwargs: Any
    ) -> xr.DataArray:
        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        if alpha.ndim == 0:
            alpha = alpha.expand_dims(alpha.name)

        uv, xv, _alpha0, mbar_obj = self._default_params("state", alpha.name)

        dims = xv.dims
        x = xv.to_numpy()
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)
        u = uv.to_numpy().reshape(-1)

        out = np.array(
            [
                mbar_obj.compute_multiple_expectations(x_flat.T, b * u)["mu"]
                for b in alpha.values
            ]
        )

        # reshape
        shape = (out.shape[0], *x.shape[2:])
        return xr.DataArray(
            out.reshape(shape), dims=(alpha.name, *dims[2:])
        ).assign_coords(alpha=alpha)

    @override
    def resample(self, *args: Any, **kwargs: Any) -> Self:
        msg = "resample not implemented for this class"
        raise NotImplementedError(msg)

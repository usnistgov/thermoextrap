"""
General extrapolation/interpolation models (:mod:`~thermoextrap.models`)
========================================================================
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

import attrs
import cmomy
import numpy as np
import pandas as pd
import xarray as xr
from attrs import converters as attc
from attrs import field
from attrs import validators as attv
from module_utilities import cached

from .core._attrs_utils import (
    MyAttrsMixin,
    convert_mapping_or_none_to_dict,
)
from .core._imports import has_pymbar
from .core._imports import sympy as sp
from .core.sputils import get_default_indexed, get_default_symbol
from .core.xrutils import xrwrap_alpha
from .data import AbstractData
from .docstrings import DOCFILLER_SHARED

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from cmomy.core.typing import Sampler

    from thermoextrap.core.typing_compat import Self

docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap")


__all__ = [
    "Derivatives",
    "ExtrapModel",
    "ExtrapWeightedModel",
    "InterpModel",
    "InterpModelPiecewise",
    "MBARModel",
    "PerturbModel",
    "StateCollection",
]


################################################################################
# Structure(s) to deal with analytic derivatives, etc
################################################################################
class SymFuncBase(sp.Function):
    """
    Base class to define a sympy function for user defined derivatives.


    See Also
    --------
    :class:`thermoextrap.models.SymDerivBase`

    """

    @classmethod
    def deriv_args(cls) -> None:
        """
        Symbol arguments of function.

        This is used by Data class to create a 'lambdfied' callable function.

        See Also
        --------
        sympy.utilities.lambdify.lambdify
        """
        msg = "must specify in subclass"
        raise NotImplementedError(msg)

    def fdiff(self, argindex=1) -> None:
        """Derivative of function.  This will be used by :class:`thermoextrap.models.SymDerivBase`."""
        msg = "must specify in subclass"
        raise NotImplementedError(msg)

    @classmethod
    def eval(cls, beta) -> None:
        """
        Evaluate function.

        We use the convention of passing in `beta='None'` to evaluate the
        function to an indexed variable.
        """
        msg = "must specify in subclass"
        raise NotImplementedError(msg)


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

    def __init__(self, func, args=None, expand=True, post_func=None) -> None:
        if args is None:
            args = func.deriv_args()

        self._func_orig = func
        self._post_func = post_func

        if post_func is not None:
            if isinstance(post_func, str):
                if post_func == "minus_log":
                    post_func = lambda f: -sp.log(f)
                elif post_func.startswith("pow_"):
                    i = int(post_func.split("_")[-1])
                    post_func = lambda f: pow(f, i)
                else:
                    msg = "post_func must be callable or in {minus_log, pow_1, pow_2, ...}"
                    raise ValueError(msg)
            func = post_func(func)

        self.func = func
        self.args = args
        self.expand = expand
        self._cache: dict[str, Any] = {}

    @cached.meth
    def __getitem__(self, order):
        if order == 0:
            out = self.func
        else:
            out = self[order - 1].diff(self.beta, 1)
            if self.expand:
                out = out.expand()
        return out


@attrs.define
class SymSubs:
    """
    Class to handle substitution on :class:`thermoextrap.models.SymDerivBase`.

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

    funcs: Sequence[sp.Function] = field()
    subs: Sequence | None = field(default=None)
    subs_final: Sequence | None = field(default=None)
    subs_all: Mapping | None = field(default=None)
    recursive: bool = field(default=True)
    simplify: bool = field(default=False)
    expand: bool = field(default=True)

    _cache: dict = field(init=False, repr=False, factory=dict)

    @cached.meth
    def __getitem__(self, order):
        func = self.funcs[order]

        if self.subs is not None:
            if self.recursive:
                for o in range(order, -1, -1):
                    func = func.subs(self.subs[o])
            else:
                func = func.subs(self.subs[order])

        if self.subs_final is not None:
            func = func.subs(self.subs_final[order])

        if self.subs_all is not None:
            func = func.subs(self.subs_all)

        if self.simplify:
            func = func.simplify()

        if self.expand:
            func = func.expand()

        return func


@attrs.define
class Lambdify:
    """
    Create python function from list of expressions.

    Parameters
    ----------
    exprs : sequence of symFunction
        array of sympy expressions to ``lambdify``
    args : sequence of Symbol
        array of symbols which will be in args of the resulting function
    lambdify_kws : dict
        extra arguments to ``lambdify``

    See Also
    --------
    sympy.utilities.lambdify.lambdify
    """

    exprs: Sequence[sp.Function] = field()
    args: Sequence | None = field(default=None)
    lambdify_kws: Mapping | None = field(
        kw_only=True,
        default=None,
        converter=attc.default_if_none(factory=dict),
    )

    _cache: dict = field(init=False, repr=False, factory=dict)

    @cached.meth
    def __getitem__(self, order):
        return sp.lambdify(self.args, self.exprs[order], **self.lambdify_kws)

    @classmethod
    def from_u_xu(cls, exprs, **lambdify_kws):
        """Factory for u/xu args."""
        u, xu = get_default_indexed("u", "xu")
        # args = (u, xu)
        return cls(exprs=exprs, args=(u, xu), lambdify_kws=lambdify_kws)

    @classmethod
    def from_du_dxdu(cls, exprs, xalpha=False, **lambdify_kws):
        """Factory for du/dxdu args."""
        x1 = get_default_indexed("x1") if xalpha else get_default_symbol("x1")
        du, dxdu = get_default_indexed("du", "dxdu")
        return cls(exprs=exprs, args=(x1, du, dxdu), lambdify_kws=lambdify_kws)


# -log<X>
class SymMinusLog:
    """Class to compute derivatives of Y = -log(<X>)."""

    X, dX = get_default_indexed("X", "dX")

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    @cached.meth
    def __getitem__(self, order):
        if order == 0:
            return -sp.log(self.X[0])

        expr = 0
        for k in range(1, order + 1):
            expr += (
                sp.factorial(k - 1) * (-1 / self.X[0]) ** k * sp.bell(order, k, self.dX)
            )
        # subber
        subs = {self.dX[j]: self.X[j + 1] for j in range(order + 1)}
        return expr.subs(subs).expand().simplify()


@lru_cache(5)
def factory_minus_log():
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
    funcs: Sequence[Callable] = field()
    #: Sequence of sympy expressions, optional
    exprs: Sequence[sp.Function] | None = field(kw_only=True, default=None)
    #: Arguments
    args: Sequence | None = field(kw_only=True, default=None)

    @staticmethod
    def _apply_minus_log(X, order):
        func = factory_minus_log()
        return [func[i](X) for i in range(order + 1)]

    def derivs(
        self,
        data=None,
        order=None,
        args=None,
        minus_log=False,
        order_dim="order",
        concat_kws=None,
        norm=False,
    ):
        """
        Calculate derivatives for orders range(0, order+1).

        Parameters
        ----------
        data : object
            Data object.
            If passed, use `args=data.derivs_args`
        order : int, optional
            If pass `data` and `order` is `None`, then `order=data.order`
            Otherwise, must mass order
        args : tuple
            arguments passed to ``self.funcs[i](*args)``
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
        if data is not None:
            args = data.derivs_args
            if order is None:
                order = data.order

        if args is None:
            msg = "must specify args or data"
            raise ValueError(msg)

        if order is None:
            msg = "must specify order or data"
            raise ValueError(msg)

        out = [self.funcs[i](*args) for i in range(order + 1)]

        if minus_log:
            out = self._apply_minus_log(X=out, order=order)

        if norm:
            out = [x / math.factorial(i) for i, x in enumerate(out)]

        if order_dim is not None:
            if concat_kws is None:
                concat_kws = {}
            out = xr.concat(out, dim=order_dim, **concat_kws)
        return out

    def coefs(
        self, data=None, args=None, order=None, minus_log=False, order_dim="order"
    ):
        """
        Alias to `self.derivs(..., norm=True)`.

        See Also
        --------
        derivs
        """
        return self.derivs(
            data=data,
            args=args,
            order=order,
            minus_log=minus_log,
            order_dim=order_dim,
            norm=True,
        )

    @classmethod
    def from_sympy(cls, exprs, args):
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
def taylor_series_norm(order, order_dim="order"):
    """``taylor_series_coefficients = derivs * taylor_series_norm``."""
    out = np.array([1 / math.factorial(i) for i in range(order + 1)])
    if order_dim is not None:
        out = xr.DataArray(out, dims=order_dim)
    return out


@attrs.define
class ExtrapModel(MyAttrsMixin):
    """Apply taylor series extrapolation."""

    #: Alpha value data is evaluated at
    alpha0: float = field(converter=float)

    #: Data object
    data: AbstractData = field(validator=attv.instance_of(AbstractData))

    #: Derivatives object
    derivatives: Derivatives = field(validator=attv.instance_of(Derivatives))

    #: Maximum order of expansion
    order: int | None = field(default=attrs.Factory(lambda self: self.data.order))
    #: Whether to apply `X <- -log(X)`.
    minus_log: bool | None = field(
        kw_only=True,
        default=False,
        converter=attc.default_if_none(False),
    )
    #: Name of `alpha`
    alpha_name: str = field(kw_only=True, default="alpha", converter=str)

    _cache: dict = field(init=False, repr=False, factory=dict)

    @cached.meth
    def _derivs(self, order, order_dim, minus_log):
        return self.derivatives.derivs(
            data=self.data,
            order=order,
            norm=False,
            minus_log=minus_log,
            order_dim=order_dim,
        )

    def derivs(self, order=None, order_dim="order", minus_log=None, norm=False):
        if minus_log is None:
            minus_log = self.minus_log
        if order is None:
            order = self.order
        out = self._derivs(order=order, order_dim=order_dim, minus_log=minus_log)
        if norm:
            return out * taylor_series_norm(order, order_dim)
        return out

    def coefs(self, order=None, order_dim="order", minus_log=None):
        return self.derivs(
            order=order, order_dim=order_dim, minus_log=minus_log, norm=True
        )

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(
        self,
        alpha,
        order=None,
        order_dim="order",
        cumsum=False,
        no_sum=False,
        minus_log=None,
        alpha_name=None,
        dalpha_coords="dalpha",
        alpha0_coords=True,
    ):
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
        alpha0_coords : bool, default=True
            If True, add ``alpha0`` to the coordinates of the results.

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
        if dalpha_coords is not None:
            coords[dalpha_coords] = dalpha

        if alpha0_coords:
            if not isinstance(alpha0_coords, str):
                alpha0_coords = alpha_name + "0"
            coords[alpha0_coords] = self.alpha0

        # coords = {"dalpha": dalpha, alpha_name + "0": self.alpha0}

        out = (prefac * coefs.sel(**{order_dim: prefac[order_dim]})).assign_coords(
            **coords
        )

        if no_sum:
            pass
        elif cumsum:
            out = out.cumsum(order_dim)
        else:
            out = out.sum(order_dim)

        return out

    def resample(self, sampler: Sampler, **kws):
        """Create new object with resampled data."""
        return self.new_like(
            order=self.order,
            alpha0=self.alpha0,
            derivatives=self.derivatives,
            data=self.data.resample(sampler=sampler, **kws),
            minus_log=self.minus_log,
            alpha_name=self.alpha_name,
        )


# TODO(wpk): rename StateCollection to ModelSequence?
@attrs.define
class StateCollection(MyAttrsMixin):
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

    states: Sequence = field()
    kws: dict[str, Any] = field(
        kw_only=True, converter=convert_mapping_or_none_to_dict, default=None
    )

    _cache: dict = field(init=False, repr=False, factory=dict)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    @property
    def alpha_name(self):
        return getattr(self[0], "alpha_name", "alpha")

    def resample(self, sampler: Sampler | Sequence[Sampler], **kws: Any) -> Self:
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
        elif len(sampler) != len(self):
            msg = f"{len(sampler)=} must equal {len(self)=}"
            raise ValueError(msg)

        return type(self)(
            states=tuple(
                state.resample(sampler=sampler, **kws)
                for state, sampler in zip(self.states, sampler)
            ),
            **self.kws,
        )

    def map(self, func, *args, **kwargs):
        """
        Apply a function to elements self.
        ``out = [func(s, *args, **kwargs) for s in self]``.

        if func is a str, then
        ``out = [getattr(s, func)(*args, **kwargs) for s in self]``
        """
        if isinstance(func, str):
            out = [getattr(s, func)(*args, **kwargs) for s in self]
        else:
            out = [func(s, *args, **kwargs) for s in self]

        return out

    def map_concat(self, func, concat_dim=None, concat_kws=None, *args, **kwargs):
        """
        Apply function and concat output.

        defaults to concat with dim=pd.Index(self.alpha0, name=self.alpha_name)
        """
        out = self.map(func, *args, **kwargs)
        if isinstance(out[0], (xr.DataArray, xr.Dataset)):
            if concat_dim is None:
                concat_dim = pd.Index(self.alpha0, name=self.alpha_name)
            if concat_kws is None:
                concat_kws = {}
            out = xr.concat(out, dim=concat_dim, **concat_kws)
        return out

    def append(self, states, sort=True, key=None, **kws):
        """
        Create new object with states appended to self.states.

        Parameters
        ----------
        states : list
            states to append to self.states
        sort : bool, default=True
            if true, sort states by key `alpha0`
        key : callable, optional
            callable function to use as key.
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
                key = lambda x: x.alpha0
            new_states = sorted(new_states, key=key, **kws)
        return type(self)(new_states, **self.kws)

    @property
    def order(self):
        return min(m.order for m in self)

    @property
    def alpha0(self):
        return [m.alpha0 for m in self]

    def _check_alpha(self, alpha, bounded=False) -> None:
        if bounded:
            try:
                seq = iter(alpha)
            except TypeError:
                seq = [alpha]

            lb, ub = self[0].alpha0, self[-1].alpha0

            for a in seq:
                if a < lb or a > ub:
                    msg = f"{a} outside of bounds [{lb}, {ub}]"
                    raise ValueError(msg)


def xr_weights_minkowski(deltas, m=20, dim="state"):
    deltas_m = deltas**m
    return 1.0 - deltas_m / deltas_m.sum(dim)


@attrs.define
class PiecewiseMixin:
    """Provide methods for Piecewise state collection."""

    def _indices_between_alpha(self, alpha):
        idx = np.digitize(alpha, self.alpha0, right=False) - 1
        if idx < 0:
            idx = 0
        elif idx == len(self) - 1:
            idx = len(self) - 2
        return [idx, idx + 1]

    def _indices_nearest_alpha(self, alpha):
        dalpha = np.abs(np.array(self.alpha0) - alpha)
        # two lowest
        return np.argsort(dalpha)[:2]

    def _indices_alpha(self, alpha, method):
        if method is None or method == "between":
            return self._indices_between_alpha(alpha)
        if method == "nearest":
            return self._indices_nearest_alpha(alpha)
        msg = f"unknown method {method}"
        raise ValueError(msg)

    def _states_alpha(self, alpha, method):
        return [self[i] for i in self._indices_alpha(alpha, method)]


@attrs.define
@docfiller_shared.inherit(StateCollection)
class ExtrapWeightedModel(StateCollection, PiecewiseMixin):
    """
    Weighted extrapolation model.

    Parameters
    ----------
    states : sequence of ExtrapModel
        Extrap models to consider.
    """

    states: Sequence[ExtrapModel] = field()

    def predict(
        self,
        alpha,
        order=None,
        order_dim="order",
        cumsum=False,
        minus_log=None,
        alpha_name=None,
        method=None,
        bounded=False,
    ):
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
        self._check_alpha(alpha, bounded)

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        if len(self) == 2:
            states = self.states

        else:
            # multiple states
            if np.array(alpha).ndim > 0:
                # have multiple alphas
                # recursively call
                return xr.concat(
                    (
                        self.predict(
                            alpha=a,
                            order=order,
                            order_dim=order_dim,
                            cumsum=cumsum,
                            minus_log=minus_log,
                            alpha_name=alpha_name,
                            method=method,
                        )
                        for a in alpha
                    ),
                    dim=alpha_name,
                )

            states = self._states_alpha(alpha, method)

        out = xr.concat(
            [
                m.predict(
                    alpha,
                    order=order,
                    order_dim=order_dim,
                    cumsum=cumsum,
                    minus_log=minus_log,
                    alpha_name=alpha_name,
                )
                for m in states
            ],
            dim="state",
        )

        w = xr_weights_minkowski(np.abs(out.dalpha))
        return (out * w).sum("state") / w.sum("state")


@attrs.define
@docfiller_shared.inherit(StateCollection)
class InterpModel(StateCollection):
    """Interpolation model."""

    @cached.meth
    def coefs(self, order=None, order_dim="porder", minus_log=None):
        from scipy.special import factorial as sp_factorial

        if order is None:
            order = self.order

        porder = len(self) * (order + 1) - 1

        # keep track of these to reconstruct index later
        states = []
        orders = []

        # construct mat[porder, porder]
        # by stacking
        mat = []
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
                mat.append(val)
                states.append(istate)
                orders.append(j)

        mat = np.array(mat)
        mat[np.isinf(mat)] = 0.0

        mat_inv = np.linalg.inv(mat)
        mat_inv = (
            xr.DataArray(mat_inv, dims=[order_dim, "state_order"])
            .assign_coords(state=("state_order", states))
            .assign_coords(order=("state_order", orders))
            .set_index(state_order=["state", "order"])
            .unstack()
        )

        coefs = xr.concat(
            [
                m.derivs(order, norm=False, minus_log=minus_log, order_dim="order")
                for m in self.states
            ],
            dim="state",
        )
        if isinstance(coefs, xr.Dataset):
            coefs = xr.Dataset({k: xr.dot(mat_inv, v) for k, v in coefs.items()})
        else:
            coefs = xr.dot(mat_inv, coefs)

        return coefs

    def predict(
        self, alpha, order=None, order_dim="porder", minus_log=None, alpha_name=None
    ):
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
class InterpModelPiecewise(StateCollection, PiecewiseMixin):
    """Apposed to the multiple model InterpModel, perform a piecewise interpolation."""

    # @cached.meth
    # def single_interpmodel(self, state0, state1):
    #     return InterpModel([state0, state1])
    @cached.meth
    def single_interpmodel(self, *state_indices):
        state0, state1 = (self[i] for i in state_indices)
        return InterpModel([state0, state1])

    def predict(
        self,
        alpha,
        order=None,
        order_dim="porder",
        minus_log=None,
        alpha_name=None,
        method=None,
        bounded=False,
    ):
        """
        Parameters
        ----------
        alpha : float or sequence of float

        """
        self._check_alpha(alpha, bounded)

        if alpha_name is None:
            alpha_name = self.alpha_name

        if len(self) == 2:
            # model = self.single_interpmodel(self[0], self[1])
            model = self.single_interpmodel(0, 1)

            out = model.predict(
                alpha=alpha,
                order=order,
                order_dim=order_dim,
                minus_log=minus_log,
                alpha_name=alpha_name,
            )

        else:
            try:
                seq = iter(alpha)
            except TypeError:
                seq = [alpha]

            out = []
            for a in seq:
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
                    )
                )

            out = out[0] if len(out) == 1 else xr.concat(out, dim=alpha_name)

        return out


@attrs.define
class PerturbModel(MyAttrsMixin):
    """Perturbation model."""

    alpha0: float = field(converter=float)
    data: AbstractData = field(validator=attv.instance_of(AbstractData))
    alpha_name: str | None = field(
        default="alpha", converter=attc.default_if_none("alpha")
    )

    def predict(self, alpha, alpha_name=None):
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
        expvals = np.exp(dalpha_uv_diff)

        num = xr.dot(expvals, xv, dims=rec_dim) / len(xv[rec_dim])
        den = expvals.mean(rec_dim)

        return num / den

    def resample(self, sampler: Sampler, **kws):
        return self.__class__(
            alpha0=self.alpha0,
            data=self.data.resample(sampler=sampler, **kws),
            alpha_name=self.alpha_name,
        )


@attrs.define
@docfiller_shared.inherit(StateCollection)
class MBARModel(StateCollection):
    """Sadly, this doesn't work as beautifully."""

    def __attrs_pre_init__(self):
        if not has_pymbar():
            msg = "need pymbar to use this"
            raise ImportError(msg)

    @cached.meth
    def _default_params(self, state_dim="state", alpha_name="alpha"):
        import pymbar

        # all xvalues:
        xv = xr.concat([m.data.xv for m in self], dim=state_dim)
        uv = xr.concat([m.data.uv for m in self], dim=state_dim)
        alpha0 = xrwrap_alpha([m.alpha0 for m in self], name=alpha_name)

        # make sure uv, xv in correct orde
        rec_dim = self[0].data.rec_dim
        xv = xv.transpose(state_dim, rec_dim, ...)
        uv = uv.transpose(state_dim, rec_dim, ...)

        # alpha[alpha] * uv[state, rec_dim] = out[alpha, state, rec_dim]
        ukn = (alpha0 * uv).values.reshape(len(self), -1)
        n = np.ones(len(self)) * len(xv[rec_dim])
        mbar_obj = pymbar.mbar.MBAR(ukn, n)

        return uv, xv, alpha0, mbar_obj

    def predict(self, alpha, alpha_name=None):
        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        if alpha.ndim == 0:
            alpha = alpha.expand_dims(alpha.name)

        uv, xv, _alpha0, mbar_obj = self._default_params("state", alpha.name)

        dims = xv.dims
        x = np.array(xv, order="c")
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)

        u = uv.values.reshape(-1)

        out = np.array(
            [
                mbar_obj.compute_multiple_expectations(x_flat.T, b * u)["mu"]
                for b in alpha.values
            ]
        )

        # reshape
        shape = (out.shape[0],) + x.shape[2:]
        return xr.DataArray(
            out.reshape(shape), dims=(alpha.name,) + dims[2:]
        ).assign_coords(alpha=alpha)

    def resample(self, *args, **kwargs) -> None:
        msg = "resample not implemented for this class"
        raise NotImplementedError(msg)

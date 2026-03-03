# pyright: reportMissingTypeStubs=false
# pylint: disable=duplicate-code
r"""
Inverse temperature expansion of macrostate distribution (:mod:`~thermoextrap.lnpi`)
====================================================================================

This is used to extrapolate, in inverse temperature :math:`\beta = (k_{\rm B} T)^{-1}`, the macrostate distribution function :math:`\ln\Pi` from transition matrix Monte Carlo simulations.

See :ref:`examples/usage/basic/macrostate_dist_extrap:macrostate distribution extrapolation` for example usage.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Generic

import attrs
import cmomy
import numpy as np
import xarray as xr

# from attrs import converters as attc
from attrs import field
from attrs import validators as attv
from module_utilities import cached

from . import beta as beta_xpan
from .core._attrs_utils import convert_dims_to_tuple
from .core.docstrings import DOCFILLER_SHARED
from .core.sputils import get_default_indexed, get_default_symbol
from .core.typing import DataT
from .core.typing_compat import override
from .core.validate import validator_xarray_typevar
from .data import DataCallbackABC
from .models import Derivatives, ExtrapModel, SymFuncBase

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    from cmomy import IndexSampler
    from sympy.core.expr import Expr
    from sympy.core.numbers import Number
    from sympy.core.symbol import Symbol
    from sympy.tensor.indexed import IndexedBase

    from .core.typing import DataDerivArgs, OptionalKwsAny, PostFunc, SupportsData
    from .core.typing_compat import Self

docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap", "beta").decorate


################################################################################
# lnPi correction stuff
################################################################################
class lnPi_func_central(SymFuncBase):
    r"""
    Special case of u_func_central.

    For lnPi, have:

    .. math::

        \newcommand{\ave}[1]{\langle #1 \rangle}

        (\ln \Pi)' = \frac{d \ln \Pi}{d \beta} = \mu N - \ave{u} + \ave{u - \mu N}_{\rm GC}

    where :math:`\ave{}` and :math:`\ave{}_{\rm GC}` are the canonical and grand canonical (GC) ensemble averages.
    We ignore the GC average term, as it does not depend on N.  Note that this is not
    necessarily the case for molecular systems.
    So, the first derivative of this function is :func:`thermoextrap.beta.u_func_central`.
    We consider only a correction of the form:

    .. math::

        (\ln\Pi)_{\text{energy}} = \ln\Pi - \beta \mu N = \ln Q - \ln \Xi

    where :math:`Q\text{ and }\Xi` are the canonical and GC partition functions, respectively. thus,

    .. math::

        \begin{align*}
          (\ln\Pi)_{\text{energy}}'  &= - U \\
          (\ln\Pi)_{\text{energy}}'' &=  -U' \\
              &\,\,\vdots
        \end{align*}

    """

    nargs = 1
    lnPi0 = get_default_symbol("lnPi0")
    mudotN = get_default_symbol("mudotN")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[Symbol, IndexedBase, Symbol, Symbol]:
        return (*beta_xpan.u_func_central.deriv_args(), cls.lnPi0, cls.mudotN)

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        (beta,) = self.args
        return self.mudotN - beta_xpan.u_func_central.tcall(beta)

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        return self.lnPi0

    @classmethod
    @override
    def eval(cls, beta: Symbol) -> Any:  # noqa: ARG003
        return None

    @classmethod
    @override
    def tcall(cls, beta: Symbol) -> Expr:
        return cls(beta)


class lnPi_func_raw(SymFuncBase):
    """Raw moments version."""

    nargs = 1
    u = get_default_indexed("u")
    lnPi0 = get_default_symbol("lnPi0")
    mudotN = get_default_symbol("mudotN")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[IndexedBase, Symbol, Symbol]:
        return (*beta_xpan.u_func.deriv_args(), cls.lnPi0, cls.mudotN)

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        (beta,) = self.args
        return self.mudotN - beta_xpan.u_func.tcall(beta, n=1)

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        return self.lnPi0

    @classmethod
    @override
    def eval(cls, beta: Symbol) -> Any:  # noqa: ARG003
        return None

    @classmethod
    @override
    def tcall(cls, beta: Symbol) -> Expr:
        return cls(beta)


@lru_cache(5)
@docfiller_shared
def factory_derivatives(
    name: str = "lnPi",
    n: int | None = None,
    d: int | None = None,
    xalpha: bool = False,
    central: bool = False,
    expand: bool = True,
    post_func: PostFunc = None,
) -> Derivatives:
    """
    Expansion for ln(Pi/Pi_0) (ignore bad parts of stuff).

    Parameters
    ----------
    name : str, default='lnPi'
        If name is `'lnPi'`, then get derivatives of lnPi.
        Otherwise, get derivative object for general `X`.
    {n_order}
    {d_order}
    {xalpha}
    {central}
    {expand}
    {post_func}

    Returns
    -------
    ~thermoextrap.models.Derivatives

    See Also
    --------
    thermoextrap.beta.factory_derivatives
    """
    if name == "lnPi":
        beta = get_default_symbol("beta")
        func = lnPi_func_central(beta) if central else lnPi_func_raw(beta)
        derivs = beta_xpan.SymDerivBeta(func=func, expand=expand, post_func=post_func)

        return Derivatives.from_sympy(derivs.doit(), args=derivs.args)
    return beta_xpan.factory_derivatives(
        name=name,
        n=n,
        d=d,
        xalpha=xalpha,
        central=central,
        post_func=post_func,
        expand=expand,
    )


def _convert_ncoords(
    ncoords: xr.DataArray | None, self_: attrs.AttrsInstance
) -> xr.DataArray:
    if ncoords is not None:
        return ncoords

    if not isinstance(self_, lnPiDataCallback):
        msg = "Converter only works with `lnPiDataCallback`"
        raise TypeError(msg)

    ncoords_ = np.meshgrid(
        *tuple(self_.lnPi0[x].to_numpy() for x in self_.dims_n),  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        indexing="ij",
    )
    return xr.DataArray(np.array(ncoords_), dims=(self_.dims_comp, *self_.dims_n))


@attrs.frozen
class lnPiDataCallback(DataCallbackABC, Generic[DataT]):
    """
    Class to handle metadata callbacks for lnPi data.

    Parameters
    ----------
    lnPi0 : DataArray
        Reference value of lnPi.
    mu : DataArray
        Value of chemical potential.  Must have dimension ``dims_comp``.
    dims_n : hashable or sequence of hashable
        Dimension(s) for number of particle(s).  That is, the dimensions of lnPi0 corresponding to particle number.
    dims_comp : hashable
        Dimension corresponding to components.
    ncoords : DataArray, optional
        Count of number of particles for given particle number (vector) and component.
        Must have dimensions ``dims_comp`` and ``dims_n``.
    allow_resample : bool, default=False
        If True, allow simplified resampling of ``lnPi0`` data.

    """

    # TODO(wpk): rename dims_comp to dim_comp.

    #: lnPi data
    lnPi0: DataT = field(validator=validator_xarray_typevar)
    #: Chemical potential
    mu: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Dimensions for particle number
    dims_n: tuple[Hashable, ...] = field(converter=convert_dims_to_tuple)
    #: Dimensions for component
    dims_comp: Hashable = field()
    #: Particle number coordinates
    ncoords: xr.DataArray = field(
        converter=attrs.Converter(_convert_ncoords, takes_self=True),  # type: ignore[misc]
        validator=attv.instance_of(xr.DataArray),
        default=None,
    )
    #: Flag to allow/disallow resampling of ``lnPi0``.
    allow_resample: bool = field(default=False)

    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict[str, "Any"])
    # TODO(wpk): using dims_n, dims_comp naming because this is what is used in lnPi module

    @override
    def check(self, data: SupportsData[Any]) -> None:
        pass

    @property
    def lnPi0_ave(self) -> DataT:
        return self.lnPi0

    @cached.prop
    def mudotN(self) -> xr.DataArray:
        """Dot product of `self.mu` and `self.ncoords`, reduces along `self.dims_comp`."""
        from .core.compat import xr_dot

        return xr_dot(self.mu, self.ncoords, dim=self.dims_comp)

    @override
    def resample(
        self,
        data: SupportsData[Any],
        *,
        meta_kws: OptionalKwsAny,
        sampler: IndexSampler[Any],
        **kws: Any,
    ) -> Self:
        """Resample lnPi0 data."""
        if not self.allow_resample:
            msg = (
                "Must set `self.allow_resample` to `True` to use resampling. "
                "Resampling here is handled in an ad-hoc way, and should be "
                "used with care."
            )
            raise ValueError(msg)

        warnings.warn(
            "'Correct' resampling of lnPi should be handled externally. "
            "This resamples the average lnPi values.  Instead, it is "
            "recommended to resample based on collection matrices, and "
            "construct lnPi values based on these.",
            category=UserWarning,
            stacklevel=2,
        )

        # wrap in xarray object:
        dc = cmomy.wrap_reduce_vals(
            self.lnPi0.expand_dims(dim="_new", axis=0),
            mom=1,
            dim="_new",
            mom_dims="_mom",
        )
        # resample and reduce
        dc = dc.resample_and_reduce(sampler=sampler, **kws)
        # return new object
        return self.new_like(lnPi0=dc.obj.sel(_mom=1, drop=True))  # pyright: ignore[reportUnknownMemberType]

    @override
    def deriv_args(
        self, data: SupportsData[Any], *, deriv_args: DataDerivArgs
    ) -> DataDerivArgs:
        return (*tuple(deriv_args), self.lnPi0_ave, self.mudotN)


# Much more likely to have pre-aves here, but save that for the user
@docfiller_shared
def factory_extrapmodel_lnPi(
    beta: float,
    data: SupportsData[DataT],
    *,
    central: bool | None = None,
    order: int | None = None,
    alpha_name: str = "beta",
    derivatives: Derivatives | None = None,
    post_func: PostFunc = None,
    derivatives_kws: OptionalKwsAny = None,
) -> ExtrapModel[DataT]:
    """
    Factory function to create Extrapolation model for beta expansion.

    Parameters
    ----------
    {beta}
    data : object
        Data object.
        Should include :class:`lnPiDataCallback` object as well
    order : int, optional
        maximum order.
        If not specified, default to `data.order + 1`
    {central}
    {post_func}
    {alpha_name}
    derivatives : :class:`thermoextrap.models.Derivatives`, optional
        Derivatives object.  If not passed, construct derivatives using :func:`thermoextrap.lnpi.factory_derivatives`.
    derivatives_kws : mapping, optional
        Optional parameters to :func:`thermoextrap.lnpi.factory_derivatives`.

    Returns
    -------
    extrapmodel : :class:`~thermoextrap.models.ExtrapModel`

    See Also
    --------
    thermoextrap.lnpi.factory_derivatives
    ~thermoextrap.models.ExtrapModel
    """
    if central is None:
        central = data.central
    if order is None:
        order = data.order + 1

    if central != data.central:
        raise ValueError
    if order > data.order + 1:
        raise ValueError
    if not data.x_is_u:
        raise ValueError

    if derivatives is None:
        if derivatives_kws is None:
            derivatives_kws = {}
        derivatives = factory_derivatives(
            name="lnPi", central=central, post_func=post_func, **derivatives_kws
        )
    return ExtrapModel(
        alpha0=beta,
        data=data,
        derivatives=derivatives,
        order=order,
        # minus_log=mineus_log,
        alpha_name=alpha_name,
    )

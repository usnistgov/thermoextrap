# pyright: reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false
# ruff: noqa: ARG003  # bunch of unused arguments
# pylint: disable=duplicate-code

"""
Inverse temperature (beta) extrapolation (:mod:`~thermoextrap.beta`)
====================================================================
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from sympy.core.numbers import Number

from .core.docstrings import DOCFILLER_SHARED
from .core.sputils import (
    get_default_indexed,
    get_default_symbol,
)
from .core.typing_compat import override
from .core.validate import validate_positive_integer
from .models import (
    Derivatives,
    ExtrapModel,
    PerturbModel,
    SymDerivBase,
    SymFuncBase,
)

if TYPE_CHECKING:
    from typing import Any

    import xarray as xr
    from sympy.core.mul import Expr
    from sympy.core.symbol import Symbol
    from sympy.tensor.indexed import IndexedBase

    from .core.typing import (
        DataT,
        OptionalKwsAny,
        PostFunc,
        SupportsData,
        SymDerivNames,
    )
    from .core.typing_compat import Self

docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap", "beta")


# * Recursive derivatives for beta expansion ----------------------------------
# ** Central moments ----------------------------------------------------------
class du_func(SymFuncBase):
    r"""
    Sympy function to evaluate energy fluctuations using central moments.

    :math:`\text{du_func}(\beta, n) = \langle (u(\beta) - \langle u(\beta) \rangle)^n \rangle`

    Notes
    -----
    sub in ``{'beta': 'None'}`` to convert to indexed objects
    Have to use ``'None'`` instead of ``None`` as sympy does some magic to
    the input arguments.
    """

    nargs = 2
    du = get_default_indexed("du")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[IndexedBase]:
        return (cls.du,)

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        beta, n = self.args
        return -(
            +self.tcall(beta, n=n + 1)
            - n * self.tcall(beta, n=n - 1) * self.tcall(beta, n=2)
        )

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        n = self.args[1]
        return self.du[n]  # pyright: ignore[reportReturnType, reportUnknownVariableType]

    @classmethod
    @override
    def eval(cls, beta: Symbol, n: int | Number = 0) -> Any:
        if n == 0:
            return Number(1)
        if n == 1:
            return Number(0)
        return None

    @classmethod
    @override
    def tcall(cls, beta: Symbol, *, n: int | Number = 0) -> Expr:
        return cls(beta, n)


class u_func_central(SymFuncBase):
    r"""
    Sympy function to evaluate energy averages using central moments.

    :math:`\text{u_func_central}(beta, n) = \langle u(\beta)^n \rangle`
    """

    nargs = 1
    u = get_default_symbol("u")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[Symbol, IndexedBase]:
        return (cls.u, *du_func.deriv_args())

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        (beta,) = self.args
        return -du_func.tcall(beta, n=2)

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        return self.u

    @classmethod
    @override
    def eval(cls, beta: Symbol) -> Any:
        return None

    @classmethod
    @override
    def tcall(cls, beta: Symbol) -> Expr:
        return cls(beta)


class dxdu_func(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of observable fluctuations using central moments.

    :math:`\text{dxdu_func}(\beta, n, d) = \langle \delta  x^{(d)}(\beta)(\delta u)^n \rangle`, where :math:`x^{(k)} = d^k x / d\beta^k`.

    or

    :math:`\text{dxdu_func_nobeta}(\beta, n) = \langle \delta x (\delta u)^n \rangle`

    If ``u`` doesn't depend on ``beta``, don't pass d
    """

    nargs = (2, 3)
    dxdu = get_default_indexed("dxdu")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase]:
        return (*du_func.deriv_args(), cls.dxdu)

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        if len(self.args) == 2:
            (beta, n), d = self.args, None
        else:
            beta, n, d = self.args

        out = (
            -self.tcall(beta, n=n + 1, deriv=d)
            + n * self.tcall(beta, n=n - 1, deriv=d) * du_func.tcall(beta, n=2)
            + self.tcall(beta, n=1, deriv=d) * du_func.tcall(beta, n=n)
        )

        if d is None:
            return out
        return out + self.tcall(beta, n=n, deriv=d + 1)

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        if len(self.args) == 2:
            _, n = self.args
            return cast("Expr", self.dxdu[n])
        _, n, deriv = self.args
        return cast("Expr", self.dxdu[n, deriv])

    @classmethod
    @override
    def eval(
        cls,
        beta: Symbol,
        n: int | Number = 0,
        deriv: int | Number | None = None,
    ) -> Any:
        if n == 0:
            return Number(0)
        return None

    @classmethod
    @override
    def tcall(
        cls,
        beta: Symbol,
        *,
        n: int | Number = 0,
        deriv: int | Number | None = None,
    ) -> Expr:
        if deriv is None:
            return cls(beta, n)
        return cls(beta, n, deriv)


class x_func_central(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of observable :math:`\langle x(\beta) \rangle` using central moments.

    If ``x`` does not depend on beta, don't pass ``deriv`` argument.
    """

    nargs = (1, 2)

    # NOTE: take advantage of fact you can use an IndexedBase object
    # like a symbol.  So don't need to special case deriv_args.
    x1_indexed = get_default_indexed("x1")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase, IndexedBase]:
        return (cls.x1_indexed, *dxdu_func.deriv_args())

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        if len(self.args) == 1:
            (beta,), d = self.args, None
        else:
            beta, d = self.args

        out = -dxdu_func.tcall(beta, n=1, deriv=d)
        if d is None:
            return out
        return out + self.tcall(beta, deriv=d + 1)

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        if len(self.args) == 1:
            return self.x1_indexed
        _, deriv = self.args
        return cast("Expr", self.x1_indexed[deriv])

    @classmethod
    @override
    def eval(cls, beta: Symbol, deriv: int | Number | None = None) -> Any:
        return None

    @classmethod
    @override
    def tcall(cls, beta: Symbol, *, deriv: int | Number | None = None) -> Expr:
        if deriv is None:
            return cls(beta)
        return cls(beta, deriv)


# ** raw moments --------------------------------------------------------------
class u_func(SymFuncBase):
    r"""Sympy function to evaluate derivatives of energy :math:`\langle u \rangle` using raw moments."""

    nargs = 2
    u = get_default_indexed("u")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[IndexedBase]:
        return (cls.u,)

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        beta, n = self.args
        return -(
            self.tcall(beta, n=n + 1) - self.tcall(beta, n=n) * self.tcall(beta, n=1)
        )

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        _, n = self.args
        return cast("Expr", self.u[n])

    @classmethod
    @override
    def eval(cls, beta: Symbol, n: int | Number = 0) -> Any:
        if n == 0:
            return Number(1)
        return None

    @classmethod
    @override
    def tcall(cls, beta: Symbol, *, n: int | Number = 0) -> Expr:
        return cls(beta, n)


class xu_func(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of :math:`\langle x u^n \rangle`.

    If ``x`` is a function of ``beta``, then :math:`\text{xu_func}(\beta, n, d) = \langle x^{(d)} u^n \rangle`.
    If ``x`` is not a function of ``beta``, drop argument ``d``.
    """

    nargs = (2, 3)
    xu = get_default_indexed("xu")

    @classmethod
    @override
    def deriv_args(cls) -> tuple[IndexedBase, IndexedBase]:
        return (*u_func.deriv_args(), cls.xu)

    @override
    def fdiff(self, argindex: int | Number = 1) -> Expr:
        if len(self.args) == 2:
            (beta, n), d = self.args, None
        else:
            beta, n, d = self.args

        out = -self.tcall(beta, n=n + 1, deriv=d) + self.tcall(
            beta, n=n, deriv=d
        ) * u_func.tcall(beta, n=1)

        if d is None:
            return out
        return out + self.tcall(beta, n=n, deriv=d + 1)

    @override
    def doit(self, deep: bool = False, **hints: Any) -> Expr:
        self._doit_args(deep, **hints)
        if len(self.args) == 2:
            _, n = self.args
            return cast("Expr", self.xu[n])
        _, n, deriv = self.args
        return cast("Expr", self.xu[n, deriv])

    @classmethod
    @override
    def eval(
        cls,
        beta: Symbol,
        n: int | Number = 0,
        deriv: int | Number | None = None,
    ) -> Any:
        return None

    @classmethod
    @override
    def tcall(
        cls,
        beta: Symbol,
        *,
        n: int | Number = 0,
        deriv: int | Number | None = None,
    ) -> Expr:
        if deriv is None:
            return cls(beta, n)
        return cls(beta, n, deriv)


@docfiller_shared.inherit(SymDerivBase)
class SymDerivBeta(SymDerivBase):
    r"""Provide symbolic expressions for :math:`d^n \langle x \rangle /d\beta^n`."""

    @classmethod
    @docfiller_shared.decorate
    def x_ave(
        cls,
        xalpha: bool = False,
        central: bool | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
        beta: Symbol | None = None,
    ) -> Self:
        r"""
        General method to find derivatives of :math:`\langle x \rangle`.

        Parameters
        ----------
        {xalpha}
        {central}
        {expand}
        {post_func}
        """
        if central is None:
            central = False
        beta = beta or get_default_symbol("beta")

        if central:  # pylint: disable=consider-ternary-expression
            func = x_func_central(beta, 0) if xalpha else x_func_central(beta)
        else:
            func = xu_func(beta, 0, 0) if xalpha else xu_func(beta, 0)

        return cls(
            func=func,
            args=func.deriv_args(),
            expand=expand,
            post_func=post_func,
            beta=beta,
        )

    @classmethod
    @docfiller_shared.decorate
    def u_ave(
        cls,
        central: bool | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
        beta: Symbol | None = None,
    ) -> Self:
        r"""
        General constructor for symbolic derivatives of :math:`\langle u \rangle`.

        Parameters
        ----------
        {central}
        {expand}
        {post_func}

        """
        if central is None:
            central = False
        beta = beta or get_default_symbol("beta")

        func = u_func_central(beta) if central else u_func(beta, 1)

        return cls(func=func, expand=expand, post_func=post_func, beta=beta)

    @classmethod
    @docfiller_shared.decorate
    def dun_ave(
        cls,
        n: int,
        expand: bool = True,
        post_func: PostFunc = None,
        central: bool | None = None,
        beta: Symbol | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle (\delta u)^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {post_func}
        {central}
        """
        if central is not None and not central:
            msg = f"{central=} must be None or evaluate to True"
            raise ValueError(msg)
        beta = beta or get_default_symbol("beta")

        if (n := int(n)) <= 1:
            msg = f"{n=} must be > 1."
            raise ValueError(msg)
        func = du_func(beta, n)

        # special case for args.
        # for consistency between uave and dun_ave, also include u variable
        args = u_func_central.deriv_args()
        return cls(
            func=func,
            args=args,
            expand=expand,
            post_func=post_func,
            beta=beta,
        )

    @classmethod
    @docfiller_shared.decorate
    def dxdun_ave(
        cls,
        n: int,
        xalpha: bool = False,
        expand: bool = True,
        post_func: PostFunc = None,
        d: int | None = None,
        central: bool | None = None,
        beta: Symbol | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle \delta x \delta u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {xalpha}
        {post_func}
        {d_order}
        {central}

        Notes
        -----
        If xalpha is True, must also specify d.
        """
        # special case for args
        # for consistency between xave and dxdun_ave, also include x1
        if central is not None and not central:
            msg = f"{central=} nust be `None` or evaluate to `True`"
            raise ValueError(msg)
        beta = beta or get_default_symbol("beta")

        n = validate_positive_integer(int(n), name="n")
        func = (
            dxdu_func(beta, n, validate_positive_integer(d, "d"))
            if xalpha
            else dxdu_func(beta, n)
        )

        return cls(
            func=func,
            args=x_func_central.deriv_args(),
            expand=expand,
            post_func=post_func,
            beta=beta,
        )

    @classmethod
    @docfiller_shared.decorate
    def un_ave(
        cls,
        n: int,
        expand: bool = True,
        post_func: PostFunc = None,
        central: bool | None = None,
        beta: Symbol | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {post_func}
        {central}
        """
        if central is not None and central:
            msg = f"{central=} must be `None` or evaluate to False"
            raise ValueError(msg)
        beta = beta or get_default_symbol("beta")

        if (n := int(n)) < 1:
            msg = f"{n=} must be >=1."
            raise ValueError(msg)

        func = u_func(beta, n)
        return cls(func=func, expand=expand, post_func=post_func, beta=beta)

    @classmethod
    @docfiller_shared.decorate
    def xun_ave(
        cls,
        n: int,
        d: int | None = None,
        xalpha: bool = False,
        expand: bool = True,
        post_func: PostFunc = None,
        central: bool | None = None,
        beta: Symbol | None = None,
    ) -> Self:
        r"""
        Constructor for derivatives of :math:`\langle x^{{(d)}} u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {d_order}
        {xalpha}
        {expand}
        {post_func}
        {central}
        """
        if central is not None and central:
            msg = f"{central=} must be `None` or False"
            raise ValueError(msg)
        beta = beta or get_default_symbol("beta")

        if (n := int(n)) < 0:
            msg = f"{n=} must be >= 0"
            raise ValueError(msg)

        func = (
            xu_func(beta, n, validate_positive_integer(d, "d"))
            if xalpha
            else xu_func(beta, n)
        )

        return cls(func=func, expand=expand, post_func=post_func, beta=beta)

    @classmethod
    def from_name(
        cls,
        name: SymDerivNames,
        xalpha: bool = False,
        central: bool | None = None,
        expand: bool = True,
        post_func: PostFunc = None,
        n: int | None = None,
        d: int | None = None,
        beta: Symbol | None = None,
    ) -> Self:
        """
        Create a derivative expressions indexer by name.

        Parameters
        ----------
        name : {'x_ave', 'u_ave', 'dun_ave', 'dxdun_ave', 'un_ave', 'xun_ave', 'lnPi_energy'}
            All properties use post_func and expand parameters.

            * x_ave: general average of <x>(central, xalpha)
            * u_ave: <u>(central)
            * dun_ave: derivative of <(u - <u>)**n>(central, n)
            * dxdun_ave: derivatives of <dx^(d) * du**n>(xalpha, n, d)

            * un_ave: derivative of <u**n>(n)
            * xun_ave: derivative of <x^(d) * u**n>(xalpha, n, [d])
            * lnPi_energy: derivatives of <lnPi - beta * mu * N>(central)

        xalpha : bool, default=False
            Whether property depends on alpha (beta)
        central : bool, default=False
            Whether central moments expansion should be used
        expand : bool, default=True
            Whether expressions should be expanded
        n : int, optional
            n parameter used for dun_ave or un_ave
        d : int, optional
            d parameter for dxdun_ave
        """
        beta = beta or get_default_symbol("beta")

        if (func := getattr(cls, name, None)) is None:
            msg = f"{name} not found"
            raise ValueError(msg)

        kws: dict[str, Any] = {
            "expand": expand,
            "post_func": post_func,
            "central": central,
        }
        if name == "x_ave":
            kws.update(xalpha=xalpha)
        # elif name in ["u_ave", "lnPi_correction":
        #     kws.update(central=central)
        elif name in {"dun_ave", "un_ave"}:
            kws.update(n=n)
        elif name in {"dxdun_ave", "xun_ave"}:
            kws.update(n=n, xalpha=xalpha, d=d)

        elif name == "lnPi_energy":
            # already have central
            pass

        return cast("Self", func(**kws))


###############################################################################
# Factory functions
###############################################################################


@lru_cache(5)
@docfiller_shared.decorate
def factory_derivatives(
    name: SymDerivNames = "x_ave",
    n: int | None = None,
    d: int | None = None,
    xalpha: bool = False,
    central: bool | None = None,
    post_func: PostFunc = None,
    expand: bool = True,
) -> Derivatives:
    r"""
    Factory function to provide derivative function for expansion.

    Parameters
    ----------
    name : {{x_ave, u_ave, dxdun_ave, dun_ave, un_ave, xun_ave}}
    {xalpha}
    {central}
    {post_func}

    Returns
    -------
    derivatives : :class:`thermoextrap.models.Derivatives` instance
        Object used to calculate taylor series coefficients
    """
    derivs = SymDerivBeta.from_name(
        name=name,
        n=n,
        d=d,
        xalpha=xalpha,
        central=central,
        post_func=post_func,
        expand=expand,
    )
    return Derivatives.from_sympy(derivs.doit(), args=derivs.args)


@docfiller_shared.decorate
def factory_extrapmodel(
    beta: float,
    data: SupportsData[DataT],
    *,
    name: str = "x_ave",
    n: int | None = None,
    d: int | None = None,
    xalpha: bool | None = None,
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
    {data}
    {n_order}
    {d_order}
    {order}
    {xalpha}
    {central}
    {post_func}
    {alpha_name}
    derivatives_kws : dict
        extra arguments to `factory_derivatives`

    Returns
    -------
    extrapmodel : :class:`~thermoextrap.models.ExtrapModel`

    See Also
    --------
    ~thermoextrap.models.ExtrapModel

    Notes
    -----
    Note that default values for parameters ``order``, ``xalpha``, and ``central``
    are inferred from corresponding attributes of ``data``.
    """
    if xalpha is None:
        xalpha = data.xalpha
    if central is None:
        central = data.central
    if order is None:
        order = data.order

    # assert xalpha == data.xalpha
    # assert central == data.central
    # assert order <= data.order
    if xalpha != data.xalpha:
        msg = f"{xalpha=} must equal {data.xalpha=}"
        raise ValueError(msg)
    if central != data.central:
        msg = f"{central=} must equal {data.central=}"
        raise ValueError(msg)
    if order > data.order:
        msg = f"{order=} must be <= {data.order=}"
        raise ValueError(msg)

    if derivatives is None:
        if name in {"u_ave", "un_ave", "dun_ave"} and not data.x_is_u:
            msg = "if name in [u_ave, un_ave, dun_ave] must have data.x_is_u"
            raise ValueError(msg)

        if derivatives_kws is None:
            derivatives_kws = {}
        derivatives = factory_derivatives(
            name=name,
            n=n,
            d=d,
            xalpha=xalpha,
            central=central,
            post_func=post_func,
            **derivatives_kws,
        )
    return ExtrapModel(
        alpha0=beta,
        data=data,
        derivatives=derivatives,
        order=order,
        # minus_log=mineus_log,
        alpha_name=alpha_name,
    )


@docfiller_shared.decorate
def factory_perturbmodel(
    beta: float, uv: xr.DataArray, xv: DataT, alpha_name: str = "beta", **kws: Any
) -> PerturbModel[DataT]:
    """
    Factory function to create PerturbModel for beta expansion.

    Parameters
    ----------
    {beta}
    {uv_xv_array}
    {alpha_name}
    kws : dict
        extra arguments to :class:`~.DataCentralMomentsVals`

    Returns
    -------
    perturbmodel : :class:`thermoextrap.models.PerturbModel`


    See Also
    --------
    ~thermoextrap.models.PerturbModel
    """
    from .data import DataCentralMomentsVals

    data = DataCentralMomentsVals(uv=uv, xv=xv, order=0, resample_values=True, **kws)
    return PerturbModel(alpha0=beta, data=data, alpha_name=alpha_name)

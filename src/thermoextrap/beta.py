"""
Inverse temperature (beta) extrapolation (:mod:`~thermoextrap.beta`)
====================================================================
"""


from functools import lru_cache
from typing import Literal

from .core._docstrings import factory_docfiller_shared
from .core.models import (
    Derivatives,
    ExtrapModel,
    PerturbModel,
    SymDerivBase,
    SymFuncBase,
    SymSubs,
    get_default_indexed,
    get_default_symbol,
)

docfiller_shared = factory_docfiller_shared(names=("default", "beta"))

##############################################################################
# recursive deriatives for beta expansion
###############################################################################

####################
# Central moments
####################


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
    def deriv_args(cls):
        return [cls.du]

    def fdiff(self, argindex=1):
        beta, n = self.args
        out = -(+du_func(beta, n + 1) - n * du_func(beta, n - 1) * du_func(beta, 2))
        return out

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            out = 1
        elif n == 1:
            out = 0
        elif beta is None:
            out = cls.du[n]
        else:
            out = None
        return out


class u_func_central(SymFuncBase):
    r"""
    Sympy function to evaluate energy averages using central moments.

    :math:`\text{u_func_central}(beta, n) = \langle u(\beta)^n \rangle`
    """

    nargs = 1
    u = get_default_symbol("u")

    @classmethod
    def deriv_args(cls):
        return [cls.u] + du_func.deriv_args()

    def fdiff(self, argindex=1):
        (beta,) = self.args
        out = -du_func(beta, 2)
        return out

    @classmethod
    def eval(cls, beta):
        if beta is None:
            return cls.u
        else:
            out = None
        return out


class dxdu_func_nobeta(SymFuncBase):
    r"""
    Sympy function to evaluate observable energy fluctuations using central moments.

    :math:`\text{dxdu_func_nobeta}(\beta, n) = \langle \delta x (\delta u)^n \rangle`

    for use when x is not a function of beta.
    """

    nargs = 2
    dxdu = get_default_indexed("dxdu")

    @classmethod
    def deriv_args(cls):
        return du_func.deriv_args() + [cls.dxdu]

    def fdiff(self, argindex=1):
        beta, n = self.args
        out = (
            -dxdu_func_nobeta(beta, n + 1)
            + n * dxdu_func_nobeta(beta, n - 1) * du_func(beta, 2)
            + dxdu_func_nobeta(beta, 1) * du_func(beta, n)
        )
        return out

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            out = 0
        elif beta is None:
            out = cls.dxdu[n]
        else:
            out = None
        return out


class dxdu_func_beta(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of observable fluctuations using central moments.

    :math:`\text{dxdu_func_beta}(\beta, n, d) = \langle \delta  x^{(d)}(\beta)(\delta u)^n \rangle`, where :math:`x^{(k)} = d^k x / d\beta^k`.

    """

    nargs = 3
    dxdu = get_default_indexed("dxdu")

    @classmethod
    def deriv_args(cls):
        return du_func.deriv_args() + [cls.dxdu]

    def fdiff(self, argindex=1):
        beta, n, d = self.args
        out = (
            -dxdu_func_beta(beta, n + 1, d)
            + n * dxdu_func_beta(beta, n - 1, d) * du_func(beta, 2)
            + dxdu_func_beta(beta, n, d + 1)
            + dxdu_func_beta(beta, 1, d) * du_func(beta, n)
        )
        return out

    @classmethod
    def eval(cls, beta, n, deriv):
        if n == 0:
            out = 0
        elif beta is None:
            out = cls.dxdu[n, deriv]
        else:
            out = None
        return out


class x_func_central_nobeta(SymFuncBase):
    r"""Sympy functionn to evaluate derivatives of observable :math:`\langle x \rangle` using central moments."""

    nargs = 1
    x1_symbol = get_default_symbol("x1")

    @classmethod
    def deriv_args(cls):
        return [cls.x1_symbol] + dxdu_func_nobeta.deriv_args()

    def fdiff(self, argindex=1):
        (beta,) = self.args
        return -dxdu_func_nobeta(beta, 1)

    @classmethod
    def eval(cls, beta):
        if beta is None:
            out = cls.x1_symbol
        else:
            out = None
        return out


class x_func_central_beta(SymFuncBase):
    r"""Sympy function to evaluate derivatives of observable :math:`\langle x(\beta) \rangle` using central moments."""

    nargs = 2
    x1_indexed = get_default_indexed("x1")

    @classmethod
    def deriv_args(cls):
        return [cls.x1_indexed] + dxdu_func_beta.deriv_args()

    def fdiff(self, argindex=1):
        # xalpha
        beta, d = self.args
        out = -dxdu_func_beta(beta, 1, d) + x_func_central_beta(beta, d + 1)
        return out

    @classmethod
    def eval(cls, beta, deriv):
        if beta is None:
            out = cls.x1_indexed[deriv]
        else:
            out = None
        return out


####################
# raw moments
####################
class u_func(SymFuncBase):
    r"""Sympy function to evaluate derivatives of energy :math:`\langle u \rangle` using raw moments."""

    nargs = 2
    u = get_default_indexed("u")

    @classmethod
    def deriv_args(cls):
        return [cls.u]

    def fdiff(self, argindex=1):
        beta, n = self.args
        return -(u_func(beta, n + 1) - u_func(beta, n) * u_func(beta, 1))

    @classmethod
    def eval(cls, beta, n):
        if n == 0:
            out = 1
        elif beta is None:
            out = cls.u[n]
        else:
            out = None
        return out


class xu_func(SymFuncBase):
    r"""
    Sympy function to evaluate derivatives of :math:`\langle x u^n \rangle`.

    If ``x`` is a function of ``beta``, then :math:`\text{xu_func}(\beta, n, d) = \langle x^{(d)} u^n \rangle`.
    If ``x`` is not a function of ``beta``, drop argument ``d``.
    """

    nargs = (2, 3)
    xu = get_default_indexed("xu")

    @classmethod
    def deriv_args(cls):
        return u_func.deriv_args() + [cls.xu]

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            beta, n = self.args
            out = -xu_func(beta, n + 1) + xu_func(beta, n) * u_func(beta, 1)

        else:
            beta, n, d = self.args
            out = (
                -xu_func(beta, n + 1, d)
                + xu_func(beta, n, d + 1)
                + xu_func(beta, n, d) * u_func(beta, 1)
            )
        return out

    @classmethod
    def eval(cls, beta, n, deriv=None):
        if beta is None:
            if deriv is None:
                out = cls.xu[n]
            else:
                out = cls.xu[n, deriv]
        else:
            out = None
        return out


class SymDerivBeta(SymDerivBase):
    r"""Provide symbolic expressions for :math:`d^n \langle x \rangle /d\beta^n`."""

    beta = get_default_symbol("beta")

    @classmethod
    @docfiller_shared
    def x_ave(
        cls, xalpha=False, central=None, expand=True, post_func=None
    ):  # noqa: 417
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

        if central:
            if xalpha:
                func = x_func_central_beta(cls.beta, 0)

            else:
                func = x_func_central_nobeta(cls.beta)

        else:
            if xalpha:
                func = xu_func(cls.beta, 0, 0)
            else:
                func = xu_func(cls.beta, 0)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    @docfiller_shared
    def u_ave(cls, central=None, expand=True, post_func=None):  # noqa: D417
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

        if central:
            func = u_func_central(cls.beta)
        else:
            func = u_func(cls.beta, 1)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    @docfiller_shared
    def dun_ave(cls, n, expand=True, post_func=None, central=None):  # noqa: D417
        r"""
        Constructor for derivatives of :math:`\langle (\delta u)^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {post_func}
        {central}
        """

        if central is not None:
            assert central

        n = int(n)
        assert n > 1
        func = du_func(cls.beta, n)

        # special case for args.
        # for consistency between uave and dun_ave, also include u variable
        args = u_func_central.deriv_args()
        return cls(
            func=func,
            args=args,
            expand=expand,
            post_func=post_func,
        )

    @classmethod
    @docfiller_shared
    def dxdun_ave(
        cls, n, xalpha=False, expand=True, post_func=None, d=None, central=None
    ):
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
        if central is not None:
            assert central

        assert isinstance(n, int) and n > 0
        if xalpha:
            assert isinstance(d, int)
            func = dxdu_func_beta(cls.beta, n, d)
            args = x_func_central_beta.deriv_args()

        else:
            func = dxdu_func_nobeta(cls.beta, n)
            args = x_func_central_nobeta.deriv_args()

        return cls(
            func=func,
            args=args,
            expand=expand,
            post_func=post_func,
        )

    @classmethod
    @docfiller_shared
    def un_ave(cls, n, expand=True, post_func=None, central=None):
        r"""
        Constructor for derivatives of :math:`\langle u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {expand}
        {post_func}
        {central}
        """
        if central is not None:
            assert not central
        n = int(n)
        assert n >= 1

        func = u_func(cls.beta, n)
        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    @docfiller_shared
    def xun_ave(
        cls, n, d=None, xalpha=False, expand=True, post_func=None, central=None
    ):
        r"""
        Constructor for deriatives of :math:`\langle x^{{(d)}} u^n\rangle`.

        Parameters
        ----------
        {n_order}
        {d_order}
        {xalpha}
        {expand}
        {post_func}
        {central}
        """

        if central is not None:
            assert not central

        assert isinstance(n, int) and n >= 0

        if xalpha:
            assert isinstance(d, int) and d >= 0
            func = xu_func(cls.beta, n, d)
        else:
            func = xu_func(cls.beta, n)

        return cls(func=func, expand=expand, post_func=post_func)

    @classmethod
    def from_name(
        cls,
        name: Literal[
            "x_ave", "u_ave", "dun_ave", "dxdun_ave", "un_ave", "xun_ave", "lnPi_energy"
        ],
        xalpha=False,
        central=None,
        expand=True,
        post_func=None,
        n=None,
        d=None,
    ):
        """
        Create a derivative expressions indexer by name.

        Parameters
        ----------
        name : {'xave', 'uave', 'dun_ave', 'un_ave'}
        All properties use post_func and expand parameters.
            * x_ave: general average of <x>(central, xalpha)
            * u_ave: <u>(central)
            * dun_ave: derivative of <(u - <u>)**n>(central, n)
            * dxdun_ave: derivatives of <dx^(d) * du**n>(xalpha, n, d)

            * un_ave: derivative of <u**n>(n)
            * xun_ave: derivative of <x^(d) * u**n>(xalpha, n, [d])
            * lnPi_correction: derivatives of <lnPi - beta * mu * N>(central)

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

        func = getattr(cls, name, None)

        if func is None:
            raise ValueError("{name} not found")

        kws = {"expand": expand, "post_func": post_func, "central": central}
        if name == "x_ave":
            kws.update(xalpha=xalpha)
        # elif name in ["u_ave", "lnPi_correction":
        #     kws.update(central=central)
        elif name in ["dun_ave", "un_ave"]:
            kws.update(n=n)
        elif name in ["dxdun_ave", "xun_ave"]:
            kws.update(n=n, xalpha=xalpha, d=d)

        elif name == "lnPi_correction":
            # already have central
            pass

        return func(**kws)


###############################################################################
# Factory functions
###############################################################################


@lru_cache(5)
@docfiller_shared
def factory_derivatives(
    name="x_ave",
    n=None,
    d=None,
    xalpha=False,
    central=None,
    post_func=None,
    expand=True,
):
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
    exprs = SymSubs(
        derivs,
        subs_all={derivs.beta: "None"},
        expand=False,
        simplify=False,
    )
    return Derivatives.from_sympy(exprs, args=derivs.args)


@docfiller_shared
def factory_extrapmodel(
    beta,
    data,
    *,
    name="x_ave",
    n=None,
    d=None,
    xalpha=None,
    central=None,
    order=None,
    alpha_name="beta",
    derivatives=None,
    post_func=None,
    derivatives_kws=None,
):
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
    kws : dict
        extra arguments to `factory_data_values`

    Returns
    -------
    extrapmodel : :class:`~thermoextrap.models.ExtrapModel`


    Notes
    -----
    Note that default values for parameters ``order``, ``xalpha``, and ``central``
    are inferred from corresponding attributes of ``data``.

    See Also
    --------
    ~thermoextrap.models.ExtrapModel
    """

    if xalpha is None:
        xalpha = data.xalpha
    if central is None:
        central = data.central
    if order is None:
        order = data.order

    assert xalpha == data.xalpha
    assert central == data.central
    assert order <= data.order

    if derivatives is None:
        if name in ["u_ave", "un_ave", "dun_ave"]:
            assert data.x_is_u

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


@docfiller_shared
def factory_perturbmodel(beta, uv, xv, alpha_name="beta", **kws):
    """
    Factory function to create PerturbModel for beta expansion.

    Parameters
    ----------
    {beta}
    {uv_xv_array}
    {alpha_name}
    kws : dict
        extra arguments to data object

    Returns
    -------
    perturbmodel : :class:`thermoextrap.models.PerturbModel`


    See Also
    --------
    ~thermoextrap.models.PerturbModel
    """
    from .core.data import factory_data_values

    data = factory_data_values(uv=uv, xv=xv, order=0, central=False, **kws)
    return PerturbModel(alpha0=beta, data=data, alpha_name=alpha_name)

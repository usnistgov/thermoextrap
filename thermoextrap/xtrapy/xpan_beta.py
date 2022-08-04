"""
Routines for beta expansion(s)
"""

from __future__ import absolute_import

from functools import lru_cache

import sympy as sp

from .cached_decorators import gcached
from .data import (  # noqa: F401
    DataCentralMoments,
    DataCentralMomentsVals,
    DataValues,
    DataValuesCentral,
    resample_indicies,
)
from .models import (
    Derivatives,
    ExtrapModel,
    PerturbModel,
    SymSubs,
    _get_default_indexed,
    _get_default_symbol,
)

##############################################################################
# recursive deriatives for beta expansion
###############################################################################

####################
# Central moments
####################


class du_func(sp.Function):
    """
    du_func(beta, n) = <(u - <u>)**n>

    Note
    ----
    sub in {beta: 'None'} to convert to indexed objects
    Have to use 'None' instead of None as sympy does some magic to
    the input arguments
    """

    nargs = 2
    du = _get_default_indexed("du")

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


class u_func_central(sp.Function):
    """
    energy function
    """

    nargs = 1
    u = _get_default_indexed("u")

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


class dxdu_func(sp.Function):
    """
    dxdu_func(beta, n, d) = <du**n * (x^(d) - <x^(d)>)> = dxdu[n, d]
    or (if x != x(alpha))
    dxdu_func(beta, n) = <du**n * (x - <x>)>
    """

    nargs = (2, 3)
    dxdu = _get_default_indexed("dxdu")

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            beta, n = self.args
            out = (
                -dxdu_func(beta, n + 1)
                + n * dxdu_func(beta, n - 1) * du_func(beta, 2)
                + dxdu_func(beta, 1) * du_func(beta, n)
            )

        else:
            beta, n, d = self.args
            out = (
                -dxdu_func(beta, n + 1, d)
                + n * dxdu_func(beta, n - 1, d) * du_func(beta, 2)
                + dxdu_func(beta, n, d + 1)
                + dxdu_func(beta, 1, d) * du_func(beta, n)
            )
        return out

    @classmethod
    def eval(cls, beta, n, deriv=None):
        if n == 0:
            out = 0
        elif beta is None:
            if deriv is None:
                out = cls.dxdu[n]
            else:
                out = cls.dxdu[n, deriv]
        else:
            out = None

        return out


class x_central_func(sp.Function):
    nargs = (1, 2)

    x1_indexed = _get_default_indexed("x1")
    x1_symbol = _get_default_symbol("x1")

    def fdiff(self, argindex=1):
        if len(self.args) == 1:
            (beta,) = self.args
            return -dxdu_func(beta, 1)

        else:
            # xalpha
            beta, d = self.args
            out = -dxdu_func(beta, 1, d) + x_central_func(beta, d + 1)
            return out

    @classmethod
    def eval(cls, beta, deriv=None):
        if beta is None:
            if deriv is None:
                out = cls.x1_symbol
            else:
                out = cls.x1_indexed[deriv]
        else:
            out = None

        return out


####################
# raw moments
####################
class u_func(sp.Function):
    """
    calculate derivatives of <U**n>
    """

    nargs = 2
    u = _get_default_indexed("u")

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


class xu_func(sp.Function):
    """
    calculate derivatives of
    xu_func(beta, n) = <x * u**n> = xu[n]
    or (xalpha)
    xu_func(beta, n, d) = <x^(d) * u**n> = xu[n, d]
    """

    nargs = (2, 3)
    xu = _get_default_indexed("xu")

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


# class SymDerivBeta(object):
#     """
#     provide expressions for d^n <x> / d(beta)^n
#     """

#     beta = _get_default_symbol("beta")

#     def __init__(self, xalpha=False, central=False, expand=True):
#         if central:
#             if xalpha:
#                 x = x_central_func(self.beta, 0)
#                 args = [x.x1_indexed]
#             else:
#                 x = x_central_func(self.beta)
#                 args = [x.x1_symbol]
#             args += _get_default_indexed("du", "dxdu")

#         else:
#             if xalpha:
#                 x = xu_func(self.beta, 0, 0)
#             else:
#                 x = xu_func(self.beta, 0)
#             args = _get_default_indexed("u", "xu")

#         self.xave = x
#         self.args = args
#         self.expand = expand

#     @gcached(prop=False)
#     def __getitem__(self, order):
#         if order == 0:
#             out = self.xave
#         else:
#             out = self[order - 1].diff(self.beta, 1)
#             if self.expand:
#                 out = out.expand()
#         return out


class SymDerivBeta(object):
    """
    provide expressions for d^n <x> / d(beta)^n

    includes ability to use -ln(<x>) directly

    """

    beta = _get_default_symbol("beta")

    def __init__(self, xalpha=False, central=False, expand=True, minus_log=False):
        if central:
            if xalpha:
                x = x_central_func(self.beta, 0)
                args = [x.x1_indexed]
            else:
                x = x_central_func(self.beta)
                args = [x.x1_symbol]
            args += _get_default_indexed("du", "dxdu")

        else:
            if xalpha:
                x = xu_func(self.beta, 0, 0)
            else:
                x = xu_func(self.beta, 0)
            args = _get_default_indexed("u", "xu")

        if minus_log:
            x = -sp.log(x)

        self.xave = x
        self.args = args
        self.expand = expand

    @gcached(prop=False)
    def __getitem__(self, order):
        if order == 0:
            out = self.xave
        else:
            out = self[order - 1].diff(self.beta, 1)
            if self.expand:
                out = out.expand()
        return out


###############################################################################
# Factory functions
###############################################################################
def factory_data(
    uv,
    xv,
    order,
    central=False,
    skipna=False,
    xalpha=False,
    rec_dim="rec",
    umom_dim="umom",
    val_dims="val",
    rep_dim="rep",
    deriv_dim=None,
    chunk=None,
    compute=None,
    **kws
):
    """
    Factory function to produce a Data object

    Parameters
    ----------
    uv : array-like
        energy values
    xv : array-like
        observable values
    order : int
        highest umom_dim to calculate
    skipna : bool, default=False
        if True, skip `np.nan` values in creating averages.
        Can make some "big" calculations slow
    rec_dim, umom_dim, val_dim, rep_dim, deriv_dim : str
        names of record (i.e. time), umom_dim, value, replicate,
        and derivative (with respect to alpha)
    chunk : int or dict, optional
        If specified, perform chunking on resulting uv, xv arrays.
        If integer, chunk with {rec: chunk}
        otherwise, should be a mapping of form {dim_0: chunk_0, dim_1: chunk_1, ...}
    compute : bool, optional
        if compute is True, do compute averages greedily.
        if compute is False, and have done chunking, then defer calculation of averages (i.e., will be dask future objects).
        Default is to do greedy calculation
    kws : dict, optional
        extra arguments
    """

    if central:
        cls = DataValuesCentral
    else:
        cls = DataValues

    if xalpha and deriv_dim is None:
        raise ValueError("if xalpha, must pass string name of derivative")

    return cls.from_vals(
        uv=uv,
        xv=xv,
        order=order,
        skipna=skipna,
        rec_dim=rec_dim,
        umom_dim=umom_dim,
        val_dims=val_dims,
        rep_dim=rep_dim,
        deriv_dim=deriv_dim,
        chunk=chunk,
        compute=compute,
        **kws
    )


@lru_cache(5)
def factory_derivatives(xalpha=False, central=False, minus_log=False):
    """
    factory function to provide derivative function for expansion

    Parameters
    ----------
    xalpha : bool, default=False
        whether x = func(beta) or not
    central : bool, default=False
        whether to use central moments or not

    Returns
    -------
    derivatives : Derivatives object used to calculate taylor series coefficients
    """

    derivs = SymDerivBeta(xalpha=xalpha, central=central, minus_log=minus_log)
    exprs = SymSubs(
        derivs,
        subs_all={derivs.beta: "None"},
        expand=False,
        simplify=False,
    )
    return Derivatives.from_sympy(exprs, args=derivs.args)


def factory_extrapmodel(
    beta,
    data,
    xalpha=None,
    central=None,
    order=None,
    minus_log=False,
    alpha_name="beta",
):
    """
    factory function to create Extrapolation model for beta expansion

    Parameters
    ----------
    beta : float
        reference value of inverse temperature
    data : Data object
        data object to consider.
        See data.AbstractData
    order : int, optional
        maximum order.
        If not specified, infer from `data`
    xalpha : bool, optional
        Whether or not x = func(alpha).
        If not specified, infer from `data`
    central : bool, optional
        Whether or not to use central moments
        If not specified, infer from `data`
    minus_log : bool, default=False
        Wheter or not we are expanding x = -log <x>
    alpha_name, str, default='beta'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_data`

    Returns
    -------
    extrapmodel : ExtrapModel object
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

    derivatives = factory_derivatives(
        xalpha=xalpha, central=central, minus_log=minus_log
    )
    return ExtrapModel(
        alpha0=beta,
        data=data,
        derivatives=derivatives,
        order=order,
        # minus_log=minus_log,
        alpha_name=alpha_name,
    )


def factory_perturbmodel(beta, uv, xv, alpha_name="beta", **kws):
    """
    factory function to create PerturbModel for beta expansion

    Parameters
    ----------
    beta : float
        reference value of inverse temperature
    uv, xv : array-like
        values of u and x
    alpha_name : str, default='beta'
        name of expansion parameter
    kws : dict
        extra arguments to `models.Data`

    Returns
    -------
    perturbmodel : PerturbModel object
    """
    data = factory_data(uv=uv, xv=xv, order=0, central=False, **kws)
    return PerturbModel(alpha0=beta, data=data, alpha_name=alpha_name)

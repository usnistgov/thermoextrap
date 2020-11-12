"""
Routines for beta expansion(s)
"""

from __future__ import absolute_import

from functools import lru_cache

import xarray as xr
import sympy as sp

from .cached_decorators import gcached
from .core import _get_default_symbol, _get_default_indexed
from .core import (
    DataTemplateValues,
    DatasetSelector,
    DataStatsCov,
    DataStatsCovVals,
)
from .core import SymSubs, Coefs
from .core import ExtrapModel, PerturbModel

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


class SymDerivBeta(object):
    """
    provide expressions for d^n <x> / d(beta)^n
    """

    beta = _get_default_symbol("beta")

    def __init__(self, xalpha=False, central=False, expand=True):
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
# Data
###############################################################################
def build_aves_xu(
    uv,
    xv,
    order,
    rec="rec",
    moment="moment",
    val="val",
    rep="rep",
    deriv="deriv",
    skipna=False,
    u_name=None,
    xu_name=None,
    xalpha=False,
    merge=False,
    transpose=False,
):
    """
    build averages from values uv, xv up to order `order`

    Parameters
    ----------
    moment : dimension with moment order
    deriv : dimension with derivative order

    skipna : bool, default=False
        if True, then handle nan values correctly.  Note that skipna=True
        can drastically slow down the calculations

    """

    assert isinstance(uv, xr.DataArray)
    assert isinstance(xv, (xr.DataArray, xr.Dataset))

    u_order = (moment, ...)
    if xalpha:
        x_order = (deriv,) + u_order
    else:
        x_order = u_order

    # do averageing

    # this is faster is some cases then the
    # simplier
    u = []
    xu = []

    uave = uv.mean(rec, skipna=skipna)
    xave = xv.mean(rec, skipna=skipna)
    for i in range(order + 1):
        if i == 0:
            # <u**0>
            u.append(xr.ones_like(uave))
            # <x * u**0> = <x>
            xu.append(xave)

        elif i == 1:
            u_n = uv.copy()
            xu_n = xv * uv

            u.append(uave)
            xu.append(xu_n.mean(rec, skipna=skipna))
        else:
            u_n *= uv
            xu_n *= uv
            u.append(u_n.mean(rec, skipna=skipna))
            xu.append(xu_n.mean(rec, skipna=skipna))

    u = xr.concat(u, dim=moment)
    xu = xr.concat(xu, dim=moment)

    # simple, but sometimes slow....
    # nvals = xr.DataArray(np.arange(order + 1), dims=[moment])
    # un = uv**nvals
    # u = (un).mean(rec, skipna=skipna)
    # xu = (un * xv).mean(rec, skipna=skipna)

    if transpose:
        u = u.trapsose(*u_order)
        xu = xu.transpose(*x_order)

    if u_name is not None:
        u = u.rename(u_name)

    if xu_name is not None and isinstance(xu, xr.DataArray):
        xu = xu.rename(xu_name)

    if merge:
        return xr.merge((u, xu))
    else:
        return u, xu


def build_aves_dxdu(
    uv,
    xv,
    order,
    rec="rec",
    moment="moment",
    val="val",
    rep="rep",
    deriv="deriv",
    skipna=False,
    du_name=None,
    dxdu_name=None,
    xave_name=None,
    xalpha=False,
    merge=False,
    transpose=False,
):
    """
    build central moments from values uv, xv up to order `order`

    Parameters
    ----------
    moment : dimension with moment order
    deriv : dimension with derivative order

    skipna : bool, default=False
        if True, then handle nan values correctly.  Note that skipna=True
        can drastically slow down the calculations

    """

    assert isinstance(uv, xr.DataArray)
    assert isinstance(xv, (xr.DataArray, xr.Dataset))

    u_order = (moment, ...)
    if xalpha:
        x_order = (deriv,) + u_order
    else:
        x_order = u_order

    xave = xv.mean(rec, skipna=skipna)
    uave = uv.mean(rec, skipna=skipna)

    # i=0
    # <du**0> = 1
    # <dx * du**0> = 0
    duave = []
    dxduave = []
    du = uv - uave

    for i in range(order + 1):
        if i == 0:
            # <du ** 0> = 1
            # <dx * du**0> = 0
            duave.append(xr.ones_like(uave))
            dxduave.append(xr.zeros_like(xave))

        elif i == 1:
            # <du**1> = 0
            # (dx * du**1> = ...

            du_n = du.copy()
            dxdu_n = (xv - xave) * du
            duave.append(xr.zeros_like(uave))
            dxduave.append(dxdu_n.mean(rec, skipna=skipna))

        else:
            du_n *= du
            dxdu_n *= du
            duave.append(du_n.mean(rec, skipna=skipna))
            dxduave.append(dxdu_n.mean(rec, skipna=skipna))

    duave = xr.concat(duave, dim=moment)
    dxduave = xr.concat(dxduave, dim=moment)

    if transpose:
        duave = duave.transpose(*u_order)
        dxduave = dxduave.transpoe(*x_order)

    if du_name is not None:
        duave = duave.rename(du_name)
    if dxdu_name is not None and isinstance(dxduave, xr.DataArray):
        dxduave = dxduave.rename(dxdu_name)
    if xave_name is not None and isinstance(xave, xr.DataArray):
        xave = xave.renamae(xave_name)

    if merge:
        return xr.merge((xave, duave, dxduave))
    else:
        return xave, duave, dxduave


class Data(DataTemplateValues):
    """
    Class to hold uv/xv data
    """

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves_xu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            xalpha=self.xalpha,
            rep=self._rep,
            rec=self._rec,
            val=self._val,
            moment=self._moment,
            deriv=self._deriv,
            **self._kws
        )

    @gcached()
    def u(self):
        out = self._mean()[0]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def xu(self):
        out = self._mean()[1]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def u_selector(self):
        return DatasetSelector(self.u, deriv=self._deriv, moment=self._moment)

    @gcached()
    def xu_selector(self):
        return DatasetSelector(self.xu, deriv=self._deriv, moment=self._moment)

    @property
    def _xcoefs_args(self):
        return (self.u_selector, self.xu_selector)


class DataCentral(DataTemplateValues):
    """
    Hold uv/xv data and produce central momemnts

    Attributes:
    xave : DataArray or Dataset
        <xv>
    du : DataArray
        <(u-<u>)**moment>
    dxdu : DataArray
        <(x-<x>) * (u-<u>)**moment>
    """

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves_dxdu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            xalpha=self.xalpha,
            rep=self._rep,
            rec=self._rec,
            val=self._val,
            moment=self._moment,
            deriv=self._deriv,
            **self._kws
        )

    @gcached()
    def xave(self):
        out = self._mean()[0]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def du(self):
        out = self._mean()[1]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def dxdu(self):
        out = self._mean()[2]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def du_selector(self):
        return DatasetSelector(self.du, deriv=self._deriv, moment=self._moment)

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector(self.dxdu, deriv=self._deriv, moment=self._moment)

    @gcached()
    def xave_selector(self):
        if self.xalpha:
            return DatasetSelector(self.xave, dims=[self._deriv])
        else:
            return self.xave

    @property
    def _xcoefs_args(self):
        return (self.xave_selector, self.du_selector, self.dxdu_selector)


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
    rec="rec",
    moment="moment",
    val="val",
    rep="rep",
    deriv="deriv",
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
        highest moment to calculate
    skipna : bool, default=False
        if True, skip `np.nan` values in creating averages.
        Can make some "big" calculations slow
    rec, moment, val, rep, deriv : str
        names of record (i.e. time), moment, value, replicate,
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
        cls = DataCentral
    else:
        cls = Data
    return cls(
        uv=uv,
        xv=xv,
        order=order,
        skipna=skipna,
        xalpha=xalpha,
        rec=rec,
        moment=moment,
        val=val,
        rep=rep,
        deriv=deriv,
        chunk=chunk,
        compute=compute,
        **kws
    )


@lru_cache(5)
def factory_coefs(xalpha=False, central=False):
    """
    factory function to provide coefficients of expansion

    Parameters
    ----------
    xalpha : bool, default=False
        whether x = func(beta) or not
    central : bool, default=False
        whether to use central moments or not

    Returns
    -------
    coefs : Coefs object used to calculate moments
    """

    derivs = SymDerivBeta(xalpha=xalpha, central=central)
    exprs = SymSubs(
        derivs, subs_all={derivs.beta: "None"}, expand=False, simplify=False
    )
    return Coefs.from_sympy(exprs, args=derivs.args)


def factory_extrapmodel(
    alpha0,
    order=None,
    data=None,
    uv=None,
    xv=None,
    xalpha=False,
    central=False,
    minus_log=False,
    alpha_name="beta",
    **kws
):
    """
    factory function to create Extrapolation model for beta expanssion

    Parameters
    ----------
    order : int
        maximum order
    alpha0 : float
        reference value of alpha (beta)
    data : Data object

    uv, xv : array-like
        values for u and x
    xalpha : bool, default=False
        Whether or not x = func(alpha)
    central : bool, default=False
        Whether or not to use central moments
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

    if data is None:
        data = factory_data(
            uv=uv, xv=xv, order=order, central=central, xalpha=xalpha, **kws
        )


    coefs = factory_coefs(xalpha=xalpha, central=central)
    return ExtrapModel(
        alpha0=alpha0,
        data=data,
        coefs=coefs,
        order=data.order,
        minus_log=minus_log,
        alpha_name=alpha_name,
    )


def factory_perturbmodel(alpha0, uv, xv, alpha_name="beta", **kws):
    """
    factory function to create PerturbModel for beta expansion

    Parameters
    ----------
    alpha0 : float
        reference value of beta
    uv, xv : array-like
        values of u and x
    alpha_name : str, default='beta'
        name of expansion parameter
    kws : dict
        extra arguments to `core.Data`

    Returns
    -------
    perturbmodel : PerturbModel object
    """
    data = Data(uv=uv, xv=xv, order=0, **kws)
    return PerturbModel(alpha0=alpha0, data=data, alpha_name=alpha_name)

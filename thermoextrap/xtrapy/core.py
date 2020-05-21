from __future__ import absolute_import

from functools import lru_cache

import numpy as np
import sympy as sp
import xarray as xr
from scipy.special import factorial as sp_factorial

from .cached_decorators import gcached

try:
    from pymbar import mbar

    _HAS_PYMBAR = True
except ImportError:
    _HAS_PYMBAR = False

# NOTE: General scheme:
# uv, xv -> samples (values) for u, x
# u, xu -> averages of u and x*u
# u[i] = <u**i>
# xu[i] = <x * u**i>
# xu[i, j] = <d^i x/d beta^i * u**j>


###############################################################################
# Structure(s) to handle data
###############################################################################
def _check_xr(x, dims, strict=True, name=None):
    if isinstance(x, xr.Dataset):
        # don't do anything to datasets
        pass

    elif not isinstance(x, xr.DataArray):
        x = np.array(x)
        if isinstance(dims, dict):
            dims = dims[x.ndim]
        x = xr.DataArray(x, dims=dims, name=name)
    elif strict:
        if isinstance(dims, dict):
            dims = dims[x.ndim]
        for d in dims:
            if d not in x.dims:
                raise ValueError("{} not in dims".format(d))
    return x


def xrwrap_uv(uv, dims=None, rec="rec", rep="rep", name="u", stict=True):
    """
    wrap uv (energy values) array

    assumes uv[rec], or uv[rep, rec] where rec is recored (or time) and rep is replicate
    """
    if dims is None:
        dims = {1: [rec], 2: [rep, rec]}
    return _check_xr(uv, dims, strict=stict, name=name)


def xrwrap_xv(
    xv,
    dims=None,
    rec="rec",
    rep="rep",
    deriv="deriv",
    val="val",
    xbeta=False,
    name="x",
    strict=None,
):
    """
    wraps xv (x values) array

    if xbeta is False, assumes xv[rec], xv[rec, val], xv[rep, rec, val]
    if xbeta is True, assumes xv[rec, deriv], xv[rec,deriv, val], xv[rep,rec,deriv,val]
    """
    if not xbeta:
        if strict is None:
            strict = False
        if dims is None:
            dims = {1: [rec], 2: [rec, val], 3: [rep, rec, val]}

    else:
        if strict is None:
            strict = False
        if dims is None:
            dims = {2: [rec, deriv], 3: [rec, deriv, val], 4: [rep, rec, deriv, val]}
    return _check_xr(xv, dims=dims, strict=strict, name=name)


def xrwrap_beta(beta, dims=None, stict=False, name="beta"):
    """
    wrap beta values
    """
    if isinstance(beta, xr.DataArray):
        pass
    else:
        beta = np.array(beta)
        if dims is None:
            dims = "beta"

        if beta.ndim == 0:
            beta = xr.DataArray(beta, coords={dims: beta}, name=name)
        elif beta.ndim == 1:
            beta = xr.DataArray(beta, dims=dims, coords={dims: beta}, name=name)
        else:
            beta = xr.DataArray(beta, dims=dims, name=name)
    return beta


def build_aves(
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
    xbeta=False,
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
    if xbeta:
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


def build_aves_central(
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
    xbeta=False,
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
    if xbeta:
        x_order = (deriv,) + u_order
    else:
        x_order = u_order

    xave = xv.mean(rec, skipna=skipna)
    uave = uv.mean(rec, skipna=skipna)

    DU = []
    DXDU = []

    # i=0
    # <du**0> = 1
    # <dx * du**0> = 0
    dU = []
    dXdU = []
    du = uv - uave

    for i in range(order + 1):
        if i == 0:
            # <du ** 0> = 1
            # <dx * du**0> = 0
            dU.append(xr.ones_like(uave))
            dXdU.append(xr.zeros_like(xave))

        elif i == 1:
            # <du**1> = 0
            # (dx * du**1> = ...

            du_n = du.copy()
            dxdu_n = (xv - xave) * du
            dU.append(xr.zeros_like(uave))
            dXdU.append(dxdu_n.mean(rec, skipna=skipna))

        else:
            du_n *= du
            dxdu_n *= du
            dU.append(du_n.mean(rec, skipna=skipna))
            dXdU.append(dxdu_n.mean(rec, skipna=skipna))

    dU = xr.concat(dU, dim=moment)
    dXdU = xr.concat(dXdU, dim=moment)

    if transpose:
        dU = dU.transpose(*u_order)
        dXdU = dXdU.transpoe(*x_order)

    if du_name is not None:
        dU = dU.rename(dU_name)
    if dxdu_name is not None and isinstance(dxdu, xr.DataArray):
        dXdU = dXdU.rename(dxdu_name)
    if xave_name is not None and isinstance(xave, xr.DataArray):
        xave = xave.renamae(xave_name)

    if merge:
        return xr.merge((xave, dU, dXdU))
    else:
        return xave, dU, dXdU


def resample_indicies(size, nrep, rec="rec", rep="rep"):
    """
    get indexing DataArray
    """
    return (
        xr.DataArray(
            [np.random.choice(size, size=size, replace=True) for _ in range(nrep)],
            dims=[rep, rec],
        )
        # things are faster with
        .transpose("rec", ...)
    )


class DatasetSelector(object):
    """
    wrap dataset so can index like ds[i, j]

    Needed for calling sympy.lambdify functions
    """

    def __init__(self, data, dims=None, deriv="deriv", moment="moment"):

        # Default dims
        if dims is None:
            if deriv in data.dims:
                dims = [deriv, moment]
            else:
                dims = [moment]

        if isinstance(dims, str):
            dims = [dims]

        self.data = data
        self.dims = dims

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        assert len(idx) == len(self.dims)
        selector = dict(zip(self.dims, idx))
        return self.data.isel(**selector)


class _DataBase(object):
    def __init__(
        self,
        uv,
        xv,
        order,
        skipna=False,
        xbeta=False,
        rec="rec",
        moment="moment",
        val="val",
        rep="rep",
        deriv="deriv",
        chunk=None,
        compute=None,
        **kws
    ):

        uv = xrwrap_uv(uv, rec=rec, rep=rep)
        xv = xrwrap_xv(xv, rec=rec, rep=rep, deriv=deriv, val=val, xbeta=xbeta)

        if chunk is not None:
            if isinstance(chunk, int):
                chunk = {rec: chunk}

            uv = uv.chunk(chunk)
            xv = xv.chunk(chunk)

        if compute is None:
            # default compute
            # if not chunk, default compute is False
            # if chunk, default compute is True
            if chunk is None:
                compute = False
            else:
                compute = True

        self.uv = uv
        self.xv = xv
        self.chunk = chunk
        self.compute = compute

        self.order = order
        self.skipna = skipna
        self.xbeta = xbeta

        self._rec = rec
        self._rep = rep
        self._val = val
        self._moment = moment
        self._deriv = deriv
        self._kws = kws

    def __len__(self):
        return len(self.uv["rec"])

    def resample(self, nrep, idx=None, chunk=None, compute="None"):

        if chunk is None:
            chunk = self.chunk

        if compute is "None":
            compute = None
        elif compute is None:
            compute = self.compute

        shape = len(self.uv[self._rec])

        if idx is None:
            idx = resample_indicies(shape, nrep, rec=self._rec, rep=self._rep)

        uv = self.uv.compute()[idx]
        # allow for Dataset
        xv = self.xv.compute().isel(**{self._rec: idx})

        return self.__class__(
            uv=uv,
            xv=xv,
            order=self.order,
            rec=self._rec,
            rep=self._rep,
            val=self._val,
            moment=self._moment,
            deriv=self._deriv,
            xbeta=self.xbeta,
            skipna=self.skipna,
            chunk=chunk,
            compute=compute,
            **self._kws
        )


class Data(_DataBase):
    """
    Class to hold uv/xv data
    """

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            xbeta=self.xbeta,
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


class DataCentral(_DataBase):
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

        return build_aves_central(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            xbeta=self.xbeta,
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
        if self.xbeta:
            return DatasetSelector(self.xave, dims=[self._deriv])
        else:
            return self.xave


def factory_data(
    uv,
    xv,
    order,
    central=False,
    skipna=False,
    xbeta=False,
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
        xbeta=xbeta,
        rec=rec,
        moment=moment,
        val=val,
        rep=rep,
        deriv=deriv,
        chunk=chunk,
        compute=compute,
        **kws
    )


################################################################################
# Structure(s) to deal with analytic derivatives, etc
################################################################################

def _set_default_symbols():
    d = {}
    # symbols:
    for key in ['b', 'k', 'u1', 'x1_symbol']:
        d[key] = sp.symbols(key)

    # funcs of b
    for key in ['f', 'z']:
        d[key] = sp.Function(key)(d['b'])

    d["Q"] = d["f"] / d["z"]

    for key in ['u', 'x', 'xu', 'du', 'dxdu', 'x1_indexed']:
        d[key] = sp.IndexedBase(key)
    return d


_DEFAULT_SYMBOLS = _set_default_symbols()


@lru_cache(20)
def _get_default_symbols(*args):
    out = [_DEFAULT_SYMBOLS[k] for k in args]
    if len(args) == 1:
        out = out[0]
    return out





class _SymDeriv2(object):
    _ave_func, b = _get_default_symbols('Q','b')

    @gcached(prop=False)
    def __getitem__(self, order):
        # recusive get derivative
        if order == 0:
            return self._ave_func
        else:
            return self[order - 1].diff(self.b, 1)


class _Subs2(object):
    f, z, b = _get_default_symbols('f', 'z', 'b')

    def __init__(self, u=None, xu=None):
        if u is None:
            u = _get_default_symbols('u')
        if xu is None:
            xu = _get_default_symbols('xu')
        self.u = u
        self.xu = xu

        self._init_data()

    def _init_data(self):
        self._data = [[(self.f, self.xu[0] * self.z)]]

    @property
    def order(self):
        return len(self._data) - 1

    def _add_order(self):
        order = self.order + 1
        new = []
        # f derivative:
        lhs = self.f.diff(self.b, order)
        rhs = (-1) ** order * self.xu[order] * self.z
        new.append((lhs, rhs))

        # z deriative:
        lhs = self.z.diff(self.b, order)
        rhs = (-1) ** order * self.u[order] * self.z
        new.append((lhs, rhs))
        self._data.append(new)

    #    @gcached(prop=False)
    def __getitem__(self, order):
        assert order >= 0
        while order > self.order:
            self._add_order()

        # give up to order
        # and in reversed order
        return sum(self._data[: order + 1], [])[-1::-1]


class _Subsxbeta2(_Subs2):
    k = _get_default_symbols('k')

    def _init_data(self):
        self._data = [[(self.f, self.xu[0, 0] * self.z)]]

    def _add_order(self):
        order = self.order + 1

        new = []

        # f deriv:
        lhs = self.f.diff(self.b, order)
        rhs = (
            sp.Sum(
                (
                    (-1) ** self.k
                    * sp.binomial(order, self.k)
                    * self.xu[order - self.k, self.k]
                ),
                (self.k, 0, order),
            ).doit()
            * self.z
        )
        new.append((lhs, rhs))

        # z deriv:
        lhs = self.z.diff(self.b, order)
        rhs = (-1) ** order * self.u[order] * self.z
        new.append((lhs, rhs))

        self._data.append(new)



class _CentralBase(object):

    u1, x, k, du, dxdu = _get_default_symbols('u1','x','k','du','dxdu')


class _Central_u_dxdu(_CentralBase):
    """
    u = _Central_u_dxdu()

    u[i] = u({du}, {dxdu})
    """

    @gcached(prop=False)
    def _get_ubar_of_dubar(self, n):
        expr = (
            sp.Sum(
                sp.binomial(n, self.k) * self.du[self.k] * self.u1 ** (n - self.k),
                (self.k, 0, n),
            )
            .doit()
            .subs({self.du[0]: 1, self.du[1]: 0})
            .simplify()
        )
        return expr

    def __getitem__(self, n):
        return self._get_ubar_of_dubar(n)


central_u_dxdu = _Central_u_dxdu()


class _Central_xu_dxdu(_CentralBase):
    """
    xu = _Central_xu_dxdu()

    xu[i] = xu({x1}, {du}, {dxdu})
    """
    x1 = _get_default_symbols('x1_symbol')

    @gcached(prop=False)
    def _get_xubar_of_dxdubar(self, n):
        expr = sp.Sum(
            sp.binomial(n, self.k) * self.u1 ** (n - self.k) * self.dxdu[self.k],
            (self.k, 0, n),
        ) + self.x1 * central_u_dxdu[n]
        return expr.doit().subs({self.dxdu[0]: 0}).expand().simplify()

    def __getitem__(self, n):
        return self._get_xubar_of_dxdubar(n)

central_xu_dxdu = _Central_xu_dxdu()


class _Central_xu_dxdu_xbeta(_CentralBase):

    x1 = _get_default_symbols('x1_indexed')

    @gcached(prop=False)
    def _get_xubar_of_dxdubar(self, deriv, n):
        expr = sp.Sum(
            sp.binomial(n, self.k) * self.u1 ** (n - self.k) * self.dxdu[deriv, self.k],
            (self.k, 0, n),
        ) + self.x1[deriv] * central_u_dxdu[n]
        return expr.doit().subs({self.dxdu[deriv, 0]: 0}).expand().simplify()

    def __getitem__(self, idx):
        return self._get_xubar_of_dxdubar(*idx)

central_xu_dxdu_xbeta = _Central_xu_dxdu_xbeta()




class _BaseSym(object):
    """
    Base class for symbolic computations
    """

    # symbols:
    b = sp.symbols("b")
    f = sp.Function("f")(b)
    z = sp.Function("z")(b)
    _ave_func = f / z

    u = sp.IndexedBase("u")
    xu = sp.IndexedBase("xu")


class _BaseIndex(object):
    """
    base class for index-able stuff
    """

    def __init__(self):
        self._data = {}

    def _get_order(self, order):
        raise NotImplementedError("to be implemented in subclass")

    def __getitem__(self, order):
        return self._get_order(order)

    def __repr__(self):
        if len(self._data) > 0:
            return "<{}, order={}>".format(
                self.__class__.__name__, max(self._data.keys())
            )
        else:
            return "<{}>".format(self.__class__.__name__)


class _SymDeriv(_BaseIndex, _BaseSym):
    """
    object to handle symbolic differentiation
    """

    def __init__(self):
        super(_SymDeriv, self).__init__()
        self._data[0] = self._ave_func

    def _get_order(self, order):
        # recursively calculate derivative
        if order not in self._data:
            self._data[order] = self._get_order(order - 1).diff(self.b, 1)
        return self._data[order]


class _Subs(_BaseIndex, _BaseSym):
    """
    object to handle substitution values
    """

    def __init__(self):
        super(_Subs, self).__init__()
        self._data[0] = {self.f: self.xu[0] * self.z}

    def _get_order(self, order):
        # NOTE: instead of querying the derivative,
        # here, we just build up all f, z derivatives
        # which is equivalent.
        assert order >= 0
        if order not in self._data:
            subs = {}
            # f deriv:
            lhs = self.f.diff(self.b, order)
            rhs = (-1) ** order * self.xu[order] * self.z
            subs[lhs] = rhs

            # z deriv:
            lhs = self.z.diff(self.b, order)
            rhs = (-1) ** order * self.u[order] * self.z
            subs[lhs] = rhs

            self._data[order] = subs
        return self._data[order]


class _Subsxbeta(_BaseIndex, _BaseSym):
    """
    substitutions with x = func(beta)
    """

    k = sp.symbols("k")

    def __init__(self):
        super(_Subsxbeta, self).__init__()
        self._data[0] = {self.f: self.xu[0, 0] * self.z}

    def _get_order(self, order):

        # NOTE: instead of querying the derivative,
        # here, we just build up all f, z derivatives
        # which is equivalent.
        assert order >= 0
        if order not in self._data:
            subs = {}
            # f deriv:
            lhs = self.f.diff(self.b, order)
            rhs = (
                sp.Sum(
                    (
                        (-1) ** self.k
                        * sp.binomial(order, self.k)
                        * self.xu[order - self.k, self.k]
                    ),
                    (self.k, 0, order),
                ).doit()
                * self.z
            )
            subs[lhs] = rhs

            # z deriv:
            lhs = self.z.diff(self.b, order)
            rhs = (-1) ** order * self.u[order] * self.z
            subs[lhs] = rhs

            self._data[order] = subs
        return self._data[order]


class _SubsCentralMoments(_BaseSym):

    k = sp.symbols("k")
    du = sp.IndexedBase("du")
    dxdu = sp.IndexedBase("dxdu")
    x = sp.IndexedBase("x")

    u1, x1 = sp.symbols("u1, x1")

    def __init__(self):
        self._data = {}

        self._order = 0
        self._data[self.u[0]] = self._get_ubar_of_dubar(0)
        self._data[self.xu[0]] = self._get_xubar_of_dxdubar(0)

    @gcached(prop=False)
    def _get_ubar_of_dubar(self, n):
        expr = (
            sp.Sum(
                sp.binomial(n, self.k) * self.du[self.k] * self.u1 ** (n - self.k),
                (self.k, 0, n),
            )
            .doit()
            .subs({self.du[0]: 1, self.du[1]: 0})
            .simplify()
        )
        return expr

    @gcached(prop=False)
    def _get_xubar_of_dxdubar(self, n):
        expr = sp.Sum(
            sp.binomial(n, self.k) * self.u1 ** (n - self.k) * self.dxdu[self.k],
            (self.k, 0, n),
        ) + self.x1 * self._get_ubar_of_dubar(n)
        return expr.doit().subs({self.dxdu[0]: 0}).expand().simplify()

    def _update(self, order):
        if order > self._order:
            for i in range(self._order + 1, order + 1):
                self._data[self.u[i]] = self._get_ubar_of_dubar(i)
                self._data[self.xu[i]] = self._get_xubar_of_dxdubar(i)

    def __getitem__(self, order):
        self._update(order)
        return self._data


class _SubsCentralMomentsxbeta(_SubsCentralMoments):

    x1 = sp.IndexedBase("x1")

    def __init__(self):

        self._data = {}

        self._order = 0
        self._data[self.u[0]] = self._get_ubar_of_dubar(0)
        self._data[self.xu[0, 0]] = self._get_xubar_of_dxdubar(0, 0)

    @gcached(prop=False)
    def _get_xubar_of_dxdubar(self, deriv, n):
        expr = sp.Sum(
            sp.binomial(n, self.k) * self.u1 ** (n - self.k) * self.dxdu[deriv, self.k],
            (self.k, 0, n),
        ) + self.x1[deriv] * self._get_ubar_of_dubar(n)
        return expr.doit().subs({self.dxdu[deriv, 0]: 0}).expand().simplify()

    def _update(self, order):
        if order > self._order:
            for o in range(self._order + 1, order + 1):
                self._data[self.u[o]] = self._get_ubar_of_dubar(o)
            for o in range(order + 1):
                for d in range(order + 1):
                    key = self.xu[d, o]
                    if key not in self._data:
                        self._data[key] = self._get_xubar_of_dxdubar(d, o)


class _SymSubs(_BaseIndex):
    def __init__(self, funcs, subs, subs_final=None):
        self.funcs = funcs
        self.subs = subs
        self.subs_final = subs_final
        super(_SymSubs, self).__init__()

    def _get_order(self, order):
        if order not in self._data:
            func = self.funcs[order]

            if self.subs is not None:
                for o in range(order, -1, -1):
                    func = func.subs(self.subs[o])

            if self.subs_final is not None:
                func = func.subs(self.subs_final[order])

            self._data[order] = func.simplify().expand()
        return self._data[order]


class _LambdifyBase(_BaseIndex):
    def __init__(self, funcs, args=None, **opts):
        self.funcs = funcs

        self.args = args
        self.opts = opts
        super(_LambdifyBase, self).__init__()

    def _get_order(self, order):
        if order not in self._data:
            self._data[order] = sp.lambdify(self.args, self.funcs[order], **self.opts)
        return self._data[order]


class _Lambdify(_LambdifyBase, _BaseSym):
    def __init__(self, funcs, args=None, **opts):
        if args is None:
            args = (self.u, self.xu)
        super(_Lambdify, self).__init__(funcs=funcs, args=args, **opts)


# -Log(X)
class _SymMinusLog(_BaseIndex):
    X = sp.IndexedBase("X")
    dX = sp.IndexedBase("dX")
    k = sp.symbols("k")

    def __init__(self):
        super(_SymMinusLog, self).__init__()
        self._data[0] = -sp.log(self.X[0])
        self._subs = {}
        self._order = 0

    def _add_order(self):
        order = self._order + 1
        expr = 0
        for k in range(1, order + 1):
            expr += (
                sp.factorial(k - 1) * (-1 / self.X[0]) ** k * sp.bell(order, k, self.dX)
            )

        self._order = order
        self._subs[self.dX[order - 1]] = self.X[order]
        self._data[order] = expr.subs(self._subs).simplify()

    def _get_order(self, order):
        while order > self._order:
            self._add_order()
        return self._data[order]


class _LambdifyMinusLog(_LambdifyBase):
    def __init__(self, funcs, args=None, **opts):
        if args is None:
            args = funcs.X
        super(_LambdifyMinusLog, self).__init__(funcs=funcs, args=args, **opts)


class CoefsMinusLog(object):
    def __init__(self):
        self.exprs = _SymMinusLog()
        self.funcs = _LambdifyMinusLog(self.exprs, args=(self.exprs.X,))

    def coefs(self, X, order=None):
        if order is None:
            order = len(X) - 1
        return [self.funcs[i](X) for i in range(order + 1)]


@lru_cache(20)
def factory_CoefsMinusLog():
    return CoefsMinusLog()


class _CoefsBase(object):
    def __init__(self, xbeta=False, subs_final=None, args=None, minus_log=None):

        self.derivs = _SymDeriv()
        if xbeta:
            self.subs = _Subsxbeta()
        else:
            self.subs = _Subs()

        self.subs_final = subs_final
        self.exprs = _SymSubs(self.derivs, self.subs, self.subs_final)
        self.funcs = _Lambdify(self.exprs, args=args)

        if minus_log is True:
            minus_log = factory_CoefsMinusLog().coefs
        self.minus_log = minus_log


class Coefs(_CoefsBase):
    def __init__(self, xbeta=False, minus_log=None):
        super(Coefs, self).__init__(
            xbeta=xbeta, subs_final=None, args=None, minus_log=minus_log
        )

    def coefs(self, u, xu, order, norm=True):
        """
        coefficients of exapnsion up to specified order
        """

        out = [self.funcs[i](u, xu) for i in range(order + 1)]

        if self.minus_log is not None:
            out = self.minus_log(out)

        if norm:
            out = [x / np.math.factorial(i) for i, x in enumerate(out)]
        return out

    def xcoefs(
        self,
        ds,
        order=None,
        u="u_selector",
        xu="xu_selector",
        order_name="order",
        norm=True,
    ):
        if order is None:
            order = ds.order
        out = self.coefs(u=getattr(ds, u), xu=getattr(ds, xu), order=order, norm=norm)
        return xr.concat(out, dim=order_name)


class CoefsCentral(_CoefsBase):
    def __init__(self, xbeta=False, minus_log=None):

        if xbeta:
            subs_final = _SubsCentralMomentsxbeta()
        else:
            subs_final = _SubsCentralMoments()

        args = (subs_final.x1, subs_final.du, subs_final.dxdu)

        super(CoefsCentral, self).__init__(
            xbeta=xbeta, subs_final=subs_final, args=args, minus_log=minus_log
        )

    def coefs(self, xave, du, dxdu, order, norm=True):
        """
        coefficients of exapnsion up to specified order
        """
        out = [self.funcs[i](xave, du, dxdu) for i in range(order + 1)]

        if self.minus_log is not None:
            out = self.minus_log(out)

        if norm:
            out = [x / np.math.factorial(i) for i, x in enumerate(out)]
        return out

    def xcoefs(
        self,
        ds,
        order=None,
        du="du_selector",
        dxdu="dxdu_selector",
        xave="xave_selector",
        order_name="order",
        norm=True,
    ):
        if order is None:
            order = ds.order
        out = self.coefs(
            xave=getattr(ds, xave),
            du=getattr(ds, du),
            dxdu=getattr(ds, dxdu),
            order=order,
            norm=norm,
        )
        return xr.concat(out, dim=order_name)


# will usually use the same coeffs
@lru_cache(10)
def factory_coefs(xbeta=False, central=False, minus_log=None):
    if not central:
        return Coefs(xbeta=xbeta, minus_log=minus_log)
    else:
        return CoefsCentral(xbeta, minus_log=minus_log)


class ExtrapModel(object):
    def __init__(self, order, beta0, data, coefs=None, minus_log=None):
        self.order = order
        self.beta0 = beta0
        self.data = data

        if coefs is None:
            if hasattr(self.data, "xbeta"):
                coefs = factory_coefs(xbeta=self.data.xbeta, minus_log=minus_log)
            else:
                raise ValueError("must speficy coefs")
        self.coefs = coefs

    @gcached(prop=False)
    def _default_xcoefs(self, order=None, order_name="order", norm=True):
        if order is None:
            order = self.order
        return self.coefs.xcoefs(self.data, order, order_name=order_name, norm=norm)

    def xcoefs(self, order=None, order_name="order", norm=True):
        return self._default_xcoefs(order, order_name, norm=norm)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(
        self, beta, order=None, xcoefs=None, order_name="order", cumsum=False,
    ):
        if order is None:
            order = self.order
        if xcoefs is None:
            xcoefs = self._default_xcoefs(order, order_name, norm=True)

        beta = xrwrap_beta(beta)

        dbeta = beta - self.beta0
        p = xr.DataArray(np.arange(order + 1), dims=order_name)
        prefac = dbeta ** p

        out = (prefac * xcoefs.sel(**{order_name: prefac[order_name]})).assign_coords(
            dbeta=dbeta, beta0=self.beta0
        )

        if cumsum:
            out = out.cumsum(order_name)
        else:
            out = out.sum(order_name)

        # if delta:
        #     out = (out, dbeta)

        return out

    def resample(self, nrep, idx=None, **kws):
        return self.__class__(
            order=self.order,
            beta0=self.beta0,
            coefs=self.coefs,
            data=self.data.resample(nrep=nrep, idx=idx, **kws),
        )

    @classmethod
    def from_values(
        cls, order, beta0, uv, xv, xbeta=False, central=False, minus_log=None, **kws
    ):
        """
        build a model from data
        """
        data = factory_data(
            uv=uv, xv=xv, order=order, xbeta=xbeta, central=central, **kws
        )
        coefs = factory_coefs(xbeta=xbeta, central=central, minus_log=minus_log)
        return cls(order=order, beta0=beta0, coefs=coefs, data=data)


class _StateCollection(object):
    def __init__(self, states):
        self.states = states

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    def resample(self, nrep, idxs=None, **kws):
        if idxs is None:
            idxs = [None] * len(self)
        assert len(idxs) == len(self)

        return self.__class__(
            states=tuple(
                state.resample(nrep=nrep, idx=idx, **kws)
                for state, idx in zip(self.states, idxs)
            )
        )

    @gcached()
    def order(self):
        return min([m.order for m in self])


def xr_weights_minkowski(deltas, m=20, dim="state"):
    deltas_m = deltas ** m
    return 1.0 - deltas_m / deltas_m.sum(dim)


class ExtrapWeightedModel(_StateCollection):
    def predict(self, beta, order=None, xcoefs=None, order_name="order", cumsum=False):

        if order is None:
            order = self.order
        if xcoefs is None:
            xcoefs = [None] * len(self)
        assert len(xcoefs) == len(self)

        out = xr.concat(
            [
                m.predict(
                    beta, order=order, xcoefs=c, order_name=order_name, cumsum=cumsum
                )
                for (m, c) in zip(self.states, xcoefs)
            ],
            dim="state",
        )

        w = xr_weights_minkowski(np.abs(out.dbeta))
        out = (out * w).sum("state") / w.sum("state")
        return out


class InterpModel(_StateCollection):
    @gcached(prop=False)
    def _default_coefs(self, order=None, order_name="porder"):

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
            beta = m.beta0
            for j in range(order + 1):
                with np.errstate(divide="ignore"):
                    val = (
                        (beta ** (power - j))
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
            xr.DataArray(mat_inv, dims=[order_name, "state_order"])
            .assign_coords(state=("state_order", states))
            .assign_coords(order=("state_order", orders))
            .set_index(state_order=["state", "order"])
            .unstack()
        )

        coefs = xr.concat(
            [m._default_xcoefs(order, norm=False) for m in self.states], dim="state"
        )
        if isinstance(coefs, xr.Dataset):
            coefs = xr.Dataset({k: xr.dot(mat_inv, v) for k, v in coefs.items()})
        else:
            coefs = xr.dot(mat_inv, coefs)

        return coefs

    def predict(self, beta, order=None, order_name="porder"):

        if order is None:
            order = self.order

        xcoefs = self._default_coefs(order, order_name)
        beta = xrwrap_beta(beta)

        porder = len(xcoefs[order_name]) - 1

        p = xr.DataArray(np.arange(porder + 1), dims=order_name)
        prefac = beta ** p

        out = (prefac * xcoefs).sum(order_name)
        return out


class PerturbModel(object):
    def __init__(self, beta0, data):

        self.beta0 = beta0
        self.data = data

    def predict(self, beta):

        beta = xrwrap_beta(beta)
        uv = self.data.uv
        xv = self.data.xv

        beta0 = self.beta0

        rec = self.data._rec
        dbeta = beta - beta0

        dbeta_uv = (-1.0) * dbeta * uv
        dbeta_uv_diff = dbeta_uv - dbeta_uv.max(rec)
        expvals = np.exp(dbeta_uv_diff)

        num = xr.dot(expvals, xv, dims="rec") / len(xv[rec])
        den = expvals.mean("rec")

        return num / den

    def resample(self, nrep, idx=None, **kws):
        return self.__class__(
            beta0=self.beta0, data=self.data.resample(nrep=nrep, idx=idx, **kws)
        )

    @classmethod
    def from_values(cls, beta0, uv, xv, **kws):
        data = Data(uv, xv, order=0, **kws)
        return cls(beta0=beta0, data=data)


class MBARModel(_StateCollection):
    """
    Sadly, this doesn't work as beautifully.
    """

    def __init__(self, states):
        if not _HAS_PYMBAR:
            raise ImportError("need pymbar to use this")
        super(MBARModel, self).__init__(states)

    @gcached(prop=False)
    def _default_params(self, state_name="state", beta_name="beta"):

        # all xvalues:
        xv = xr.concat([m.data.xv for m in self], dim=state_name)
        uv = xr.concat([m.data.uv for m in self], dim=state_name)
        beta0 = xrwrap_beta([m.beta0 for m in self], name=beta_name)

        # make sure uv, xv in correct order
        rec = self[0].data._rec
        xv = xv.transpose(state_name, rec, ...)
        uv = uv.transpose(state_name, rec, ...)

        # beta[beta] * uv[state, rec] = out[beta, state, rec]
        Ukn = (beta0 * uv).values.reshape(len(self), -1)
        N = np.ones(len(self)) * len(xv["rec"])
        mbar_obj = mbar.MBAR(Ukn, N)

        return uv, xv, beta0, mbar_obj

    def predict(self, beta):
        beta = xrwrap_beta(beta)
        if beta.ndim == 0:
            beta = beta.expand_dims(beta.name)

        uv, xv, beta0, mbar_obj = self._default_params("state", beta.name)

        dims = xv.dims
        x = np.array(xv, order="c")
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)

        U = uv.values.reshape(-1)

        out = []
        for b in beta.values:
            out.append(mbar_obj.computeMultipleExpectations(x_flat.T, b * U)[0])

        out = np.array(out)
        # reshape
        shape = (out.shape[0],) + x.shape[2:]
        out = xr.DataArray(
            out.reshape(shape), dims=(beta.name,) + dims[2:]
        ).assign_coords(beta=beta)

        return out


# def extrap_to_poly(B0, derivs):
#     """Converts an extrapolation around a reference point to a polynomial over all real
#      numbers by collecting terms. Input is the reference state point and the derivatives
#      at that state point (starting with the zeroth derivative, which is just the
#      observable value). Only works for SINGLE observable element if observable is a
#      vector (so derivs must be a 1D array).
#   """
#     coeffs = np.zeros(len(derivs))
#     for k, d in enumerate(derivs):
#         for l in range(k + 1):
#             coeffs[l] += (
#                 (-B0)**(k - l)) * d * sp.binom(k, l) / np.math.factorial(k)
#     return coeffs

# def _build_aves(u, x, order):
#     x = np.array(x)
#     u = np.array(u)

#     # u[rec, rep]
#     # x[rec, val, rep]

#     if u.ndim == 1:
#         u = u[:, None]
#     if x.ndim == 1:
#         x = x[:, None, None]
#     elif x.ndim == 2:
#         x = x[..., None]

#     assert u.ndim == 2
#     assert x.ndim == 3
#     assert u.shape[0] == x.shape[0]
#     assert u.shape[-1] == x.shape[-1]

#     # un[rec, order, rep] = u[rec, None, rep] ** n[None,:, None]
#     n = np.arange(order+1)[None, :, None]
#     un = u[:, None, :] ** n

#     # u[order, rep]
#     u_mean = un.mean(0)
#     # xu[order, val, rep] <- un[rec, order, None, rep] * x[rec, None, val, rep]
#     xu_mean = (un[:, :, None, :] * x[:,None, :, :]).mean(axis=0)
#     return u_mean, xu_mean

# def _build_aves_xbeta(u, x, order=None):
#     x = np.array(x)
#     u = np.array(u)

#     # u[rec, rep]
#     if u.ndim == 1:
#         u = u[:, None]

#     # x[rec, deriv, val, rep]
#     if x.ndim == 3:
#         x = x[..., None]

#     assert u.ndim == 2
#     assert x.ndim == 4
#     assert u.shape[0] == x.shape[0]
#     assert u.shape[-1] == x.shape[-1]

#     if order is None:
#         order = x.shape[1] - 1
#     if x.shape[1] < order + 1:
#         raise ValueError('request order beyond derivatives')

#     # un[rec, order, rep]
#     n = np.arange(order+1)[None, :, None]
#     un = u[:, None, :] ** n

#     # u_mean[order, rep]
#     u_mean = un.mean(0)

#     # xu[deriv, order, val, rep] <-
#     #      un[rec, None, order, None, rep] *
#     #       x[rec, deriv, None, val, rep]
#     xu_mean = (un[:, None, :, None, :] * x[:, :, None, :, :]).mean(axis=0)
#     return u_mean, xu_mean

# def build_aves(u, x, order, xbeta=False, squeeze_rep=True):
#     """
#     Build averages

#     Parameters
#     ----------
#     x : array-like
#         shape=(nrec) or (nrec, nrep)
#     u : array-like
#         if xbeta is False, then
#         shape=(nrec,nval) or (nrec, nval, nrep)
#         if xbeta is True, then
#         shape=(nrec,nderiv,nval) or (nrec, nderiv, nval, nrep)
#     """
#     if xbeta:
#         func = _build_aves_xbeta
#     else:
#         func = _build_aves

#     U, XU =  func(u=u, x=x, order=order)
#     if squeeze_rep and U.shape[-1] == 1:
#         U = U[..., 0]
#         XU = XU[..., 0]

#     return U, XU

# def resampl_values(u, x, nrep, ret_idx):
#     size = x.shape[0]

#     X = []
#     U = []
#     I = []
#     for i in range(nrep):
#         idx = np.random.choice(size, size=size, replace=True)
#         I.append(idx)
#         X.append(x[idx])
#         U.append(u[idx])

#     X = np.stack(X, axis=-1)
#     U = np.stack(U, axis=-1)

#     if ret_idx:
#         return U, X, I
#     else:
#         return U, X

# class _DictWrapper(object):
#     """
#     container for dict functions

#     NOTE: Don't actually need this because switched to IndexedBase for u, xu
#     """
#     def __init__(self, data):
#         self.data = data
#     def __call__(self, order):
#         return self.data[order]

# class SymDeriv(object):
#     def __init__(self):
#         # symbols:
#         self._b = sp.symbols('b')
#         self._f = sp.Function('f')(self._b)
#         self._z = sp.Function('z')(self._b)

#         self._u = sp.IndexedBase('u')
#         self._xu = sp.IndexedBase('xu')

#         # self._u  = sp.Function('u')
#         # self._xu = sp.Function('xu')
#         self._ave_func = self._f / self._z

#         self._derivs = {0: self._ave_func}

#         self._funcs = {}
#         self._funcs_np = {}

#     def deriv(self, order):
#         # recursive differentiation
#         if order not in self._derivs:
#             self._derivs[order] = self.deriv(order-1).diff(self._b, 1)
#         return self._derivs[order]

#     def _set_func(self, order):
#         deriv = self.deriv(order) #self._ave_func.diff(self._b, order)

#         tosub = deriv.atoms(sp.Function, sp.Derivative)
#         #When we sub in, must do in order of highest to lowest derivatives, then functions
#         #Otherwise substitution doesn't work because derivatives computed recursively by sympy
#         for o in range(order + 1)[::-1]:
#             subvals = {}
#             if o == 0:
#                 for d in tosub:
#                     if isinstance(d, sp.Function):
#                         if str(d) == str(self._f): #'f(b)':
#                             subvals[d] = self._xu[0] * self._z

#             else:
#                 for d in tosub:
#                     if isinstance(d, sp.Derivative) and d.derivative_count == o:
#                         if str(d.expr) == str(self._f): #'f(b)':
#                             subvals[d] = (
#                                 ((-1)**d.derivative_count) *
#                                 self._xu[d.derivative_count] * self._z)

#                         elif str(d.expr) == str(self._z): #'z(b)':
#                             subvals[d] = (
#                                 ((-1)**d.derivative_count) *
#                                 self._u[d.derivative_count] * self._z)

#             #Substitute derivatives for functions u and xu at this order
#             deriv = deriv.subs(subvals)

#         #To allow for vector-valued function inputs and to gain speed, lambdify
#         deriv = deriv.simplify().expand()
#         self._funcs[order] = deriv

#     def expr(self, order):
#         if order not in self._funcs:
#             self._set_func(order)
#         return self._funcs[order]

#     def func(self, order):
#         if order not in self._funcs_np:
#             self._funcs_np[order] = sp.lambdify((self._u, self._xu), self.expr(order))
#         return self._funcs_np[order]

""" to handle data objects
"""


from __future__ import absolute_import

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from .cached_decorators import gcached
from .xrutils import xrwrap_uv, xrwrap_xv

try:
    import cmomy.xcentral as xcentral

    _HAS_CMOMY = True
except ImportError:
    _HAS_CMOMY = False


# NOTE: General scheme:
# uv, xv -> samples (values) for u, x
# u, xu -> averages of u and x*u
# u[i] = <u**i>
# xu[i] = <x * u**i>
# xu[i, j] = <d^i x/d beta^i * u**j


def resample_indicies(size, nrep, rec_dim="rec", rep_dim="rep", replace=True):
    """
    get indexing DataArray

    Parameters
    ----------
    size : int
        size of axis to bootstrap along
    nrep : int
        number of replicates
    rec_dim : str, default='rec'
        name of record dimension.
        That is, the dimension to sample along
    rep_dim : str, default='rep'
        name of replicate dimension.
        That is, the new resampledd dimension.
    replace : bool, default=True
        if True, sample with replacement.
    transpose : bool, default=False
        see `out`

    Returns
    -------
    indices : xarray.DataArray
        if transpose, shape=(size, nrep)
        else, shape=(nrep, size)
    """
    indices = xr.DataArray(
        data=np.random.choice(size, size=(nrep, size), replace=replace),
        dims=[rep_dim, rec_dim],
    )

    # if transpose:
    #     indices = indices.transpose(rec_dim, rep_dim)
    return indices


class DatasetSelector(object):
    """
    wrap dataset so can index like ds[i, j]

    Needed for calling sympy.lambdify functions
    """

    def __init__(self, data, dims=None, mom_dim="moment", deriv_dim=None):

        # Default dims
        if dims is None:
            if deriv_dim is not None:
                dims = [mom_dim, deriv_dim]
            else:
                dims = [mom_dim]

        # if dims is None:
        #     if deriv_dim in data.dims:
        #         dims = [mom_dim, deriv_dim]
        #     else:
        #         dims = [mom_dim]

        if isinstance(dims, str):
            dims = [dims]

        for d in dims:
            if d not in data.dims:
                raise ValueError("{} not in dims {}".format(d, data.dims))
        self.data = data
        self.dims = dims

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != len(self.dims):
            raise ValueError("bad idx {}, vs dims {}".format(idx, self.dims))
        selector = dict(zip(self.dims, idx))
        return self.data.isel(**selector, drop=True)

    def __repr__(self):
        return repr(self.data)


class NewLikeAssignMixin(ABC):
    @property
    @abstractmethod
    def param_names(self):
        """
        returns sequence of string names to be copied over
        """
        raise NotImplementedError

    def new_like(self, **kws):
        """
        create new object with optional parameters
        """
        kws_default = {k: getattr(self, k) for k in self.param_names}
        kws = dict(kws_default, **kws)
        return type(self)(**kws)

    def assign_params(self, **kws):
        """
        create new object with optional parameters

        Same as self.new_like
        """
        return self.new_like(**kws)

    def set_params(self, **kws):
        """
        set parameters of self, and retun self (for chaining)
        """

        for name in kws.keys():
            assert hasattr(self, name)

        for name, val in kws.items():
            setattr(self, name, val)
        return self


class DataCallbackABC(NewLikeAssignMixin):
    """
    Base class for handling callbacks to adjust data.

    For some cases, the default Data classes don't quite cut it.
    For example, for volume extrapolation, extrap parameters need to
    be included in the derivatives.  To handle this generally,
    the Data class include `self.meta` which performs these actions.

    DataCallback can be subclassed to fine tune things.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def check(self, data):
        """Perform any consistency checks between self and data"""
        pass

    @abstractmethod
    def derivs_args(self, data, derivs_args):
        """adjust derivs args from data class

        should return a tuple
        """
        return derivs_args

    # define these to raise error instead
    # of forcing usage.
    def resample(self, data, meta_kws, **kws):
        """adjust create new object

        Should return new instance of class or self no change
        """
        raise NotImplementedError

    def block(self, data, meta_kws, **kws):
        raise NotImplementedError

    def reduce(self, data, meta_kws, **kws):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class DataCallback(DataCallbackABC):
    """
    Basic version of DataCallbackABC.

    Implemented to pass things through unchanged.  Will be used for default construction

    """

    def __init__(self, *args, **kwargs):
        pass

    def check(self, data):
        """Perform any consistency checks between self and data"""
        pass

    @property
    def param_names(self):
        """returns sequence of string names to be copied over"""
        return ()

    def derivs_args(self, data, derivs_args):
        """adjust derivs args from data class

        should return a tuple
        """
        return derivs_args

    def resample(self, data, meta_kws, **kws):
        """adjust create new object

        Should return new instance of class or self no change
        """
        return self

    def block(self, data, meta_kws, **kws):
        return self

    def reduce(self, data, meta_kws, **kws):
        return self


class AbstractData(NewLikeAssignMixin):
    @property
    @abstractmethod
    def order(self):
        pass

    @property
    @abstractmethod
    def central(self):
        pass

    @property
    @abstractmethod
    def derivs_args(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def resample(self, indices=None, nrep=None, **kwargs):
        pass

    @property
    def xalpha(self):
        """Whether X has explicit dependence on `alpha`

        That is, if `self.deriv_dim` is not `None`
        """
        return self.deriv_dim is not None

    # metheds for working with meta
    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, m):
        """
        set value and check meta
        """
        if m is None:
            m = DataCallback()
        elif not isinstance(m, DataCallbackABC):
            raise ValueError("meta must be None or subclass of DataCallbackABC")
        m.check(data=self)
        self._meta = m

    @property
    def x_isnot_u(self):
        return not self.x_is_u

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    # def __repr__(self):
    #     return f'<{self.__class__.__name__}>'


class DataValuesBase(AbstractData):
    def __init__(
        self,
        uv,
        xv,
        order,
        rec_dim="rec",
        umom_dim="umom",
        deriv_dim=None,
        skipna=False,
        chunk=None,
        compute=None,
        build_aves_kws=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Parameters
        ----------
        uv : xarray.DataArray
            raw values of u (energy)
        xv : xarray.DataArray
            raw values of x (observable)
        order : int
            maximum order of moments to calculate
        rec_dim : str, default='rec'
            Name of dimension to average along.
        umom_dim : str, default='umom',
            Name of moment dimension < u ** umom >
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        skipna : bool, default=False
            if True, skip nan values
        chunk : bool, optional
            chunking of xarray objects
        compute : bool, optional
            whether to perform compute step on xarray outputs
        meta : dict, optional
            extra meta data/parameters to be caried along with object and child objects.
            if 'checker' in meta, then perform a callback of the form meta['checker](self, meta)
            this can also be used to hotwire things like derivs_args.
        x_is_u : bool, default=False
            if True, treat `xv = uv` and do adjust u/du accordingly
        Values passed through method `resample_meta`
        """

        assert isinstance(uv, xr.DataArray)
        self.x_is_u = x_is_u

        if xv is not None:
            assert isinstance(xv, (xr.DataArray, xr.Dataset))

        if chunk is not None:
            if isinstance(chunk, int):
                chunk = {rec_dim: chunk}

            uv = uv.chunk(chunk)
            if xv is not None or self.x_isnot_u:
                xv = xv.chunk(chunk)

        if compute is None:
            # default compute
            # if not chunk, default compute is False
            # if chunk, default compute is True
            if chunk is None:
                compute = False
            else:
                compute = True

        if build_aves_kws is None:
            build_aves_kws = {}

        self.build_aves_kws = build_aves_kws

        self.uv = uv
        if xv is None or self.x_is_u:
            xv = uv
        self.xv = xv

        self._order = order

        self.chunk = chunk
        self.compute = compute
        self.skipna = skipna

        # dimension names
        self.rec_dim = rec_dim
        self.umom_dim = umom_dim
        self.deriv_dim = deriv_dim
        self.meta = meta

    @property
    def param_names(self):
        return [
            "uv",
            "xv",
            "order",
            "rec_dim",
            "umom_dim",
            "deriv_dim",
            "skipna",
            "chunk",
            "compute",
            "build_aves_kws",
            "meta",
            "x_is_u",
        ]

    @classmethod
    def from_vals(
        cls,
        xv,
        uv,
        order,
        rec_dim="rec",
        umom_dim="umom",
        rep_dim="rep",
        deriv_dim=None,
        val_dims="val",
        skipna=False,
        chunk=None,
        compute=None,
        build_aves_kws=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Constructor from arrays

        Parameters
        ----------
        uv : array-like
            raw values of u (energy)
            if not DataArray, wrap with `xrwrap_uv`
        xv : xarray.DataArray
            raw values of x (observable)
            if not DataArray, wrap with `xrwrap_xv`
        order : int
            maximum order of moments to calculate
        rec_dim : str, default='rec'
            Name of dimension to average along.
        umom_dim : str, default='umom',
            Name of moment dimension <u**umom_dim>
        val_dims : str or list-like
            names of extra dimensions
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        skipna : bool, default=False
            if True, skip nan values
        chunk : bool, optional
            chunking of xarray objects
        compute : bool, optional
            whether to perform compute step on xarray outputs
        meta : DataCallback, optional
            extra keyword arguments
        """

        # make sure "val" is a list
        if isinstance(val_dims, str):
            val_dims = [val_dims]
        elif not isinstance(val_dims, list):
            val_dims = list(val_dims)

        uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)

        if xv is not None:
            xv = xrwrap_xv(
                xv,
                rec_dim=rec_dim,
                rep_dim=rep_dim,
                deriv_dim=deriv_dim,
                val_dims=val_dims,
            )

        return cls(
            uv=uv,
            xv=xv,
            order=order,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            deriv_dim=deriv_dim,
            skipna=skipna,
            chunk=chunk,
            compute=compute,
            build_aves_kws=build_aves_kws,
            meta=meta,
            x_is_u=x_is_u,
        )

    @property
    def order(self):
        return self._order

    @property
    def central(self):
        return self._CENTRAL

    def __len__(self):
        return len(self.uv[self.rec_dim])

    def resample(
        self,
        indices=None,
        nrep=None,
        rep_dim="rep",
        chunk=None,
        compute="None",
        meta_kws=None,
    ):
        """
        resample object

        Parameters
        ----------
        indices : array-like, shape=(nrep, nrec), optional
            if present, use this to resample
        nrep : int, optional
            construct resampling with `nrep` samples
        rep_dim : str, default='rep'
            dimension name for repetition dimension
        chunk : optional
            chunk size
        compute : optional
            if compute is the string 'None', then pass compute=None to constructor.
            Otherwise, inheret compute


        """

        if chunk is None:
            chunk = self.chunk

        if compute == "None":
            compute = None
        elif compute is None:
            compute = self.compute

        if rep_dim is None:
            rep_dim = self.rep_dim

        if indices is None:
            assert nrep is not None
            indices = resample_indicies(
                len(self), nrep, rec_dim=self.rec_dim, rep_dim=rep_dim
            )
        elif not isinstance(indices, xr.DataArray):
            indices = xr.DataArray(indices, dims=(rep_dim, self.rec_dim))

        assert indices.sizes[self.rec_dim] == len(self)

        uv = self.uv.compute()[indices]
        if self.x_isnot_u:
            xv = self.xv.compute().isel(**{self.rec_dim: indices})
        else:
            xv = None

        meta = self.meta.resample(
            data=self,
            meta_kws=meta_kws,
            indices=indices,
            nrep=nrep,
            rep_dim=rep_dim,
            chunk=chunk,
            compute=compute,
        )

        return self.__class__(
            uv=uv,
            xv=xv,
            order=self.order,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            skipna=self.skipna,
            chunk=chunk,
            compute=compute,
            build_aves_kws=self.build_aves_kws,
            meta=meta,
            x_is_u=self.x_is_u,
        )


###############################################################################
# Data
###############################################################################
def build_aves_xu(
    uv,
    xv,
    order,
    rec_dim="rec",
    umom_dim="umom",
    deriv_dim=None,
    skipna=False,
    u_name=None,
    xu_name=None,
    merge=False,
    transpose=False,
):
    """
    build averages from values uv, xv up to order `order`

    Parameters
    ----------
    umom_dim : dimension with moment order
    deriv_dim : dimension with derivative order

    skipna : bool, default=False
        if True, then handle nan values correctly.  Note that skipna=True
        can drastically slow down the calculations

    """

    assert isinstance(uv, xr.DataArray)
    assert isinstance(xv, (xr.DataArray, xr.Dataset))

    # do averageing

    # this is faster is some cases then the
    # simplier
    u = []
    xu = []

    uave = uv.mean(rec_dim, skipna=skipna)
    xave = xv.mean(rec_dim, skipna=skipna)
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
            xu.append(xu_n.mean(rec_dim, skipna=skipna))
        else:
            u_n *= uv
            xu_n *= uv
            u.append(u_n.mean(rec_dim, skipna=skipna))
            xu.append(xu_n.mean(rec_dim, skipna=skipna))

    u = xr.concat(u, dim=umom_dim)
    xu = xr.concat(xu, dim=umom_dim)

    # simple, but sometimes slow....
    # nvals = xr.DataArray(np.arange(order + 1), dims=[umom_dim])
    # un = uv**nvals
    # u = (un).mean(rec_dim, skipna=skipna)
    # xu = (un * xv).mean(rec_dim, skipna=skipna)

    if transpose:
        u_order = (umom_dim, ...)
        if deriv_dim is not None:
            x_order = (umom_dim, deriv_dim, ...)
        else:
            x_order = u_order
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
    rec_dim="rec",
    umom_dim="umom",
    deriv_dim=None,
    skipna=False,
    du_name=None,
    dxdu_name=None,
    xave_name=None,
    merge=False,
    transpose=False,
):
    """
    build central moments from values uv, xv up to order `order`

    Parameters
    ----------
    umom_dim : dimension with moment order
    deriv_dim : dimension with derivative order

    skipna : bool, default=False
        if True, then handle nan values correctly.  Note that skipna=True
        can drastically slow down the calculations

    """

    assert isinstance(uv, xr.DataArray)
    assert isinstance(xv, (xr.DataArray, xr.Dataset))

    xave = xv.mean(rec_dim, skipna=skipna)
    uave = uv.mean(rec_dim, skipna=skipna)

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
            dxduave.append(dxdu_n.mean(rec_dim, skipna=skipna))

        else:
            du_n *= du
            dxdu_n *= du
            duave.append(du_n.mean(rec_dim, skipna=skipna))
            dxduave.append(dxdu_n.mean(rec_dim, skipna=skipna))

    duave = xr.concat(duave, dim=umom_dim)
    dxduave = xr.concat(dxduave, dim=umom_dim)

    if transpose:
        u_order = (umom_dim, ...)
        if deriv_dim is not None:
            x_order = (deriv_dim,) + u_order
        else:
            x_order = u_order

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


def _xu_to_u(xu, dim="umom"):
    """
    for case where x = u, shift umom and add umom=0
    """

    n = xu.sizes[dim]

    out = xu.assign_coords(**{dim: lambda x: x[dim] + 1}).reindex(**{dim: range(n + 1)})

    # add 0th element
    out.loc[{dim: 0}] = 1.0
    return out


class DataValues(DataValuesBase):
    """
    Class to hold uv/xv data
    """

    _CENTRAL = False

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves_xu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            **self.build_aves_kws,
        )

    @gcached()
    def xu(self):
        out = self._mean()[1]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def u(self):
        if self.x_isnot_u:
            out = self._mean()[0]
            if self.compute:
                out = out.compute()
        else:
            out = _xu_to_u(self.xu, self.umom_dim)

        return out

    @gcached()
    def u_selector(self):
        return DatasetSelector(self.u, deriv_dim=None, mom_dim=self.umom_dim)

    @gcached()
    def xu_selector(self):
        return DatasetSelector(self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim)

    @property
    def derivs_args(self):
        if self.x_isnot_u:
            out = (self.u_selector, self.xu_selector)
        else:
            out = (self.u_selector,)
        return self.meta.derivs_args(data=self, derivs_args=out)


class DataValuesCentral(DataValuesBase):
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

    _CENTRAL = True

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves_dxdu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            **self.build_aves_kws,
        )

    @gcached()
    def xave(self):
        out = self._mean()[0]
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
    def du(self):
        if self.x_isnot_u:
            out = self._mean()[1]
            if self.compute:
                out = out.compute()
        else:
            out = _xu_to_u(self.dxdu, dim=self.umom_dim)

        return out

    @gcached()
    def du_selector(self):
        return DatasetSelector(self.du, deriv_dim=None, mom_dim=self.umom_dim)

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @gcached()
    def xave_selector(self):
        if self.deriv_dim is None:
            return self.xave
        else:
            return DatasetSelector(self.xave, dims=[self.deriv_dim])

    @property
    def derivs_args(self):
        if self.x_isnot_u:
            out = (self.xave_selector, self.du_selector, self.dxdu_selector)
        else:
            out = (self.xave_selector, self.du_selector)

        return self.meta.derivs_args(self, out)


################################################################################
# StatsCov objects
################################################################################
class DataCentralMomentsBase(AbstractData):
    def __init__(
        self,
        dxduave,
        umom_dim="umom",
        xmom_dim="xmom",
        rec_dim="rec",
        deriv_dim=None,
        central=False,
        meta=None,
        x_is_u=False,
    ):
        """
        Data object based on central co-moments array

        Parameters
        ----------
        dxduave : cmomy.xcentral.xStatsAccum object
            central object to work with
        rec_dim : str, default='rec'
            Name of dimension to average along.
        umom_dim : str, default='umom',
            Name of moment dimension on `u`, < u ** umom>
        xmom_dim : str, default='xmom'
            Name of moment dimension on `x` variable
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        central : bool, default=False
            if True, use central moments.
            if False, use raw moments
        meta : dict, optional
            extra meta data/parameters to be passed to child objects.  To be used with care.
        x_is_u : bool, default=False
            If `True` treat x as the same as u, and expose special methods
        """

        self.x_is_u = x_is_u

        self.dxduave = dxduave
        self.umom_dim = umom_dim
        self.xmom_dim = xmom_dim
        self.rec_dim = rec_dim
        self.deriv_dim = deriv_dim
        self.central = central
        self.meta = meta

    @property
    def param_names(self):
        return (
            "dxduave",
            "xmom_dim",
            "umom_dim",
            "rec_dim",
            "deriv_dim",
            "central",
            "meta",
            "x_is_u",
        )

    @property
    def central(self):
        return self._central

    @central.setter
    def central(self, val):
        self._central = val

    @property
    def order(self):
        return self.dxduave.sizes[self.umom_dim] - 1

    @property
    def values(self):
        return self.dxduave.values

    @gcached(prop=False)
    def rmom(self):
        """
        raw co-moments
        """
        return self.dxduave.rmom()

    @gcached(prop=False)
    def cmom(self):
        """
        central co-moments
        """
        return self.dxduave.cmom()

    @gcached()
    def xu(self):
        return self.rmom().sel(**{self.xmom_dim: 1}, drop=True)

    @gcached()
    def u(self):
        if self.x_isnot_u:
            out = self.rmom().sel(**{self.xmom_dim: 0}, drop=True)
            if self.xalpha:
                out = out.sel(**{self.deriv_dim: 0}, drop=True)
        else:
            out = _xu_to_u(self.xu, self.umom_dim)

        return out

    @gcached()
    def xave(self):
        return self.dxduave.values.sel(
            **{self.umom_dim: 0, self.xmom_dim: 1}, drop=True
        )

    @gcached()
    def dxdu(self):
        return self.cmom().sel(**{self.xmom_dim: 1}, drop=True)

    @gcached()
    def du(self):
        if self.x_isnot_u:
            out = self.cmom().sel(**{self.xmom_dim: 0}, drop=True)
            if self.xalpha:
                out = out.sel(**{self.deriv_dim: 0}, drop=True)
        else:
            out = _xu_to_u(self.dxdu, self.umom_dim)

        return out

    @gcached()
    def u_selector(self):
        return DatasetSelector(self.u, deriv_dim=None, mom_dim=self.umom_dim)

    @gcached()
    def xu_selector(self):
        return DatasetSelector(self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim)

    @gcached()
    def xave_selector(self):
        if self.deriv_dim is None:
            return self.xave
        else:
            return DatasetSelector(self.xave, dims=[self.deriv_dim])

    @gcached()
    def du_selector(self):
        return DatasetSelector(self.du, deriv_dim=None, mom_dim=self.umom_dim)

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    def derivs_args(self):
        if self.x_isnot_u:
            if self.central:
                out = (self.xave_selector, self.du_selector, self.dxdu_selector)

            else:
                out = (self.u_selector, self.xu_selector)
        else:
            if self.central:
                out = (self.xave_selector, self.du_selector)
            else:
                out = (self.u_selector,)

        return self.meta.derivs_args(data=self, derivs_args=out)


class DataCentralMoments(DataCentralMomentsBase):
    def __len__(self):
        return self.values.sizes[self.rec_dim]

    def block(self, block_size, axis=None, meta_kws=None, **kwargs):
        """
        block resample along axis

        Paramters
        ---------
        block_size : int
            number of sample to block average together
        axis : int or str, default=self.rec_dim
            axis or dimension to block average along
        **kwargs : dict
            extra arguments to cmomy.xCentralMoments.block
        """

        if axis is None:
            axis = self.rec_dim

        kws = dict(block_size=block_size, axis=axis, **kwargs)
        return self.new_like(
            dxduave=self.dxduave.block(**kws),
            meta=self.meta.block(data=self, meta_kws=meta_kws, **kws),
        )

    def reduce(self, axis=None, meta_kws=None, **kwargs):
        """
        reduce along axis

        Parameters
        ----------
        axis : int or str, default=self.rec_dim
            axis to reduce (combine moments) along
        """
        if axis is None:
            axis = self.rec_dim
        kws = dict(axis=axis, **kwargs)

        return self.new_like(
            dxduave=self.dxduave.reduce(**kws),
            meta=self.meta.reduce(data=self, meta_kws=meta_kws, **kws),
        )

    def resample(
        self,
        freq=None,
        indices=None,
        nrep=None,
        axis=None,
        rep_dim="rep",
        parallel=True,
        resample_kws=None,
        meta_kws=None,
        **kwargs,
    ):

        """
        resample data

        Parameters
        ----------
        freq : array-like, shape=(nrec, nrep)
            frequency table for resampling.
        indices : array-like, shape=(nrep, nrec)
            if specified and `freq` is None, construct frequency table from this.
        nrep : int
            if `freq` and `indices` is None, construct frequency table from this.
        axis : int or string, default=self.rec_dim
            axis or dimension to resample along
        rep_dim : str, default='rep'
            Name of repetition dimension
        parallel : bool, default=True
            If true, perform resampling in parallel
        resample_kws : dict, optional
            dictionary of values to pass to self.dxduave.resample_and_reduce
        """

        if axis is None:
            axis = self.rec_dim

        kws = dict(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            rep_dim=rep_dim,
            parallel=True,
            resample_kws=None,
            **kwargs,
        )

        dxdu_new = self.dxduave.resample_and_reduce(**kws)

        # make new with new data
        new = self.new_like(dxduave=dxdu_new)

        # set meta of new objects
        return new.set_params(
            meta=new.meta.resample(data=new, meta_kws=meta_kws, **kws)
        )

    # TODO : update from_raw from_data to
    # include a mom_dims arguments
    # that defaults to (xmom_dim, umom_dim)
    # so if things are in wrong order, stuff still works out

    @classmethod
    def from_raw(
        cls,
        raw,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Parameters
        ----------
        raw : array-like
            raw moments.  The form of this array is such that
            raw[..., i, j] = weight,        i = j = 0
                           = <x**i * u**j>, otherwise
           The shape should be (..., 2, order+1)
        """

        if x_is_u:
            raw = xr.concat(
                [
                    (
                        raw.sel(**{umom_dim: s}).assign_coords(
                            **{umom_dim: lambda x: range(x.sizes[umom_dim])}
                        )
                    )
                    for s in [slice(None, -1), slice(1, None)]
                ],
                dim=xmom_dim,
            )
            raw = raw.transpose(..., xmom_dim, umom_dim)

        dxduave = xcentral.xCentralMoments.from_raw(
            raw=raw,
            mom=mom,
            mom_ndim=2,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    def from_vals(
        cls,
        xv,
        uv,
        order,
        xmom_dim="xmom",
        umom_dim="umom",
        rec_dim="rec",
        deriv_dim=None,
        central=False,
        w=None,
        axis=0,
        broadcast=True,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        create DataCentralMoments object from individual (unaveraged) samples

        uv, xv are wrapped before execution
        """

        if xv is None or x_is_u:
            xv = uv

        dxduave = xcentral.xCentralMoments.from_vals(
            x=(xv, uv),
            w=w,
            axis=axis,
            mom=(1, order),
            broadcast=broadcast,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    def from_data(
        cls,
        data,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Create DataCentralMoments object from data

        data[..., i, j] = weight                          i = j = 0
                        = < x >                           i = 1 and j = 0
                        = < u >                           i = 0 and j = 1
                        = <(x - <x>)**i * (u - <u>)**j >  otherwise
        """

        if x_is_u:
            # convert from central moments to central co-moments
            # out_0 = data.sel(**{umom_dim: slice(None, -1)})
            # out_1 = data.sel(**{umom_dim: slice(1, None)})
            data = xr.concat(
                [
                    (
                        data.sel(**{umom_dim: s}).assign_coords(
                            **{umom_dim: lambda x: range(x.sizes[umom_dim])}
                        )
                    )
                    for s in [slice(None, -1), slice(1, None)]
                ],
                dim=xmom_dim,
            )

        dxduave = xcentral.xCentralMoments.from_data(
            data=data,
            mom_ndim=2,
            mom=mom,
            val_shape=val_shape,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    def from_resample_vals(
        cls,
        xv,
        uv,
        order,
        freq=None,
        indices=None,
        nrep=None,
        xmom_dim="xmom",
        umom_dim="umom",
        rep_dim="rep",
        deriv_dim=None,
        central=False,
        w=None,
        axis=0,
        broadcast=True,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        resample_kws=None,
        parallel=True,
        meta=None,
        meta_kws=None,
        x_is_u=False,
    ):

        """
        create DataCentralMoments object from unaveraged samples with resampling

        Parameters
        ----------
        axis : string or int
            axis or dimension to resample along
        rep_dim : string, default='rep'
            name of repetition dimension.  This will be the rec_dim dimension of the resulting object

        """
        if xv is None or x_is_u:
            xv = uv

        kws = dict(
            x=(xv, uv),
            w=w,
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            mom=(1, order),
            parallel=parallel,
            resample_kws=resample_kws,
            broadcast=broadcast,
            dtype=dtype,
            dims=dims,
            rep_dim=rep_dim,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
        )

        dxduave = xcentral.xCentralMoments.from_resample_vals(**kws)

        out = cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rep_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

        out = out.set_params(meta=out.meta.resample(data=out, meta_kws=meta_kws, **kws))

        return out

    @classmethod
    def from_ave_raw(
        cls,
        u,
        xu,
        w=None,
        axis=-1,
        umom_axis=None,
        xumom_axis=None,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        create object with <u**n>, <x * u**n> arrays

        Parameters
        ----------
        u : array-like
            u[n] = <u**n>.
        xu : array_like
            xu[n] = <x * u**n>.
        w : array-like, optional
            sample weights
        umom_axis : int, optional
            axis of `u` array corresponding to moments.
            Ignored if xu is an xarray.DataArray object
        xumom_axis : int, optional
            axis of `xu` array corresponding to moments
            Ignored if xu is an xarray.DataArray object
        axis : int, default=-1
            if `umom_axis` or `xumom_axis` is None, set to axis
            Ignored if xu is an xarray.DataArray object
        """

        if xu is None or x_is_u:

            raw = u

            if w is not None:
                raw.loc[{umom_dim: 0}] = w
            raw = raw.transpose(..., umom_dim)
            # xu, u = [
            #     (
            #         u.isel(**{umom_dim: s}).assign_coords(
            #             **{umom_dim: lambda x: range(x.sizes[umom_dim])}
            #         )
            #     )
            #     for s in [slice(1, None), slice(None, -1)]
            # ]

        elif isinstance(xu, xr.DataArray):
            raw = xr.concat((u, xu), dim=xmom_dim)
            if w is not None:
                raw.loc[{umom_dim: 0, xmom_dim: 0}] = w
            # make sure in correct order
            raw = raw.transpose(..., xmom_dim, umom_dim)
            # return raw
            # raw.data = np.ascontiguousarray(raw.data)
        else:
            if axis is None:
                axis = -1
            if umom_axis is None:
                umom_axis = axis
            if xumom_axis is None:
                xumom_axis = axis

            u = np.swapaxes(u, umom_axis, -1)
            xu = np.swapaxes(xu, xumom_axis, -1)

            shape = xu.shape[:-1]
            shape_moments = (2, min(u.shape[-1], xu.shape[-1]))
            shape_out = shape + shape_moments

            if dtype is None:
                dtype = xu.dtype

            raw = np.empty(shape_out, dtype=dtype)
            raw[..., 0, :] = u
            raw[..., 1, :] = xu
            if w is not None:
                raw[..., 0, 0] = w
            # raw = np.ascontiguousarray(raw)

        return cls.from_raw(
            raw=raw,
            deriv_dim=deriv_dim,
            rec_dim=rec_dim,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            mom=mom,
            central=central,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    def from_ave_central(
        cls,
        du,
        dxdu,
        w=None,
        xave=None,
        uave=None,
        axis=-1,
        umom_axis=None,
        xumom_axis=None,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        constructor from central moments, with reduction along axis

        Parameters
        ----------
        du : array-like
            du[0] = 1 or weight,
            du[1] = <u> or uave
            du[n] = <(u-<u>)**n>, n >= 2
        dxdu : array-like
            dxdu[0] = <x> or xave,
            dxdu[n] = <(x-<x>) * (u - <u>)**n>, n >= 1
        w : array-like, optional
            sample weights
        xave : array-like, optional
            if present, set dxdu[0] to xave
        uave : array-like, optional
            if present, set du[0] to uave
        umom_axis : int, optional
            axis of `u` array corresponding to moments.
            Ignored if xu is an xarray.DataArray object
        xumom_axis : int, optional
            axis of `xu` array corresponding to moments
            Ignored if xu is an xarray.DataArray object
        axis : int, default=-1
            if `umom_axis` or `xumom_axis` is None, set to axis
            Ignored if xu is an xarray.DataArray object

        """

        if dxdu is None or x_is_u:
            dxdu, du = [
                (
                    du.sel(**{umom_dim: s}).assign_coords(
                        **{umom_dim: lambda x: range(x.sizes[umom_dim])}
                    )
                )
                for s in [slice(1, None), slice(None, -1)]
            ]

        if (xave is None or x_is_u) and uave is not None:
            xave = uave

        if isinstance(dxdu, xr.DataArray):
            data = xr.concat((du, dxdu), dim=xmom_dim)
            if w is not None:
                data.loc[{umom_dim: 0, xmom_dim: 0}] = w
            if xave is not None:
                data.loc[{umom_dim: 0, xmom_dim: 1}] = xave
            if uave is not None:
                data.loc[{umom_dim: 1, xmom_dim: 0}] = uave
            data = data.transpose(..., xmom_dim, umom_dim)

        else:
            if axis is None:
                axis = -1
            if umom_axis is None:
                umom_axis = axis
            if xumom_axis is None:
                xumom_axis = axis

            du = np.swapaxes(du, umom_axis, -1)
            dxdu = np.swapaxes(dxdu, xumom_axis, -1)

            shape = dxdu.shape[:-1]
            shape_moments = (2, min(du.shape[-1], dxdu.shape[-1]))
            shape_out = shape + shape_moments

            if dtype is None:
                dtype = dxdu.dtype

            data = np.empty(shape_out, dtype=dtype)
            data[..., 0, :] = du
            data[..., 1, :] = dxdu
            if w is not None:
                data[..., 0, 0] = w
            if xave is not None:
                data[..., 1, 0] = xave
            if uave is not None:
                data[..., 0, 1] = uave

        dxduave = xcentral.xCentralMoments.from_data(
            data=data,
            mom=mom,
            mom_ndim=2,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )


class DataCentralMomentsVals(DataCentralMomentsBase):
    def __init__(
        self,
        uv,
        xv,
        order=None,
        dxduave=None,
        w=None,
        rec_dim="rec",
        umom_dim="umom",
        xmom_dim="xmom",
        deriv_dim=None,
        central=False,
        from_vals_kws=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Parameters
        ----------
        uv : xarray.DataArray
            raw values of u (energy)
        xv : xarray.DataArray
            raw values of x (observable)
        w : array-like, optional
            optional weigth array.  Note that this array/xarray must be conformable to uv, xv
        order : int
            maximum order of moments to calculate
        rec_dim : str, default='rec'
            Name of dimension to average along.
        umom_dim : str, default='umom'
            Name of moment dimension <u**umom_dim>
        xmom_dim : str, default='xmom'
            Name of the x moment dimension <u**umom_dim * x**xmom_dim>
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        central : bool, default=False
            Whether this is for central or raw moments
        from_vals_kws : dict, optional
            extra arguments passed to xcentral.xCentralMoments.from_vals
        meta : dict, optional
            extra keyword arguments.  To be used in subclasses with care.
        """

        self.x_is_u = x_is_u

        assert isinstance(uv, xr.DataArray)
        self.uv = uv

        if xv is None or self.x_is_u:
            xv = uv
        else:
            assert isinstance(xv, xr.DataArray)
        self.xv = xv

        self.w = w

        if from_vals_kws is None:
            from_vals_kws = {}
        self.from_vals_kws = from_vals_kws

        if dxduave is None:
            if order is None:
                raise ValueError("must pass order if calculating dxduave")

            dxduave = xcentral.xCentralMoments.from_vals(
                x=(self.xv, self.uv),
                w=self.w,
                axis=rec_dim,
                mom=(1, order),
                broadcast=True,
                mom_dims=(xmom_dim, umom_dim),
                **self.from_vals_kws,
            )

        super(DataCentralMomentsVals, self).__init__(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @property
    def param_names(self):
        return (
            "uv",
            "xv",
            "w",
            "dxduave",
            "order",
            "rec_dim",
            "umom_dim",
            "xmom_dim",
            "deriv_dim",
            "central",
            "meta",
            "x_is_u",
        )

    @classmethod
    def from_vals(
        cls,
        xv,
        uv,
        order,
        w=None,
        rec_dim="rec",
        umom_dim="umom",
        xmom_dim="xmom",
        rep_dim="rep",
        deriv_dim=None,
        val_dims="val",
        central=False,
        from_vals_kws=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Constructor from arrays

        Parameters
        ----------
        uv : array-like
            raw values of u (energy)
            if not DataArray, wrap with `xrwrap_uv`
        xv : xarray.DataArray
            raw values of x (observable)
            if not DataArray, wrap with `xrwrap_xv`
        order : int
            maximum order of moments to calculate
        w : array-like, optional
            optional weigth array.  Note that this array/xarray must be conformable to uv, xv
        rec_dim : str, default='rec'
            Name of dimension to average along.
        umom_dim : str, default='umom',
            Name of moment dimension <u**umom>
        xmom_dim : str, default='xmom'
            Name of moment dimension on x
        val_dims : str or list-like
            names of extra dimensions
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        central : bool, default=False
            Whether this is for central or raw moments
        from_vals_kws : dict, optional
            extra arguments passed to xcentral.xCentralMoments.from_vals
        meta : dict
            extra keyword arguments
        """

        # make sure "val" is a list
        if isinstance(val_dims, str):
            val_dims = [val_dims]
        elif not isinstance(val_dims, list):
            val_dims = list(val_dims)

        uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)

        if xv is not None and not x_is_u:
            xv = xrwrap_xv(
                xv,
                rec_dim=rec_dim,
                rep_dim=rep_dim,
                deriv_dim=deriv_dim,
                val_dims=val_dims,
            )

        return cls(
            uv=uv,
            xv=xv,
            order=order,
            w=w,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            xmom_dim=xmom_dim,
            deriv_dim=deriv_dim,
            central=central,
            from_vals_kws=from_vals_kws,
            meta=meta,
            x_is_u=x_is_u,
        )

    def __len__(self):
        return len(self.uv[self.rec_dim])

    def resample(
        self,
        indices=None,
        nrep=None,
        freq=None,
        resample_kws=None,
        parallel=True,
        axis=None,
        rep_dim="rep",
        meta_kws=None,
    ):
        """
        Resample data
        """

        if axis is None:
            axis = self.rec_dim

        kws = dict(
            indices=indices,
            nrep=nrep,
            freq=freq,
            resample_kws=resample_kws,
            parallel=True,
            axis=axis,
            rep_dim=rep_dim,
        )

        dxduave = xcentral.xCentralMoments.from_resample_vals(
            x=(self.xv, self.uv),
            w=self.w,
            mom=(1, self.order),
            broadcast=True,
            mom_dims=(self.xmom_dim, self.umom_dim),
            **kws,
        )

        meta = self.meta.resample(data=self, meta_kws=meta_kws, **kws)

        return self.new_like(dxduave=dxduave, rec_dim=rep_dim, meta=meta)

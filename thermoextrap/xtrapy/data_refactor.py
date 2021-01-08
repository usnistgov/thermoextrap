"""
Routines to handle data objects
"""


from __future__ import absolute_import

from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import xarray as xr

from .cached_decorators import gcached

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


def xrwrap_uv(uv, dims=None, rec_dim="rec", rep_dim="rep", name="u", stict=True):
    """
    wrap uv (energy values) array

    assumes uv[rec_dim], or uv[rep_dim, rec_dim] where rec_dim is recored (or time) and rep_dim is replicate
    """
    if dims is None:
        dims = {1: [rec_dim], 2: [rep_dim, rec_dim]}
    return _check_xr(uv, dims, strict=stict, name=name)


def xrwrap_xv(
    xv,
    dims=None,
    rec_dim="rec",
    rep_dim="rep",
    deriv_dim=None,
    val_dims="val",
    name="x",
    strict=None,
):
    """
    wraps xv (x values) array

    if deriv_dim is None, assumes xv[rec_dim], xv[rec_dim, vals], xv[rep_dim, rec_dim, val_dims]
    if deriv_dim is not None, assumes xv[rec_dim, deriv_dim], xv[rec_dim,deriv_dim, val_dims], xv[rep_dim,rec_dim,deriv_dim,val_dims]
    """

    if isinstance(val_dims, str):
        val_dims = [val_dims]
    elif not isinstance(val_dims, list):
        val_dims = list(val_dims)

    if deriv_dim is None:
        if strict is None:
            strict = False
        if dims is None:
            rec_val = [rec_dim] + val_dims
            rep_val = [rep_dim, rec_dim] + val_dims

            dims = {
                1: [rec_dim],
                len(rec_val): [rec_dim] + val_dims,
                len(rep_val): [rep_dim, rec_dim] + val_dims,
            }

    else:
        if strict is None:
            strict = False
        if dims is None:
            rec_val = [rec_dim, deriv_dim] + val_dims
            rep_val = [rep_dim, rec_dim, deriv_dim] + val_dims
            dims = {
                2: [rec_dim, deriv_dim],
                len(rec_val): [rec_dim, deriv_dim] + val_dims,
                len(rep_val): [rep_dim, rec_dim, deriv_dim] + [val_dims],
            }
    return _check_xr(xv, dims=dims, strict=strict, name=name)


def xrwrap_alpha(alpha, dims=None, stict=False, name="alpha"):
    """
    wrap alpha values
    """
    if isinstance(alpha, xr.DataArray):
        pass
    else:
        alpha = np.array(alpha)
        if dims is None:
            dims = name

        if alpha.ndim == 0:
            alpha = xr.DataArray(alpha, coords={dims: alpha}, name=name)
        elif alpha.ndim == 1:
            alpha = xr.DataArray(alpha, dims=dims, coords={dims: alpha}, name=name)
        else:
            alpha = xr.DataArray(alpha, dims=dims, name=name)
    return alpha


def resample_indicies(
    size, nrep, rec_dim="rec", rep_dim="rep", replace=True, transpose=False
):
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
        dims=[rep_dim, "rec"],
    )

    if transpose:
        indices = indices.transpose(rec_dim, rep_dim)
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


class AbstractData(ABC):
    @abstractproperty
    def order(self):
        pass

    @abstractproperty
    def central(self):
        pass

    @abstractproperty
    def xcoefs_args(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def resample(self, indices=None, nrep=None, **kwargs):
        pass

    @property
    def xalpha(self):
        return self.deriv_dim is not None


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
        **kws,
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
        umom_dim : str, default='umom_dim',
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
        **kws : dict
            extra keyword arguments.
            To be used in subclasses with care
        """

        assert isinstance(uv, xr.DataArray)
        assert isinstance(xv, (xr.DataArray, xr.Dataset))

        if chunk is not None:
            if isinstance(chunk, int):
                chunk = {rec_dim: chunk}

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

        if build_aves_kws is None:
            build_aves_kws = {}

        self.build_aves_kws = build_aves_kws

        self.uv = uv
        self.xv = xv
        self._order = order

        self.chunk = chunk
        self.compute = compute
        self.skipna = skipna

        # dimension names
        self.rec_dim = rec_dim
        self.umom_dim = umom_dim
        self.deriv_dim = deriv_dim

        self.kws = kws

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
        **kws,
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
        umom_dim : str, default='umom_dim',
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
        **kws : dict
            extra keyword arguments
        """

        # make sure "val" is a list
        if isinstance(val_dims, str):
            val_dims = [val_dims]
        elif not isinstance(val_dims, list):
            val_dims = list(val_dims)

        uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)
        xv = xrwrap_xv(
            xv, rec_dim=rec_dim, rep_dim=rep_dim, deriv_dim=deriv_dim, val_dims=val_dims
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
            **kws,
        )

    @property
    def order(self):
        return self._order

    @property
    def central(self):
        return self._CENTRAL

    def __len__(self):
        return len(self.uv[self.rec_dim])

    def resample_other_params(self, indices):
        """
        incase any other values are to be considered,
        then this is where they should be resampled

        Returns
        -------
        other_params : transformed version of self.other_params
        """
        return self.kws

    def resample(
        self, indices=None, nrep=None, rep_dim="rep", chunk=None, compute="None"
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
        xv = self.xv.compute().isel(**{self.rec_dim: indices})
        kws = self.resample_other_params(indices)

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
            **kws,
        )


###############################################################################
# Data
###############################################################################
def build_aves_xu(
    uv,
    xv,
    order,
    rec_dim="rec",
    moment="umom",
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
    moment : dimension with moment order
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

    u = xr.concat(u, dim=moment)
    xu = xr.concat(xu, dim=moment)

    # simple, but sometimes slow....
    # nvals = xr.DataArray(np.arange(order + 1), dims=[moment])
    # un = uv**nvals
    # u = (un).mean(rec_dim, skipna=skipna)
    # xu = (un * xv).mean(rec_dim, skipna=skipna)

    if transpose:
        u_order = (moment, ...)
        if deriv_dim is not None:
            x_order = (moment, deriv_dim, ...)
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
    moment="umom",
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
    moment : dimension with moment order
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

    duave = xr.concat(duave, dim=moment)
    dxduave = xr.concat(dxduave, dim=moment)

    if transpose:
        u_order = (moment, ...)
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
            moment=self.umom_dim,
            deriv_dim=self.deriv_dim,
            **self.build_aves_kws,
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
        return DatasetSelector(self.u, deriv_dim=None, moment=self.umom_dim)

    @gcached()
    def xu_selector(self):
        return DatasetSelector(self.xu, deriv_dim=self.deriv_dim, moment=self.umom_dim)

    @property
    def xcoefs_args(self):
        return (self.u_selector, self.xu_selector)


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
            moment=self.umom_dim,
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
        return DatasetSelector(self.du, deriv_dim=None, moment=self.umom_dim)

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector(
            self.dxdu, deriv_dim=self.deriv_dim, moment=self.umom_dim
        )

    @gcached()
    def xave_selector(self):
        if self.deriv_dim is None:
            return self.xave
        else:
            return DatasetSelector(self.xave, dims=[self.deriv_dim])

    @property
    def xcoefs_args(self):
        return (self.xave_selector, self.du_selector, self.dxdu_selector)


################################################################################
# StatsCov objects
################################################################################
class DataCentralMomentsBase(AbstractData):
    def __init__(
        self,
        dxduave,
        umom_dim="umom",
        mom_x="mom_x",
        rec_dim="rec",
        deriv_dim=None,
        central=False,
        **kws,
    ):
        """
        Data object based on central co-moments array

        Parameters
        ----------
        dxduave : cmomy.xcentral.xStatsAccum object
            central object to work with
        rec_dim : str, default='rec'
            Name of dimension to average along.
        umom_dim : str, default='umom_dim',
            Name of moment dimension <u**umom_dim>
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        central : bool, default=False
            if True, use central moments.
            if False, use raw moments
        """

        self.dxduave = dxduave
        self.umom_dim = umom_dim
        self.mom_x = mom_x
        self.rec_dim = rec_dim
        self.deriv_dim = deriv_dim
        self._central = central
        self.kws = kws

    @property
    def central(self):
        return self._central

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
    def u(self):
        out = self.rmom().sel(**{self.mom_x: 0}, drop=True)
        if self.xalpha:
            out = out.sel(**{self.deriv_dim: 0}, drop=True)
        return out

    @gcached()
    def xu(self):
        return self.rmom().sel(**{self.mom_x: 1}, drop=True)

    @gcached()
    def xave(self):
        return self.dxduave.values.sel(**{self.umom_dim: 0, self.mom_x: 1}, drop=True)

    @gcached()
    def du(self):
        out = self.cmom().sel(**{self.mom_x: 0}, drop=True)
        if self.xalpha:
            out = out.sel(**{self.deriv_dim: 0}, drop=True)
        return out

    @gcached()
    def dxdu(self):
        return self.cmom().sel(**{self.mom_x: 1}, drop=True)

    @gcached()
    def u_selector(self):
        return DatasetSelector(self.u, deriv_dim=None, moment=self.umom_dim)

    @gcached()
    def xu_selector(self):
        return DatasetSelector(self.xu, deriv_dim=self.deriv_dim, moment=self.umom_dim)

    @gcached()
    def xave_selector(self):
        if self.deriv_dim is None:
            return self.xave
        else:
            return DatasetSelector(self.xave, dims=[self.deriv_dim])

    @gcached()
    def du_selector(self):
        return DatasetSelector(self.du, deriv_dim=None, moment=self.umom_dim)

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector(
            self.dxdu, deriv_dim=self.deriv_dim, moment=self.umom_dim
        )

    @property
    def xcoefs_args(self):
        if self.central:
            return (self.xave_selector, self.du_selector, self.dxdu_selector)

        else:
            return (self.u_selector, self.xu_selector)


class DataCentralMoments(DataCentralMomentsBase):
    def __len__(self):
        return self.values.sizes[self.rec_dim]

    def new_like(self, **kws):
        kws = dict(
            # default dict
            dict(
                dxduave=self.dxduave,
                mom_x=self.mom_x,
                umom_dim=self.umom_dim,
                rec_dim=self.rec_dim,
                deriv_dim=self.deriv_dim,
                central=self.central,
                **self.kws,
            ),
            **kws,
        )
        return type(self)(**kws)

    def block(self, block_size, axis=None, **kwargs):
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
        return self.new_like(
            dxduave=self.dxduave.block(block_size=block_size, axis=axis, **kwargs)
        )

    def reduce(self, axis=None, *args, **kwargs):
        """
        reduce along axis

        Parameters
        ----------
        axis : int or str, default=self.rec_dim
            axis to reduce (combine moments) along
        """
        if axis is None:
            axis = self.rec_dim
        return self.new_like(dxduave=self.dxduave.reduce(axis=axis, *args, **kwargs))

    def resample(
        self,
        freq=None,
        indices=None,
        nrep=None,
        axis=None,
        rep_dim="rep",
        parallel=True,
        resample_kws=None,
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

        dxdu_new = self.dxduave.resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            rep_dim=rep_dim,
            resample_kws=resample_kws,
            parallel=parallel,
            **kwargs,
        )
        return self.new_like(dxduave=dxdu_new, rec_dim=rep_dim)

    # TODO : update from_raw from_data to
    # include a mom_dims arguments
    # that defaults to (mom_x, umom_dim)
    # so if things are in wrong order, stuff still works out

    @classmethod
    def from_raw(
        cls,
        raw,
        rec_dim="rec",
        mom_x="mom_x",
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
        **kws,
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
            mom_dims=(mom_x, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            mom_x=mom_x,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            **kws,
        )

    @classmethod
    def from_vals(
        cls,
        xv,
        uv,
        order,
        mom_x="mom_x",
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
        **kws,
    ):
        """
        create DataCentralMoments object from individual (unaveraged) samples

        uv, xv are wrapped before execution
        """

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
            mom_dims=(mom_x, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            mom_x=mom_x,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            **kws,
        )

    @classmethod
    def from_data(
        cls,
        data,
        rec_dim="rec",
        mom_x="mom_x",
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
        **kws,
    ):
        """
        Create DataCentralMoments object from data

        data[..., i, j] = weight                          i = j = 0
                        = < x >                           i = 1 and j = 0
                        = < u >                           i = 0 and j = 1
                        = <(x - <x>)**i * (u - <u>)**j >  otherwise
        """
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
            mom_dims=(mom_x, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            mom_x=mom_x,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            **kws,
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
        mom_x="mom_x",
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
        **kws,
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

        dxduave = xcentral.xCentralMoments.from_resample_vals(
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
            mom_dims=(mom_x, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            mom_x=mom_x,
            umom_dim=umom_dim,
            rec_dim=rep_dim,
            deriv_dim=deriv_dim,
            central=central,
            **kws,
        )

    @classmethod
    def from_ave_raw(
        cls,
        u,
        xu,
        w=None,
        axis=-1,
        axis_mom_u=None,
        axis_mom_xu=None,
        rec_dim="rec",
        mom_x="mom_x",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        **kws,
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
        axis_mom_u : int, optional
            axis of `u` array corresponding to moments.
            Ignored if xu is an xarray.DataArray object
        axis_mom_xu : int, optional
            axis of `xu` array corresponding to moments
            Ignored if xu is an xarray.DataArray object
        axis : int, default=-1
            if `axis_mom_u` or `axis_mom_xu` is None, set to axis
            Ignored if xu is an xarray.DataArray object
        """

        if isinstance(xu, xr.DataArray):
            raw = xr.concat((u, xu), dim=mom_x)
            if w is not None:
                raw.loc[{umom_dim: 0, mom_x: 0}] = w
            # make sure in correct order
            raw = raw.transpose(..., mom_x, umom_dim)
        else:
            if axis is None:
                axis = -1
            if axis_mom_u is None:
                axis_mom_u = axis
            if axis_mom_xu is None:
                axis_mom_xu = axis

            u = np.swapaxes(u, axis_mom_u, -1)
            xu = np.swapaxes(xu, axis_mom_xu, -1)

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

        return cls.from_raw(
            raw=raw,
            deriv_dim=deriv_dim,
            rec_dim=rec_dim,
            mom_x=mom_x,
            umom_dim=umom_dim,
            mom=mom,
            central=central,
            shape=shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            **kws,
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
        axis_mom_u=None,
        axis_mom_xu=None,
        rec_dim="rec",
        mom_x="mom_x",
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
        **kws,
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
        axis_mom_u : int, optional
            axis of `u` array corresponding to moments.
            Ignored if xu is an xarray.DataArray object
        axis_mom_xu : int, optional
            axis of `xu` array corresponding to moments
            Ignored if xu is an xarray.DataArray object
        axis : int, default=-1
            if `axis_mom_u` or `axis_mom_xu` is None, set to axis
            Ignored if xu is an xarray.DataArray object

        """

        if isinstance(dxdu, xr.DataArray):
            data = xr.concat((du, dxdu), dim=mom_x)
            if w is not None:
                data.loc[{umom_dim: 0, mom_x: 0}] = w
            if xave is not None:
                data.loc[{umom_dim: 0, mom_x: 1}] = xave
            if uave is not None:
                data.loc[{umom_dim: 1, mom_x: 0}] = uave
            data = data.transpose(..., mom_x, umom_dim)

        else:
            if axis is None:
                axis = -1
            if axis_mom_u is None:
                axis_mom_u = axis
            if axis_mom_xu is None:
                axis_mom_xu = axis

            du = np.swapaxes(du, axis_mom_u, -1)
            dxdu = np.swapaxes(dxdu, axis_mom_xu, -1)

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
            mom_dims=(mom_x, umom_dim),
        )

        return cls(
            dxduave=dxduave,
            mom_x=mom_x,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            **kws,
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
        mom_x="mom_x",
        deriv_dim=None,
        central=False,
        from_vals_kws=None,
        **kws,
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
        umom_dim : str, default='umom_dim'
            Name of moment dimension <u**umom_dim>
        mom_x : str, default='mom_x'
            Name of the x moment dimension <u**umom_dim * x**mom_x>
        deriv_dim : str, default=None
            if deriv_dim is a string, then this is the name of the derivative dimension
            and xarray objects will have a derivative
        central : bool, default=False
            Whether this is for central or raw moments
        from_vals_kws : dict, optional
            extra arguments passed to xcentral.xCentralMoments.from_vals
        **kws : dict
            extra keyword arguments
        """

        assert isinstance(uv, xr.DataArray)
        assert isinstance(xv, xr.DataArray)

        self.uv = uv
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
                mom_dims=(mom_x, umom_dim),
                **self.from_vals_kws,
            )

        super(DataCentralMomentsVals, self).__init__(
            dxduave=dxduave,
            mom_x=mom_x,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            **kws,
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
        mom_x="mom_x",
        rep_dim="rep",
        deriv_dim=None,
        val_dims="val",
        central=False,
        from_vals_kws=None,
        **kws,
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
        umom_dim : str, default='umom_dim',
            Name of moment dimension <u**umom_dim>
        mom_x : str, default='mom_x'
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
        **kws : dict
            extra keyword arguments
        """

        # make sure "val" is a list
        if isinstance(val_dims, str):
            val_dims = [val_dims]
        elif not isinstance(val_dims, list):
            val_dims = list(val_dims)

        uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)
        xv = xrwrap_xv(
            xv, rec_dim=rec_dim, rep_dim=rep_dim, deriv_dim=deriv_dim, val_dims=val_dims
        )

        return cls(
            uv=uv,
            xv=xv,
            order=order,
            w=w,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            mom_x=mom_x,
            deriv_dim=deriv_dim,
            central=central,
            from_vals_kws=from_vals_kws,
            **kws,
        )

    def new_like(self, **kws):
        kws = dict(
            dict(
                uv=self.uv,
                xv=self.xv,
                w=self.w,
                dxduave=self.dxduave,
                order=self.order,
                rec_dim=self.rec_dim,
                umom_dim=self.umom_dim,
                mom_x=self.mom_x,
                deriv_dim=self.deriv_dim,
                central=self.central,
                **self.kws,
            ),
            **kws,
        )
        return type(self)(**kws)

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
    ):
        """
        Resample data
        """

        if axis is None:
            axis = self.rec_dim

        dxduave = xcentral.xCentralMoments.from_resample_vals(
            x=(self.xv, self.uv),
            w=self.w,
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            mom=(1, self.order),
            parallel=parallel,
            resample_kws=resample_kws,
            broadcast=True,
            rep_dim=rep_dim,
            mom_dims=(self.mom_x, self.umom_dim),
        )

        return self.new_like(dxduave=dxduave, rec_dim=rep_dim)

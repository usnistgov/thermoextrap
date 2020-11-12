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
    xalpha=False,
    name="x",
    strict=None,
):
    """
    wraps xv (x values) array

    if xalpha is False, assumes xv[rec], xv[rec, val], xv[rep, rec, val]
    if xalpha is True, assumes xv[rec, deriv], xv[rec,deriv, val], xv[rep,rec,deriv,val]
    """
    if not xalpha:
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


def resample_indicies(size, nrep, rec="rec", rep="rep", replace=True, transpose=False):
    """
    get indexing DataArray

    Parameters
    ----------
    size : int
        size of axis to bootstrap along
    nrep : int
        number of replicates
    rec : str, default='rec'
        name of record dimension
    rep : str, default='rep'
        name of replicate dimension
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
        dims=[rep, "rec"],
    )

    if transpose:
        indices = indices.transpose(rec, rep)

    return indices


class DatasetSelector(object):
    """
    wrap dataset so can index like ds[i, j]

    Needed for calling sympy.lambdify functions
    """

    def __init__(self, data, dims=None, moment="moment", deriv="deriv"):

        # Default dims
        if dims is None:
            if deriv in data.dims:
                dims = [moment, deriv]
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


from abc import ABC, abstractmethod, abstractproperty


class AbstractData(ABC):

    @abstractproperty
    def order(self):
        pass

    @abstractproperty
    def xcoefs_args(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def resample(self, *args, **kwargs):
        pass





class DataTemplateValues(object):
    def __init__(
        self,
        uv,
        xv,
        order,
        skipna=False,
        xalpha=False,
        rec="rec",
        moment="moment",
        val="val",
        rep="rep",
        deriv="deriv",
        chunk=None,
        compute=None,
        **kws,
    ):

        uv = xrwrap_uv(uv, rec=rec, rep=rep)
        xv = xrwrap_xv(xv, rec=rec, rep=rep, deriv=deriv, val=val, xalpha=xalpha)

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
        self.xalpha = xalpha

        self._rec = rec
        self._rep = rep
        self._val = val
        self._moment = moment
        self._deriv = deriv
        self._kws = kws

    def __len__(self):
        return len(self.uv[self._rec])

    def resample(self, indices=None, nrep=None, chunk=None, compute="None"):
        """
        resample object

        Parameters
        ----------
        indices : array-like, shape=(nrep, nrec), optional
            if present, use this to resample
        nrep : int, optional
            construct resampling with `nrep` samples
        chunk : optional
            chunk size
        compute : optional

        """

        if chunk is None:
            chunk = self.chunk

        if compute is "None":
            compute = None
        elif compute is None:
            compute = self.compute

        shape = len(self.uv[self._rec])

        if indices is None:
            assert nrep is not None
            indices = resample_indicies(shape, nrep, rec=self._rec, rep=self._rep)
        elif not isinstance(indices, xr.DataArray):
            indices = xr.DataArray(indices, dims=(self._rep, self._rec))

        assert indices.sizes[self._rec] == len(self)

        uv = self.uv.compute()[indices]
        # allow for Dataset
        xv = self.xv.compute().isel(**{self._rec: indices})

        return self.__class__(
            uv=uv,
            xv=xv,
            order=self.order,
            rec=self._rec,
            rep=self._rep,
            val=self._val,
            moment=self._moment,
            deriv=self._deriv,
            xalpha=self.xalpha,
            skipna=self.skipna,
            chunk=chunk,
            compute=compute,
            **self._kws,
        )


class DataStatsCovBase(object):
    def __init__(
        self,
        dxduave,
        xalpha=False,
        mom_x="mom_x",
        moment="moment",
        rec="rec",
        rep="rep",
        deriv="deriv",
        central=False,
        **kws,
    ):

        self._dxduave = dxduave
        self.xalpha = xalpha

        self._central = central
        self._mom_x = mom_x
        self._moment = moment
        self._rec = rec
        self._rep = rep
        self._deriv = deriv
        self._kws = kws

    @property
    def dxduave(self):
        """
        xStatsAccumCov object
        """
        return self._dxduave

    @property
    def values(self):
        return self._dxduave.values

    @property
    def order(self):
        return self.dxduave.sizes[self._moment] - 1

    @gcached(prop=False)
    def rmom(self):
        return self.dxduave.rmom()

    @gcached(prop=False)
    def cmom(self):
        return self.dxduave.cmom()

    @gcached()
    def u(self):
        return self.rmom().sel(**{self._mom_x: 0})

    @gcached()
    def xu(self):
        return self.rmom().sel(**{self._mom_x: 1})

    @gcached()
    def xave(self):
        return self.dxduave.values.sel(**{self._moment: 0, self._mom_x: 1})

    @gcached()
    def du(self):
        return self.cmom().sel(**{self._mom_x: 0})

    @gcached()
    def dxdu(self):
        return self.cmom().sel(**{self._mom_x: 1})

    @gcached()
    def u_selector(self):
        return DatasetSelector(self.u, deriv=self._deriv, moment=self._moment)

    @gcached()
    def xu_selector(self):
        return DatasetSelector(self.xu, deriv=self._deriv, moment=self._moment)

    @gcached()
    def xave_selector(self):
        if self.xalpha:
            return DatasetSelector(self.xave, dims=[self._deriv])
        else:
            return self.xave

    @gcached()
    def du_selector(self):
        return DatasetSelector(self.du, deriv=self._deriv, moment=self._moment)

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector(self.dxdu, deriv=self._deriv, moment=self._moment)

    @property
    def _xcoefs_args(self):
        if self._central:
            return (self.xave_selector, self.du_selector, self.dxdu_selector)

        else:
            return (self.u_selector, self.xu_selector)


class DataStatsCov(DataStatsCovBase):
    def __len__(self):
        return self.dxduave.values.sizes[self._rec]

    def new_like(self, **kws):
        # if dxduave is None:
        #     dxduave = self.dxduave
        # return type(self)(
        #     dxduave=dxduave,
        #     xalpha=self.xalpha,
        #     mom_x=self._mom_x,
        #     moment=self._moment,
        #     rec=self._rec,
        #     rep=self._rep,
        #     deriv=self._deriv,
        #     central=self._central,
        #     **self._kws,
        # )

        kws = dict(
            # default dict
            dict(
                dxduave=self.dxduave,
                xalpha=self.xalpha,
                mom_x=self._mom_x,
                moment=self._moment,
                rec=self._rec,
                rep=self._rep,
                deriv=self._deriv,
                central=self._central,
                **self._kws,
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

        axis : int or str
            axis or dimension to block average along
        **kwargs : dict
            extra arguments to cmomy.xStatsAccumCov.block
        """
        return self.new_like(
            dxduave=self.dxduave.block(block_size=block_size, axis=axis, **kwargs)
        )

    def reduce(self, axis=0, *args, **kwargs):
        """
        reduce along axis
        """
        return self.new_like(dxduave=self.dxduave.reduce(axis=axis, *args, **kwargs))

    def resample(
        self,
        freq=None,
        indices=None,
        nrep=None,
        resample_kws=None,
        axis=None,
        dim_rep=None,
        parallel=True,
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
        parallel : bool, default=True
            If true, perform resampling in parallel 
        """

        if axis is None:
            axis = self._rec

        if dim_rep is None:
            dim_rep = self._rep

        if resample_kws is None:
            resample_kws = {}
        resample_kws["parallel"] = parallel

        dxdu_new = self.dxduave.resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            dim_rep=dim_rep,
            resample_kws=resample_kws,
            **kwargs,
        )

        return self.new_like(dxduave=dxdu_new)

    @classmethod
    def from_raw(
        cls,
        raw,
        xalpha=False,
        rep="rep",
        deriv="deriv",
        rec="rec",
        mom_x="mom_x",
        moment="moment",
        central=False,
        moments=None,
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
        Parameters
        ----------
        raw : array-like
            raw moments.  The form of this array is such that
            raw[..., i, j] = <x**i * u**j>
           The shape should be (..., 2, order+1)
        """

        import cmomy.xcentral as xcentral

        dxduave = xcentral.xStatsAccumCov.from_raw(
            raw=raw,
            moments=moments,
            shape=shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            moments_dims=(mom_x, moment),
        )

        return cls(
            dxduave=dxduave,
            xalpha=xalpha,
            mom_x=mom_x,
            moment=moment,
            rec=rec,
            rep=rep,
            deriv=deriv,
            central=central,
            **kws,
        )

    @classmethod
    def from_vals(
        cls,
        xv,
        uv,
        order,
        xalpha=False,
        mom_x="mom_x",
        moment="moment",
        rec="rec",
        rep="rep",
        deriv="deriv",
        central=False,
        w=None,
        axis=0,
        broadcast=True,
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
        create DataStatsCov object from individual (unaveraged) samples
        """

        import cmomy.xcentral as xcentral

        dxduave = xcentral.xStatsAccumCov.from_vals(
            x0=xv,
            x1=uv,
            w=w,
            axis=axis,
            moments=(1, order),
            broadcast=broadcast,
            shape=shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            moments_dims=(mom_x, moment),
        )

        return cls(
            dxduave=dxduave,
            xalpha=xalpha,
            mom_x=mom_x,
            moment=moment,
            rec=rec,
            rep=rep,
            deriv=deriv,
            central=central,
            **kws,
        )

    @classmethod
    def from_resample_vals(
        cls,
        xv,
        uv,
        order,
        xalpha=False,
        freq=None,
        indices=None,
        nrep=None,
        mom_x="mom_x",
        moment="moment",
        rec="rec",
        rep="rep",
        deriv="deriv",
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
        create DataStatsCov object from unaveraged samples with resampling
        """

        import cmomy.xcentral as xcentral

        if resample_kws is None:
            resample_kws = {}
        resample_kws["parallel"] = parallel

        dxduave = xcentral.xStatsAccumCov.from_resample_vals(
            x0=xv,
            x1=uv,
            w=w,
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            moments=(1, order),
            resample_kws=resample_kws,
            broadcast=broadcast,
            dtype=dtype,
            dims=dims,
            dim_rep=rep,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            moments_dims=(mom_x, moment),
        )

        return cls(
            dxduave=dxduave,
            xalpha=xalpha,
            mom_x=mom_x,
            moment=moment,
            rec=rec,
            rep=rep,
            deriv=deriv,
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
        xalpha=False,
        rep="rep",
        deriv="deriv",
        rec="rec",
        mom_x="mom_x",
        moment="moment",
        central=False,
        moments=None,
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
                raw.loc[{moment: 0, mom_x: 0}] = w
            # make sure in correct order
            raw = raw.transpose(..., mom_x, moment)
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
            xalpha=xalpha,
            rep=rep,
            deriv=deriv,
            rec=rec,
            mom_x=mom_x,
            moment=moment,
            moments=moments,
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
        xalpha=False,
        rep="rep",
        deriv="deriv",
        rec="rec",
        mom_x="mom_x",
        moment="moment",
        central=False,
        moments=None,
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
        import cmomy.xcentral as xcentral

        if isinstance(dxdu, xr.DataArray):
            data = xr.concat((du, dxdu), dim="mom_x")
            if w is not None:
                data.loc[{moment: 0, mom_x: 0}] = w
            if xave is not None:
                data.loc[{moment: 0, mom_x: 1}] = xave
            if uave is not None:
                data.loc[{moment: 1, mom_x: 0}] = uave
            data = data.transpose(..., mom_x, moment)

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

        dxduave = xcentral.xStatsAccumCov.from_data(
            data=data,
            moments=moments,
            shape=shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            moments_dims=(mom_x, moment),
        )

        return cls(
            dxduave=dxduave,
            xalpha=xalpha,
            mom_x=mom_x,
            moment=moment,
            rec=rec,
            rep=rep,
            deriv=deriv,
            central=central,
            **kws,
        )


class DataStatsCovVals(DataStatsCovBase):
    def __init__(
        self,
        uv,
        xv,
        dxduave=None,
        order=None,
        xalpha=False,
        w=None,
        rec="rec",
        moment="moment",
        val="val",
        rep="rep",
        deriv="deriv",
        mom_x="mom_x",
        central=False,
        axis=None,
        from_vals_kws=None,
        **kws,
    ):
        import cmomy.xcentral as xcentral

        uv = xrwrap_uv(uv, rec=rec, rep=rep)
        xv = xrwrap_xv(xv, rec=rec, rep=rep, deriv=deriv, val=val, xalpha=xalpha)

        self.uv = uv
        self.xv = xv
        self.w = w
        self._val = val

        if dxduave is None:
            if order is None:
                raise ValueError("must pass order if calculating dxduave")
            if axis is None:
                axis = rec

            if from_vals_kws is None:
                from_vals_kws = {}

            dxduave = xcentral.xStatsAccumCov.from_vals(
                x0=xv,
                x1=uv,
                w=w,
                axis=axis,
                moments=(1, order),
                broadcast=True,
                moments_dims=(mom_x, moment),
                **from_vals_kws
            )

        super(DataStatsCovVals, self).__init__(
            dxduave=dxduave,
            xalpha=xalpha,
            mom_x=mom_x,
            moment=moment,
            rec=rec,
            rep=rep,
            deriv=deriv,
            central=central,
            **kws,
        )

    def new_like(self, **kws):
        # if uv is None:
        #     uv = self.uv
        # if xv is None:
        #     xv = self.xv
        # if w is None:
        #     w = self.w
        # if dxduave is None:
        #     dxduave = self.dxduave

        # return type(self)(
        #     uv=uv,
        #     xv=xv,
        #     w=w,
        #     dxduave=dxduave,
        #     order=self.order,
        #     rec=self._rec,
        #     rep=self._rep,
        #     val=self._val,
        #     moment=self._moment,
        #     mom_x=self._mom_x,
        #     deriv=self._deriv,
        #     xalpha=self.xalpha,
        #     central=self._central,
        #     **self._kws,
        # )

        kws = dict(
            dict(
                uv=self.uv,
                xv=self.xv,
                w=self.w,
                dxduave=self.dxduave,
                order=self.order,
                rec=self._rec,
                rep=self._rep,
                val=self._val,
                moment=self._moment,
                mom_x=self._mom_x,
                deriv=self._deriv,
                xalpha=self.xalpha,
                central=self._central,
                **self._kws,
            ),
            **kws,
        )

        return type(self)(**kws)

    def __len__(self):
        return len(self.uv[self._rec])

    def resample(
        self,
        indices=None,
        nrep=None,
        freq=None,
        resample_kws=None,
        parallel=True,
        axis=None,
    ):

        import cmomy.xcentral as xcentral

        if resample_kws is None:
            resample_kws = {}
        resample_kws["parallel"] = parallel

        if axis is None:
            axis = self._rec

        dxduave = xcentral.xStatsAccumCov.from_resample_vals(
            x0=self.xv,
            x1=self.uv,
            w=self.w,
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            moments=(1, self.order),
            resample_kws=resample_kws,
            broadcast=True,
            dim_rep=self._rep,
            moments_dims=(self._mom_x, self._moment),
        )

        return self.new_like(dxduave=dxduave)

    # @classmethod
    # def from_vals(
    #     cls,
    #     xv,
    #     uv,
    #     order,
    #     xalpha=False,
    #     mom_x="mom_x",
    #     moment="moment",
    #     rec="rec",
    #     rep="rep",
    #     deriv="deriv",
    #     central=False,
    #     w=None,
    #     axis=0,
    #     broadcast=True,
    #     shape=None,
    #     dtype=None,
    #     dims=None,
    #     attrs=None,
    #     coords=None,
    #     indexes=None,
    #     name=None,
    #     **kws,
    # ):

    #     import cmomy.xcentral as xcentral

    #     dxduave = xcentral.xStatsAccumCov.from_vals(
    #         x0=xv,
    #         x1=uv,
    #         w=w,
    #         axis=axis,
    #         moments=(1, order),
    #         broadcast=broadcast,
    #         shape=shape,
    #         dtype=dtype,
    #         dims=dims,
    #         attrs=attrs,
    #         coords=coords,
    #         indexes=indexes,
    #         name=name,
    #         moments_dims=(mom_x, moment),
    #     )

    #     return cls(
    #         uv=uv,
    #         xv=xv,
    #         dxduave=dxduave,
    #         order=order,
    #         xalpha=xalpha,
    #         w=w,
    #         rec=rec,
    #         moment=moment,
    #         val=val,
    #         rep=rep,
    #         deriv=deriv,
    #         mom_x=mom_x,
    #         central=central,
    #         **kws,
    #     )


################################################################################
# Structure(s) to deal with analytic derivatives, etc
################################################################################


@lru_cache(100)
def _get_default_symbol(*args):
    return sp.symbols(",".join(args))


@lru_cache(100)
def _get_default_indexed(*args):
    out = [sp.IndexedBase(key) for key in args]
    if len(out) == 1:
        out = out[0]
    return out


@lru_cache(100)
def _get_default_function(*args):
    out = [sp.Function(key) for key in args]
    if len(out) == 1:
        out = out[0]
    return out


class SymSubs(object):
    def __init__(
        self,
        funcs,
        subs=None,
        subs_final=None,
        subs_all=None,
        recursive=True,
        simplify=False,
        expand=True,
    ):
        """
        perform substitution on stuff
        """

        self.funcs = funcs
        self.subs = subs
        self.subs_final = subs_final
        self.subs_all = subs_all

        self.recursive = recursive
        self.simplify = simplify
        self.expand = expand

    @gcached(prop=False)
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


class Lambdify(object):
    """
    create python function from list of sympy expression
    """

    def __init__(self, exprs, args=None, **opts):
        """
        Parameters
        ----------
        exprs : array-like
            array of sympy expressions to lambdify
        args : array-like
            array of sympy symbols which will be in args of the resulting function
        opts : dict
            extra arguments to sympy.lambdify
        """

        self.exprs = exprs
        self.args = args
        self.opts = opts

    @gcached(prop=False)
    def __getitem__(self, order):
        return sp.lambdify(self.args, self.exprs[order], **self.opts)

    @classmethod
    def from_u_xu(cls, exprs, **opts):
        """factory for u/xu args"""
        u, xu = _get_default_indexed("u", "xu")
        args = (u, xu)
        return cls(exprs=exprs, args=(u, xu), **opts)

    @classmethod
    def from_du_dxdu(cls, exprs, xalpha=False, **opts):
        """factory for du/dxdu args"""
        if xalpha:
            x1 = _get_default_indexed("x1")
        else:
            x1 = _get_default_symbol("x1")
        du, dxdu = _get_default_indexed("du", "dxdu")
        return cls(exprs=exprs, args=(x1, du, dxdu), **opts)


# -log<X>
class SymMinusLog(object):
    """class to take -log(X)"""

    X, dX = _get_default_indexed("X", "dX")

    @gcached(prop=False)
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


class Coefs(object):
    """class to handle coefficients in taylor expansion"""

    def __init__(self, funcs, exprs=None):
        """
        Parameters
        ----------
        funcs : array-like
            array of functions.
            funcs[n] is the nth derivative of function
        exprs : array-like, optional
            optional placeholder for sympy expressions corresponding to the
            lambdified functions in `funcs`
        """
        self.funcs = funcs
        self.exprs = exprs

    def _apply_minus_log(self, X, order):
        func = factory_minus_log()
        return [func[i](X) for i in range(order + 1)]

    def coefs(self, *args, order, norm=True, minus_log=False):
        out = [self.funcs[i](*args) for i in range(order + 1)]
        if minus_log:
            out = self._apply_minus_log(X=out, order=order)

        if norm:
            out = [x / np.math.factorial(i) for i, x in enumerate(out)]
        return out

    def xcoefs(self, data, order=None, norm=True, minus_log=False, order_name="order"):
        if order is None:
            order = data.order
        out = self.coefs(
            *data._xcoefs_args, order=order, norm=norm, minus_log=minus_log
        )
        return xr.concat(out, dim=order_name)

    @classmethod
    def from_sympy(cls, exprs, args):
        funcs = Lambdify(exprs, args=args)
        return cls(funcs=funcs, exprs=exprs)


class ExtrapModel(object):
    """
    apply taylor series extrapolation
    """

    def __init__(
        self, alpha0, data, coefs, order=None, minus_log=False, alpha_name="alpha"
    ):
        self.alpha0 = alpha0
        self.data = data
        self.coefs = coefs

        if order is None:
            order = self.order
        if minus_log is None:
            minus_log = False

        self.minus_log = minus_log
        self.order = order

        if alpha_name is None:
            alpha_name = "alpha"
        self.alpha_name = alpha_name

    @gcached(prop=False)
    def xcoefs(self, order=None, order_name="order", norm=True, minus_log=None):
        if minus_log is None:
            minus_log = self.minus_log
        if order is None:
            order = self.order
        return self.coefs.xcoefs(
            self.data,
            order=order,
            order_name=order_name,
            norm=norm,
            minus_log=minus_log,
        )

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(
        self,
        alpha,
        order=None,
        order_name="order",
        cumsum=False,
        minus_log=None,
        alpha_name=None,
    ):
        if order is None:
            order = self.order

        if alpha_name is None:
            alpha_name = self.alpha_name

        xcoefs = self.xcoefs(
            order=order, order_name=order_name, norm=True, minus_log=minus_log
        )

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        dalpha = alpha - self.alpha0
        p = xr.DataArray(np.arange(order + 1), dims=order_name)
        prefac = dalpha ** p

        coords = {"dalpha": dalpha, alpha_name + "0": self.alpha0}

        out = (prefac * xcoefs.sel(**{order_name: prefac[order_name]})).assign_coords(
            **coords
        )

        if cumsum:
            out = out.cumsum(order_name)
        else:
            out = out.sum(order_name)

        return out

    def resample(self, indices=None, nrep=None, **kws):
        return self.__class__(
            order=self.order,
            alpha0=self.alpha0,
            coefs=self.coefs,
            data=self.data.resample(nrep=nrep, indices=indices, **kws),
            minus_log=self.minus_log,
            alpha_name=self.alpha_name,
        )

    # @classmethod
    # def from_values_beta(
    #     cls, order, alpha0, uv, xv, xalpha=False, central=False, minus_log=False, **kws
    # ):
    #     """
    #     build a model from beta extraploation from data
    #     """

    #     data = factory_data(
    #         uv=uv, xv=xv, order=order, xalpha=xalpha, central=central, **kws
    #     )

    #     coefs = factory_coefs_beta(xalpha=xalpha, central=central)

    #     return cls(
    #         order=order, alpha0=alpha0, coefs=coefs, data=data, minus_log=minus_log
    #     )


class StateCollection(object):
    def __init__(self, states):
        self.states = states

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    @property
    def alpha_name(self):
        try:
            alpha_name = self[0].alpha_name
        except:
            alpha_name = "alpha"
        return alpha_name

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


class ExtrapWeightedModel(StateCollection):
    def predict(
        self,
        alpha,
        order=None,
        order_name="order",
        cumsum=False,
        minus_log=None,
        alpha_name=None,
    ):

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        out = xr.concat(
            [
                m.predict(
                    alpha,
                    order=order,
                    order_name=order_name,
                    cumsum=cumsum,
                    minus_log=minus_log,
                    alpha_name=alpha_name,
                )
                for m in self.states
            ],
            dim="state",
        )

        w = xr_weights_minkowski(np.abs(out.dalpha))
        out = (out * w).sum("state") / w.sum("state")
        return out


class InterpModel(StateCollection):
    @gcached(prop=False)
    def xcoefs(self, order=None, order_name="porder", minus_log=None):

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
            xr.DataArray(mat_inv, dims=[order_name, "state_order"])
            .assign_coords(state=("state_order", states))
            .assign_coords(order=("state_order", orders))
            .set_index(state_order=["state", "order"])
            .unstack()
        )

        coefs = xr.concat(
            [m.xcoefs(order, norm=False, minus_log=minus_log) for m in self.states],
            dim="state",
        )
        if isinstance(coefs, xr.Dataset):
            coefs = xr.Dataset({k: xr.dot(mat_inv, v) for k, v in coefs.items()})
        else:
            coefs = xr.dot(mat_inv, coefs)

        return coefs

    def predict(
        self, alpha, order=None, order_name="porder", minus_log=None, alpha_name=None
    ):

        if order is None:
            order = self.order
        if alpha_name is None:
            alpha_name = self.alpha_name

        xcoefs = self.xcoefs(order=order, order_name=order_name, minus_log=minus_log)
        alpha = xrwrap_alpha(alpha, name=alpha_name)

        porder = len(xcoefs[order_name]) - 1

        p = xr.DataArray(np.arange(porder + 1), dims=order_name)
        prefac = alpha ** p

        out = (prefac * xcoefs).sum(order_name)
        return out


class PerturbModel(object):
    def __init__(self, alpha0, data, alpha_name="alpha"):

        self.alpha0 = alpha0
        self.data = data

        if alpha_name is None:
            alpha_name = "alpha"
        self.alpha_name = alpha_name

    def predict(self, alpha, alpha_name=None):

        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        uv = self.data.uv
        xv = self.data.xv

        alpha0 = self.alpha0

        rec = self.data._rec
        dalpha = alpha - alpha0

        dalpha_uv = (-1.0) * dalpha * uv
        dalpha_uv_diff = dalpha_uv - dalpha_uv.max(rec)
        expvals = np.exp(dalpha_uv_diff)

        num = xr.dot(expvals, xv, dims="rec") / len(xv[rec])
        den = expvals.mean("rec")

        return num / den

    def resample(self, nrep, idx=None, **kws):
        return self.__class__(
            alpha0=self.alpha0,
            data=self.data.resample(nrep=nrep, idx=idx, **kws),
            alpha_name=self.alpha_name,
        )


class MBARModel(StateCollection):
    """
    Sadly, this doesn't work as beautifully.
    """

    def __init__(self, states):
        if not _HAS_PYMBAR:
            raise ImportError("need pymbar to use this")
        super(MBARModel, self).__init__(states)

    @gcached(prop=False)
    def _default_params(self, state_name="state", alpha_name="alpha"):

        # all xvalues:
        xv = xr.concat([m.data.xv for m in self], dim=state_name)
        uv = xr.concat([m.data.uv for m in self], dim=state_name)
        alpha0 = xrwrap_alpha([m.alpha0 for m in self], name=alpha_name)

        # make sure uv, xv in correct order
        rec = self[0].data._rec
        xv = xv.transpose(state_name, rec, ...)
        uv = uv.transpose(state_name, rec, ...)

        # alpha[alpha] * uv[state, rec] = out[alpha, state, rec]
        Ukn = (alpha0 * uv).values.reshape(len(self), -1)
        N = np.ones(len(self)) * len(xv["rec"])
        mbar_obj = mbar.MBAR(Ukn, N)

        return uv, xv, alpha0, mbar_obj

    def predict(self, alpha, alpha_name=None):
        if alpha_name is None:
            alpha_name = self.alpha_name

        alpha = xrwrap_alpha(alpha, name=alpha_name)
        if alpha.ndim == 0:
            alpha = alpha.expand_dims(alpha.name)

        uv, xv, alpha0, mbar_obj = self._default_params("state", alpha.name)

        dims = xv.dims
        x = np.array(xv, order="c")
        x_flat = x.reshape(x.shape[0] * x.shape[1], -1)

        U = uv.values.reshape(-1)

        out = []
        for b in alpha.values:
            out.append(mbar_obj.computeMultipleExpectations(x_flat.T, b * U)[0])

        out = np.array(out)
        # reshape
        shape = (out.shape[0],) + x.shape[2:]
        out = xr.DataArray(
            out.reshape(shape), dims=(alpha.name,) + dims[2:]
        ).assign_coords(alpha=alpha)

        return out

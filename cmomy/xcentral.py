"""
Thin wrapper around central routines with xarray support
"""
from __future__ import absolute_import

import numpy as np
import xarray as xr

from . import central
from .cached_decorators import gcached
from .utils import _xr_order_like, _xr_wrap_like


###############################################################################
# central mom/comom routine
###############################################################################
def _xcentral_moments(
    x,
    mom,
    w=None,
    axis=0,
    last=True,
    mom_dims=None,
):

    assert isinstance(x, xr.DataArray)

    if isinstance(mom, tuple):
        mom = mom[0]

    if mom_dims is None:
        mom_dims = ("mom_0",)
    if isinstance(mom_dims, str):
        mom_dims = (mom_dims,)
    assert len(mom_dims) == 1

    if w is None:
        w = xr.ones_like(x)
    else:
        w = xr.DataArray(w).broadcast_like(x)

    if isinstance(axis, int):
        dim = x.dims[axis]
    else:
        dim = axis

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xave = xr.dot(w, x, dims=dim) * wsum_inv

    p = xr.DataArray(np.arange(0, mom + 1), dims=mom_dims)
    dx = (x - xave) ** p
    out = xr.dot(w, dx, dims=dim) * wsum_inv

    out.loc[{mom_dims[0]: 0}] = wsum
    out.loc[{mom_dims[0]: 1}] = xave

    if last:
        out = out.transpose(..., *mom_dims)
    return out


def _xcentral_comoments(
    x,
    mom,
    w=None,
    axis=0,
    last=True,
    broadcast=False,
    mom_dims=None,
):
    """
    calculate central co-mom (covariance, etc) along axis
    """

    if isinstance(mom, int):
        mom = (mom,) * 2

    mom = tuple(mom)
    assert len(mom) == 2

    assert isinstance(x, tuple) and len(x) == 2
    x, y = x

    assert isinstance(x, xr.DataArray)

    if w is None:
        w = xr.ones_like(x)
    else:
        w = xr.DataArray(w).broadcast_like(x)

    if broadcast:
        y = xr.DataArray(y).broadcast_like(x)
    else:
        assert isinstance(y, xr.DataArray)

        y = y.transpose(*x.dims)
        assert y.shape == x.shape
        assert x.dims == y.dims

    if isinstance(axis, int):
        dim = x.dims[axis]
    else:
        dim = axis

    if mom_dims is None:
        mom_dims = ["mom_0", "mom_1"]
    assert len(mom_dims) == 2

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    x = (x, y)

    xave = [xr.dot(w, xx, dims=dim) * wsum_inv for xx in x]

    p = [
        xr.DataArray(np.arange(0, mom + 1), dims=dim) for mom, dim in zip(mom, mom_dims)
    ]

    dx = [(xx - xxave) ** pp for xx, xxave, pp in zip(x, xave, p)]

    out = xr.dot(w, dx[0], dx[1], dims=dim) * wsum_inv

    out.loc[{mom_dims[0]: 0, mom_dims[1]: 0}] = wsum
    out.loc[{mom_dims[0]: 1, mom_dims[1]: 0}] = xave[0]
    out.loc[{mom_dims[0]: 0, mom_dims[1]: 1}] = xave[1]

    if last:
        out = out.transpose(..., *mom_dims)
    return out


def xcentral_moments(
    x,
    mom,
    w=None,
    axis=0,
    last=True,
    mom_dims=None,
    broadcast=False,
):
    """
    calculate central mom along axis

    Parameters
    ----------
    x : xarray.DataArray or tuple of xarray.Datarray
        input data
    mom : int
        number of mom to calculate
    w : array-like, optional
        if passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    axis : int, default=0
        axis to reduce along
    last : bool, default=True
        if True, put mom as last dimension.
        Otherwise, mom will be in first dimension
    dtype, order : options to np.asarray
    out : array
        if present, use this for output data
        Needs to have shape of either (mom,) + shape or shape + (mom,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape shape + (mom,) or (mom,) + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:]. Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment
    """

    if isinstance(mom, int):
        mom = (mom,)

    kws = dict(x=x, mom=mom, w=w, axis=axis, last=last, mom_dims=mom_dims)
    if len(mom) == 1:
        func = _xcentral_moments
    else:
        func = _xcentral_comoments
        kws["broadcast"] = broadcast
    return func(**kws)


def _attributes_from_xr(da, dim=None, mom_dims=None, **kws):
    if isinstance(da, xr.DataArray):
        if dim is not None:
            # reduce along this dim
            da = da.isel(**{dim: 0}, drop=True)
        out = {k: getattr(da, k) if v is None else v for k, v in kws.items()}
    else:
        out = kws.copy()

    out["mom_dims"] = mom_dims
    return out


def _check_xr_input(x, axis=None, mom_dims=None, **kws):
    if isinstance(x, xr.DataArray):
        if axis is None:
            dim = None
        elif isinstance(axis, str):
            dim = axis
            axis = x.get_axis_num(dim)
        else:
            dim = x.dims[axis]

        values = x.values
    else:
        if axis is None:
            dim = None
        else:
            dim = axis
        values = x

    kws = _attributes_from_xr(x, dim=dim, mom_dims=mom_dims, **kws)
    return kws, axis, values


class xCentralMoments(central.CentralMoments):
    """
    xarray.DataArray wrapper of `cmomy.central.CentralMoments`

    Most methods are wrapped to accept `xarray.DataArray` object

    Notes:
    ------
    unlike xarray, most methods take only the `axis` parameter
    instead of both an `axis` (for positional) and `dim` (for names)
    parameter for reduction.  If `axis` is a integer, then positional
    reduction is assumed, otherwise named reduction is done.  In the
    case that `dims` have integer values, this will lead to only positional
    reduction.
    """

    __slots__ = "_xdata"

    def __init__(
        self,
        data,
        mom_ndim,
        dims=None,
        coords=None,
        attrs=None,
        name=None,
        indexes=None,
        mom_dims=None,
        **kws,
    ):

        if isinstance(data, xr.DataArray):
            if mom_dims is None:
                mom_dims = data.dims[-mom_ndim:]
            else:
                if isinstance(mom_dims, str):
                    mom_dims = [mom_dims]
                assert len(mom_dims) == mom_ndim
                order = (...,) + tuple(mom_dims)

                if data.dims != order:
                    data = data.transpose(*order)
            self._xdata = data

        else:
            data = self._wrap_data_with_xr(
                data=data,
                mom_ndim=mom_ndim,
                dims=dims,
                coords=coords,
                attrs=attrs,
                name=name,
                indexes=indexes,
                mom_dims=mom_dims,
            )

        self._xdata = data
        # TODO: data.data or data.values?
        super(xCentralMoments, self).__init__(data=data.data, mom_ndim=mom_ndim)

    @property
    def values(self):
        return self._xdata

    # xarray attriburtes
    @property
    def attrs(self):
        return self._xdata.attrs

    @property
    def dims(self):
        return self._xdata.dims

    @property
    def coords(self):
        return self._xdata.coords

    @property
    def name(self):
        return self._xdata.name

    @property
    def indexes(self):
        return self._xdata.indexes

    @property
    def sizes(self):
        return self._xdata.sizes

    ###########################################################################
    # SECTION: top level creation/copy/new
    ###########################################################################
    @gcached()
    def _template_val(self):
        """template for values part of data"""
        return self._xdata[self._weight_index]

    # def _wrap_like_template(self, x):
    #     return _wrap_like(self._template_val, x)

    def _wrap_like(self, x):
        return _xr_wrap_like(self._xdata, x)

    @property
    def val_dims(self):
        """dim names of values"""
        return self.dims[: -self.mom_ndim]

    @property
    def mom_dims(self):
        return self.dims[-self.mom_ndim :]

    @property
    def _one_like_val(self):
        return xr.ones_like(self._template_val)

    @property
    def _zeros_like_val(self):
        return xr.zeros_like(self._template_val)

    def new_like(
        self, data=None, verify=False, check=False, copy=False, copy_kws=None, **kws
    ):
        """
        create new object like self, with new data


        Parameters
        ----------
        data : array-like, optional
            data for new object
        verify : bool, default=False
            if True, pass data through np.asarray
        check : bool, default=True
            if True, then check that data has same total shape as self
        copy : bool, default=False
            if True, perform copy of data
        *args, **kwargs : extra arguments
            arguments to data.copy
        """

        if data is None:
            data = xr.zeros_like(self._xdata)

        else:
            if verify:
                if not isinstance(data, xr.DataArray):
                    data = self._wrap_like(data)

            if check:
                assert data.shape == self.shape
                data.dims == self.dims

            if copy:
                if copy_kws is None:
                    copy_kws = {}
                data = data.copy(**copy_kws)
        return type(self)(data=data, mom_ndim=self.mom_ndim, **kws)

    @classmethod
    def _wrap_data_with_xr(
        cls,
        data,
        mom_ndim,
        dims=None,
        coords=None,
        attrs=None,
        name=None,
        indexes=None,
        mom_dims=None,
    ):
        """
        given an array x, wrap with appropriate stuff
        for an xarray
        """

        ndim = data.ndim
        if dims is not None:
            if isinstance(dims, str):
                dims = [dims]
        else:
            dims = [f"dim_{i}" for i in range(ndim - mom_ndim)]
        dims = tuple(dims)

        if len(dims) == ndim:
            dims_total = dims
        elif len(dims) == ndim - mom_ndim:
            if mom_dims is None:
                # default mom dims
                mom_dims = [f"mom_{i}" for i in range(mom_ndim)]
            elif isinstance(mom_dims, str):
                mom_dims = [mom_dims]

            assert len(mom_dims) == mom_ndim
            dims_total = dims + tuple(mom_dims)
        else:
            raise ValueError("bad dims {}, moment_dims {}".format(dims, mom_dims))

        # xarray object
        return xr.DataArray(
            data,
            dims=dims_total,
            coords=coords,
            attrs=attrs,
            name=name,
            indexes=indexes,
        )

    def zeros_like(self, *args, **kwargs):
        return self.new_like(data=xr.zeros_like(self._xdata, *args, **kwargs))

    @classmethod
    def zeros(
        cls,
        mom=None,
        val_shape=None,
        mom_ndim=None,
        shape=None,
        dtype=None,
        zeros_kws=None,
        dims=None,
        coords=None,
        attrs=None,
        name=None,
        indexes=None,
        mom_dims=None,
        **kws,
    ):
        """
        create a new base object

        Parameters
        ----------
        shape : tuple, optional
            if passed, create object with this total shape
        mom : int or tuple
            moments.  if integer, then moments will be (mom,)
        val_shape : tuple, optional
            shape of values, excluding moments.  For example, if considering the average
            of observations `x`, then val_shape = x.shape.
            if not passed, then assume shape = ()
        dtype : nunpy dtype, default=float
        kwargs : dict
            extra arguments to numpy.zeros

        Returns
        -------
        object : instance of class `cls`

        Notes
        -----
        the resulting total shape of data is shape + mom_shape
        """

        return super(xCentralMoments, cls).zeros(
            mom=mom,
            val_shape=val_shape,
            mom_ndim=mom_ndim,
            shape=shape,
            dtype=dtype,
            zeros_kws=zeros_kws,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            **kws,
        )

    ###########################################################################
    # xarray specific methods
    ###########################################################################
    def _wrap_xarray_method(self, _method, *args, **kwargs):
        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        return self.new_like(data=xdata)

    def assign_coords(self, coords=None, **coords_kwargs):
        return self._wrap_xarray_method("assign_coords", coords=coords, **coords_kwargs)

    def assign_attrs(self, *args, **kwargs):
        return self._wrap_xarray_method("assign_attrs", *args, **kwargs)

    def rename(self, new_name_or_name_dict=None, **names):
        return self._wrap_xarray_method(
            "rename", new_name_or_name_dict=new_name_or_name_dict, **names
        )

    def _wrap_xarray_method_from_data(
        self,
        _method,
        _data_copy=False,
        _data_order=False,
        _data_kws=None,
        *args,
        **kwargs,
    ):

        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        if _data_order:
            xdata = xdata.transpose(..., *self.mom_dims)
        if _data_kws is None:
            _data_kws = {}
        _data_kws.setdefault("copy", _data_copy)
        _data_kws.setdefault("copy", _data_copy)

        return type(self).from_data(data=xdata, mom_ndim=self.mom_ndim, **_data_kws)

    def stack(
        self,
        dimensions=None,
        _data_copy=False,
        _data_kws=None,
        _data_order=True,
        **dimensions_kwargs,
    ):
        return self._wrap_xarray_method_from_data(
            "stack",
            _data_copy=_data_copy,
            _data_kws=_data_kws,
            _data_order=_data_order,
            **dimensions_kwargs,
        )

    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop=False,
        _data_kws=None,
        _data_copy=False,
        _data_order=False,
        **indexers_kws,
    ):
        return self._wrap_xarray_method_from_data(
            "sel",
            _data_copy=_data_copy,
            _data_kws=_data_kws,
            _data_order=_data_order,
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kws,
        )

    def isel(
        self,
        indexers=None,
        drop=False,
        _data_kws=None,
        _data_copy=False,
        _data_order=False,
        **indexers_kws,
    ):
        return self._wrap_xarray_method_from_data(
            "isel",
            _data_copy=_data_copy,
            _data_kws=_data_kws,
            _data_order=_data_order,
            indexers=indexers,
            drop=drop,
            **indexers_kws,
        )

    ###########################################################################
    # Push/verify
    ###########################################################################
    def _xverify_value(
        self, x, target=None, dim=None, broadcast=False, expand=False, shape_flat=None
    ):

        if isinstance(target, str):

            if dim is not None:
                if isinstance(dim, int):
                    dim = x.dims[dim]

            if target == "val":
                target = self.val_dims
            elif target == "vals":
                target = (dim,) + self.val_dims
            elif target == "data":
                target = self.dims
            elif target == "datas":
                target = (dim,) + self.dims

        if isinstance(target, tuple):
            # no broadcast in this cast
            target_dims = target

            target_shape = tuple(
                x.sizes[k] if k == dim else self.sizes[k] for k in target_dims
            )

            # make sure in correct order
            x = x.transpose(*target_dims)
            target_output = x
            values = x.values

        else:
            target_dims = target.dims
            target_shape = target.shape

            target_output = None

            if dim is not None:
                if isinstance(dim, int):
                    # this is a bit awkward, but
                    # should work
                    # assume we already have target in correct order
                    dim = target_dims[0]

            if isinstance(x, xr.DataArray):
                if broadcast:
                    x = x.broadcast_like(target)
                else:
                    x = x.transpose(*target_dims)

                values = x.values
            else:
                # only things this can be is either a scalor or
                # array with same size as target
                x = np.asarray(x)
                if x.shape == target.shape:
                    values = x
                    # have x -> target to get correct recs
                    x = target

                elif x.ndim == 0 and broadcast and expand:
                    x = xr.DataArray(x).broadcast_like(target)
                    values = x.values

                elif (
                    x.ndim == 1 and len(x) == target.sizes[dim] and broadcast and expand
                ):
                    x = xr.DataArray(x, dims=dim).broadcast_like(target)
                    values = x.values

        # check shape
        assert values.shape == target_shape
        if dim is None:
            nrec = ()
        else:
            nrec = (x.sizes[dim],)

        if shape_flat is not None:
            values = values.reshape(nrec + shape_flat)

        if values.ndim == 0:
            values = values[()]

        if target_output is None:
            return values
        else:
            return values, target_output

    def _verify_value(
        self,
        x,
        target=None,
        axis=None,
        broadcast=False,
        expand=False,
        shape_flat=None,
    ):
        if isinstance(x, xr.DataArray) or isinstance(target, xr.DataArray):
            return self._xverify_value(
                x,
                target=target,
                dim=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
            )

        else:
            return super(xCentralMoments, self)._verify_value(
                x,
                target=target,
                axis=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
            )

    def _single_index_selector(
        self, val, dim_combined="variable", coords_combined=None, select=True
    ):
        idxs = self._single_index(val)[-self.mom_ndim :]
        if coords_combined is None:
            coords_combined = self.mom_dims

        selector = {
            dim: (idx if self._mom_ndim == 1 else xr.DataArray(idx, dims=dim_combined))
            for dim, idx in zip(self.mom_dims, idxs)
        }
        if select:
            out = self.values.isel(**selector)
            if self._mom_ndim > 1:
                out = out.assign_coords(**{dim_combined: list(coords_combined)})
            return out
        else:
            return selector

        return selector

    # have to do some special stuff for mean/var
    # due to xarray special indexing
    def mean(self, dim_combined="variable", coords_combined=None):
        return self._single_index_selector(
            val=1, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def var(self, dim_combined="variable", coords_combined=None):
        return self._single_index_selector(
            val=2, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def cmom(self):
        return self._wrap_like(super(xCentralMoments, self).cmom())

    def rmom(self):
        return self._wrap_like(super(xCentralMoments, self).rmom())

    def resample_and_reduce(
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
        bootstrap resample and reduce


        Parameters
        ----------
        rep_dim : str, default='rep'
            dimension name for resampled
            if 'dims' is not passed in kwargs, then reset dims
            with replicate dimension having name 'rep_dim',
            and all other dimensions have the same names as
            the parent object
        """

        if "dims" in kwargs:
            # passed explicit dims, so just use these.
            # NOTE, don't do anything special in this case
            pass
        else:
            axis = self._wrap_axis(axis)
            dims = list(self.values.dims)
            kwargs["dims"] = [rep_dim] + dims[:axis] + dims[axis + 1 :]

        return super(xCentralMoments, self).resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            parallel=parallel,
            resample_kws=resample_kws,
            **kwargs,
        )

    def block(self, block_size, axis=None, *args, **kwargs):
        """

        block average along an axis

        Parameters
        ----------
        block_size : int
            size of blocks to average over
        axis : str or int, default=0
            axis/dimension to block average over
        args : tuple
            positional arguments to CentralMomentsBase.block
        kwargs : dict
            key-word arguments to CentralMomentsBase.block
        """

        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        dim = self.dims[axis]

        template = self.values.isel(**{dim: 0})

        kwargs["dims"] = (dim,) + template.dims

        # properties after creation
        for k in ["coords", "attrs", "name", "indexes"]:
            kwargs[k] = getattr(template, k)

        return super(xCentralMoments, self).block(
            block_size=block_size, axis=axis, **kwargs
        )

    # def resample(self, indices, axis=0, first=True, **kws):
    #     self._raise_if_scalar()
    #     axis = self._wrap_axis(axis)
    #     dim = self.

    #     if not isinstance(indices, xr.DataArray):

    def _wrap_axis(self, axis, default=0, ndim=None):
        # if isinstance(axis, str):
        #     axis = self._xdata.get_axis_num(axis)
        # return super(xCentralMoments, self)._wrap_axis(
        #     axis=axis, default=default, ndim=ndim
        # )
        if isinstance(axis, str):
            return self._xdata.get_axis_num(axis)
        else:
            return super(xCentralMoments, self)._wrap_axis(
                axis=axis, default=default, ndim=ndim
            )

    @classmethod
    def from_data(
        cls,
        data,
        mom=None,
        mom_ndim=None,
        val_shape=None,
        dtype=None,
        copy=True,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):
        """
        object from data array


        Parameters
        ----------
        dims : tuple, optional
            dimension names for resulting object
        """

        kws = _attributes_from_xr(
            data,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        verify = not isinstance(data, xr.DataArray)
        return super(xCentralMoments, cls).from_data(
            data=data,
            mom=mom,
            mom_ndim=mom_ndim,
            val_shape=val_shape,
            dtype=dtype,
            copy=copy,
            verify=verify,
            **kws,
        )

    @classmethod
    def from_datas(
        cls,
        datas,
        mom=None,
        mom_ndim=None,
        axis=0,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):
        """
        Parameters
        ----------
        dims : tuple, optional
            dimension names.
            Note that this does not include the dimension reduced over.
        """

        kws, axis, values = _check_xr_input(
            datas,
            axis=axis,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xCentralMoments, cls).from_datas(
            datas=values,
            mom=mom,
            mom_ndim=mom_ndim,
            axis=axis,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    def to_raw(self):
        return self._wrap_like(super(xCentralMoments, self).to_raw())

    @classmethod
    def from_raw(
        cls,
        raw,
        mom=None,
        mom_ndim=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):
        """
        Parameters
        ----------
        dims : tuple, optional
            dimension names
        """
        kws, _, values = _check_xr_input(
            raw,
            axis=None,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xCentralMoments, cls).from_raw(
            raw=values,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    @classmethod
    def from_raws(
        cls,
        raws,
        mom=None,
        mom_ndim=None,
        axis=0,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            raws,
            axis=axis,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        super(xCentralMoments, cls).from_raws(
            values,
            mom=mom,
            mom_ndim=mom_ndim,
            axis=axis,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    @classmethod
    def from_vals(
        cls,
        x,
        w=None,
        axis=0,
        mom=2,
        val_shape=None,
        dtype=None,
        broadcast=False,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):

        mom_ndim = cls._mom_ndim_from_mom(mom)
        x0 = x if mom_ndim == 1 else x[0]
        kws, axis, values = _check_xr_input(
            x0,
            axis=axis,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xCentralMoments, cls).from_vals(
            x,
            w=w,
            axis=axis,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            broadcast=broadcast,
            **kws,
        )

    @classmethod
    def from_resample_vals(
        cls,
        x,
        freq=None,
        indices=None,
        nrep=None,
        w=None,
        axis=0,
        mom=2,
        rep_dim="rep",
        dtype=None,
        broadcast=False,
        parallel=True,
        resample_kws=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):

        mom_ndim = cls._mom_ndim_from_mom(mom)
        if mom_ndim == 1:
            y = None
        else:
            x, y = x

        kws, axis, values = _check_xr_input(
            x,
            axis=axis,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        if kws["dims"] is not None:
            kws["dims"] = (rep_dim,) + tuple(kws["dims"])

        # reorder
        w = _xr_order_like(x, w)
        if y is not None:
            y = _xr_order_like(x, y)
            x = (x, y)

        return super(xCentralMoments, cls).from_resample_vals(
            x=x,
            freq=freq,
            indices=indices,
            nrep=nrep,
            w=w,
            axis=axis,
            mom=mom,
            dtype=dtype,
            broadcast=broadcast,
            parallel=parallel,
            resample_kws=resample_kws,
            **kws,
        )

    @classmethod
    def from_stat(
        cls,
        a,
        v=0.0,
        w=None,
        mom=2,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):

        kws, _, values = _check_xr_input(
            a,
            axis=None,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xCentralMoments, cls).from_stat(
            a=a, v=v, w=w, mom=mom, val_shape=val_shape, dtype=dtype, **kws
        )

    @classmethod
    def from_stats(
        cls,
        a,
        v=0.0,
        w=None,
        axis=0,
        mom=2,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            a,
            axis=axis,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xCentralMoments, cls).from_stat(
            a=a, v=v, w=w, axis=axis, mom=mom, val_shape=val_shape, dtype=dtype, **kws
        )

    def transpose(self, *dims, transpose_coords=None, copy=False, **kws):
        # make sure dims are last
        dims = list(dims)
        for k in self.mom_dims:
            if k in dims:
                dims.pop(dims.index(k))
        dims = tuple(dims) + self.mom_dims

        values = (
            self.values.transpose(*dims, transpose_coords=transpose_coords)
            # make sure mom are last
            # .transpose(...,*self.mom_dims)
        )
        return type(self).from_data(values, mom_ndim=self.mom_ndim, copy=copy, **kws)

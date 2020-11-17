"""
Thin wrapper around central routines with xarray support
"""
from __future__ import absolute_import

import numpy as np
import xarray as xr

from .central import StatsAccumBase, _StatsAccumMixin, _StatsAccumCovMixin
from .cached_decorators import gcached, cached_clear



###############################################################################
# central moments/comoments routine
###############################################################################
def xcentral_moments(
    x, moments, weights=None, axis=0, last=True, moments_dims=None,
):
    """
    calculate central moments along axis

    Parameters
    ----------
    x : array-like
        input data
    moments : int
        number of moments to calculate
    weights : array-like, optional
        if passed, should be able to broadcast to `x`. An exception is if
        weights is a 1d array with len(weights) == x.shape[axis]. In this case,
        weights will be reshaped and broadcast against x
    axis : int, default=0
        axis to reduce along
    last : bool, default=True
        if True, put moments as last dimension.
        Otherwise, moments will be in first dimension
    dtype, order : options to np.asarray
    out : array
        if present, use this for output data
        Needs to have shape of either (moments,) + shape or shape + (moments,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape shape + (moments,) or (moments,) + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:]. Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment
    """

    assert isinstance(x, xr.DataArray)

    if moments_dims is None:
        moments_dims = ("mom_0",)
    if isinstance(moments_dims, str):
        moments_dims = (moments_dims,)
    assert len(moments_dims) == 1

    if weights is None:
        weights = xr.ones_like(x)
    else:
        weights = xr.DataArray(weights).broadcast_like(x)

    if isinstance(axis, int):
        dim = x.dims[axis]
    else:
        dim = axis

    wsum = weights.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xave = xr.dot(weights, x, dims=dim) * wsum_inv

    p = xr.DataArray(np.arange(0, moments + 1), dims=moments_dims)
    dx = (x - xave) ** p
    out = xr.dot(weights, dx, dims=dim) * wsum_inv

    out.loc[{moments_dims[0]: 0}] = wsum
    out.loc[{moments_dims[0]: 1}] = xave

    if last:
        out = out.transpose(..., *moments_dims)

    return out


def xcentral_comoments(
    x0,
    x1,
    moments,
    weights=None,
    axis=0,
    last=True,
    broadcast=False,
    moments_dims=None,
):
    """
    calculate central co-moments (covariance, etc) along axis
    """

    if isinstance(moments, int):
        moments = (moments,) * 2

    moments = tuple(moments)
    assert len(moments) == 2

    assert isinstance(x0, xr.DataArray)

    if weights is None:
        weights = xr.ones_like(x0)
    else:
        weights = xr.DataArray(weights).broadcast_like(x0)

    if broadcast:
        x1 = xr.DataArray(x1).broadcast_like(x0)
    else:
        assert isinstance(x1, xr.DataArray)

        x1 = x1.transpose(*x0.dims)
        assert x1.shape == x0.shape
        assert x0.dims == x1.dims

    if isinstance(axis, int):
        dim = x0.dims[axis]
    else:
        dim = axis

    if moments_dims is None:
        moments_dims = ["mom_0", "mom_1"]
    assert len(moments_dims) == 2

    wsum = weights.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    x = (x0, x1)

    xave = [xr.dot(weights, xx, dims=dim) * wsum_inv for xx in x]

    p = [
        xr.DataArray(np.arange(0, mom + 1), dims=dim)
        for mom, dim in zip(moments, moments_dims)
    ]

    dx = [(xx - xxave) ** pp for xx, xxave, pp in zip(x, xave, p)]

    out = xr.dot(weights, dx[0], dx[1], dims=dim) * wsum_inv

    out.loc[{moments_dims[0]: 0, moments_dims[1]: 0}] = wsum
    out.loc[{moments_dims[0]: 1, moments_dims[1]: 0}] = xave[0]
    out.loc[{moments_dims[0]: 0, moments_dims[1]: 1}] = xave[1]

    if last:
        out = out.transpose(..., *moments_dims)

    return out


def _attributes_from_xr(da, dim=None, moments_dims=None, **kws):
    if isinstance(da, xr.DataArray):
        if dim is not None:
            # reduce along this dim
            da = da.isel(**{dim: 0}, drop=True)
        out = {k: getattr(da, k) if v is None else v for k, v in kws.items()}
    else:
        out = kws.copy()

    out["moments_dims"] = moments_dims
    return out


def _check_xr_input(x, axis=None, moments_dims=None, **kws):
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

    kws = _attributes_from_xr(x, dim=dim, moments_dims=moments_dims, **kws)
    return kws, axis, values


def _dim_axis_from_xr(da, axis):
    if isinstance(da, xr.DataArray):
        if isinstance(axis, int):
            dim = da.dims[axis]
        else:
            dim = axis
            axis = da.get


def _wrap_like(da, x):
    """
    wrap x with xarray like da
    """

    x = np.asarray(x)
    assert x.shape == da.shape

    return xr.DataArray(
        x, dims=da.dims, coords=da.coords, name=da.name, indexes=da.name
    )



def _order_like(template, *others):
    """
    given dimensions, order in same manner
    """

    if not isinstance(template, xr.DataArray):
        out = others

    else:
        dims = template.dims

        key_map = {dim: i for i, dim in enumerate(dims)}
        key = lambda x: key_map[x]

        out = []
        for other in others:
            if isinstance(other, xr.DataArray):
                # reorder
                order = sorted(other.dims, key=key)

                x = other.transpose(*order)
            else:
                x = other

            out.append(x)

    if len(out) == 1:
        out = out[0]

    return out




class xStatsAccumBase(StatsAccumBase):
    __slots__ = "_xdata"

    def __init__(
        self,
        moments,
        shape=None,
        dtype=None,
        data=None,
        dims=None,
        coords=None,
        attrs=None,
        name=None,
        indexes=None,
        moments_dims=None,
    ):

        if isinstance(data, xr.DataArray):
            if moments_dims is None:
                moments_dims = data.dims[-self._moments_len :]
            else:
                if isinstance(moments_dims, str):
                    moments_dims = [moments_dims]
                assert len(moments_dims) == self._moments_len

                order = (...,) + tuple(moments_dims)
                data = data.transpose(*order)

            # use this then
            super(xStatsAccumBase, self).__init__(
                moments=moments, shape=shape, dtype=dtype, data=data.values
            )
            self._xdata = data

        else:
            # build up
            super(xStatsAccumBase, self).__init__(
                moments=moments, shape=shape, dtype=dtype, data=data
            )

            # dims
            if dims is not None:
                if isinstance(dims, str):
                    dims = [dims]
            else:
                # default dims
                dims = [f"dim_{i}" for i in range(self.ndim)]
            dims = tuple(dims)

            if len(dims) == self.ndim + self._moments_len:
                # assume dims has all the dimensions for data and moments
                dims_total = dims

            elif len(dims) == self.ndim:
                if moments_dims is None:
                    # default moments dims
                    moments_dims = [f"mom_{i}" for i in range(self._moments_len)]
                assert len(moments_dims) == self._moments_len
                dims_total = dims + tuple(moments_dims)
            else:
                raise ValueError('bad dims {}, moment_dims {}'.format(dims, moments_dims))

            assert len(dims_total) == self.ndim + self._moments_len

            # xarray object
            self._xdata = xr.DataArray(
                self._data,
                dims=dims_total,
                coords=coords,
                attrs=attrs,
                name=name,
                indexes=indexes,
            )

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

    def _wrap_xarray_method(self, _method, *args, **kwargs):
        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        # Use new like to create a new thin wrapper
        # underlying data may still be the same between new and
        # self, but metadata can be different
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
            xdata = xdata.transpose(..., *self._dims_mom)
        if _data_kws is None:
            _data_kws = {}
        _data_kws.setdefault("copy", _data_copy)
        _data_kws.setdefault("copy", _data_copy)

        return type(self).from_data(data=xdata, **_data_kws)

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

    @gcached()
    def _template(self):
        return self._xdata[self._weight_index]

    def _wrap_like_template(self, x):
        return _wrap_like(self._template, x)

    def _wrap_like(self, x):
        return _wrap_like(self._xdata, x)

    @gcached()
    def _dims_val(self):
        """dim names of values"""
        return self.dims[: -self._moments_len]

    @gcached()
    def _dims_mom(self):
        """dim names of moments"""
        return self.dims[-self._moments_len :]

    @property
    def _one_like_val(self):
        return xr.ones_like(self._template)

    @property
    def _zeros_like_val(self):
        return xr.zeros_like(self._template)

    def new_like(self, data=None):
        """
        create new object like self

        Parameters
        ----------
        data : array-like, optional
            if passed, then this will be the data of the new object

        Returns
        -------
        output : xStats object
            same type as `self`
        """

        return type(self)(
            moments=self.moments,
            shape=self.shape,
            dtype=self.dtype,
            data=data,
            coords=self.coords,
            dims=self.dims,
            attrs=self.attrs,
            name=self.name,
            indexes=self.indexes,
        )

    def _xverify_value(
        self, x, target=None, dim=None, broadcast=False, expand=False, shape_flat=None
    ):

        if isinstance(target, str):

            if dim is not None:
                if isinstance(dim, int):
                    dim = x.dims[dim]

            if target == "val":
                target = self._dims_val
            elif target == "vals":
                target = (dim,) + self._dims_val
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

        if target_output is None:
            return values
        else:
            return values, target_output

    def _verify_value(
        self, x, target=None, axis=None, broadcast=False, expand=False, shape_flat=None,
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
            return super(xStatsAccumBase, self)._verify_value(
                x,
                target=target,
                axis=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
            )

    # have to do some special stuff for mean/var
    # due to xarray special indexing
    def _single_index_selector(
        self, val, dim_combined="variable", coords_combined=None, select=True
    ):
        idxs = self._single_index(val)[-self._moments_len :]
        if coords_combined is None:
            coords_combined = self._dims_mom

        selector = {
            dim: (
                idx if self._moments_len == 1
                else xr.DataArray(idx, dims=dim_combined)
            )
            for dim, idx in zip(self._dims_mom, idxs)
        }
        if select:
            out = self.values.isel(**selector)
            if self._moments_len > 1:
                out = out.assign_coords(
                    **{dim_combined: list(coords_combined)}
                )
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
        return self._wrap_like(super(xStatsAccumBase, self).cmom())

    def rmom(self):
        return self._wrap_like(super(xStatsAccumBase, self).rmom())


    # def mean(self, dim_combined="variable", coords_combined=None):
    #     idxs = self._single_index(1)[-self._moments_len :]

    #     if coords_combined is None:
    #         coords_combined = self._dims_mom

    #     selector = {
    #         dim: xr.DataArray(idx, dims=dim_combined)
    #         for dim, idx in zip(self._dims_mom, idxs)
    #     }
    #     return self.values.isel(**selector)

    # def var(self, dim_combined="variable", coords_combined=None):
    #     idxs = self._single_index(2)[-self._moments_len :]

    # def block1(self, block_size, axis=None, *args, **kwargs):

    #     self._raise_if_scalar()

    #     axis = self._wrap_axis(axis)
    #     dim = self.dims[axis]

    #     da = self.values

    #     n = da.sizes[dim]

    #     if block_size is None:
    #         block_size = n
    #         nblock = 1
    #     else:
    #         nblock = n // block_size

    #     da = da.isel(**{dim: slice(None, nblock * block_size, None)})

    #     z = '_tmp_{}'.format(dim)
    #     b = '{}_block'.format(dim)
    #     datas = (
    #         da
    #         .isel(**{dim : slice(None, nblock * block_size, None)})
    #         .rename({dim : z})
    #         .assign_coords(
    #             **{
    #                 dim: (z, np.repeat(np.arange(nblock), block_size)),
    #                 b:   (z, np.tile(np.arange(block_size), nblock))
    #             }
    #         )
    #         .set_index({z: (dim, b)})
    #         .unstack(z)
    #         .transpose(*(b,) + da.dims)
    #     )

    #     del datas[dim], datas[b]

    #     return type(self).from_datas(
    #         datas=datas, axis=b,
    #         moments=self.moments, *args, **kwargs
    #     )

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
            positional arguments to StatsAccumBase.block
        kwargs : dict
            key-word arguments to StatsAccumBase.block
        """


        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        dim = self.dims[axis]

        template = self.values.isel(**{dim: 0})

        kwargs["dims"] = (dim,) + template.dims

        # properties after creation
        for k in ["coords", "attrs", "name", "indexes"]:
            kwargs[k] = getattr(template, k)

        return super(xStatsAccumBase, self).block(
            block_size=block_size, axis=axis, **kwargs
        )

    def resample_and_reduce(
        self,
        freq=None,
        indices=None,
        nrep=None,
        axis=None,
        dim_rep="rep",
        resample_kws=None,
        **kwargs,
    ):
        """
        bootstrap resample and reduce


        Parameters
        ----------
        dim_rep : str, default='rep'
            dimension name for resampled
            if 'dims' is not passed in kwargs, then reset dims
            with replicate dimension having name 'dim_rep',
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
            kwargs["dims"] = [dim_rep] + dims[:axis] + dims[axis + 1 :]

        return super(xStatsAccumBase, self).resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            resample_kws=resample_kws,
            **kwargs,
        )

    @classmethod
    def from_data(
        cls,
        data,
        moments=None,
        shape=None,
        dtype=None,
        copy=True,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
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
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        verify = not isinstance(data, xr.DataArray)
        return super(xStatsAccumBase, cls).from_data(
            data=data,
            moments=moments,
            shape=shape,
            dtype=dtype,
            copy=copy,
            verify=verify,
            **kws,
        )

    @classmethod
    def from_datas(
        cls,
        datas,
        moments=None,
        axis=0,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
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
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xStatsAccumBase, cls).from_datas(
            datas=values, moments=moments, axis=axis, shape=shape, dtype=dtype, **kws
        )

    def to_raw(self):
        return self._wrap_like(super(xStatsAccumBase, self).to_raw())

    @classmethod
    def from_raw(
        cls,
        raw,
        moments=None,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
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
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xStatsAccumBase, cls).from_raw(
            raw=values, moments=moments, shape=shape, dtype=dtype, **kws
        )

    @classmethod
    def from_raws(
        cls,
        raws,
        moments=None,
        axis=0,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            raws,
            axis=axis,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        super(xStatsAccumBase, cls).from_raws(
            values, moments=moments, axis=axis, shape=shape, dtype=dtype, **kws
        )

    def _wrap_axis(self, axis, default=0, ndim=None):
        # if isinstance(axis, str):
        #     axis = self._xdata.get_axis_num(axis)
        # return super(xStatsAccumBase, self)._wrap_axis(
        #     axis=axis, default=default, ndim=ndim
        # )
        if isinstance(axis, str):
            return self._xdata.get_axis_num(axis)
        else:
            return super(xStatsAccumBase, self)._wrap_axis(
                axis=axis, default=default, ndim=ndim
            )


class xStatsAccum(xStatsAccumBase, _StatsAccumMixin):
    _moments_len = 1

    @classmethod
    def from_vals(
        cls,
        x,
        w=None,
        axis=0,
        moments=2,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            x,
            axis=axis,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        # return kws, axis, values

        return super(xStatsAccum, cls).from_vals(
            x, w=w, axis=axis, moments=moments, shape=shape, dtype=dtype, **kws
        )

    @classmethod
    def from_stat(
        cls,
        a,
        v=0.0,
        w=None,
        moments=2,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, _, values = _check_xr_input(
            x,
            axis=None,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xStatsAccum, cls).from_stat(
            a=a, v=v, w=w, moments=moments, shape=shape, dtype=dtype, **kws
        )

    @classmethod
    def from_stats(
        cls,
        a,
        v=0.0,
        w=None,
        axis=0,
        moments=2,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            a,
            axis=axis,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xStatsAccum, cls).from_stat(
            a=a, v=v, w=w, axis=axis, moments=moments, shape=shape, dtype=dtype, **kws
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
        moments=2,
        dim_rep='rep',
        dtype=None,
        resample_kws=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            x,
            axis=axis,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        if kws['dims'] is not None:
            kws['dims'] = (dim_rep,) + tuple(kws['dims'])


        # reorder
        w = _order_like(x, w)


        return super(xStatsAccum, cls).from_resample_vals(
            x=x,
            freq=freq,
            indices=indices,
            nrep=nrep,
            w=w,
            axis=axis,
            moments=moments,
            dtype=dtype,
            resample_kws=resample_kws,
            **kws,
        )

    def transpose(self, *dims, transpose_coords=None, copy=False, **kws):

        # make sure dims are last
        dims = list(dims)
        for k in self._dims_mom:
            if k in dims:
                dims.pop(dims.index(k))
        dims = tuple(dims) + self._dims_mom

        values = (
            self.values.transpose(*dims, transpose_coords=transpose_coords)
            # make sure moments are last
            # .transpose(...,*self._dims_mom)
        )

        return type(self).from_data(values, copy=copy, **kws)


class xStatsAccumCov(xStatsAccumBase, _StatsAccumCovMixin):
    _moments_len = 2

    # def _single_index_selector(
    #     self, val, dim_combined="variable", coords_combined=None, select=True
    # ):
    #     idxs = self._single_index(1)[-self._moments_len :]
    #     if coords_combined is None:
    #         coords_combined = self._dims_mom

    #     selector = {
    #         dim: xr.DataArray(idx, dims=dim_combined)
    #         for dim, idx in zip(self._dims_mom, idxs)
    #     }
    #     if select:
    #         return self.values.isel(**selector).assign_coords(
    #             **{dim_combined: list(coords_combined)}
    #         )
    #     else:
    #         return selector

    #     return selector

    # # have to do some special stuff for mean/var
    # # due to xarray special indexing
    # def mean(self, dim_combined="variable", coords_combined=None):
    #     return self._single_index_selector(
    #         val=1, dim_combined=dim_combined, coords_combined=coords_combined
    #     )

    # def var(self, dim_combined="variable", coords_combined=None):
    #     return self._single_index_selector(
    #         val=2, dim_combined=dim_combined, coords_combined=coords_combined
    #     )

    # def cmom(self):
    #     return self._wrap_like(super(xStatsAccumCov, self).cmom())

    # def rmom(self):
    #     return self._wrap_like(super(xStatsAccumCov, self).rmom())

    @classmethod
    def from_vals(
        cls,
        x0,
        x1,
        w=None,
        axis=0,
        moments=2,
        broadcast=False,
        shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            x0,
            axis=axis,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        return super(xStatsAccumCov, cls).from_vals(
            x0=x0,
            x1=x1,
            w=w,
            axis=axis,
            moments=moments,
            broadcast=broadcast,
            shape=shape,
            dtype=dtype,
            **kws,
        )

    @classmethod
    def from_resample_vals(
        cls,
        x0,
        x1,
        freq=None,
        indices=None,
        nrep=None,
        w=None,
        axis=0,
        moments=2,
        dim_rep='rep',
        broadcast=False,
        dtype=None,
        resample_kws=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        moments_dims=None,
    ):

        kws, axis, values = _check_xr_input(
            x0,
            axis=axis,
            moments_dims=moments_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
        )

        if kws['dims'] is not None:
            kws['dims'] = (dim_rep,) + tuple(kws['dims'])

        x1, w = _order_like(x0, x1, w)

        return super(xStatsAccumCov, cls).from_resample_vals(
            x0=x0,
            x1=x1,
            freq=freq,
            indices=indices,
            nrep=nrep,
            w=w,
            axis=axis,
            moments=moments,
            broadcast=broadcast,
            dtype=dtype,
            resample_kws=resample_kws,
            **kws,
        )




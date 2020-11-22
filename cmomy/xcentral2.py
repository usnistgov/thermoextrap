"""
Thin wrapper around central routines with xarray support
"""
from __future__ import absolute_import

import numpy as np
import xarray as xr

# from .central import StatsAccumBase, _StatsAccumMixin, _StatsAccumCovMixin
from .cached_decorators import gcached, cached_clear
from .pushers import factory_pushers
from .resample import resample_data, resample_vals, randsamp_freq
from . import convert


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


class StatsAccumBase(object):
    _moments_len = 1
    __slots__ = (
        "_xdata",
        "_cache",
        "_data",
        "_data_flat",
        "_push",
    )

    def __init__(
        self, xdata,
    ):
        """
        wrap an xarray.DataArray for handling moment calculation
        """

        if not isinstance(xdata, xr.DataArray):
            raise ValueError("must pass an xarray.DataArray object")

        assert xdata.ndim >= self._moments_len

        self._xdata = xdata
        self.data = xdata.data

        # setup pushers
        vec = len(self.shape) > 0
        cov = self._moments_len == 2
        self._push = factory_pushers(cov=cov, vec=vec)

    @property
    def xdata(self):
        return self._xdata

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._data_flat = data.reshape(self._shape_data_flat)

    @property
    def shape(self):
        """total shape, including moments"""
        return self._data.shape

    @property
    def shape_val(self):
        """
        shape, less moment dimensions
        """
        return self._data.shape[: -self._moments_len]

    @property
    def shape_mom(self):
        """shape of moments"""
        return self._data.shape[-self._moments_len :]

    @gcached()
    def mom(self):
        """moments"""
        return tuple(s - 1 for s in self._data.shape[-self._moments_len :])

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def values(self):
        return self._xdata

    @property
    def ndim(self):
        return len(self._data.ndim)

    @gcached()
    def _shape_vals_flat(self):
        """
        flat shape for values
        """
        shape_val = self.shape_val
        if shape_val == ():
            return ()
        else:
            return (np.prod(shape_val),)

    @gcached()
    def _shape_data_flat(self):
        """shape of flat data"""
        return self._shape_vals_flat + self.shape_moms

    # convertes
    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    # not sure I want to actually implements this
    # could lead to all sorts of issues applying
    # ufuncs to underlying data
    # def __array_wrap__(self, obj, context=None):
    #     return self, obj, context
    # xarray attriburtes

    # New/copy
    def new_like(self, xdata=None):
        if xdata is None:
            xdata = self.xdata
        return type(self)(xdata=xdata)

    def copy(self, xdata=None, *args, **kwargs):
        """
        create copy of object

        Parameters
        ----------
        args : tuple
            position arguments to xarray.DataArray.copy
        kwargs : dict
            keyword arguments to xarray.DataArray.copy

        See Also
        --------
        xarray.DataArray.copy
        """
        if xdata is None:
            xdata = self.xdata
        return type(self)(xdata.copy(*args, **kwargs))

    # xarray stuff
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
        return self.new_like(data=getattr(self._xdata, _method)(*args, **kwargs))

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
        _copy_kws=None,
        *args,
        **kwargs,
    ):

        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        if _data_order:
            xdata = xdata.transpose(..., *self._dims_mom)
        if _copy_kws is None:
            _copy_kws = {}
        if _data_copy:
            xdata = xdata.copy(**_copy_kws)
        return self.new_like(xdata=xdata)

    def stack(
        self,
        dimensions=None,
        _data_copy=False,
        _copy_kws=None,
        _data_order=True,
        **dimensions_kwargs,
    ):
        return self._wrap_xarray_method_from_data(
            "stack",
            _data_copy=_data_copy,
            _copy_kws=_copy_kws,
            _data_order=_data_order,
            **dimensions_kwargs,
        )

    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop=False,
        _copy_kws=None,
        _data_copy=False,
        _data_order=False,
        **indexers_kws,
    ):
        return self._wrap_xarray_method_from_data(
            "sel",
            _data_copy=_data_copy,
            _copy_kws=_copy_kws,
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
        _copy_kws=None,
        _data_copy=False,
        _data_order=False,
        **indexers_kws,
    ):
        return self._wrap_xarray_method_from_data(
            "isel",
            _data_copy=_data_copy,
            _copy_kws=_copy_kws,
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
    def dims_val(self):
        """dim names of values"""
        return self.dims[: -self._moments_len]

    @gcached()
    def dims_mom(self):
        """dim names of moments"""
        return self.dims[-self._moments_len :]

    @property
    def _one_like_val(self):
        return xr.ones_like(self._template)

    @property
    def _zeros_like_val(self):
        return xr.zeros_like(self._template)

    @property
    def _unit_weight(self):
        return _cached_ones(self.shape, dtype=self.dtype)

    def _single_index(self, val):
        dims = len(self.moments)
        if dims == 1:
            index = [val]
        else:
            # this is a bit more complicated
            index = [[0] * dims for _ in range(dims)]
            for i in range(dims):
                index[i][i] = val
        return index

    @gcached()
    def _weight_index(self):
        index = [0] * len(self.moments)
        if self.ndim > 0:
            index = [...] + index
        return tuple(index)

    @gcached(prop=False)
    def _single_index(self, val):
        # index with things like
        # data[1,0 ,...]
        # data[0,1 ,...]

        # so build a total with indexer like
        # data[indexer]
        # with
        # index = ([1,0],[0,1],...)
        dims = len(self.moments)
        if dims == 1:
            index = [val]
        else:
            # this is a bit more complicated
            index = [[0] * dims for _ in range(dims)]
            for i in range(dims):
                index[i][i] = val

        if self.ndim > 0:
            index = [...] + index

        return tuple(index)

    def _single_index_selector(
        self, val, dim_combined="variable", coords_combined="infer", select=True
    ):

        idxs = self._single_index(val)[-self._moments_len :]

        if coords_combined == "infer" and self._moments_len > 1:
            coords_combined = self._dims_mom

        selector = {
            dim: (
                idx if self._moments_len == 1 else xr.DataArray(idx, dims=dim_combined)
            )
            for dim, idx in zip(self._dims_mom, idxs)
        }
        if select:
            out = self.values.isel(**selector)

            if coords_combined is not None:
                out = out.assign_coords(**{dim_combined: list(coords_combined)})
            return out
        else:
            return selector
        return selector

    def weight(self):
        return self.values[self._weight_index]

    def mean(self, dim_combined="variable", coords_combined=None):
        return self._single_index_selector(
            val=1, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def var(self, dim_combined="variable", coords_combined=None):
        return self._single_index_selector(
            val=2, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def cmom(self):
        out = self.values.copy()
        out[self._weight_index] = 1
        out.loc[self._single_index_selector(val=1)] = 0
        return out

    def rmom(self):
        out = self.to_raw()
        out.loc[self._weight_index] = 1
        return out

    def _check_other(self, b):
        """check other object"""
        assert type(self) == type(b)
        assert self.shape == b.shape

    def __iadd__(self, b):
        self._check_other(b)
        self.push_data(b.data)
        return self

    def __add__(self, b):
        self._check_other(b)
        new = self.copy()
        new.push_data(b.data)
        return new

    def __isub__(self, b):
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        data = b.data.copy()
        data[self._weight_index] *= -1
        self.push_data(data)
        return self

    def __sub__(self, b):
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        new = b.copy()
        # NOTE: consider adding push_data_scale...
        new._data[self._weight_index] *= -1
        new.push_data(self.data)
        return new

    def __mul__(self, scale):
        """
        new object with weights scaled by scale
        """
        scale = float(scale)
        new = self.copy()
        new._data[self._weight_index] *= scale
        return new

    def __imul__(self, scale):
        scale = float(scale)
        self._data[self._weight_index] *= scale
        return self

    # Universal pushers
    def push_data(self, data):
        data = self.check_data(data)
        self._push.data(self._data_flat, data)

    def push_datas(self, datas, axis=0):
        datas = self.check_datas(datas, axis)
        self._push.datas(self._data_flat, datas)



    # NOTE: A work in progress from here....
    # utilities
    def _wrap_axis(self, axis, default=0, ndim=None):
        """wrap axis to po0sitive value and check"""
        if axis is None:
            axis = default
        if ndim is None:
            ndim = self.ndim
        if axis < 0:
            axis += ndim
        assert 0 <= axis < ndim
        return axis

    @classmethod
    def _check_mom(cls, moments, shape=None):
        if moments is None:
            if shape is not None:
                moments = tuple(x - 1 for x in shape[-cls._moments_len :])
            else:
                raise ValueError("must specify moments")

        if isinstance(moments, int):
            moments = (moments,) * cls._moments_len
        else:
            moments = tuple(moments)
        assert len(moments) == cls._moments_len
        return moments

    @classmethod
    def _datas_axis_to_first(cls, datas, axis):
        datas = np.asarray(datas)
        ndim = datas.ndim - cls._moments_len
        if axis < 0:
            axis += ndim
        assert 0 <= axis < ndim

        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        return datas, axis

    @property
    def _is_vector(self):
        return self.ndim > 0

    def _raise_if_scalar(self, message=None):
        if not self._is_vector:
            if message is None:
                message = "not implemented for scalar"
            raise ValueError(message)

    def _reshape_flat(self, x, nrec=None, shape_flat=None):
        if shape_flat is None:
            shape_flat = self._shape_flat
        if nrec is None:
            x = x.reshape(self._shape_flat)
        else:
            x = x.reshape(*((nrec,) + self._shape_flat))
        if x.ndim == 0:
            x = x[()]
        return x

    def _asarray(self, val):
        return np.asarray(val)

    def _get_target_shape(self, nrec=None, axis=None, data=False):
        """
        return shape of targert object array
        """
        shape = self.shape
        if data:
            shape += self._moments_shape
        if axis is not None:
            shape = _shape_insert_axis(shape, axis, nrec)
        return shape

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
    def mean(self, dim_combined="variable", coords_combined=None):
        idxs = self._single_index(1)[-self._moments_len :]

        if coords_combined is None:
            coords_combined = self._dims_mom

        selector = {
            dim: xr.DataArray(idx, dim=dim_combined)
            for dim, idx in zip(self._dims_mom, idxs)
        }
        return self.values.isel(**selector)

    def var(self, dim_combined="variable", coords_combined=None):
        idxs = self._single_index(2)[-self._moments_len :]

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
        dim_rep="rep",
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

        if kws["dims"] is not None:
            kws["dims"] = (dim_rep,) + tuple(kws["dims"])

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

    def _single_index_selector(
        self, val, dim_combined="variable", coords_combined=None, select=True
    ):
        idxs = self._single_index(1)[-self._moments_len :]
        if coords_combined is None:
            coords_combined = self._dims_mom

        selector = {
            dim: xr.DataArray(idx, dims=dim_combined)
            for dim, idx in zip(self._dims_mom, idxs)
        }
        if select:
            return self.values.isel(**selector).assign_coords(
                **{dim_combined: list(coords_combined)}
            )
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
        return self._wrap_like(super(xStatsAccumCov, self).cmom())

    def rmom(self):
        return self._wrap_like(super(xStatsAccumCov, self).rmom())

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
        dim_rep="rep",
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

        if kws["dims"] is not None:
            kws["dims"] = (dim_rep,) + tuple(kws["dims"])

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

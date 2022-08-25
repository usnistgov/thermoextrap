# Type: ignore
"""Thin wrapper around central routines with xarray support."""


from __future__ import annotations

from typing import (  # TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)
from warnings import warn

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from ._typing import T_CENTRALMOMENTS, T_MOM
from .cached_decorators import gcached
from .central import CentralMoments
from .utils import _xr_order_like  # , _xr_wrap_like

# from numpy import float64, ndarray


# from xarray.core.coordinates import DataArrayCoordinates
# from xarray.core.dataarray import DataArray
# from xarray.core.indexes import Indexes
# from xarray.core.utils import Frozen


T_MOM_DIMS = Union[Hashable, Tuple[Hashable, Hashable]]
T_DIMS = Union[Hashable, Tuple[Hashable, ...]]
_T_XRVALS = Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]

# if TYPE_CHECKING:
#     from .central import CentralMoments


###############################################################################
# central mom/comom routine
###############################################################################


def _select_axis_dim(
    dims: Tuple[Hashable, ...],
    axis: int | str | None = None,
    dim: Hashable | None = None,
    default_axis: int | None = None,
    default_dim: Hashable | None = None,
) -> Tuple[int, Hashable]:
    """Produce axis/dim from input."""

    if axis is None and dim is None:
        if default_axis is None and default_dim is None:
            raise ValueError("must specify axis or dim")
        elif default_axis is not None and default_dim is not None:
            raise ValueError("can only specify one of default_axis or default_dim")
        elif default_axis:
            axis = default_axis
        else:
            dim = default_dim

    elif axis is not None and dim is not None:
        raise ValueError("can only specify axis or dim")

    if dim is not None:
        if dim in dims:
            axis = dims.index(dim)
        else:
            raise ValueError(f"did not find '{dim}' in {dims}")
    elif axis is not None:
        if isinstance(axis, str):
            warn(
                "Using string value for axis is deprecated.  Please use `dim` option instead."
            )
            dim = axis
            axis = dims.index(dim)
        else:
            dim = dims[axis]
    else:
        raise ValueError(f"unknown dim {dim} and axis {axis}")

    return axis, dim


@no_type_check
def _xcentral_moments(
    vals: xr.DataArray,
    mom: int | Tuple[int],
    w: xr.DataArray | None = None,
    axis: int | str | None = None,
    dim: Hashable | None = None,
    last: bool = True,
    mom_dims: T_MOM_DIMS | None = None,
):  # -> xr.DataArray:

    x = vals
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

    axis, dim = _select_axis_dim(x.dims, axis, dim, default_axis=0)
    # if isinstance(axis, int):
    #     dim = x.dims[axis]
    # else:
    #     dim = axis

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


@no_type_check
def _xcentral_comoments(
    vals: Tuple[xr.DataArray, xr.DataArray],
    mom: int | Tuple[int, int],
    w: xr.DataArray | None = None,
    axis: int | str | None = None,
    dim: Hashable | None = None,
    last: bool = True,
    broadcast: bool = False,
    mom_dims: Tuple[Hashable, Hashable] | None = None,
):
    """Calculate central co-mom (covariance, etc) along axis."""

    if isinstance(mom, int):
        mom = (mom,) * 2
    else:
        mom = tuple(mom)  # type: ignore

    assert len(mom) == 2  # type: ignore

    assert isinstance(vals, tuple) and len(vals) == 2
    x, y = vals

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

    axis, dim = _select_axis_dim(x.dims, axis, dim, default_axis=0)

    if mom_dims is None:
        mom_dims = ("mom_0", "mom_1")

    assert len(mom_dims) == 2

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xy = (x, y)

    xave = [xr.dot(w, xx, dims=dim) * wsum_inv for xx in xy]

    p = [
        xr.DataArray(np.arange(0, mom + 1), dims=dim) for mom, dim in zip(mom, mom_dims)
    ]

    dx = [(xx - xxave) ** pp for xx, xxave, pp in zip(xy, xave, p)]

    out = xr.dot(w, dx[0], dx[1], dims=dim) * wsum_inv

    out.loc[{mom_dims[0]: 0, mom_dims[1]: 0}] = wsum
    out.loc[{mom_dims[0]: 1, mom_dims[1]: 0}] = xave[0]
    out.loc[{mom_dims[0]: 0, mom_dims[1]: 1}] = xave[1]

    if last:
        out = out.transpose(..., *mom_dims)
    return out


def xcentral_moments(
    x: _T_XRVALS,
    mom: T_MOM,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    last: bool = True,
    mom_dims: T_MOM_DIMS | None = None,
    broadcast: bool = False,
) -> xr.DataArray:
    """Calculate central mom along axis.

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

    kws = dict(vals=x, mom=mom, w=w, axis=axis, dim=dim, last=last, mom_dims=mom_dims)
    if len(mom) == 1:
        out = _xcentral_moments(**kws)  # type: ignore
    else:
        kws["broadcast"] = broadcast
        out = _xcentral_comoments(**kws)  # type: ignore

    return cast(xr.DataArray, out)


@no_type_check
def _attributes_from_xr(
    da: xr.DataArray | np.ndarray,
    dim: Hashable | None = None,
    mom_dims: T_MOM_DIMS | None = None,
    **kws,
) -> Dict[str, Any]:
    if isinstance(da, xr.DataArray):
        if dim is not None:
            # reduce along this dim
            da = da.isel({dim: 0}, drop=True)
        out = {k: getattr(da, k) if v is None else v for k, v in kws.items()}
    else:
        out = kws.copy()

    out["mom_dims"] = mom_dims
    return out


@no_type_check
def _check_xr_input(
    x: xr.DataArray | np.ndarray,
    axis: int | str | None = None,
    dim: Hashable | None = None,
    mom_dims: T_MOM_DIMS | None = None,
    _kws_in: Optional[Dict[Any, Any]] = None,
    **kws,
) -> Any:
    if isinstance(x, xr.DataArray):
        # MIGRATION DIM
        # axis, dim = _select_axis_dim(x.dims, axis, dim)
        if axis is None and dim is None:
            pass
        else:
            axis, dim = _select_axis_dim(x.dims, axis, dim)
        values = x.values
    else:
        if axis is None:
            dim = None
        else:
            dim = axis  # type: ignore
        values = x
    kws = _attributes_from_xr(x, dim=dim, mom_dims=mom_dims, **kws)

    if _kws_in is not None and len(_kws_in) > 0:
        kws = dict(kws, **_kws_in)

    return kws, axis, dim, values


@no_type_check
def _optional_wrap_data(
    data: xr.DataArray | np.ndarray,
    mom_ndim: int,
    template: Any = None,
    dims: Tuple[Hashable, ...] | None = None,
    coords: Mapping | None = None,
    name: Hashable | None = None,
    attrs: Mapping | None = None,
    indexes: Any = None,
    mom_dims: T_MOM_DIMS | None = None,
    dtype: DTypeLike | None = None,
    copy: bool = False,
    copy_kws: Mapping | None = None,
    verify: bool = True,
    # verify_mom_dims=True,
) -> xr.DataArray:
    """Wrap data with xarray."""

    if isinstance(data, xr.DataArray):
        if mom_dims is not None:
            if isinstance(mom_dims, str):
                mom_dims = (mom_dims,)
            else:
                mom_dims = tuple(mom_dims)

    elif template is not None:
        data = template.copy(data=data)

    else:
        # wrap data with DataArray
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
                mom_dims = tuple(f"mom_{i}" for i in range(mom_ndim))
            elif isinstance(mom_dims, str):
                mom_dims = (mom_dims,)
            else:
                mom_dims = tuple(mom_dims)

            dims_total = dims + mom_dims
        else:
            raise ValueError("bad dims {}, moment_dims {}".format(dims, mom_dims))

        # xarray object
        data = xr.DataArray(
            data,
            dims=dims_total,
            coords=coords,
            attrs=attrs,
            name=name,
            # skip this option.  Breaks with some versions of xarray
            # indexes=indexes,
        )

    # if verify_mom_dims:
    #     if data.dims[-mom_ndim:] != mom_dims:
    #         data = data.transpose(*((..., ) + mom_dims))
    if mom_dims is not None:
        if data.dims[-mom_ndim:] != mom_dims:
            raise ValueError(f"last dimensions {data.dims} do not match {mom_dims}")

    if verify:
        vals = np.asarray(data.values, dtype=dtype, order="c")
    else:
        vals = data.values

    if copy:
        if copy_kws is None:
            copy_kws = {}

        if vals is data.values:
            vals = vals.copy(**copy_kws)

        data = data.copy(data=vals)

    elif vals is not data.values:
        # data.values = vals
        # Above leads to overwriting the data object in cases where we are updating things.
        # Instead, create a new object with the correct data
        data = data.copy(data=vals)

    return data


# from .abstract_central import CentralMomentsABC


class xCentralMoments(CentralMoments):
    """Wrap cmomy.CentralMoments with xarray.

    Most methods are wrapped to accept xarray.DataArray object.

    Notes
    -----
    unlike xarray, most methods take only the `axis` parameter
    instead of both an `axis` (for positional) and `dim` (for names)
    parameter for reduction.  If `axis` is a integer, then positional
    reduction is assumed, otherwise named reduction is done.  In the
    case that `dims` have integer values, this will lead to only positional
    reduction.
    """

    __slots__ = "_xdata"

    def __init__(self, data: xr.DataArray, mom_ndim: int = 1) -> None:

        if not isinstance(data, xr.DataArray):
            raise ValueError(
                "data must be a xarray.DataArray. "
                "See xCentralMoments.from_data for wrapping numpy arrays"
            )

        self._xdata = data

        # TODO: data.data or data.values?
        super(xCentralMoments, self).__init__(data=data.data, mom_ndim=mom_ndim)

    @property
    def values(self):
        """Underlying data."""
        return self._xdata

    # xarray attriburtes
    @property
    def attrs(self):
        """Attributes of values."""
        return self._xdata.attrs

    @property
    def dims(self) -> Tuple[Hashable, ...]:
        """Dimensions of values."""
        return self._xdata.dims

    @property
    def coords(self):
        """Coordinates of values."""
        return self._xdata.coords

    @property
    def name(self):
        """Name of values."""
        return self._xdata.name

    @property
    def indexes(self):
        """Indexes of values."""
        return self._xdata.indexes

    @property
    def sizes(self):
        """Sizes of values."""
        return self._xdata.sizes

    ###########################################################################
    # SECTION: top level creation/copy/new
    ###########################################################################
    @gcached()
    def _template_val(self) -> xr.DataArray:
        """Template for values part of data."""
        return self._xdata[self._weight_index]

    # def _wrap_like_template(self, x):
    #     return _wrap_like(self._template_val, x)

    def _wrap_like(self, x) -> xr.DataArray:
        # return _xr_wrap_like(self._xdata, x)
        return self._xdata.copy(data=x)

    @property
    def val_dims(self) -> Tuple[Hashable, ...]:
        """Names of value dimensions."""
        return self.dims[: -self.mom_ndim]

    @property
    def mom_dims(self) -> Tuple[Hashable, ...]:
        """Names of moment dimensions."""
        return self.dims[-self.mom_ndim :]

    @property
    def _one_like_val(self):  # -> xr.DataArray:
        return xr.ones_like(self._template_val)

    @property
    def _zeros_like_val(self):  # -> xr.DataArray:
        return xr.zeros_like(self._template_val)

    @gcached()
    def _array_view(self):
        """Create CentralMoments view."""
        return self.to_centralmoments()

    @no_type_check
    def new_like(
        self: T_CENTRALMOMENTS,
        data: np.ndarray | xr.DataArray | None = None,
        copy: bool = False,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
        strict: bool = False,
        **kws,
    ) -> T_CENTRALMOMENTS:  # type: ignore
        """Create new object like self, with new data.

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
        **kws : dict
            extra arguments to self.from_data
        """

        if data is None:
            data = xr.zeros_like(self._xdata)
            copy = verify = check_shape = False

        kws.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kws = {
                "mom": self.mom,
                "val_shape": self.val_shape,
                "dtype": self.dtype,
                **kws,
            }

        if isinstance(data, np.ndarray):
            kws.setdefault("template", self._xdata)

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            check_shape=check_shape,
            **kws,
        )

    @no_type_check
    @classmethod
    def zeros(
        cls: Type[T_CENTRALMOMENTS],
        mom: T_MOM | None = None,
        val_shape: Tuple[int, ...] | None = None,
        mom_ndim: int | None = None,
        shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping | None = None,
        dims: Tuple[Hashable, ...] | None = None,
        coords: Mapping | None = None,
        attrs: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any | None = None,
        mom_dims: T_MOM_DIMS | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create a new base object.

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

        Returns
        -------
        object : instance of class `cls`

        Notes
        -----
        the resulting total shape of data is shape + mom_shape
        """

        return CentralMoments.zeros(
            mom=mom,
            val_shape=val_shape,
            mom_ndim=mom_ndim,
            shape=shape,
            dtype=dtype,
            zeros_kws=zeros_kws,
        ).to_xcentralmoments(
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
        )

    ###########################################################################
    # xarray specific methods
    ###########################################################################
    def _wrap_xarray_method(self, _method, *args, **kwargs):
        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        return self.new_like(data=xdata, strict=False)

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign coordinates to data and return new object."""
        return self._wrap_xarray_method("assign_coords", coords=coords, **coords_kwargs)

    def assign_attrs(self, *args, **kwargs):
        """Assign attributes to data and return new object."""
        return self._wrap_xarray_method("assign_attrs", *args, **kwargs)

    def rename(self, new_name_or_name_dict=None, **names):
        """Rename object."""
        return self._wrap_xarray_method(
            "rename", new_name_or_name_dict=new_name_or_name_dict, **names
        )

    def _wrap_xarray_method_from_data(
        self,
        _method: str,
        _data_copy: bool = False,
        _data_order: bool = False,
        _data_kws: Mapping | None = None,
        _data_verify: bool = False,
        *args,
        **kwargs,
    ) -> "xCentralMoments":

        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        if _data_order:
            xdata = xdata.transpose(..., *self.mom_dims)
        if _data_kws is None:
            _data_kws = {}
        else:
            _data_kws = dict(_data_kws)
        _data_kws.setdefault("copy", _data_copy)
        _data_kws.setdefault("copy", _data_copy)

        out = type(self).from_data(
            data=xdata, mom_ndim=self.mom_ndim, verify=_data_verify, **_data_kws
        )

        return cast("xCentralMoments", out)

    def stack(
        self,
        dimensions: Mapping[Any, Sequence[Hashable]] | None = None,
        _data_copy: bool = False,
        _data_kws: Mapping | None = None,
        _data_order: bool = True,
        _data_verify: bool = False,
        **dimensions_kwargs,
    ) -> "xCentralMoments":
        """Stack dimensions.

        See xarray.DataArray.stack
        """
        return self._wrap_xarray_method_from_data(
            "stack",
            dimensions=dimensions,
            _data_copy=_data_copy,
            _data_kws=_data_kws,
            _data_order=_data_order,
            _data_verify=_data_verify,
            **dimensions_kwargs,
        )

    def unstack(
        self,
        dim: Hashable | Sequence[Hashable] | None = None,
        fill_value: Any = np.nan,
        sparse: bool = False,
        _data_copy=False,
        _data_kws=None,
        _data_order=True,
        _data_verify=False,
    ) -> "xCentralMoments":
        """Unstack dimensions.

        See xarray.DataArray.unstack
        """
        return self._wrap_xarray_method_from_data(
            "unstack",
            _data_copy=_data_copy,
            _data_kws=_data_kws,
            _data_order=_data_order,
            _data_verify=_data_verify,
            dim=dim,
            fill_value=fill_value,
            sparse=sparse,
        )

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        _data_kws=None,
        _data_copy=False,
        _data_order=False,
        _data_verify=False,
        **indexers_kws,
    ) -> "xCentralMoments":
        """Select subset of data.

        See xarray.DataArray.sel
        """
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
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        _data_kws: None = None,
        _data_copy: bool = False,
        _data_order: bool = False,
        _data_verify: bool = False,
        **indexers_kws,
    ) -> "xCentralMoments":
        """Select subset of data by position.

        See xarray.DataArray.isel
        """
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
    @no_type_check
    def _xverify_value(
        self,
        x: Union[xr.DataArray, float],
        target: xr.DataArray | str | None = None,
        dim: Hashable | None = None,
        axis: int | str | None = None,
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Optional[Any] = None,
    ) -> Union[
        Tuple[np.ndarray, xr.DataArray], np.ndarray, Tuple[float, xr.DataArray], float
    ]:

        if isinstance(target, str):

            # if dim is not None:
            #     if isinstance(dim, int):
            #         dim = x.dims[dim]

            if dim is not None and axis is not None:
                axis = None

            if dim is not None or axis is not None:
                axis, dim = _select_axis_dim(x.dims, axis, dim)

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

            if dim is not None and axis is not None:
                axis = None

            if dim is not None or axis is not None:
                dim = target_dims[0]

                # axis_a, dim_a = _select_axis_dim(target_dims, axis, dim)
                # assert dim_a == target_dims[0], f'Error, {axis}, {dim}, {target_dims} {x.dims}'
                # axis, dim = axis_a, dim_a

            # if dim is not None:
            #     if isinstance(dim, int):
            #         # this is a bit awkward, but
            #         # should work
            #         # assume we already have target in correct order
            #         dim = target_dims[0]

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

    @no_type_check
    def _verify_value(
        self,
        x: float | np.ndarray | xr.DataArray,
        target: str | np.ndarray | xr.DataArray = None,
        axis: int | str | None = None,
        dim: Hashable | None = None,
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Tuple[int, ...] | None = None,
    ) -> Any:
        if isinstance(x, xr.DataArray) or isinstance(target, xr.DataArray):

            return self._xverify_value(
                x,
                target=target,
                axis=axis,
                dim=dim,
                # dim=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
            )

        else:
            assert axis is None or isinstance(
                axis, int
            ), f"Error with axis value {axis}"

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

    def mean(self, dim_combined="variable", coords_combined=None):
        """Return mean/first moment(s) of data."""
        return self._single_index_selector(
            val=1, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def var(self, dim_combined="variable", coords_combined=None):
        """Return variance (second central moment) of data."""
        return self._single_index_selector(
            val=2, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def cmom(self):  # -> xr.DataArray:
        """Return central moments.

        Structure is cmom[i, j] = <(x-<x>)**i (y-<y>)**u>.

        Note that is strict, so, `cmom[1, 0] = cmom[0, 1] = 0` and
        `cmom[0, 0] = 1`
        """
        return self._wrap_like(super(xCentralMoments, self).cmom())

    def rmom(self):  # -> xr.DataArray:
        """Return raw moments.

        Structure is rmom[i, j] = <x**i y**j>.
        """
        return self._wrap_like(super(xCentralMoments, self).rmom())

    @no_type_check
    def resample_and_reduce(
        self,
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: int | None = None,
        axis: int | str | None = None,
        dim: Hashable | None = None,
        rep_dim: str = "rep",
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        full_output: bool = False,
        **kws,
    ) -> "xCentralMoments":
        """Bootstrap resample and reduce.

        Parameters
        ----------
        rep_dim : str, default='rep'
            dimension name for resampled
            if 'dims' is not passed in kwargs, then reset dims
            with replicate dimension having name 'rep_dim',
            and all other dimensions have the same names as
            the parent object
        """
        assert not (axis is None and dim is None)

        kws, axis, *_ = _check_xr_input(
            self._xdata,
            axis=axis,
            dim=dim,
            mom_dims=None,
            dims=None,
            attrs=None,
            coords=None,
            indexes=None,
            name=None,
            _kws_in=kws,
        )

        axis = self._wrap_axis(axis)

        # new dims after resample
        kws["dims"] = [rep_dim] + list(kws["dims"])

        return super(xCentralMoments, self).resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            parallel=parallel,
            resample_kws=resample_kws,
            full_output=full_output,
            **kws,
        )

    @no_type_check
    def reduce(
        self: T_CENTRALMOMENTS,
        axis: int | str | None = None,
        dim: Hashable | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create new object reducealong axis."""
        self._raise_if_scalar()
        axis, dim = _select_axis_dim(self.dims, axis, dim)
        axis = self._wrap_axis(axis)
        return type(self).from_datas(
            self.values, mom_ndim=self.mom_ndim, axis=axis, **kws
        )

    @no_type_check
    def block(
        self,
        block_size: int,
        axis: int | str | None = None,
        dim: Hashable | None = None,
        coords_policy: Literal["first", "last", None] = "first",
        **kws,
    ) -> "xCentralMoments":
        """Block average along an axis.

        Parameters
        ----------
        block_size : int
            size of blocks to average over
        axis : str or int, default=0
            axis/dimension to block average over
        args : tuple
            positional arguments to CentralMomentsBase.block
        coords_policy : {'first','last',None}
            Policy for handling coordinates along `axis`.
            If not coordinates do nothing.  Otherwise use:

            * 'first': select first value of coordinate for each block
            * 'last': select last value of coordate for each block
            * None: drop any coordates

        kws : dict
            key-word arguments to CentralMomentsBase.block
        """

        self._raise_if_scalar()
        axis, dim = _select_axis_dim(self.dims, axis, dim, default_axis=0)
        axis = self._wrap_axis(axis)
        # dim = self.dims[axis]

        check_kws = dict(
            mom_dims=None, dims=None, attrs=None, coords=None, indexes=None
        )
        if coords_policy in ["first", "last"]:
            if block_size is None:
                block_size = self.sizes[dim]
                nblock = 1
            else:
                nblock = self.sizes[dim] // block_size

            if coords_policy == "first":
                start = 0
            else:
                start = block_size - 1

            data = self.values.isel(
                **{dim: slice(start, block_size * nblock, block_size)}
            ).transpose(dim, ...)
            kws_default, *_ = _check_xr_input(data, axis=None, **check_kws)

        else:
            kws_default, *_ = _check_xr_input(self.values, axis=axis, **check_kws)
            kws_default["dims"] = ["dim"] + list(kws_default["dims"])

        kws = dict(kws_default, **kws)

        return super(xCentralMoments, self).block(
            block_size=block_size, axis=axis, **kws
        )

    @no_type_check
    def _wrap_axis(
        self, axis: int | str, default: int = 0, ndim: None = None, **kws
    ) -> int:
        # if isinstance(axis, str):
        #     axis = self._xdata.get_axis_num(axis)
        # return super(xCentralMoments, self)._wrap_axis(
        #     axis=axis, default=default, ndim=ndim
        # )

        if isinstance(axis, str):
            raise ValueError("shouldnt get string axis here")
            # return self._xdata.get_axis_num(axis)
        else:
            return super(xCentralMoments, self)._wrap_axis(
                axis=axis, default=default, ndim=ndim
            )

    def to_centralmoments(self):
        """Create a CentralMoments object from xCentralMoments."""
        return CentralMoments(data=self.data, mom_ndim=self.mom_ndim)

    @classmethod
    def from_centralmoments(
        cls,
        obj: "CentralMoments",
        dims: Hashable | Sequence[Hashable] | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: Hashable | Tuple[Hashable, Hashable] | None = None,
    ) -> "xCentralMoments":
        """Create and xCentralMoments object from CentralMoments."""

        data = obj.to_xarray(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
        )
        return cls(data=data, mom_ndim=obj.mom_ndim)

    @classmethod
    @no_type_check
    def from_data(
        cls,
        data: np.ndarray | xr.DataArray,
        mom: T_MOM | None = None,
        mom_ndim: int | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        template: Any | None = None,
        dims: T_DIMS | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any = None,
        name: Hashable | None = None,
        mom_dims: T_MOM_DIMS | None = None,
        # verify_mom_dims=True,
    ) -> "xCentralMoments":
        """Object from data array.

        Parameters
        ----------
        dims : tuple, optional
            dimension names for resulting object
        """

        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        data_verified = _optional_wrap_data(
            data=data,
            mom_ndim=mom_ndim,
            template=template,
            dims=dims,
            coords=coords,
            name=name,
            attrs=attrs,
            indexes=indexes,
            mom_dims=mom_dims,
            dtype=dtype,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            # verify_mom_dims=verify_mom_dims
        )

        if check_shape:
            if val_shape is None:
                val_shape = data_verified.shape[:-mom_ndim]
            mom = cls._check_mom(mom, mom_ndim, data_verified.shape)

            if data_verified.shape != val_shape + tuple(x + 1 for x in mom):
                raise ValueError(
                    f"{data.shape} does not conform to {val_shape} and {mom}"
                )
        return cls(data=data_verified, mom_ndim=mom_ndim)

    @classmethod
    @no_type_check
    def from_datas(
        cls,
        datas: np.ndarray | xr.DataArray,
        mom: T_MOM | None = None,
        mom_ndim: int | None = None,
        axis: int | str | None = None,
        dim: Hashable | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dims: T_DIMS | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: T_MOM_DIMS | None = None,
        **kws,
    ) -> "xCentralMoments":
        """Create object from multiple datas.

        Parameters
        ----------
        dims : tuple, optional
            dimension names.
            Note that this does not include the dimension reduced over.
        """

        if axis is None and dim is None:
            raise ValueError("must specify axis or dim")

        kws, axis, dim, values = _check_xr_input(
            datas,
            axis=axis,
            dim=dim,
            _kws_in=kws,
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

    @no_type_check
    def to_raw(self) -> xr.DataArray:
        """Convert underlying central data to raw data.

        See `CentralMoments.to_raw`
        """
        return self._wrap_like(super(xCentralMoments, self).to_raw())

    @no_type_check
    @classmethod
    def from_raw(
        cls,
        raw: np.ndarray | xr.DataArray,
        mom: T_MOM | None = None,
        mom_ndim: int | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        dims: T_DIMS | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: T_MOM_DIMS | None = None,
        **kws,
    ) -> "xCentralMoments":
        """Create object from raw moment data.

        Parameters
        ----------
        dims : tuple, optional
            dimension names

        See Also
        --------
        CentralMoments.from_raw
        """
        kws, _, values = _check_xr_input(
            raw,
            axis=None,
            _kws_in=kws,
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
            convert_kws=convert_kws,
            **kws,
        )

    @no_type_check
    @classmethod
    def from_raws(
        cls,
        raws: np.ndarray | xr.DataArray,
        mom: T_MOM | None = None,
        mom_ndim: int | None = None,
        axis: int | str | None = None,
        dim: Hashable | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        dims: T_DIMS | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Mapping | None = None,
        name: Hashable | None = None,
        mom_dims: T_MOM_DIMS = None,
        **kws,
    ):
        """Create object from multiple raw values.

        See `CentralMoments.from_raw`
        """

        assert not (axis is None and dim is None)

        kws, axis, dim, values = _check_xr_input(
            raws,
            axis=axis,
            dim=dim,
            _kws_in=kws,
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
            convert_kws=convert_kws,
            **kws,
        )

    @no_type_check
    @classmethod
    def from_vals(
        cls,
        x: np.ndarray
        | Tuple[np.ndarray, np.ndarray]
        | xr.DataArray
        | Tuple[xr.DataArray, xr.DataArray],
        w: float | np.ndarray | xr.Datarray | None = None,
        axis: str | int | None = None,
        dim: Hashable | None = None,
        mom: T_MOM | None = 2,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        dims: T_DIMS | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: T_MOM_DIMS | None = None,
        **kws,
    ) -> "xCentralMoments":
        """Create object from values.

        See CentralMomentsfrom_vals
        """

        # specify dim or axis
        assert not (axis is None and dim is None)

        mom_ndim = cls._mom_ndim_from_mom(mom)
        x0 = x if mom_ndim == 1 else x[0]
        kws, axis, dim, values = _check_xr_input(
            x0,
            axis=axis,
            dim=dim,
            _kws_in=kws,
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
            dim=dim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            broadcast=broadcast,
            **kws,
        )

    @no_type_check
    @classmethod
    def from_resample_vals(
        cls,
        x: np.ndarray
        | Tuple[np.ndarray, np.ndarray]
        | xr.DataArray
        | Tuple[xr.DataArray, xr.DataArray],
        freq=None,
        indices=None,
        nrep=None,
        w=None,
        axis=None,
        dim=None,
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
        full_output=False,
        **kws,
    ):
        """Create from resampling values.

        See CentralMomentsfrom_values
        """

        assert not (axis is None and dim is None)

        mom_ndim = cls._mom_ndim_from_mom(mom)
        if mom_ndim == 1:
            y = None
        else:
            x, y = x

        kws, axis, dim, values = _check_xr_input(
            x,
            axis=axis,
            dim=dim,
            _kws_in=kws,
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
            full_output=full_output,
            **kws,
        )

    @no_type_check
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
        **kws,
    ):
        """Create from single observation of statisitcs (mean, variance).

        See CentralMoment.from_stat
        """

        kws, *_ = _check_xr_input(
            a,
            axis=None,
            _kws_in=kws,
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

    @no_type_check
    @classmethod
    def from_stats(
        cls,
        a,
        v=0.0,
        w=None,
        axis=None,
        dim=None,
        mom=2,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        mom_dims=None,
        **kws,
    ):
        """Create from collection of statisitcs.

        See CentralMoments.from_stats
        """

        assert not (axis is None and dim is None)

        kws, axis, dim, values = _check_xr_input(
            a,
            axis=axis,
            dim=dim,
            mom_dims=mom_dims,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            _kws_in=kws,
        )

        return super(xCentralMoments, cls).from_stats(
            a=a,
            v=v,
            w=w,
            axis=axis,
            dim=dim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    @no_type_check
    def transpose(
        self, *dims, transpose_coords=None, copy=False, **kws
    ) -> "xCentralMoments":
        """Transpose dimensions of data.

        See xarray.DataArray.transpose
        """
        # make sure dims are last
        dims = list(dims)  # type: ignore
        for k in self.mom_dims:
            if k in dims:
                dims.pop(dims.index(k))
        dims = tuple(dims) + self.mom_dims  # type: ignore

        values = (
            self.values.transpose(*dims, transpose_coords=transpose_coords)
            # make sure mom are last
            # .transpose(...,*self.mom_dims)
        )
        return type(self).from_data(values, mom_ndim=self.mom_ndim, copy=copy, **kws)

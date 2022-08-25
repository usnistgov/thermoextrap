"""Central moments/comoments routines."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
)

import numpy as np
import xarray as xr
from numpy.core.numeric import normalize_axis_index  # type: ignore
from numpy.typing import ArrayLike, DTypeLike

from . import convert
from ._typing import ASARRAY_ORDER, T_CENTRALMOMENTS, T_MOM
from .resample import randsamp_freq, resample_vals
from .utils import _axis_expand_broadcast  # _cached_ones,; _my_broadcast,
from .utils import _shape_insert_axis, _shape_reduce

if TYPE_CHECKING:
    from .xcentral import xCentralMoments


###############################################################################
# central mom/comoments routines
###############################################################################
def _central_moments(
    vals: ArrayLike,
    mom: int | Tuple[int],
    w: np.ndarray | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ASARRAY_ORDER | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate central mom along axis."""

    if isinstance(mom, tuple):
        mom = mom[0]

    x = np.asarray(vals, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    if w is None:
        w = np.ones_like(x)
    else:
        w = _axis_expand_broadcast(
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = (mom + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try rolling
            out = np.moveaxis(out, -1, 0)
        assert out.shape == shape

    wsum = w.sum(axis=0)
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", w, x) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, mom + 1).reshape(*shape)

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", w, dx) * wsum_inv

    if last:
        out = np.moveaxis(out, 0, -1)
    return out


def _central_comoments(
    vals: Tuple[np.ndarray, np.ndarray],
    mom: int | Sequence[int],
    w: Optional[np.ndarray] = None,
    axis: int = 0,
    last: bool = True,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ASARRAY_ORDER | None = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate central co-mom (covariance, etc) along axis."""

    if isinstance(mom, int):
        mom = (mom,) * 2

    mom = tuple(mom)
    assert len(mom) == 2

    # change x to tuple of inputs
    assert isinstance(vals, tuple) and len(vals) == 2
    x, y = vals

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    y = _axis_expand_broadcast(
        y,
        x.shape,
        axis,
        roll=False,
        broadcast=broadcast,
        expand=broadcast,
        dtype=dtype,
        order=order,
    )

    if w is None:
        w = np.ones_like(x)
    else:
        w = _axis_expand_broadcast(
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    assert w.shape == x.shape
    assert y.shape == x.shape

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        y = np.moveaxis(y, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = tuple(x + 1 for x in mom) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try moving axis
            out = np.moveaxis(out, [-2, -1], [0, 1])
        assert out.shape == shape

    wsum = w.sum(axis=0)
    wsum_inv = 1.0 / wsum

    xave = np.einsum("r...,r...->...", w, x) * wsum_inv
    yave = np.einsum("r...,r...->...", w, y) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p0 = np.arange(0, mom[0] + 1).reshape(*shape)
    p1 = np.arange(0, mom[1] + 1).reshape(*shape)

    dx = (x[None, ...] - xave) ** p0
    dy = (y[None, ...] - yave) ** p1

    out[...] = np.einsum("r...,ir...,jr...->ij...", w, dx, dy) * wsum_inv

    out[0, 0, ...] = wsum
    out[1, 0, ...] = xave
    out[0, 1, ...] = yave

    if last:
        out = np.moveaxis(out, [0, 1], [-2, -1])
    return out


def central_moments(
    x: np.ndarray | Tuple[np.ndarray, np.ndarray],
    mom: T_MOM,
    w: np.ndarray | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ASARRAY_ORDER | None = None,
    out: np.ndarray | None = None,
    broadcast: bool = False,
) -> np.ndarray:
    """Calculate central moments or comoments along axis.

    Parameters
    ----------
    vals : array-like or tuple of array-like
        if calculating moments, then this is the input array.
        if calculating comoments, then pass in tuple of values of form (x, y)
    mom : int or tuple
        number of moments to calculate.  If tuple, then this specifies that
        comoments will be calculated.
    w : array-like, optional
        Weights. If passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    axis : int, default=0
        axis to reduce along
    last : bool, aaefault=True
        if True, put mom as last dimension.
        Otherwise, mom will be in first dimension
    dtype, order : options to np.asarray
    broadcast : bool, default=False
        if True and calculating comoments, then the x[1] will be broadcast against x[0].
    out : array
        if present, use this for output data
        Needs to have shape of either (mom,) + shape or shape + (mom,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape=shape + mom_shape or mom_shape + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:], and `mom_shape` is the shape of
        the moment part, either (mom+1,) or (mom0+1, mom1+1).  Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment.
    """

    if isinstance(mom, int):
        mom = (mom,)

    kws = dict(
        vals=x, mom=mom, w=w, axis=axis, last=last, dtype=dtype, order=order, out=out
    )
    if len(mom) == 1:
        return _central_moments(**kws)  # type: ignore
    else:
        kws["broadcast"] = broadcast
        return _central_comoments(**kws)  # type: ignore


from .abstract_central import CentralMomentsABC


###############################################################################
# Classes
###############################################################################
class CentralMoments(CentralMomentsABC[np.ndarray]):
    """Base class for moments accumulation.

    Parameters
    ----------
    data : numpy data array
        data should have the shape `val_shape + mom_shape`
        where `val_shape` is the shape of a single observation,
        and mom_shape is the shape of the moments
    mom_ndim : int, {1, 2}
        number of dimensions of moments.
        * 1 : central moments of single variable
        * 2 : central comoments of two variables
    kws : dict
        optional arguments to be used in subclasses
    """

    __slots__ = (
        "_mom_ndim",
        "_cache",
        "_data",
        "_data_flat",
    )

    def __init__(self, data: np.ndarray, mom_ndim: int = 1) -> None:
        if mom_ndim not in (1, 2):
            raise ValueError(
                "mom_ndim must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )
        self._mom_ndim = mom_ndim

        if data.ndim < self.mom_ndim:
            raise ValueError("not enough dimensions in data")

        self._data = data
        self._data_flat = self._data.reshape(self.shape_flat)

        if any(m <= 0 for m in self.mom):
            raise ValueError("moments must be positive")

    @property
    def values(self) -> np.ndarray:
        """Access underlying central moments data."""
        return self._data

    @property
    def data(self) -> np.ndarray:
        """Accessor to numpy underlying data.

        By convention data has the following meaning for the moments indexes

        * `data[...,i=0,j=0]`, weights
        * `data[...,i=1,j=0]]`, if only one moment indice is one and all
        others zero, then this is the average value of the variable with unit index.

        * all other cases, the central moments `<(x0-<x0>)**i0 * (x1 - <x1>)**i1 * ...>`
        """
        return self._data

    ###########################################################################
    # SECTION: top level creation/copy/new
    ###########################################################################
    def new_like(
        self: T_CENTRALMOMENTS,
        data: np.ndarray | None = None,
        copy: bool = False,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        strict: bool = False,
        **kws,
    ) -> T_CENTRALMOMENTS:
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
        copy_kws : dict, optional
            key-word arguments to `self.data.copy`
        **kws : extra arguments
            arguments to classmethod `from_data`
        """

        if data is None:
            data = np.zeros_like(self._data, order="C")
            copy = verify = check_shape = False

        kws.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kws = dict(
                dict(
                    mom=self.mom,
                    val_shape=self.val_shape,
                    dtype=self.dtype,
                ),
                **kws,
            )

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            check_shape=check_shape,
            **kws,
        )

    @classmethod
    def zeros(
        cls: Type[T_CENTRALMOMENTS],
        mom: T_MOM | None = None,
        val_shape: Tuple[int, ...] | None = None,
        mom_ndim: int | None = None,
        shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create a new base object.

        Parameters
        ----------
        mom : int or tuple
            moments.
            if integer, or length one tuple, then moments of single variable.
            if tuple of length 2, then comoments of two variables.
        val_shape : tuple, optional
            shape of values, excluding moments.  For example, if considering the average
            of observations `x`, then `val_shape = x.shape`
            if not passed, then assume val_shape = ()
        shape : tuple, optional
            if passed, create object with this total shape
        mom_ndim : int {1, 2}, optional
            number of variables.
            if pass `shape`, then must pass mom_ndim
        dtype : nunpy dtype, default=float

        **kws : dict
            extra arguments to cls.from_data

        Returns
        -------
        object : instance of class `cls`

        Notes
        -----
        the resulting total shape of data is shape + (mom + 1)
        """

        if shape is None:
            assert mom is not None
            if isinstance(mom, int):
                mom = (mom,)
            if mom_ndim is None:
                mom_ndim = len(mom)
            assert len(mom) == mom_ndim

            if val_shape is None:
                val_shape = ()
            elif isinstance(val_shape, int):
                val_shape = (val_shape,)
            shape = val_shape + tuple(x + 1 for x in mom)

        else:
            assert mom_ndim is not None

        if dtype is None:
            dtype = float

        if zeros_kws is None:
            zeros_kws = {}
        data = np.zeros(shape=shape, dtype=dtype, **zeros_kws)

        kws = dict(kws, verify=False, copy=False, check_shape=False)
        return cls.from_data(data=data, mom_ndim=mom_ndim, **kws)

    ###########################################################################
    # SECTION: Access to underlying statistics
    ###########################################################################
    def to_xarray(
        self,
        dims: Hashable | Sequence[Hashable] | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: Hashable | Tuple[Hashable, Hashable] | None = None,
    ) -> xr.DataArray:
        """Create an xr.DataArray representation of underlying data.

        Parameters
        ----------
        dims : Hashable or Sequence of Hashables
            Dimensions of resulting DataArray.

            * If len(dims) == self.ndim, then dims specifies all dimensions.
            * If len(dims) == self.val_ndim, dims = dims + mom_dims
        attrs : Mapping
            Attributes of output
        coords : Mapping
            Coordinates of output
        name : Hashable
            Name of output
        index : Any
        mom_dims : Hashable or Sequence of Hashable
            Name of mom_dims.  Defaults to (mom_0, [mom_1])

        Results
        --------
        output : xr.DataArray
        """

        if dims is None:
            dims = tuple(f"dim_{i}" for i in range(self.val_ndim))
        elif isinstance(dims, str):
            dims = (dims,)
        dims = tuple(dims)  # type: ignore

        if len(dims) == self.ndim:
            dims_output = dims  # type: ignore

        elif len(dims) == self.val_ndim:
            if mom_dims is None:
                mom_dims = tuple(f"mom_{i}" for i in range(self.mom_ndim))
            elif isinstance(mom_dims, Hashable):
                mom_dims = (mom_dims,)
            mom_dims = tuple(mom_dims)

            assert len(mom_dims) == self.mom_ndim

            dims_output = dims + mom_dims

        else:
            raise ValueError(
                f"Problem with {dims}, {mom_dims}.  Total length should be {self.ndim}"
            )
        return xr.DataArray(
            self.data, dims=dims_output, coords=coords, attrs=attrs, name=name
        )

    def to_xcentralmoments(
        self,
        dims: Hashable | Sequence[Hashable] | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: Hashable | Tuple[Hashable, Hashable] | None = None,
    ) -> xCentralMoments:
        """Create an xr.DataArray representation of underlying data.

        Parameters
        ----------
        dims : Hashable or Sequence of Hashables
            Dimensions of resulting DataArray.

            * If len(dims) == self.ndim, then dims specifies all dimensions.
            * If len(dims) == self.val_ndim, dims = dims + mom_dims
        attrs : Mapping
            Attributes of output
        coords : Mapping
            Coordinates of output
        name : Hashable
            Name of output
        index : Any
        mom_dims : Hashable or Sequence of Hashable
            Name of mom_dims.  Defaults to (mom_0, [mom_1])

        Results
        --------
        output : xCentralMoments
        """
        from .xcentral import xCentralMoments

        data = self.to_xarray(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
        )
        return xCentralMoments(data=data, mom_ndim=self.mom_ndim)

    ###########################################################################
    # SECTION: pushing routines
    ###########################################################################
    #  -> np.ndarray | float | Tuple[float|np.ndarray, None|float|np.ndarray] :
    def _verify_value(
        self,
        x: np.ndarray | float,
        target: np.ndarray | Tuple[int, ...] | str | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,  # included here for consistency
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Tuple[int, ...] | None = None,
        other: np.ndarray | None = None,
        **kwargs,
    ):  # type: ignore
        """Verify input values.

        Parameters
        ----------
        x : array
        target : tuple or array
            If tuple, this is the target shape to be used to Make target.
            If array, this is the target array
        Optinal target that has already been rolled.  If this is passed, and
        x will be broadcast/expanded, can expand to this shape without the need
        to reorder,
        """

        x = np.asarray(x, dtype=self.dtype)

        if isinstance(target, str):
            if target == "val":
                target_shape = self.val_shape
            elif target == "vals":
                target_shape = _shape_insert_axis(self.val_shape, axis, x.shape[axis])  # type: ignore
            elif target == "data":
                target_shape = self.shape
            elif target == "datas":
                # make sure axis in limits
                axis = normalize_axis_index(axis, self.val_ndim + 1)
                # if axis < 0:
                #     axis += self.ndim - self.mom_ndim
                target_shape = _shape_insert_axis(self.shape, axis, x.shape[axis])  # type: ignore
            elif target == "var":
                target_shape = self.shape_var
            elif target == "vars":
                assert other is not None
                target_shape = _shape_insert_axis(self.shape_var, axis, other.shape[axis])  # type: ignore
            else:
                raise ValueError(f"unknown string target name {target}")

            target_output = x

        elif isinstance(target, tuple):
            target_shape = target
            target_output = x

        elif isinstance(target, np.ndarray):
            target_shape = target.shape
            target_output = None

        else:
            raise ValueError("unknown target type")

        x = _axis_expand_broadcast(
            x,
            target_shape,
            axis,
            verify=False,
            expand=expand,
            broadcast=broadcast,
            dtype=self.dtype,
            roll=False,
        )

        # check shape:
        assert (
            x.shape == target_shape
        ), f"x.shape = {x.shape} not equal target_shape={target_shape}"

        # move axis
        if axis is not None:
            if axis != 0:
                x = np.moveaxis(x, axis, 0)
            nrec = (x.shape[0],)  # type: ignore
        else:
            nrec = ()  # type: ignore

        if shape_flat is not None:
            x = x.reshape(nrec + shape_flat)

        if x.ndim == 0:
            x = x[()]

        if target_output is None:
            return x
        else:
            return x, target_output

    def push_data(self: T_CENTRALMOMENTS, data: np.ndarray) -> T_CENTRALMOMENTS:
        """Push data object to moments.

        Parameters
        ----------
        data : array-like, `shape=self.shape`
            array storing moment information
        Returns
        -------
        self

        See Also
        --------
        cmomy.CentralMoments.data
        """
        data = self._check_data(data)
        self._push.data(self._data_flat, data)
        return self

    def push_datas(
        self: T_CENTRALMOMENTS,
        datas: np.ndarray,
        axis: int = 0,
    ) -> T_CENTRALMOMENTS:
        """Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like
            this should have shape like `(nrec,) + self.shape`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        axis : int, default=0
            axis to reduce along
        dim : Hashable, optional
            For use in xCentralMoments only.

        Returns
        -------
        self
        """
        datas = self._check_datas(datas, axis=axis)
        self._push.datas(self._data_flat, datas)
        return self

    def push_val(
        self: T_CENTRALMOMENTS,
        x: float | np.ndarray | Tuple[float, float] | Tuple[np.ndarray, np.ndarray],
        w: np.ndarray | float | None = None,
        broadcast: bool = False,
    ) -> T_CENTRALMOMENTS:
        """Push single sample to central moments.

        Parameters
        ----------
        x : array-like or tuple of arrays
            if `self.mom_ndim == 1`, then this is the value to consider
            if `self.mom_ndim == 2`, then `x = (x0, x1)`
            `x.shape == self.val_shape`

        w : int, float, array-like, optional
            optional weight of each sample
        broadcast : bool, default = False
            If true, do smart broadcasting for `x[1:]`

        Returns
        -------
        self
        """

        if self.mom_ndim == 1:
            ys = ()
        else:
            assert isinstance(x, tuple) and len(x) == self.mom_ndim
            x, *ys = x  # type: ignore

        xr, target = self._check_val(x, "val")  # type: ignore
        yr = tuple(self._check_val(y, target=target, broadcast=broadcast) for y in ys)  # type: ignore
        wr = self._check_weight(w, target)  # type: ignore
        self._push.val(self._data_flat, *((wr, xr) + yr))
        return self

    def push_vals(
        self: T_CENTRALMOMENTS,
        x: np.ndarray | Tuple[np.ndarray, np.ndarray],
        w: np.ndarray | None = None,
        axis: int = 0,
        broadcast: bool = False,
    ) -> T_CENTRALMOMENTS:
        """Push multiple samples to central moments.

        Parameters
        ----------
        x : array-like or tuple of arrays
            if `self.mom_ndim` == 1, then this is the value to consider
            if `self.mom_ndim` == 2, then x = (x0, x1)
            `x.shape[:axis] + x.shape[axis+1:] == self.val_shape`

        w : int, float, array-like, optional
            optional weight of each sample
        axis : int, default=0
            axis to reduce along
        broadcast : bool, default = False
            If true, do smart broadcasting for `x[1:]`
        """
        if self.mom_ndim == 1:
            ys = ()
        else:
            assert len(x) == self.mom_ndim
            x, *ys = x  # type: ignore

        xr, target = self._check_vals(x, axis=axis, target="vals")  # type: ignore
        yr = tuple(  # type: ignore
            self._check_vals(y, target=target, axis=axis, broadcast=broadcast)  # type: ignore
            for y in ys  # type: ignore
        )  # type: ignore
        wr = self._check_weights(w, target=target, axis=axis)
        self._push.vals(self._data_flat, *((wr, xr) + yr))
        return self

    ###########################################################################
    # SECTION: Constructors
    ###########################################################################
    @classmethod
    def from_data(
        cls: Type[T_CENTRALMOMENTS],
        data: np.ndarray,
        mom_ndim: int | None = None,
        mom: T_MOM | None = None,
        val_shape: Tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
        # **kws,
    ) -> T_CENTRALMOMENTS:
        """Create new object from `data` array with additional checks.

        Parameters
        ----------
        data : np.np.ndarray
            shape should be val_shape + mom.
        mom_ndim : int, optional
            Number of moment dimensions.
            `mom_dim=1` for moments, `mom_dim=2` for comoments.
        mom : int or tuple, optional
            Moments. Defaults to data.shape[-mom_ndim:].
            Must specify either `mom_ndim` or `mom`.
            Verify data has correct shape.
        val_shape : tuple, optional
            shape of non-moment dimensions.  Used to check `data`
        copy : bool, default=True.
            If True, copy `data`.  If False, try to not copy.
        copy_kws : dict, optional
            parameters to np.np.ndarray.copy
        verify : bool, default=True
            If True, force data to have 'c' order
        check_shape : bool, default=True
            If True, check that `data` has correct shape (based on `mom` and `val_shape`)
        dtype : np.dtype, optional

        Returns
        -------
        out : CentralMoments instance
        """
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if verify:
            data_verified = np.asarray(data, dtype=dtype, order="C")
        else:
            data_verified = data

        if check_shape:
            if val_shape is None:
                val_shape = data_verified.shape[:-mom_ndim]
            mom = cls._check_mom(mom, mom_ndim, data_verified.shape)

            if data_verified.shape != val_shape + tuple(x + 1 for x in mom):  # type: ignore
                raise ValueError(
                    f"{data.shape} does not conform to {val_shape} and {mom}"
                )

        if copy and data_verified is data:
            if copy_kws is None:
                copy_kws = {}
            data_verified = data_verified.copy(**copy_kws)

        return cls(data=data_verified, mom_ndim=mom_ndim)

    @classmethod
    def from_datas(
        cls: Type[T_CENTRALMOMENTS],
        datas: np.ndarray,
        mom_ndim: int | None = None,
        axis: int = 0,
        mom: T_MOM | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = True,
        check_shape: bool = True,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create object from multiple data arrays.

        Parameters
        ----------
        datas : np.np.ndarray
            Array of multiple Moment arrays.
            datas[..., i, ...] is the ith data array, where i is
            in position `axis`.

        See Also
        --------
        CentralMoments.from_data
        """

        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if verify:
            datas = np.asarray(datas, dtype=dtype)
        datas, axis = cls._datas_axis_to_first(datas, axis=axis, mom_ndim=mom_ndim)
        if check_shape:
            if val_shape is None:
                val_shape = datas.shape[1:-mom_ndim]

            mom = cls._check_mom(mom, mom_ndim, datas.shape)
            assert datas.shape[1:] == val_shape + tuple(x + 1 for x in mom)  # type: ignore

        if dtype is None:
            dtype = datas.dtype

        return cls.zeros(
            shape=datas.shape[1:], mom_ndim=mom_ndim, dtype=dtype, **kws
        ).push_datas(datas=datas, axis=0)

    @classmethod
    def from_vals(
        cls: Type[T_CENTRALMOMENTS],
        x: np.ndarray | Tuple[np.ndarray, np.ndarray],
        w: np.ndarray | None = None,
        axis: int = 0,
        dim: Hashable | None = None,
        mom: T_MOM = 2,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create from observations/values.

        Parameters
        ----------
        x : array-like or tuple of array-like
            For moments, pass single array-like objects.
            For comoments, pass tuple of array-like objects.
        w : array-like, optional
            Optional weights.
        axis : int, default=0
            axis to reduce along.
        mom : int or tuple of ints
            For moments, pass an int.  For comoments, pass a tuple of ints.
        val_shape : tuple, optional
            shape array of values part of resulting object
        broadcast : bool, default=False
            If True, and doing comoments, broadcast x[1] to x[0]
        kws : dict
            optional arguments passed to cls.zeros

        Returns
        -------
        out : CentralMoments object
        """

        mom_ndim = cls._mom_ndim_from_mom(mom)
        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast(np.ndarray, x0)
        if val_shape is None:
            val_shape = _shape_reduce(x0.shape, axis)
        if dtype is None:
            dtype = x0.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_vals(
            x=x, axis=axis, w=w, broadcast=broadcast
        )

    @classmethod
    def from_resample_vals(
        cls: Type[T_CENTRALMOMENTS],
        x: np.ndarray | Tuple[np.ndarray, np.ndarray],
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: int | None = None,
        w: np.ndarray | None = None,
        axis: int = 0,
        mom: T_MOM = 2,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        full_output: bool = False,
        **kws,
    ) -> T_CENTRALMOMENTS | Tuple[T_CENTRALMOMENTS, np.ndarray]:
        """Create from resample observations/values.

        This effectively resamples `x`.

        Parameters
        ----------
        x : array or tuple of arrays
            See CentralMoments.from_vals
        freq : array, optional
            Array of shape (nrep, size), where nrep is the number of
            replicates, and size is `x.shape(axis)`.
            `freq` is the weight that each sample contributes to the resampled values.
            See resample.randsamp_freq
        indices : array, optional
            Array of shape (nrep, size).  If passed, create `freq` from indices.
            See randsamp_freq.
        nrep : int, optional
            Number of replicates.  Create `freq` with this many replicates.
            See randsamp_freq
        w : array, optional.
            Optional weights associated with `x`.
        axis : int, default=0.
            Dimension to reduce/sample along.
        dtype : np.dtype, optional
            dtype of created output
        broadcast : bool, default=False
            If True, and calculating comoments, broadcast x[1] to x[0].shape
        parallel : bool, default=True
            If True, perform resampling in parallel.
        resample_kws : dict
            Extra arguments to resample.resample_vals
        kws : dict
            Extra arguments to CentralMoments.from_data
        full_output : bool, default=False
            If True, also return freq.

        Returns
        -------
        out : CentralMoments instance
        freq : array, optional
        """

        mom_ndim = cls._mom_ndim_from_mom(mom)

        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast(np.ndarray, x0)
        freq = randsamp_freq(
            nrep=nrep,
            freq=freq,
            indices=indices,
            size=x0.shape[axis],
            check=True,
        )

        if resample_kws is None:
            resample_kws = {}

        data = resample_vals(
            x,
            freq=freq,
            mom=mom,
            axis=axis,
            w=w,
            mom_ndim=mom_ndim,
            parallel=parallel,
            **resample_kws,
            broadcast=broadcast,
        )
        out = cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            verify=True,
            check_shape=True,
            copy=False,
            **kws,
        )

        if full_output:
            return out, freq
        else:
            return out

        return out

    @classmethod
    def from_raw(
        cls: Type[T_CENTRALMOMENTS],
        raw: np.ndarray,
        mom_ndim: int | None = None,
        mom: T_MOM | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create object from raw.

        raw[..., i, j] = <x**i y**j>.
        raw[..., 0, 0] = {weight}


        Parameters
        ----------
        raw : np.np.ndarray
            Raw moment array.
        mom_ndim : int, optional
            Number of moment dimensions.
        mom : int or tuple, optional
            number of moments.
            Must specify `mom_ndim` or `mom`.
        val_shape : tuple, optional
            shape of non-moment dimensions.
        dtype : np.dtype
            dtype of output
        convert_kws : dict
            arguments to central to raw converter
        kws : dict
            Extra arguments to cls.from_data

        Returns
        -------
        out : instance of cls

        See Also
        --------
        convert.to_central_moments
        convert.to_central_comoments
        """

        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if convert_kws is None:
            convert_kws = {}

        if mom_ndim == 1:
            data = convert.to_central_moments(raw, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            data = convert.to_central_comoments(raw, dtype=dtype, **convert_kws)
        else:
            raise ValueError(f"unknown mom_ndim {mom_ndim}")

        kws = dict(dict(verify=True, check_shape=True), **kws)

        return cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            copy=False,
            **kws,
        )

    @classmethod
    def from_raws(
        cls: Type[T_CENTRALMOMENTS],
        raws: np.ndarray,
        mom_ndim: int | None = None,
        mom: T_MOM | None = None,
        axis: int = 0,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create object from multipel `raw` moment arrays.

        Parameters
        ----------
        raws : array
            raws[...,i,...] is the ith sample of a `raw` array,
            Note that raw[...,i,j] = <x0**i, x1**j>
        where `i` is in position `axis`
        axis : int, default=0

        See Also
        --------
        CentralMoments.from_raw : called by from_raws
        CentralMoments.from_datas : similar constructor for central moments
        """
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if convert_kws is None:
            convert_kws = {}
        if mom_ndim == 1:
            datas = convert.to_central_moments(raws, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            datas = convert.to_central_comoments(raws, dtype=dtype, **convert_kws)
        else:
            raise ValueError(f"unknown mom_ndim {mom_ndim}")

        return cls.from_datas(
            datas=datas,
            axis=axis,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    ###########################################################################
    # SECTION: Manipulation
    ###########################################################################
    def reshape(
        self: T_CENTRALMOMENTS,
        shape: Tuple[int, ...],
        copy: bool = True,
        copy_kws: Mapping | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create a new object with reshaped data.

        Parameter
        ---------
        shape : tuple
            shape of values part of data
        """
        self._raise_if_scalar()
        new_shape = shape + self.mom_shape
        data = self._data.reshape(new_shape)

        return type(self).from_data(
            data=data,
            mom_ndim=self.mom_ndim,
            mom=self.mom,
            val_shape=None,
            copy=copy,
            copy_kws=copy_kws,
            verify=True,
            check_shape=True,
            dtype=self.dtype,
            **kws,
        )
        # return self.new_like(
        #     data=data, verify=False, check=False, copy=copy, copy_kws=copy_kws, **kws
        # )

    def moveaxis(
        self: T_CENTRALMOMENTS,
        source: int | Tuple[int, ...],
        destination: int | Tuple[int, ...],
        copy: bool = True,
        copy_kws: Mapping | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Move axis from source to destination."""
        self._raise_if_scalar()

        def _internal_check_val(v) -> Tuple[int, ...]:
            if isinstance(v, int):
                v = (v,)
            else:
                v = tuple(v)
            return tuple(self._wrap_axis(x) for x in v)

        source = _internal_check_val(source)
        destination = _internal_check_val(destination)
        data = np.moveaxis(self.data, source, destination)

        # use from data for extra checks
        # return self.new_like(data=data, copy=copy, *args, **kwargs)
        return type(self).from_data(
            data,
            mom=self.mom,
            mom_ndim=self.mom_ndim,
            val_shape=data.shape[: -self.mom_ndim],
            copy=copy,
            copy_kws=copy_kws,
            **kws,
        )

    # --------------------------------------------------
    # mom_ndim == 1 specific
    # --------------------------------------------------

    @staticmethod
    def _raise_if_not_1d(mom_ndim: int) -> None:
        if mom_ndim != 1:
            raise NotImplementedError("only available for mom_ndim == 1")

    # special, 1d only methods
    def push_stat(
        self: T_CENTRALMOMENTS,
        a: np.ndarray | float,
        v: np.ndarray | float = 0.0,
        w: np.ndarray | float | None = None,
        broadcast: bool = True,
    ) -> T_CENTRALMOMENTS:
        """Push statisics onto self."""
        self._raise_if_not_1d(self.mom_ndim)

        ar, target = self._check_val(a, target="val")
        vr = self._check_var(v, broadcast=broadcast)
        wr = self._check_weight(w, target=target)
        self._push.stat(self._data_flat, wr, ar, vr)
        return self

    def push_stats(
        self: T_CENTRALMOMENTS,
        a: np.ndarray,
        v: np.ndarray | float = 0.0,
        w: np.ndarray | float | None = None,
        axis: int = 0,
        broadcast: bool = True,
    ) -> T_CENTRALMOMENTS:
        """Push multiple statistics onto self."""
        self._raise_if_not_1d(self.mom_ndim)

        ar, target = self._check_vals(a, target="vals", axis=axis)
        vr = self._check_vars(v, target=target, axis=axis, broadcast=broadcast)
        wr = self._check_weights(w, target=target, axis=axis)
        self._push.stats(self._data_flat, wr, ar, vr)
        return self

    @classmethod
    def from_stat(
        cls: Type[T_CENTRALMOMENTS],
        a: ArrayLike | float,
        v: np.ndarray | float = 0.0,
        w: np.ndarray | float | None = None,
        mom: T_MOM = 2,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        order: ASARRAY_ORDER | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create object from single weight, average, variance/covariance."""
        mom_ndim = cls._mom_ndim_from_mom(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        if val_shape is None and isinstance(a, np.ndarray):
            val_shape = a.shape
        if dtype is None:
            dtype = a.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_stat(
            w=w, a=a, v=v
        )

    @classmethod
    def from_stats(
        cls: Type[T_CENTRALMOMENTS],
        a: np.ndarray,
        v: np.ndarray,
        w: np.ndarray | float | None = None,
        axis: int = 0,
        dim=None,
        mom: T_MOM = 2,
        val_shape: Tuple[int, ...] = None,
        dtype: DTypeLike | None = None,
        order: ASARRAY_ORDER | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create object from several statistics.

        Weights, averages, variances/covarainces along
        axis.
        """

        mom_ndim = cls._mom_ndim_from_mom(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        # get val_shape
        if val_shape is None:
            val_shape = _shape_reduce(a.shape, axis)
        return cls.zeros(val_shape=val_shape, dtype=dtype, mom=mom, **kws).push_stats(
            a=a,
            v=v,
            w=w,
            axis=axis,
        )

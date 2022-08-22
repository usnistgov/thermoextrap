"""Central moments/comoments routines."""
from __future__ import annotations

from typing import (
    Any,
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

import numpy as np
from numpy.core.numeric import normalize_axis_index  # type: ignore
from numpy.typing import ArrayLike, DTypeLike

from . import convert
from ._typing import (  # , T_XVAL_LIKE, T_XVAL_STRICT
    ASARRAY_ORDER,
    T_CENTRALMOMENTS,
    T_MOM,
)
from .cached_decorators import gcached
from .pushers import factory_pushers
from .resample import randsamp_freq, resample_data, resample_vals
from .utils import _axis_expand_broadcast  # _cached_ones,; _my_broadcast,
from .utils import _shape_insert_axis, _shape_reduce


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


###############################################################################
# Classes
###############################################################################
class CentralMoments(object):
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
        "_push",
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

        # if not data.flags['C_CONTIGUOUS']:
        #     raise ValueError('data must be c contiguous')

        self._data = data
        # check moments make sense
        # skip this as it comes from array shape
        # if not all([x > 0 for x in self.mom]):
        #     raise ValueError("All moments must positive")

        self._data_flat = self._data.reshape(self.shape_flat)

        if any(m <= 0 for m in self.mom):
            raise ValueError("moments must be positive")

        # setup pushers
        vec = len(self.val_shape) > 0
        cov = self.mom_ndim == 2
        self._push = factory_pushers(cov=cov, vec=vec)

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

    @property
    def shape(self) -> Tuple[int, ...]:
        """self.data.shape."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """self.data.ndim."""
        return self._data.ndim

    @property
    def dtype(self):
        """self.data.dtype."""
        return self._data.dtype

    @property
    def mom(self) -> Tuple[int] | Tuple[int, int]:
        """Number of moments."""  # noqa D401
        return tuple(x - 1 for x in self.mom_shape)  # type: ignore

    @property
    def mom_shape(self) -> Tuple[int] | Tuple[int, int]:
        """Shape of moments part."""
        return self._data.shape[-self.mom_ndim :]  # type: ignore

    @property
    def mom_ndim(self) -> Literal[1, 2]:
        """Length of moments.

        if `mom_ndim` == 1, then single variable
        moments if `mom_ndim` == 2, then co-moments.
        """
        return self._mom_ndim  # type: ignore

    @property
    def val_shape(self) -> Tuple[int, ...]:
        """Shape of values dimensions.

        That is shape less moments dimensions.
        """
        return self._data.shape[: -self.mom_ndim]

    @property
    def val_ndim(self) -> int:
        """Number of value dimensions."""  # noqa D401
        return len(self.val_shape)

    @property
    def val_shape_flat(self) -> Tuple[int, ...]:
        """Shape of values part flattened."""
        if self.val_shape == ():
            return ()
        else:
            return (np.prod(self.val_shape),)

    @property
    def shape_flat(self) -> Tuple[int, ...]:
        """Shape of flattened data."""
        return self.val_shape_flat + self.mom_shape

    @property
    def mom_shape_var(self) -> Tuple[int, ...]:
        """Shape of moment part of variance."""
        return tuple(x - 1 for x in self.mom)

    # variance shape
    @property
    def shape_var(self) -> Tuple[int, ...]:
        """Total variance shape."""
        return self.val_shape + self.mom_shape_var

    @property
    def shape_flat_var(self) -> Tuple[int, ...]:
        """Shape of flat variance."""
        return self.val_shape_flat + self.mom_shape_var

    # I think this is for pickling
    # probably don't need it anymore
    # def __setstate__(self, state):
    #     self.__dict__ = state
    #     # make sure datar points to data
    #     self._data_flat = self._data.reshape(self.shape_flat)

    # not sure I want to actually implements this
    # could lead to all sorts of issues applying
    # ufuncs to underlying data
    # def __array_wrap__(self, obj, context=None):
    #     return self, obj, context

    def __repr__(self):
        """Repr for class."""
        s = "<CentralMoments(val_shape={}, mom={})>\n".format(self.val_shape, self.mom)
        return s + repr(self.values)

    def __array__(self, dtype: DTypeLike | None = None) -> np.ndarray:
        """Used by np.array(self)."""  # noqa D401
        return np.asarray(self._data, dtype=dtype)

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

        return cast(
            T_CENTRALMOMENTS,
            type(self).from_data(
                data=data,
                copy=copy,
                copy_kws=copy_kws,
                verify=verify,
                check_shape=check_shape,
                **kws,
            ),
        )

    def zeros_like(self: T_CENTRALMOMENTS) -> T_CENTRALMOMENTS:
        """Create new object empty object like self."""
        return self.new_like()

    def copy(self: T_CENTRALMOMENTS, **copy_kws) -> T_CENTRALMOMENTS:
        """Create a new object with copy of data."""
        return self.new_like(
            data=self.values,
            verify=False,
            check_shape=False,
            copy=True,
            copy_kws=copy_kws,
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
    @gcached()
    def _weight_index(self):
        index = (0,) * len(self.mom)
        if self.val_ndim > 0:
            index = (...,) + index
        return index

    @gcached(prop=False)
    def _single_index(self, val):
        # index with things like data[..., 1,0] data[..., 0,1]
        # index = (...,[1,0],[0,1])
        dims = len(self.mom)
        if dims == 1:
            index = [val]
        else:
            # this is a bit more complicated
            index = [[0] * dims for _ in range(dims)]
            for i in range(dims):
                index[i][i] = val

        if self.val_ndim > 0:
            index = [...] + index

        return tuple(index)

    def weight(self):
        """Weight data."""
        return self.values[self._weight_index]

    def mean(self):
        """Mean (first moment)."""
        return self.values[self._single_index(1)]

    def var(self):
        """Variance (second central moment)."""
        return self.values[self._single_index(2)]

    def std(self):
        """Standard deviation."""  # noqa D401
        return np.sqrt(self.var())

    def cmom(self) -> np.ndarray:
        """Central moments.

        cmom[..., i0, i1] = < (x0 - <x0>)**i0 * (x1 - <x1>)**i1>
        Note that this is scrict, so `cmom[..., i, j] = 0` if `i+j = 0`
        and `cmom[...,0, 0] = 1`.

        """
        out = self.data.copy()
        # zeroth central moment
        out[self._weight_index] = 1
        # first central moment
        out[self._single_index(1)] = 0
        return out

    # convert to/from raw moments
    def to_raw(self) -> np.ndarray:
        """Convert central moments to raw moments.

        raw[...,i, j] = weight,           i = j = 0
                      = <x0**i * x1**j>,  otherwise
        """
        if self.mom_ndim == 1:
            return convert.to_raw_moments(x=self.data)
        elif self.mom_ndim == 2:
            return convert.to_raw_comoments(x=self.data)

    def rmom(self) -> np.ndarray:
        """Raw moments.

        rmom[..., i, j] = <x0 ** i * x1 ** j>
        """
        out = self.to_raw()
        out[self._weight_index] = 1
        return out

    ###########################################################################
    # SECTION: pushing routines
    ###########################################################################
    def _asarray(self, val: np.ndarray | float) -> np.ndarray:
        return np.asarray(val, dtype=self.dtype)

    # @property
    # def _unit_weight(self):
    #     """internally cached unit weight"""
    #     return _cached_ones(self.shape, dtype=self.dtype)

    #  -> np.ndarray | float | Tuple[float|np.ndarray, None|float|np.ndarray] :
    def _verify_value(
        self,
        x: np.ndarray | float,
        target: np.ndarray | Tuple[int, ...] | str | None = None,
        axis: int | None = None,
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Tuple[int, ...] | None = None,
        other: np.ndarray | None = None,
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

        x = self._asarray(x)

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

    def _check_weight(self, w: np.ndarray | float | None, target: np.ndarray):  # type: ignore
        if w is None:
            w = 1.0
        return self._verify_value(
            w,
            target=target,
            axis=None,
            broadcast=True,
            expand=True,
            shape_flat=self.val_shape_flat,
        )

    def _check_weights(
        self, w: float | np.ndarray | None, target: np.ndarray, axis: int = 0
    ):  # type: ignore
        if w is None:
            w = 1.0
        return self._verify_value(
            w,
            target=target,
            axis=axis,
            broadcast=True,
            expand=True,
            shape_flat=self.val_shape_flat,
        )

    def _check_val(self, x: np.ndarray | float, target: str, broadcast: bool = False):  # type: ignore
        return self._verify_value(
            x,
            target=target,
            broadcast=broadcast,
            expand=False,
            shape_flat=self.val_shape_flat,
        )

    def _check_vals(
        self,
        x: np.ndarray,
        target: np.ndarray | str,
        axis: int = 0,
        broadcast: bool = False,
    ):  # type: ignore
        return self._verify_value(
            x,
            target=target,
            axis=axis,
            broadcast=broadcast,
            expand=broadcast,
            shape_flat=self.val_shape_flat,
        )

    def _check_var(self, v: np.ndarray | float, broadcast: bool = False):  # type: ignore
        return self._verify_value(
            v,
            target="var",  # self.shape_var,
            broadcast=broadcast,
            expand=False,
            shape_flat=self.shape_flat_var,
        )[0]

    def _check_vars(
        self,
        v: np.ndarray | float,
        target: np.ndarray,
        axis: int = 0,
        broadcast: bool = False,
    ):  # type: ignore
        return self._verify_value(
            v,
            target="vars",
            axis=axis,
            broadcast=broadcast,
            expand=broadcast,
            shape_flat=self.shape_flat_var,
            other=target,
        )[0]

    def _check_data(self, data: np.ndarray):  # type: ignore
        return self._verify_value(data, target="data", shape_flat=self.shape_flat)[0]

    def _check_datas(self, datas: np.ndarray, axis: int = 0):  # type: ignore
        return self._verify_value(
            datas, target="datas", axis=axis, shape_flat=self.shape_flat
        )[0]

    def fill(self: T_CENTRALMOMENTS, value: Any = 0) -> T_CENTRALMOMENTS:
        """Fill data with value."""
        self._data.fill(value)
        return self

    def zero(self: T_CENTRALMOMENTS) -> T_CENTRALMOMENTS:
        """Zero out underlying data."""
        return self.fill(value=0)

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
        self: T_CENTRALMOMENTS, datas: np.ndarray, axis: int = 0
    ) -> T_CENTRALMOMENTS:
        """Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like
            this should have shape like `(nrec,) + self.shape`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        axis : int, default=0
            axis to reduce along

        Returns
        -------
        self
        """
        datas = self._check_datas(datas, axis)
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
    # SECTION: Operators
    ###########################################################################
    def _check_other(self: T_CENTRALMOMENTS, b: T_CENTRALMOMENTS) -> None:
        """Check other object."""
        assert type(self) == type(b)
        assert self.mom_ndim == b.mom_ndim
        assert self.shape == b.shape

    def __iadd__(
        self: T_CENTRALMOMENTS,
        b: T_CENTRALMOMENTS,
    ) -> T_CENTRALMOMENTS:  # noqa D105
        self._check_other(b)
        # self.push_data(b.data)
        # return self
        return self.push_data(b.data)

    def __add__(
        self: T_CENTRALMOMENTS,
        b: T_CENTRALMOMENTS,
    ) -> T_CENTRALMOMENTS:
        """Add objects to new object."""
        self._check_other(b)
        # new = self.copy()
        # new.push_data(b.data)
        # return new
        return self.copy().push_data(b.data)

    def __isub__(
        self: T_CENTRALMOMENTS,
        b: T_CENTRALMOMENTS,
    ) -> T_CENTRALMOMENTS:
        """Inplace substraction."""
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        data = b.data.copy()
        data[self._weight_index] *= -1
        # self.push_data(data)
        # return self
        return self.push_data(data)

    def __sub__(
        self: T_CENTRALMOMENTS,
        b: T_CENTRALMOMENTS,
    ) -> T_CENTRALMOMENTS:
        """Subtract objects."""
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        new = b.copy()
        new._data[self._weight_index] *= -1
        # new.push_data(self.data)
        # return new
        return new.push_data(self.data)

    def __mul__(self: T_CENTRALMOMENTS, scale: float | int) -> T_CENTRALMOMENTS:
        """New object with weights scaled by scale."""  # noqa D401
        scale = float(scale)
        new = self.copy()
        new._data[self._weight_index] *= scale
        return new

    def __imul__(self: T_CENTRALMOMENTS, scale: float | int) -> T_CENTRALMOMENTS:
        """Inplace multiply."""
        scale = float(scale)
        self._data[self._weight_index] *= scale
        return self

    ###########################################################################
    # SECTION: Constructors
    ###########################################################################
    @classmethod
    @no_type_check
    def _check_mom(
        cls, moments: T_MOM, mom_ndim: int, shape: Tuple[int, ...] | None = None
    ) -> Union[Tuple[int], Tuple[int, int]]:  # type: ignore
        """Check moments for correct shape.

        If moments is None, infer from
        shape[-mom_ndim:] if integer, convert to tuple.
        """

        if moments is None:
            if shape is not None:
                if mom_ndim is None:
                    raise ValueError(
                        "must speficy either moments or shape and mom_ndim"
                    )
                moments = tuple(x - 1 for x in shape[-mom_ndim:])
            else:
                raise ValueError("must specify moments")

        if isinstance(moments, int):
            if mom_ndim is None:
                mom_ndim = 1
            moments = (moments,) * mom_ndim

        else:
            moments = tuple(moments)
            if mom_ndim is None:
                mom_ndim = len(moments)

        assert len(moments) == mom_ndim
        return moments

    @staticmethod
    def _datas_axis_to_first(
        datas: np.ndarray, axis: int, mom_ndim: int
    ) -> Tuple[np.ndarray, int]:
        """Move axis to first first position."""
        # NOTE: removinvg this. should be handles elsewhere
        # datas = np.asarray(datas)
        # ndim = datas.ndim - mom_ndim
        # if axis < 0:
        #     axis += ndim
        # assert 0 <= axis < ndim
        axis = normalize_axis_index(axis, datas.ndim - mom_ndim)
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        return datas, axis

    def _wrap_axis(
        self, axis: int | None, default: int = 0, ndim: int | None = None
    ) -> int:
        """Wrap axis to positive value and check."""
        if axis is None:
            axis = default
        if ndim is None:
            ndim = self.val_ndim

        axis = cast(int, normalize_axis_index(axis, ndim))
        # if axis < 0:
        #     axis += ndim
        # assert 0 <= axis < ndim
        return axis

    @classmethod
    def _mom_ndim_from_mom(cls, mom: Union[Tuple[int], Tuple[int, int], int]) -> int:
        if isinstance(mom, int):
            return 1
        elif isinstance(mom, tuple):
            return len(mom)
        else:
            raise ValueError("mom must be int or tuple")

    @classmethod
    def _choose_mom_ndim(
        cls,
        mom: T_MOM | None,
        mom_ndim: int | None,
    ) -> int:
        if mom is not None:
            mom_ndim = cls._mom_ndim_from_mom(mom)

        if mom_ndim is None:
            raise ValueError("must specify mom_ndim or mom")

        return mom_ndim

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
        datas, axis = cls._datas_axis_to_first(datas, axis, mom_ndim)
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
        **kws,
    ) -> T_CENTRALMOMENTS:
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

        Returns
        -------
        out : CentralMoments instance
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
        return cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            verify=True,
            check_shape=True,
            copy=False,
            **kws,
        )

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
    @property
    def _is_vector(self) -> bool:
        return self.val_ndim > 0

    def _raise_if_scalar(self, message: str | None = None) -> None:
        if not self._is_vector:
            if message is None:
                message = "not implemented for scalar"
            raise ValueError(message)

    # Universal reducers
    def resample_and_reduce(
        self: T_CENTRALMOMENTS,
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: None = None,
        axis: int | None = None,
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Bootstrap resample and reduce.

        Parameter
        ----------
        freq : array-like, shape=(nrep, nrec), optional
            frequence table.  freq[i, j] is the weight of the jth record to the ith
            replicate indices : array-like, shape=(nrep, nrec), optional
            resampling array.  idx[i, j] is the record index of the original array to
            place in new sample[i, j]. if specified, create freq array from idx
        nrep : int, optional
            if specified, create idx array with this number of replicates
        axis : int, Default=0
            axis to resample and reduce along
        parallel : bool, default=True
            flags to `numba.njit`
        resample_kws : dict
            extra arguments to `cmomy.resample.resample_and_reduce`
        kws : dict
            extra key-word arguments to from_data method
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        if resample_kws is None:
            resample_kws = {}

        freq = randsamp_freq(
            nrep=nrep, indices=indices, freq=freq, size=self.val_shape[axis], check=True
        )
        data = resample_data(
            self.data, freq, mom=self.mom, axis=axis, parallel=parallel, **resample_kws
        )
        return type(self).from_data(data, mom_ndim=self.mom_ndim, copy=False, **kws)

    def resample(
        self: T_CENTRALMOMENTS,
        indices: np.ndarray,
        axis: int = 0,
        first: bool = True,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Create a new object sampled from index.

        Parameters
        ----------
        indicies : array-like
        axis : int, default=0
            axis to resample
        first : bool, default=True
            if True, and axis != 0, the move the axis to first position.
            This makes results similar to resample and reduce
            If `first` False, then resampled array can have odd shape

        Returns
        -------
        output : accumulator object
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)

        data = self.data
        if first and axis != 0:
            data = np.moveaxis(data, axis, 0)
            axis = 0

        out = np.take(data, indices, axis=axis)

        return type(self).from_data(
            data=out,
            mom_ndim=self.mom_ndim,
            mom=self.mom,
            copy=False,
            verify=True,
            **kws,
        )

    def reduce(self: T_CENTRALMOMENTS, axis: int = 0, **kws) -> T_CENTRALMOMENTS:
        """Create new object reducealong axis."""
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        return type(self).from_datas(
            self.values, mom_ndim=self.mom_ndim, axis=axis, **kws
        )

    def block(
        self: T_CENTRALMOMENTS,
        block_size: int | None = None,
        axis: int | None = None,
        **kws,
    ) -> T_CENTRALMOMENTS:
        """Block average reduction.

        Parameters
        ----------
        block_size : int
            number of consecutive records to combine
        axis : int, default=0
            axis to reduce along
        kws : dict
            extral key word arguments to `from_datas` method
        """

        self._raise_if_scalar()

        axis = self._wrap_axis(axis)
        data = self.data

        # move axis to first
        if axis != 0:
            data = np.moveaxis(data, axis, 0)

        n = data.shape[0]

        if block_size is None:
            block_size = n
            nblock = 1

        else:
            nblock = n // block_size

        datas = data[: (nblock * block_size), ...].reshape(
            (nblock, block_size) + data.shape[1:]
        )
        return type(self).from_datas(datas=datas, mom_ndim=self.mom_ndim, axis=1, **kws)

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
            a=a, v=v, w=w, axis=axis
        )

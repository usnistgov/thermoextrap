"""Base class for central moments calculations."""


from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Mapping,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import numpy as np
import xarray as xr
from custom_inherit import DocInheritMeta
from numpy.core.numeric import normalize_axis_index  # type: ignore
from numpy.typing import ArrayLike, DTypeLike

from . import convert
from ._docstrings import docfiller_shared
from ._typing import Moments, T_Array, T_CentralMoments
from .cached_decorators import gcached
from .options import DOC_SUB
from .pushers import factory_pushers
from .resample import randsamp_freq, resample_data

# * TODO Main
# TODO: Total rework is called for to handle typing correctly.


def _get_metaclass():
    if DOC_SUB:
        return DocInheritMeta(style="numpy_with_merge")
    else:
        return ABCMeta


class CentralMomentsABC(Generic[T_Array], metaclass=_get_metaclass()):
    r"""Wrapper to calculate central moments.

    Base data has the form

    .. math::

        data[..., i, j] = \begin{cases}
            \text{weight} & i = j = 0 \\
            \langle x \rangle & i = 1, j = 0 \\
            \langle (x - \langle x \rangle^i) (y - \langle y \rangle^j) \rangle & i + j > 0
        \end{cases}

    Parameters
    ----------
    data : xr.DataArray / np.ndarray
        Moment collection array.
    mom_ndim : {1, 2}
        Number of dimensions for moment part of `data`.
        * 1 : central moments of single variable
        * 2 : central comoments of two variables

    """

    __slots__ = (
        "_mom_ndim",
        "_cache",
        "_data",
        "_data_flat",
    )

    # Override __new__ to make signature correct
    # Better to do this in subclasses.
    # otherwise, signature for data will be 'T_Array``
    # def __new__(cls, data: T_Array, mom_ndim: Literal[1, 2] = 1):
    #     return super().__new__(cls, data=data, mom_ndim=mom_ndim)

    def __init__(self, data: T_Array, mom_ndim: Literal[1, 2] = 1) -> None:

        self._data = cast(np.ndarray, data)
        self._data_flat = self._data

        self._mom_ndim = mom_ndim
        self._cache: Mapping[Any, Any] = {}

    @property
    def data(self) -> np.ndarray:
        """Accessor to numpy array underlying data.

        By convention data has the following meaning for the moments indexes

        * `data[...,i=0,j=0]`, weights
        * `data[...,i=1,j=0]]`, if only one moment indice is one and all others zero, then this is the average value of the variable with unit index.
        * all other cases, the central moments `<(x0-<x0>)**i0 * (x1 - <x1>)**i1 * ...>`

        """
        return cast(np.ndarray, self._data)

    @property
    @abstractmethod
    def values(self) -> T_Array:
        """Access underlying central moments array."""
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        """self.data.shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """self.data.ndim."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        """self.data.dtype."""
        # Not sure why I have to cast
        return cast(np.dtype, self.data.dtype)

    @property
    def mom_ndim(self) -> Literal[1, 2]:
        """Length of moments.

        if `mom_ndim` == 1, then single variable
        moments if `mom_ndim` == 2, then co-moments.
        """
        return self._mom_ndim

    @property
    def mom_shape(self) -> Tuple[int] | Tuple[int, int]:
        """Shape of moments part."""
        return cast(
            Union[Tuple[int], Tuple[int, int]], self.data.shape[-self.mom_ndim :]
        )

    @property
    def mom(self) -> Tuple[int] | Tuple[int, int]:
        """Number of moments."""  # noqa D401
        return tuple(x - 1 for x in self.mom_shape)  # type: ignore

    @property
    def val_shape(self) -> Tuple[int, ...]:
        """Shape of values dimensions.

        That is shape less moments dimensions.
        """
        return self.data.shape[: -self.mom_ndim]

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

    @property
    def shape_var(self) -> Tuple[int, ...]:
        """Total variance shape."""
        return self.val_shape + self.mom_shape_var

    @property
    def shape_flat_var(self) -> Tuple[int, ...]:
        """Shape of flat variance."""
        return self.val_shape_flat + self.mom_shape_var

    @gcached()
    def _push(self):
        vec = len(self.val_shape) > 0
        cov = self.mom_ndim == 2
        return factory_pushers(cov=cov, vec=vec)

    def __repr__(self):
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(val_shape={self.val_shape}, mom={self.mom})>\n"
        return s + repr(self.values)

    def __array__(self, dtype: DTypeLike | None = None) -> np.ndarray:
        """Used by np.array(self)."""  # noqa D401
        return np.asarray(self.data, dtype=dtype)

    ###########################################################################
    # ** top level creation/copy/new
    ###########################################################################
    @abstractmethod
    @docfiller_shared
    def new_like(
        self: T_CentralMoments,
        *,
        data: ArrayLike | xr.DataArray | None = None,
        copy: bool = False,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = False,
        strict: bool = False,
        **kws,
    ) -> T_CentralMoments:
        """
        Create new object like self, with new data.

        Parameters
        ----------
        data : array-like, optional
            data for new object
        {copy}
        {copy_kws}
        {verify}
        {check_shape}
        strict : bool, default=False
            If True, verify that `data` has correct shape
        **kws
            arguments to classmethod :meth:`from_data`


        See Also
        --------
        from_data

        """
        pass

    def zeros_like(self: T_CentralMoments) -> T_CentralMoments:
        """Create new empty object like self.

        Returns
        -------
        output : same as object
            Object with same attributes as caller, but with zerod out data.

        See Also
        --------
        new_like
        from_data
        """
        return self.new_like()

    def copy(self: T_CentralMoments, **copy_kws) -> T_CentralMoments:
        """Create a new object with copy of data.

        Parameters
        ----------
        **copy_kws
            passed to parameter ``copy_kws`` in method :meth:`new_like`

        Returns
        -------
        output : same as object
            Object with same attributes as caller, but with new underlying data.


        See Also
        --------
        new_like
        zeros_like
        """
        return self.new_like(
            data=self.values,
            verify=False,
            check_shape=False,
            copy=True,
            copy_kws=copy_kws,
        )

    ###########################################################################
    # ** Access to underlying statistics
    ###########################################################################

    @gcached()
    def _weight_index(self):
        index = (0,) * len(self.mom)
        if self.val_ndim > 0:
            return (...,) + index
        else:
            return index

    @gcached(prop=False)
    def _single_index(self, val) -> Tuple[List[int], ...]:
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

    def weight(self) -> float | T_Array:
        """Weight data."""
        return cast(Union[float, T_Array], self.values[self._weight_index])

    def mean(self) -> float | T_Array:
        """Mean (first moment)."""
        return cast(Union[float, T_Array], self.values[self._single_index(1)])

    def var(self) -> float | T_Array:
        """Variance (second central moment)."""
        return cast(Union[float, T_Array], self.values[self._single_index(2)])

    def std(self) -> float | T_Array:
        """Standard deviation."""  # noqa D401
        return cast(Union[float, T_Array], np.sqrt(self.var()))

    def _wrap_like(self, x: np.ndarray, *args, **kwargs) -> T_Array:
        return cast(T_Array, x)

    def cmom(self) -> T_Array:
        r"""Central moments.

        Strict central moments of the form

        .. math::

            \text{cmom[..., n, m]} =
            \langle (x - \langle x \rangle^n) (y - \langle y \rangle^m) \rangle

        Returns
        -------
        output : ndarray or DataArray
        """
        out = self.data.copy()
        # zeroth central moment
        out[self._weight_index] = 1
        # first central moment
        out[self._single_index(1)] = 0
        return self._wrap_like(out)

    def to_raw(self, weights=None) -> T_Array:
        r"""
        Raw moments accumulation array.

        .. math::

            \text{raw[..., n, m]} = \begin{cases}
            \text{weight} & n = m = 0 \\
            \langle x^n y ^m \rangle & \text{otherwise}
            \end{cases}

        Returns
        -------
        raw : ndarray or DataArray

        See Also
        --------
        from_raw
        """
        if self.mom_ndim == 1:
            out = convert.to_raw_moments(x=self.data)
        elif self.mom_ndim == 2:
            out = convert.to_raw_comoments(x=self.data)

        if weights is not None:
            out[self._weight_index] = weights

        return self._wrap_like(out)

    def rmom(self) -> T_Array:
        r"""Raw moments.

        .. math::
            \text{rmom[..., n, m]} = \langle x^n y^m \rangle

        Returns
        -------
        raw_moments : ndarray or DataArray

        See Also
        --------
        to_raw
        cmom
        """
        return self.to_raw(weights=1.0)

    ###########################################################################
    # ** pushing routines
    ###########################################################################
    def fill(self: T_CentralMoments, value: Any = 0) -> T_CentralMoments:
        """Fill data with value.

        Parameters
        ----------
        value : scalar
            Value to insert into `self.data`

        Returns
        -------
        self : same as object
            Same object as caller with data filled with `values`
        """
        self._data.fill(value)
        return self

    def zero(self: T_CentralMoments) -> T_CentralMoments:
        """Zero out underlying data.

        Returns
        -------
        self : same as object
            Same object with data filled with zeros.

        See Also
        --------
        fill
        """
        return self.fill(value=0.0)

    @abstractmethod
    def _verify_value(
        self,
        x: Any,
        target: Any,
        axis: int | None = None,
        broadcast: bool = False,
        expand: bool = False,
        other: Any | None = None,
        *args,
        **kwargs,
    ):
        pass

    def _check_weight(self, w, target, **kwargs):  # type: ignore
        if w is None:
            w = 1.0
        return self._verify_value(
            w,
            target=target,
            broadcast=True,
            expand=True,
            shape_flat=self.val_shape_flat,
            **kwargs,
        )

    def _check_weights(
        self,
        w,
        target,
        axis: int = None,
        **kwargs,
    ):
        # type: ignore
        if w is None:
            w = 1.0
        return self._verify_value(
            w,
            target=target,
            axis=axis,
            broadcast=True,
            expand=True,
            shape_flat=self.val_shape_flat,
            **kwargs,
        )

    def _check_val(self, x, target, broadcast=False, **kwargs):  # type: ignore
        return self._verify_value(
            x,
            target=target,
            broadcast=broadcast,
            expand=False,
            shape_flat=self.val_shape_flat,
            **kwargs,
        )

    def _check_vals(self, x, target, axis, broadcast=False, **kwargs):  # type: ignore
        return self._verify_value(
            x,
            target=target,
            axis=axis,
            broadcast=broadcast,
            expand=broadcast,
            shape_flat=self.val_shape_flat,
            **kwargs,
        )

    def _check_var(self, v, broadcast=False, **kwargs):
        return self._verify_value(
            v,
            target="var",  # self.shape_var,
            broadcast=broadcast,
            expand=False,
            shape_flat=self.shape_flat_var,
            **kwargs,
        )[0]

    def _check_vars(
        self, v, target, axis, broadcast: bool = False, **kwargs
    ):  # type: ignore
        return self._verify_value(
            v,
            target="vars",
            axis=axis,
            broadcast=broadcast,
            expand=broadcast,
            shape_flat=self.shape_flat_var,
            other=target,
            **kwargs,
        )[0]

    def _check_data(self, data, **kwargs):  # type: ignore
        return self._verify_value(
            data, target="data", shape_flat=self.shape_flat, **kwargs
        )[0]

    def _check_datas(self, datas, axis, **kwargs):  # type: ignore
        return self._verify_value(
            datas,
            target="datas",
            axis=axis,
            shape_flat=self.shape_flat,
            **kwargs,
        )[0]

    @docfiller_shared
    def push_data(self: T_CentralMoments, data: Any) -> T_CentralMoments:
        """
        Push data object to moments.

        Parameters
        ----------
        data : array-like
            Accumulation array of same form as ``self.data``

        Returns
        -------
        {pushed}
        """
        data = self._check_data(data)
        self._push.data(self._data_flat, data)
        return self

    @docfiller_shared
    def push_datas(
        self: T_CentralMoments,
        datas,
        axis: int,
        **kwargs,
    ) -> T_CentralMoments:
        """
        Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like
            Collection of accumulation arrays to push onto ``self``.
            This should have shape like `(nrec,) + self.shape`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        {axis}

        Returns
        -------
        {pushed}
        """

        datas = self._check_datas(datas=datas, axis=axis, **kwargs)
        self._push.datas(self._data_flat, datas)
        return self

    @docfiller_shared
    def push_val(
        self: T_CentralMoments, x, w=None, broadcast: bool = False, **kwargs
    ) -> T_CentralMoments:
        """
        Push single sample to central moments.

        Parameters
        ----------
        x : array-like or tuple of array-like
            Pass single array `x=x0`if accumulating moments.
            Pass tuple of arrays `(x0, x1)` if accumulating comoments.
            `x0.shape == self.val_shape`
        w : int, float, array-like, optional
            Weight of each sample.  If scalar, broadcast `w.shape` to `x0.shape`.
        broadcast : bool, optional
            If True, and `x1` present, attempt to broadcast `x1.shape` to `x0.shape`

        Returns
        -------
        {pushed}

        Notes
        -----
        Array `x0` should have same shape as `self.val_shape`.
        """

        if self.mom_ndim == 1:
            ys = ()
        else:
            assert isinstance(x, tuple) and len(x) == self.mom_ndim
            x, *ys = x  # type: ignore

        xr, target = self._check_val(x, "val", **kwargs)  # type: ignore
        yr = tuple(self._check_val(y, target=target, broadcast=broadcast) for y in ys)  # type: ignore
        wr = self._check_weight(w, target)  # type: ignore
        self._push.val(self._data_flat, *((wr, xr) + yr))
        return self

    @docfiller_shared
    def push_vals(
        self: T_CentralMoments,
        x,
        w=None,
        axis: int | None = None,
        broadcast: bool = False,
        **kwargs,
    ) -> T_CentralMoments:
        """
        Push multiple samples to central moments.

        Parameters
        ----------
        x : array-like or tuple of array-like
            Pass single array `x=x0`if accumulating moments.
            Pass tuple of arrays `(x0, x1)` if accumulating comoments.
            `x0.shape` less axis should be same as `self.val_shape`.
        w : int, float, array-like, optional
            Weight of each sample.  If scalar, broadcast to `x0.shape`
        {broadcast}
        {axis}

        Returns
        -------
        {pushed}
        """
        if self.mom_ndim == 1:
            ys = ()
        else:
            assert len(x) == self.mom_ndim
            x, *ys = x  # type: ignore

        xr, target = self._check_vals(x, axis=axis, target="vals", **kwargs)  # type: ignore
        yr = tuple(  # type: ignore
            self._check_vals(y, target=target, axis=axis, broadcast=broadcast, **kwargs)  # type: ignore
            for y in ys  # type: ignore
        )  # type: ignore
        wr = self._check_weights(w, target=target, axis=axis, **kwargs)
        self._push.vals(self._data_flat, *((wr, xr) + yr))
        return self

    ###########################################################################
    # ** Manipulation
    ###########################################################################
    def pipe(
        self,
        func_or_method: Callable | str,
        *args,
        _order: bool = True,
        _copy: bool = False,
        _verify: bool = False,
        _check_mom: bool = True,
        _kws: Mapping | None = None,
        **kwargs,
    ) -> T_CentralMoments:
        """
        Apply `func_or_method` to underlying data and wrap results in `xCentralMoments` object.

        This is usefull for calling any not implemnted methods on ndarray or DataArray data.

        Parameters
        ----------
        func_or_method : str or callable
            If callable, then apply ``values = func_or_method(self.values, *args, **kwargs)``.
            If string is passed, then ``values = getattr(self.values, func_or_method)(*args, **kwargs)``.
        _order : bool, default = True
            If True, reorder the data such that ``mom_dims`` are last.
        _copy : bool, default = False
            If True, copy the resulting data.  Otherwise, try to use a view.
            This is passed as ``copy=_copy`` to :meth:`from_data`.
        _verify: bool, default=False
            If True, ensure underlying data is contiguous.  Passed as ``verify=_verify`` to :meth:`from_data`
        _check_mom: bool, default = True
            If True, check the resulting object has the same moment shape as the current object.
        _kws : Mapping, optional
            Extra arguments to :meth:`from_data`.
        *args
            Extra positional arguments to `func_or_method`
        **kwargs
            Extra keyword arguments to `func_or_method`

        Returns
        -------
        output : xCentralMoments
            New :class:`xCentralMoments` object after `func_or_method` is applies to `self.values`


        Notes
        -----
        Use leading underscore for `_order`, `_copy` to avoid name possible name clashes.


        See Also
        --------
        from_data
        """

        if isinstance(func_or_method, str):
            values = getattr(self.values, func_or_method)(*args, **kwargs)
        else:
            values = func_or_method(self.values, *args, **kwargs)

        if _order:
            values = values.transpose(..., *self.mom_dims)

        if _kws is None:
            _kws = {}
        else:
            _kws = dict(_kws)
        _kws.setdefault("copy", _copy)
        _kws.setdefault("verify", _verify)
        _kws.setdefault("mom_ndim", self.mom_ndim)
        if _check_mom:
            _kws["mom"] = self.mom
            _kws["check_shape"] = True

        out = type(self).from_data(data=values, **_kws)

        return cast(T_CentralMoments, out)

    @property
    def _is_vector(self) -> bool:
        return self.val_ndim > 0

    def _raise_if_scalar(self, message: str | None = None) -> None:
        if not self._is_vector:
            if message is None:
                message = "not implemented for scalar"
            raise ValueError(message)

    # * Universal reducers
    @docfiller_shared
    def resample_and_reduce(
        self: T_CentralMoments,
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: None = None,
        axis: int | None = None,
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        full_output: bool = False,
        **kws,
    ) -> T_CentralMoments | Tuple[T_CentralMoments, np.ndarray]:
        """
        Bootstrap resample and reduce.

        Parameters
        ----------
        {freq}
        {indices}
        {nrep}
        {axis}
        parallel : bool, default=True
            flags to `numba.njit`
        {resample_kws}
        {full_output}
        **kws
            Extra key-word arguments to :meth:`from_data`

        Returns
        -------
        output : instance of calling class
            Note that new object will have val_shape = (nrep,) + val_shape[:axis] + val_shape[axis+1:]

        See Also
        --------
        :meth:`resample`
        :meth:`reduce`
        ~resample.randsamp_freq
        ~resample.freq_to_indices
        ~resample.indices_to_freq
        ~resample.resample_data
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis, **kws)
        if resample_kws is None:
            resample_kws = {}

        freq = randsamp_freq(
            nrep=nrep, indices=indices, freq=freq, size=self.val_shape[axis], check=True
        )
        data = resample_data(
            self.data, freq, mom=self.mom, axis=axis, parallel=parallel, **resample_kws
        )
        out = type(self).from_data(data, mom_ndim=self.mom_ndim, copy=False, **kws)

        if full_output:
            return out, freq
        else:
            return out

    @docfiller_shared
    def resample(
        self: T_CentralMoments,
        indices: np.ndarray,
        axis: int = 0,
        first: bool = True,
        **kws,
    ) -> T_CentralMoments:
        """
        Create a new object sampled from index.

        Parameters
        ----------
        {indices}
        {axis}
        first : bool, default=True
            if True, and axis != 0, the move the axis to first position.
            This makes results similar to resample and reduce
            If `first` False, then resampled array can have odd shape

        Returns
        -------
        output : instance of calling class
            The new object will have shape
            `(nrep, ndat, ...) + self.shape[:axis] + self.shape[axis+1:]`


        See Also
        --------
        from_data

        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis, **kws)

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

    @docfiller_shared
    def reduce(
        self: T_CentralMoments, axis: int | None = None, **kws
    ) -> T_CentralMoments:
        """
        Create new object reducealong axis.

        Parameters
        ----------
        {axis}
        **kws
            Extra parameters to :meth:`from_data`

        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis, **kws)
        return type(self).from_datas(
            self.values, mom_ndim=self.mom_ndim, axis=axis, **kws
        )

    @docfiller_shared
    def block(
        self: T_CentralMoments,
        block_size: int | None = None,
        axis: int | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Block average reduction.

        Parameters
        ----------
        block_size : int
            number of consecutive records to combine
        {axis}
        **kws
            Extral key word arguments to :meth:`from_datas` method

        Returns
        -------
        output : new instance of calling class
            Shape of output will be
            `(nblock,) + self.shape[:axis] + self.shape[axis+1:]`.

        Notes
        -----
        The block averaged `axis` will be moved to the front of the output data.

        See Also
        --------
        reshape
        moveaxis
        reduce
        """

        self._raise_if_scalar()

        axis = self._wrap_axis(axis, **kws)
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

    ###########################################################################
    # ** Operators
    ###########################################################################
    def _check_other(self: T_CentralMoments, b: T_CentralMoments) -> None:
        """Check other object."""
        assert type(self) == type(b)
        assert self.mom_ndim == b.mom_ndim
        assert self.shape == b.shape

    def __iadd__(  # type: ignore
        self: T_CentralMoments,
        b: T_CentralMoments,
    ) -> T_CentralMoments:  # noqa D105
        self._check_other(b)
        # self.push_data(b.data)
        # return self
        return self.push_data(b.data)

    def __add__(
        self: T_CentralMoments,
        b: T_CentralMoments,
    ) -> T_CentralMoments:
        """Add objects to new object."""
        self._check_other(b)
        # new = self.copy()
        # new.push_data(b.data)
        # return new
        return self.copy().push_data(b.data)

    def __isub__(  # type: ignore
        self: T_CentralMoments,
        b: T_CentralMoments,
    ) -> T_CentralMoments:
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
        self: T_CentralMoments,
        b: T_CentralMoments,
    ) -> T_CentralMoments:
        """Subtract objects."""
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        new = b.copy()
        new._data[self._weight_index] *= -1
        # new.push_data(self.data)
        # return new
        return new.push_data(self.data)

    def __mul__(self: T_CentralMoments, scale: float | int) -> T_CentralMoments:
        """New object with weights scaled by scale."""  # noqa D401
        scale = float(scale)
        new = self.copy()
        new._data[self._weight_index] *= scale
        return new

    def __imul__(self: T_CentralMoments, scale: float | int) -> T_CentralMoments:
        """Inplace multiply."""
        scale = float(scale)
        self._data[self._weight_index] *= scale
        return self

    ###########################################################################
    # ** Constructors
    ###########################################################################
    # *** Utils
    @classmethod
    @no_type_check
    def _check_mom(
        cls, moments: Moments, mom_ndim: int, shape: Tuple[int, ...] | None = None
    ) -> Tuple[int] | Tuple[int, int]:  # type: ignore
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
        datas, axis: int, mom_ndim: int, **kws
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
        self, axis: int | None, default: int = 0, ndim: int | None = None, **kws
    ) -> int:
        """Wrap axis to positive value and check."""
        if axis is None:
            axis = default
        if ndim is None:
            ndim = self.val_ndim

        return cast(int, normalize_axis_index(axis, ndim))

    @classmethod
    def _mom_ndim_from_mom(cls, mom: Moments) -> int:
        if isinstance(mom, int):
            return 1
        elif isinstance(mom, tuple):
            return len(mom)
        else:
            raise ValueError("mom must be int or tuple")

    @classmethod
    def _choose_mom_ndim(
        cls,
        mom: Moments | None,
        mom_ndim: int | None,
    ) -> int:
        if mom is not None:
            mom_ndim = cls._mom_ndim_from_mom(mom)

        if mom_ndim is None:
            raise ValueError("must specify mom_ndim or mom")

        return mom_ndim

    # *** Core
    @classmethod
    @abstractmethod
    @docfiller_shared
    def zeros(
        cls: Type[T_CentralMoments],
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        shape: Tuple[int, ...] | None = None,
        mom_ndim: int | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Create a new base object.

        Parameters
        ----------
        {mom}
        {mom_ndim}
        {val_shape}
        {shape}
        {dtype}
        {zeros_kws}
        **kws
            extra arguments to :meth:`from_data`


        Notes
        -----
        The resulting total shape of data is shape + (mom + 1).
        Must specify either `mom` or `mom_ndim`


        See Also
        --------
        from_data : General constructor
        numpy.zeros

        """
        pass

    @classmethod
    @abstractmethod
    @docfiller_shared
    def from_data(
        cls: Type[T_CentralMoments],
        data: Any,
        mom_ndim: int | None = None,
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
    ) -> T_CentralMoments:
        """
        Create new object from `data` array with additional checks.

        Parameters
        ----------
        data : array-like
            central moments accumulation array.
        {mom_ndim}
        {mom}
        {val_shape}
        {copy}
        {copy_kws}
        {verify}
        {check_shape}
        {dtype}

        Returns
        -------
        out : caller class instance

        """

    @classmethod
    @abstractmethod
    @docfiller_shared
    def from_datas(
        cls: Type[T_CentralMoments],
        datas: Any,
        mom_ndim: int | None = None,
        axis: int | None = 0,
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = True,
        check_shape: bool = True,
        **kws,
    ) -> T_CentralMoments:
        """
        Create object from multiple data arrays.

        Parameters
        ----------
        datas : ndarray
            Array of multiple Moment arrays.
            datas[..., i, ...] is the ith data array, where i is
            in position `axis`.
        {axis}
        {mom}
        {mom_ndim}
        {val_shape}
        {dtype}
        {verify}
        {check_shape}
        **kws
            Extra arguments

        See Also
        --------
        CentralMoments.from_data
        """

    @classmethod
    @abstractmethod
    @docfiller_shared
    def from_vals(
        cls: Type[T_CentralMoments],
        x,
        w=None,
        axis: int | None = 0,
        mom: Moments = 2,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        **kws,
    ) -> T_CentralMoments:
        """
        Create from observations/values.

        Parameters
        ----------
        x : array-like or tuple of array-like
            For moments, pass single array-like objects `x=x0`.
            For comoments, pass tuple of array-like objects `x=(x0, x1)`.
        w : scalar or array-like, optional
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis}
        {mom}
        {val_shape}
        {broadcast}
        **kws
            Optional arguments passed to :meth:`zeros`

        See Also
        --------
        push_vals
        """

    @classmethod
    @abstractmethod
    @docfiller_shared
    def from_resample_vals(
        cls: Type[T_CentralMoments],
        x,
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: int | None = None,
        w: np.ndarray | None = None,
        axis: int = 0,
        mom: Moments = 2,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        full_output: bool = False,
        **kws,
    ) -> T_CentralMoments:
        """
        Create from resample observations/values.

        This effectively resamples `x`.

        Parameters
        ----------
        x : array-like or tuple of array-like
            For moments, pass single array-like objects `x=x0`.
            For comoments, pass tuple of array-like objects `x=(x0, x1)`.
        w : scalar or array-like, optional
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {freq}
        {indices}
        {nrep}
        {axis}
        {mom}
        {dtype}
        {broadcast}
        parallel : bool, default=True
            If True, perform resampling in parallel.
        {resample_kws}
        {full_output}
        **kws
            Extra arguments to CentralMoments.from_data

        Returns
        -------
        out : instance of calling class
        freq : ndarray, optional
            If `full_output` is True, also return `freq` array


        See Also
        --------
        ~resample.resample_vals
        ~resample.randsamp_freq
        ~resample.freq_to_indices
        ~resample.indices_to_freq
        """

    @classmethod
    @abstractmethod
    @docfiller_shared
    def from_raw(
        cls: Type[T_CentralMoments],
        raw,
        mom_ndim: int | None = None,
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Create object from raw.

        raw[..., i, j] = <x**i y**j>.
        raw[..., 0, 0] = `weight`


        Parameters
        ----------
        raw : ndarray
            Raw moment array.
        {mom_ndim}
        {mom}
        {val_shape}
        {dtype}
        {convert_kws}
        **kws
            Extra arguments to :meth:`from_data`

        See Also
        --------
        to_raw
        rmom
        ~convert.to_central_moments
        ~convert.to_central_comoments

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.

        """
        pass

    @classmethod
    @abstractmethod
    @docfiller_shared
    def from_raws(
        cls: Type[T_CentralMoments],
        raws,
        mom_ndim: int | None = None,
        mom: Moments | None = None,
        axis: int = 0,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Create object from multipel `raw` moment arrays.

        Parameters
        ----------
        raws : ndarray
            raws[...,i,...] is the ith sample of a `raw` array,
            Note that raw[...,i,j] = <x0**i, x1**j>
            where `i` is in position `axis`
        {axis}
        {mom_ndim}
        {mom}
        {val_shape}
        {dtype}
        {convert_kws}
        **kws
            Extra arguments to :meth:`from_datas`

        See Also
        --------
        from_raw
        from_datas
        ~convert.to_central_moments
        ~convert.to_central_comoments

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.
        """
        pass

    # --------------------------------------------------
    # mom_ndim == 1 specific
    # --------------------------------------------------

    # @staticmethod
    # def _raise_if_not_1d(mom_ndim: int) -> None:
    #     if mom_ndim != 1:
    #         raise NotImplementedError("only available for mom_ndim == 1")

    # special, 1d only methods
    # def push_stat(
    #     self: T_CentralMoments,
    #     a: np.ndarray | float,
    #     v: np.ndarray | float = 0.0,
    #     w: np.ndarray | float | None = None,
    #     broadcast: bool = True,
    # ) -> T_CentralMoments:
    #     """Push statisics onto self."""
    #     self._raise_if_not_1d(self.mom_ndim)

    #     ar, target = self._check_val(a, target="val")
    #     vr = self._check_var(v, broadcast=broadcast)
    #     wr = self._check_weight(w, target=target)
    #     self._push.stat(self._data_flat, wr, ar, vr)
    #     return self

    # def push_stats(
    #     self: T_CentralMoments,
    #     a: np.ndarray,
    #     v: np.ndarray | float = 0.0,
    #     w: np.ndarray | float | None = None,
    #     axis: int = 0,
    #     broadcast: bool = True,
    # ) -> T_CentralMoments:
    #     """Push multiple statistics onto self."""
    #     self._raise_if_not_1d(self.mom_ndim)

    #     ar, target = self._check_vals(a, target="vals", axis=axis)
    #     vr = self._check_vars(v, target=target, axis=axis, broadcast=broadcast)
    #     wr = self._check_weights(w, target=target, axis=axis)
    #     self._push.stats(self._data_flat, wr, ar, vr)
    #     return self

    # @classmethod
    # def from_stat(
    #     cls: Type[T_CentralMoments],
    #     a: ArrayLike | float,
    #     v: np.ndarray | float = 0.0,
    #     w: np.ndarray | float | None = None,
    #     mom: Moments = 2,
    #     val_shape: Tuple[int, ...] | None = None,
    #     dtype: DTypeLike | None = None,
    #     order: ArrayOrder | None = None,
    #     **kws,
    # ) -> T_CentralMoments:
    #     """Create object from single weight, average, variance/covariance."""
    #     mom_ndim = cls._mom_ndim_from_mom(mom)
    #     cls._raise_if_not_1d(mom_ndim)

    #     a = np.asarray(a, dtype=dtype, order=order)

    #     if val_shape is None and isinstance(a, np.ndarray):
    #         val_shape = a.shape
    #     if dtype is None:
    #         dtype = a.dtype

    #     return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_stat(
    #         w=w, a=a, v=v
    #     )

    # @classmethod
    # def from_stats(
    #     cls: Type[T_CentralMoments],
    #     a: np.ndarray,
    #     v: np.ndarray,
    #     w: np.ndarray | float | None = None,
    #     axis: int = 0,
    #     mom: Moments = 2,
    #     val_shape: Tuple[int, ...] = None,
    #     dtype: DTypeLike | None = None,
    #     order: ArrayOrder | None = None,
    #     **kws,
    # ) -> T_CentralMoments:
    #     """Create object from several statistics.

    #     Weights, averages, variances/covarainces along
    #     axis.
    #     """

    #     mom_ndim = cls._mom_ndim_from_mom(mom)
    #     cls._raise_if_not_1d(mom_ndim)

    #     a = np.asarray(a, dtype=dtype, order=order)

    #     # get val_shape
    #     if val_shape is None:
    #         val_shape = _shape_reduce(a.shape, axis)
    #     return cls.zeros(val_shape=val_shape, dtype=dtype, mom=mom, **kws).push_stats(
    #         a=a, v=v, w=w, axis=axis
    #     )

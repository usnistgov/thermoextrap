""""
Central moments/comoments routines
"""
from __future__ import absolute_import

import numpy as np

from .cached_decorators import gcached

from .utils import (
    _axis_expand_broadcast,
    _cached_ones,
    _my_broadcast,
    _shape_insert_axis,
)

from .pushers import factory_pushers
from .resample import resample_data, resample_vals, randsamp_freq
from . import convert

###############################################################################
# central moments/comoments routines
###############################################################################
def central_moments(
    x, moments, weights=None, axis=0, last=True, dtype=None, order=None, out=None
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
    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = _axis_expand_broadcast(
            weights, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    if axis < 0:
        axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        weights = np.moveaxis(weights, axis, 0)

    shape = (moments + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try rolling
            out = np.moveaxis(out, -1, 0)
        assert out.shape == shape

    wsum = weights.sum(axis=0)
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", weights, x) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, moments + 1).reshape(*shape)

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", weights, dx) * wsum_inv

    if last:
        out = np.moveaxis(out, 0, -1)
    return out


def central_comoments(
    x0,
    x1,
    moments,
    weights=None,
    axis=0,
    last=True,
    broadcast=False,
    dtype=None,
    order=None,
    out=None,
):
    """
    calculate central co-moments (covariance, etc) along axis
    """

    if isinstance(moments, int):
        moments = (moments,) * 2

    moments = tuple(moments)
    assert len(moments) == 2

    x0 = np.asarray(x0, dtype=dtype, order=order)
    if dtype is None:
        dtype = x0.dtype

    x1 = _axis_expand_broadcast(
        x1, x0.shape, axis, roll=False, broadcast=broadcast, dtype=dtype, order=order
    )

    if weights is None:
        weights = np.ones_like(x0)
    else:
        weights = _axis_expand_broadcast(
            weights, x0.shape, axis, roll=False, dtype=dtype, order=order
        )

    if axis < 0:
        axis += x.ndim
    if axis != 0:
        x0 = np.moveaxis(x0, axis, 0)
        x1 = np.moveaxis(x1, axis, 0)
        weights = np.moveaxis(weights, axis, 0)

    shape = tuple(x + 1 for x in moments) + x0.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try moving axis
            out = np.moveaxis(out, [-2, -1], [0, 1])
        assert out.shape == shape

    wsum = weights.sum(axis=0)
    wsum_inv = 1.0 / wsum

    x0ave = np.einsum("r...,r...->...", weights, x0) * wsum_inv
    x1ave = np.einsum("r...,r...->...", weights, x1) * wsum_inv

    shape = (-1,) + (1,) * (x0.ndim)
    p0 = np.arange(0, moments[0] + 1).reshape(*shape)
    p1 = np.arange(0, moments[1] + 1).reshape(*shape)

    dx0 = (x0[None, ...] - x0ave) ** p0
    dx1 = (x1[None, ...] - x1ave) ** p1

    out[...] = np.einsum("r...,ir...,jr...->ij...", weights, dx0, dx1) * wsum_inv

    out[0, 0, ...] = wsum
    out[1, 0, ...] = x0ave
    out[0, 1, ...] = x1ave

    if last:
        out = np.moveaxis(out, [0, 1], [-2, -1])
    return out


###############################################################################
# Classes
###############################################################################

class StatsAccumBase(object):
    """
    Base class for moments accumulation
    """

    _moments_len = 1
    __slots__ = (
        "shape",
        "moments",
        "_cache",
        "_data",
        "_data_flat",
        "_push",
    )

    def __init__(self, moments, shape=None, dtype=None, data=None):
        """
        Parameters
        ----------
        moments : int or tuple
        shape : tuple, optional
        dtype : data type
        """

        # check moments
        if isinstance(moments, int):
            moments = (moments,) * self._moments_len
        else:
            moments = tuple(moments)

        if len(moments) != self._moments_len:
            raise ValueError(
                "must supply length {} sequence or int for moments".format(
                    self._moments_len
                )
            )
        for m in moments:
            if m <= 0:
                raise ValueError("must supply values >= 0 for moments")
        self.moments = moments

        # other values
        if dtype is None:
            dtype = np.float
        if shape is None:
            shape = ()
        self.shape = shape

        # data
        if data is None:
            data = np.zeros(self._data_shape, dtype=dtype)
        else:
            if data.shape != self._data_shape:
                raise ValueError(
                    f"passed data shape {data.shape} must equal {self._data_shape}"
                )
        self._data = data
        self._data_flat = self._data.reshape(self._data_flat_shape)

        # setup pushers
        vec = len(self.shape) > 0
        cov = self._moments_len == 2
        self._push = factory_pushers(cov=cov, vec=vec)



    @property
    def dtype(self):
        return self._data.dtype

    # shape attributes
    def _moments_ndim(self):
        """number of dimensions for moments"""
        return len(self.moments)

    @gcached()
    def _moments_shape(self):
        """tuple of shape for moments"""
        return tuple(x + 1 for x in self.moments)

    @gcached()
    def _moments_shape_var(self):
        """shape of variance"""
        return tuple(x - 1 for x in self.moments)

    @gcached()
    def _shape_flat(self):
        if self.shape is ():
            return ()
        else:
            return (np.prod(self.shape),)

    @gcached()
    def _shape_var(self):
        return self.shape + self._moments_shape_var

    @gcached()
    def _shape_flat_var(self):
        return self._shape_flat + self._moments_shape_var

    @gcached()
    def _data_shape(self):
        """shape of data"""
        return self.shape + self._moments_shape

    @gcached()
    def _data_flat_shape(self):
        """shape of conformed data"""
        return self._shape_flat + self._moments_shape

    # def __setstate__(self, state):
    #     self.__dict__ = state
    #     # make sure datar points to data
    #     self._data_flat = self._data.reshape(self._data_flat_shape)

    # useful accessors
    @property
    def values(self):
        return self._data

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return len(self.shape)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    # not sure I want to actually implements this
    # could lead to all sorts of issues applying
    # ufuncs to underlying data
    # def __array_wrap__(self, obj, context=None):
    #     return self, obj, context

    @property
    def _unit_weight(self):
        return _cached_ones(self.shape, dtype=self.dtype)

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
    def _check_moments(cls, moments, shape=None):
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


    # attempt at making checking/verifying more straight forward
    # could make this a bit better
    def _verify_value(self,
                      x,
                      target,
                      axis=None,
                      broadcast=False,
                      expand=False,
                      shape_flat=None,
    ):
        """
        verify input values

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


        if isinstance(target, tuple):
            # target is the target shape
            target_shape = target
            target_output = x

        else:
            # get target_shape from target
            target_shape = target.shape
            target_output = None

        x = self._asarray(x)
        x = _axis_expand_broadcast(x, target_shape, axis,
                                   verify=False,
                                   expand=expand, broadcast=broadcast,
                                   dtype=self.dtype, roll=False)

        # check shape:
        assert x.shape == target_shape


        if axis is not None:
            if axis != 0:
                x = np.moveaxis(x, axis, 0)

        if shape_flat is not None:
            x = x.reshape(shape_flat)

        if x.ndim == 0:
            x = x[()]

        if target_output is None:
            return x, target_output
        else:
            return x

    def check_weight(self, w):#, target):
        if w is None:
            w = self._unit_weight
        else:
            w = _my_broadcast(w, self.shape)
        return self._reshape_flat(w)
        # if w is None:
        #     w = self._unit_weight
        # return self._verify_value(
        #     w,
        #     target=target,
        #     axis=None, broadcast=True,
        #     shape_flat=self._shape_flat

        # )

    def check_weights(self, w, target, axis=0):
        if w is None:
            w = self._unit_weight
            w = w.reshape(_shape_insert_axis(w.shape, axis, 1))

        #shape = _shape_insert_axis(self.shape, axis, nrec)
        w = _axis_expand_broadcast(w, target.shape, axis, dtype=self.dtype)

        assert w.shape == (target.shape[axis],) + self.shape
        return self._reshape_flat(w, w.shape[0])

    def check_val(self, x, broadcast=False):
        x = np.asarray(x, dtype=self.dtype)
        if broadcast:
            x = _my_broadcast(x, self.shape)
        assert x.shape == self.shape
        return self._reshape_flat(x)

    def check_vals(self, x, axis=0, target=None):

        if target is None:
            x = np.asarray(x, dtype=self.dtype)
            if axis != 0:
                x = np.moveaxis(x, axis, 0)
        else:
            x = _axis_expand_broadcast(
                x,
                shape=target.shape,
                axis=axis,
                expand=True,
                broadcast=True,
                roll=True,
                dtype=self.dtype,
            )

        assert x.shape[1:] == self.shape
        return self._reshape_flat(x, x.shape[0])

    def check_ave(self, a):
        return self.check_val(a)

    def check_aves(self, a, axis=0):
        return self.check_vals(a)

    def check_var(self, v):
        v = np.asarray(v, dtype=self.dtype)
        assert v.shape == self._shape_var
        return v.reshape(self._shape_flat_var)

    def check_vars(self, v, target, axis=0):
        v = np.asarray(v, dtype=self.dtype)
        if axis != 0:
            v = np.moveaxis(v, axis, 0)
        assert v.shape == (target.shape[axis],) + self._shape_var
        return v.reshape(v.shape[:1] + self._shape_flat_var)

    def check_data(self, data):
        data = np.asarray(data, dtype=self.dtype)
        assert data.shape == self._data_shape
        return data.reshape(self._data_flat_shape)

    def check_datas(self, datas, axis=0):
        datas = np.asarray(datas, dtype=self.dtype)
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        assert datas.shape[1:] == self._data_shape
        return datas.reshape(datas.shape[:1] + self._data_flat_shape)

    def zero(self):
        self._data.fill(0.0)

    # create similar objects
    def new_like(self, data=None):
        return type(self)(moments=self.moments, shape=self.shape, dtype=self.dtype, data=data)

    def zeros_like(self):
        """create zero object like self"""
        return self.new_like(data=None)


    def copy(self):
        return self.new_like(data=self.values.copy())

    ##################################################
    # indexing routines
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

    def weight(self):
        return self.values[self._weight_index]

    def mean(self):
        return self.values[self._single_index(1)]

    def var(self):
        return self.values[self._single_index(2)]

    def std(self):
        return np.sqrt(self.var(mom=2))

    def cmom(self):
        return self.values

    def _check_other(self, b):
        """check other object"""
        assert type(self) == type(b)
        assert self.shape == b.shape
        assert self.moments == b.moments

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

    # Universal factory methods
    @classmethod
    def from_data(cls, data, moments=None, shape=None, dtype=None, copy=True,
                  verify=True,
                  *args, **kwargs):

        if verify:
            data = np.asarray(data, dtype=dtype)

        if shape is None:
            shape = data.shape[: -cls._moments_len]
        moments = cls._check_moments(moments, data.shape)
        if data.shape != shape + tuple(x + 1 for x in moments):
            raise ValueError(f"{data.shape} does not conform to {shape} and {moments}")

        if dtype is None:
            dtype = data.dtype
        if copy:
            data = data.copy()

        return cls(shape=shape, dtype=dtype, moments=moments, data=data, *args, **kwargs)

    @classmethod
    def from_datas(cls, datas, moments=None, axis=0, shape=None, dtype=None, *args, **kwargs):
        """
        Data should have shape

        [..., moments] (axis!= -1)

        [..., moment, axis] (axis == -1)
        """
    
        datas, axis = cls._datas_axis_to_first(datas, axis)
        if shape is None:
            shape = datas.shape[1 : -cls._moments_len]

        moments = cls._check_moments(moments, datas.shape)
        assert datas.shape[1:] == shape + tuple(x + 1 for x in moments)

        if dtype is None:
            dtype = datas.dtype

        new = cls(shape=shape, dtype=dtype, moments=moments, *args, **kwargs)
        new.push_datas(datas=datas, axis=0)
        return new

    # convert to/from raw moments
    def to_raw(self):
        """
        convert central moments to raw moments
        """
        if self._moments_len == 1:
            func = convert.to_raw_moments
        elif self._moments_len == 2:
            func = convert.to_raw_comoments
        return func(self.data)

    @classmethod
    def from_raw(cls, raw, moments=None, shape=None, dtype=None, *args, **kwargs):
        if cls._moments_len == 1:
            func = convert.to_central_moments
        elif cls._moments_len == 2:
            func = convert.to_central_comoments
        data = func(raw)

        return cls.from_data(
            data, moments=moments, shape=shape, dtype=dtype, copy=False,
            *args, **kwargs
        )

    @classmethod
    def from_raws(cls, raws, moments=None, axis=0, shape=None, dtype=None, *args, **kwargs):
        if cls._moments_len == 1:
            func = convert.to_central_moments
        elif cls._moments_len == 2:
            func = convert.to_central_comoments
        datas = func(raws)
        return cls.from_datas(
            datas, axis=axis, moments=moments, shape=shape, dtype=dtype, copy=False,
            *args, **kwargs
        )

    # Universal reducers
    def resample_and_reduce(self, freq, axis=None, resample_kws=None, *args, **kwargs):
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        if resample_kws is None:
            resample_kws = {}

        data = resample_data(self.data, freq, moments=self.moments, axis=axis,
                             **resample_kws)
        return type(self).from_data(data, moments=self.moments, copy=False, *args, **kwargs)

    def reduce(self, axis=0, *args, **kwargs):
        """
        create new object reducealong axis
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        return type(self).from_datas(self.data, axis=axis, moments=self.moments, *args, **kwargs)

    def resample(self, indices, axis=0, roll=True, *args, **kwargs):
        """
        create a new object sampled from index

        Parameters
        ----------
        indicies : array-like
        axis : int, default=0
            axis to resample
        roll : bool, default=True
            roll axis sampling along to the first dimensions
            This makes results similar to resample and reduce
            With roll False, then resampled array can have odd shape

        Returns
        -------
        output : accumulator object
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)

        data = self.data
        if roll and axis != 0:
            data = np.moveaxis(data, axis, 0)
            axis = 0

        out = np.take(data, indices, axis=axis)
        return type(self).from_data(out, moments=self.moments, copy=False, *args, **kwargs)

    def reshape(self, shape, copy=True, *args, **kwargs):
        """
        create a new object with reshaped data
        """
        self._raise_if_scalar()
        new_shape = shape + self._moments_shape
        data = self._data.reshape(new_shape)
        return type(self).from_data(data, moments=self.moments, copy=copy, *args, **kwargs)

    def moveaxis(self, source, destination, copy=True, *args, **kwargs):
        """
        move axis from source to destination
        """
        self._raise_if_scalar()

        def _check_val(v):
            if isinstance(v, int):
                v = (v,)
            else:
                v = tuple(v)
            return tuple(self._wrap_axis(x) for x in v)

        source = _check_val(source)
        destination = _check_val(destination)
        data = np.moveaxis(self.data, source, destination)
        return type(self).from_data(data, moments=self.moments, copy=copy, *args, **kwargs)



class _StatsAccumMixin(object):
    def push_val(self, x, w=None):
        xr = self.check_val(x)
        wr = self.check_weight(w)
        self._push.val(self._data_flat, wr, xr)

    def push_vals(self, x, w=None, axis=0):
        xr = self.check_vals(x, axis)
        wr = self.check_weights(w, x, axis)
        self._push.vals(self._data_flat, wr, xr)

    def push_stat(self, a, v=0.0, w=None):
        ar = self.check_ave(a)
        vr = self.check_var(v)
        wr = self.check_weight(w)
        self._push.stat(self._data_flat, wr, ar, vr)

    def push_stats(self, a, v=0.0, w=None, axis=0):
        ar = self.check_aves(a, axis)
        vr = self.check_vars(v, a, axis)
        wr = self.check_weights(w, a, axis)
        self._push.stats(self._data_flat, wr, ar, vr)

    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_vals(cls, x, w=None, axis=0, moments=2, shape=None, dtype=None, *args, **kwargs):
        # get shape
        if shape is None:
            shape = list(x.shape)
            shape.pop(axis)
            shape = tuple(shape)
        if dtype is None:
            dtype = x.dtype
        new = cls(shape=shape, dtype=dtype, moments=moments, *args, **kwargs)
        new.push_vals(x, axis=axis, w=w)
        return new

    @classmethod
    def from_stat(cls, a, v=0.0, w=None, moments=2, shape=None, dtype=None, *args, **kwargs):
        """
        object from single weight, average, variance/covariance
        """
        if shape is None:
            shape = a.shape
        new = cls(shape=shape, moments=moments, dtype=dtype, *args, **kwargs)
        new.push_stat(w=w, a=a, v=v)
        return new

    @classmethod
    def from_stats(cls, a, v=0.0, w=None, axis=0, moments=2, shape=None, dtype=None, *args, **kwargs):
        """
        object from several weights, averages, variances/covarainces along axis
        """

        # get shape
        if shape is None:
            shape = list(A.shape)
            shape.pop(axis)
            shape = tuple(shape)

        new = cls(shape=shape, dtype=dtype, moments=moments, *args, **kwargs)
        new.push_stats(a=a, v=v, w=w, axis=axis)
        return new

    @classmethod
    def from_resample_vals(
            cls, x, freq, w=None, axis=0, moments=2, dtype=None, resample_kws=None,
            *args, **kwargs
    ):
        if isinstance(moments, int):
            moments = (moments,)
        if resample_kws is None:
            resample_kws = {}

        data = resample_vals(
            x=x, freq=freq, moments=moments, axis=axis, weights=w, **resample_kws
        )
        return cls.from_data(data, copy=False, *args, **kwargs)


class StatsAccum(StatsAccumBase, _StatsAccumMixin):
    _moments_len = 1


class _StatsAccumCovMixin(object):
    def push_val(self, x0, x1, w=None, broadcast=False):
        x0 = self.check_val(x0)
        x1 = self.check_val(x1, broadcast=broadcast)
        w = self.check_weight(w)
        self._push.val(self._data_flat, w, x0, x1)

    def push_vals(self, x0, x1, w=None, axis=0, broadcast=False):
        if broadcast:
            target = x0
        else:
            target = None

        w = self.check_weights(w, x0, axis)
        x0 = self.check_vals(x0, axis)
        x1 = self.check_vals(x1, axis, target)

        self._push.vals(self._data_flat, w, x0, x1)

    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_vals(
            cls, x0, x1, w=None, axis=0, shape=None, broadcast=False, moments=2, dtype=None, *args, **kwargs

    ):

        # get shape
        if shape is None:
            shape = list(x0.shape)
            shape.pop(axis)
            shape = tuple(shape)
        if dtype is None:
            dtype = x0.dtype

        new = cls(shape=shape, dtype=dtype, moments=moments, *args, **kwargs)
        new.push_vals(x0=x0, x1=x1, axis=axis, w=w, broadcast=broadcast)
        return new

    @classmethod
    def from_resample_vals(
        cls,
        x0,
        x1,
        freq,
        w=None,
        axis=0,
        dtype=None,
        broadcast=False,
        moments=2,
        resample_kws=None,
        *args, **kwargs
    ):
        if isinstance(moments, int):
            moments = (moments,) * cls._moments_len

        if resample_kws is None:
            resample_kws = {}

        data = resample_vals(
            x=x0,
            x1=x1,
            freq=freq,
            moments=moments,
            broadcast=broadcast,
            axis=axis,
            weights=w,
            **resample_kws,
        )
        return cls.from_data(data, copy=False, *args, **kwargs)



class StatsAccumCov(StatsAccumBase, _StatsAccumCovMixin):
    _moments_len = 2


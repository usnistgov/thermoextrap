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

from .pushers import factory_pushers_dict
from .resample import (resample_data, resample_vals, randsamp_freq)
from . import convert

###############################################################################
# central moments/comoments routines
###############################################################################
def central_moments(x, moments, weights=None, axis=0, last=True, out=None):
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
    x = np.array(x)
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = _axis_expand_broadcast(weights, x.shape, axis, roll=False)

    if axis < 0:
        axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        weights = np.moveaxis(weights, axis, 0)

    shape = (moments + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=x.dtype)
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
    x0, x1, moments, weights=None, axis=0, last=True, broadcast=False, out=None
):
    """
    calculate central co-moments (covariance, etc) along axis
    """

    if isinstance(moments, int):
        moments = (moments,) * 2

    moments = tuple(moments)
    assert len(moments) == 2

    x0 = np.array(x0)
    x1 = _axis_expand_broadcast(x1, x0.shape, axis, roll=False, broadcast=broadcast)
    if weights is None:
        weights = np.ones_like(x0)
    else:
        weights = _axis_expand_broadcast(weights, x0.shape, axis, roll=False)

    if axis < 0:
        axis += x.ndim
    if axis != 0:
        x0 = np.moveaxis(x0, axis, 0)
        x1 = np.moveaxis(x1, axis, 0)
        weights = np.moveaxis(weights, axis, 0)

    shape = tuple(x + 1 for x in moments) + x0.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=x0.dtype)
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
    _moments_len = None


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
        self._init_subclass()

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

    def _init_subclass(self):
        vec = len(self.shape) > 0
        cov = self._moments_len == 2
        pushers = factory_pushers_dict(cov=cov, vec=vec)
        for k, v in pushers.items():
            setattr(self, '_' + k, v)

    @property
    def dtype(self):
        return self._data.dtype

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

    def __setstate__(self, state):
        self.__dict__ = state
        # make sure datar points to data
        self._data_flat = self._data.reshape(self._data_flat_shape)

    @property
    def values(self):
        return self._data

    @property
    def data(self):
        return self._data

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def _unit_weight(self):
        # return np.ones(self.shape, dtype=self.dtype)
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
        datas = np.array(datas)
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

    def _reshape_flat(self, x, nrec=None):
        if nrec is None:
            x = x.reshape(self._shape_flat)
        else:
            x = x.reshape(*((nrec,) + self._shape_flat))
        if x.ndim == 0:
            x = x[()]
        return x

    def check_weight(self, w):
        if w is None:
            w = self._unit_weight
        else:
            w = _my_broadcast(w, self.shape)
        return self._reshape_flat(w)

    def check_weights(self, w, nrec, axis=0):
        if w is None:
            w = self._unit_weight
            w = w.reshape(_shape_insert_axis(w.shape, axis, 1))

        shape = _shape_insert_axis(self.shape, axis, nrec)
        w = _axis_expand_broadcast(w, shape, axis)
        return self._reshape_flat(w, nrec)

    def check_val(self, x, broadcast=False):
        x = np.array(x)
        if broadcast:
            x = np.broadcast_to(x.shape)
        assert x.shape == self.shape
        return self._reshape_flat(x)

    def check_vals(self, x, axis=0, target_shape=None):

        if target_shape is None:
            if axis != 0:
                x = np.moveaxis(x, axis, 0)
        else:
            x = _axis_expand_broadcast(
                x, shape=target_shape, axis=axis, expand=True, broadcast=True, roll=True
            )

        assert x.shape[1:] == self.shape
        return self._reshape_flat(x, x.shape[0])

    def check_ave(self, a):
        return self.check_val(a)

    def check_aves(self, a, axis=0):
        return self.check_vals(a)

    def check_var(self, v):
        v = np.array(v)
        assert v.shape == self._shape_var
        return v.reshape(self._shape_flat_var)

    def check_vars(self, v, axis=0):
        v = np.array(v)
        if axis != 0:
            v = np.moveaxis(v, axis, 0)
        assert v.shape[1:] == self._shape_var
        return v.reshape(v.shape[:1] + self._shape_flat_var)

    def check_data(self, data):
        data = np.array(data)
        assert data.shape == self._data_shape
        return data.reshape(self._data_flat_shape)

    def check_datas(self, datas, axis=0):
        datas = np.array(datas)
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        assert datas.shape[1:] == self._data_shape
        return datas.reshape(datas.shape[:1] + self._data_flat_shape)

    def zero(self):
        self._data.fill(0.0)

    # create similar objects
    def zeros_like(self):
        """create zero object like self"""
        return self.__class__(moments=self.moments, shape=self.shape, dtype=self.dtype)

    def new_like(self):
        return self.zeros_like()

    def copy(self):
        new = self.__class__(
            shape=self.shape,
            dtype=self.dtype,
            moments=self.moments,
            data=self._data.copy(),
        )
        return new

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
        return self._data[self._weight_index]

    def mean(self):
        return self._data[self._single_index(1)]

    def var(self):
        return self._data[self._single_index(2)]

    def std(self):
        return np.sqrt(self.var(mom=2))

    def cmom(self):
        return self._data

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
        self._push_data(self._data_flat, data)

    def push_datas(self, datas, axis=0):
        datas = self.check_datas(datas, axis)
        self._push_datas(self._data_flat, datas)

    # Universal factory methods
    @classmethod
    def from_data(cls, data, moments=None, shape=None, dtype=None, copy=True):
        data = np.array(data)
        if shape is None:
            shape = data.shape[: -cls._moments_len]
        moments = cls._check_moments(moments, data.shape)
        if data.shape != shape + tuple(x + 1 for x in moments):
            raise ValueError(f"{data.shape} does not conform to {shape} and {moments}")

        if dtype is None:
            dtype = data.dtype
        if copy:
            data = data.copy()

        return cls(shape=shape, dtype=dtype, moments=moments, data=data)

    @classmethod
    def from_datas(cls, datas, moments=None, axis=0, shape=None, dtype=None):
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

        new = cls(shape=shape, dtype=dtype, moments=moments)
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
    def from_raw(cls, raw, moments=None, shape=None, dtype=None):
        if cls._moments_len == 1:
            func = convert.to_central_moments
        elif cls._moments_len == 2:
            func = convert.to_central_comoments
        data = func(raw)

        return cls.from_data(data, moments=moments, shape=shape, dtype=dtype, copy=False)


    @classmethod
    def from_raws(cls, raws, moments=None, axis=0, shape=None, dtype=None):
        if cls._moments_len == 1:
            func = convert.to_central_moments
        elif cls._moments_len == 2:
            func = convert.to_central_comoments
        datas = func(raws)
        return cls.from_datas(datas, axis=axis, moments=moments, shape=shape, dtype=dtype, copy=False)




    # Universal reducers
    def resample_and_reduce(self, freq, axis=None, **kwargs):
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        data = resample_data(self.data, freq, moments=self.moments, axis=axis, **kwargs)
        return self.__class__.from_data(data, moments=self.moments, copy=False)

    def reduce(self, axis=0):
        """
        create new object reducealong axis
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        return self.__class__.from_datas(self.data, axis=axis, moments=self.moments)

    def resample(self, indices, axis=0, roll=True):
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
        return self.__class__.from_data(out, moments=self.moments, copy=False)

    def reshape(self, shape, copy=True):
        """
        create a new object with reshaped data
        """
        self._raise_if_scalar()
        new_shape = shape + self._moments_shape
        data = self.data.reshape(new_shape)
        return self.__class__.from_data(data, moments=self.moments, copy=copy)

    def moveaxis(self, source, destination, copy=True):
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
        return self.__class__.from_data(data, moments=self.moments, copy=copy)


class StatsAccum(StatsAccumBase):
    _moments_len = 1


    def push_val(self, x, w=None):
        xr = self.check_val(x)
        wr = self.check_weight(w)
        self._push_val(self._data_flat, wr, xr)

    def push_vals(self, x, w=None, axis=0):
        xr = self.check_vals(x, axis)
        wr = self.check_weights(w, xr.shape[0], axis)
        self._push_vals(self._data_flat, wr, xr)

    def push_stat(self, a, v=0.0, w=None):
        ar = self.check_ave(a)
        vr = self.check_var(v)
        wr = self.check_weight(w)
        self._push_stat(self._data_flat, wr, ar, vr)

    def push_stats(self, a, v=0.0, w=None, axis=0):
        ar = self.check_aves(a, axis)
        vr = self.check_vars(v, axis)
        wr = self.check_weights(w, ar.shape[0], axis)
        self._push_stats(self._data_flat, wr, ar, vr)

    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_vals(cls, x, w=None, axis=0, moments=2, shape=None, dtype=None):
        # get shape
        if shape is None:
            shape = list(x.shape)
            shape.pop(axis)
            shape = tuple(shape)
        if dtype is None:
            dtype = x.dtype
        new = cls(shape=shape, dtype=dtype, moments=moments)
        new.push_vals(x, axis=axis, w=w)
        return new

    @classmethod
    def from_stat(cls, a, v=0.0, w=None, moments=2, shape=None, dtype=None):
        """
        object from single weight, average, variance/covariance
        """
        if shape is None:
            shape = a.shape
        new = cls(shape=shape, moments=moments, dtype=dtype)
        new.push_stat(w=w, a=a, v=v)
        return new

    @classmethod
    def from_stats(cls, a, v=0.0, w=None, axis=0, moments=2, shape=None, dtype=None):
        """
        object from several weights, averages, variances/covarainces along axis
        """

        # get shape
        if shape is None:
            shape = list(A.shape)
            shape.pop(axis)
            shape = tuple(shape)

        if dtype is None:
            a = np.array(a)
            dtype = a.dtype

        new = cls(shape=shape, dtype=A.dtype, moments=moments)
        new.push_stats(a=a, v=v, w=w, axis=axis)
        return new

    @classmethod
    def from_resample_vals(
        cls, x, freq, w=None, axis=0, moments=2, dtype=None, **kwargs
    ):
        if isinstance(moments, int):
            moments = (moments,)
        data = resample_vals(
            x=x, freq=freq, moments=moments, axis=axis, weights=w, **kwargs
        )
        return cls.from_data(data, copy=False)


class StatsAccumCov(StatsAccumBase):
    _moments_len = 2

    # def _init_subclass(self):
    #     if self.shape == ():
    #         self._push_val = _push_val_cov
    #         self._push_vals = _push_vals_cov
    #         self._push_data = _push_data_cov
    #         self._push_datas = _push_datas_cov
    #     else:
    #         self._push_val = _push_val_cov_vec
    #         self._push_vals = _push_vals_cov_vec
    #         self._push_data = _push_data_cov_vec
    #         self._push_datas = _push_datas_cov_vec

    def push_val(self, x0, x1, w=None, broadcast=False):
        x0 = self.check_val(x0)
        x1 = self.check_val(x1, broadcast=broadcast)
        w = self.check_weight(w)
        self._push_val(self._data_flat, w, x0, x1)

    def push_vals(self, x0, x1, w=None, axis=0, broadcast=False):
        if broadcast:
            target_shape = x0.shape
        else:
            target_shape = None

        x0 = self.check_vals(x0, axis)
        x1 = self.check_vals(x1, axis, target_shape)
        w = self.check_weights(w, x0.shape[0], axis)
        self._push_vals(self._data_flat, w, x0, x1)

    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_vals(
        cls, x0, x1, w=None, axis=0, shape=None, broadcast=False, moments=2, dtype=None
    ):

        # get shape
        if shape is None:
            shape = list(x0.shape)
            shape.pop(axis)
            shape = tuple(shape)
        if dtype is None:
            dtype = x0.dtype

        new = cls(shape=shape, dtype=dtype, moments=moments)
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
        **kwargs,
    ):
        if isinstance(moments, int):
            moments = (moments,) * cls._moments_len

        data = resample_vals(
            x=x0,
            x1=x1,
            freq=freq,
            moments=moments,
            broadcast=broadcast,
            axis=axis,
            weights=w,
            **kwargs,
        )
        return cls.from_data(data, copy=False)



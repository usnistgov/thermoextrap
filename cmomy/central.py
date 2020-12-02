"""
Central moments/comoments routines
"""
from __future__ import absolute_import

import numpy as np
from .cached_decorators import gcached

from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple


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
# central mom/comoments routines
###############################################################################
def _central_moments(
    x, mom, w=None, axis=0, last=True, dtype=None, order=None, out=None
):
    """calculate central mom along axis"""

    if isinstance(mom, tuple):
        mom = mom[0]

    x = np.asarray(x, dtype=dtype, order=order)
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
    x,
    mom,
    w=None,
    axis=0,
    last=True,
    broadcast=False,
    dtype=None,
    order=None,
    out=None,
):
    """calculate central co-mom (covariance, etc) along axis"""

    if isinstance(mom, int):
        mom = (mom,) * 2

    mom = tuple(mom)
    assert len(mom) == 2

    # change x to tuple of inputs
    assert isinstance(x, tuple) and len(x) == 2
    x, y = x

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
    x, mom, w=None, axis=0, last=True, dtype=None, order=None, out=None, broadcast=False
):
    """
    calculate central moments or comoments along axis

    Parameters
    ----------
    x : array-like or tuple of array-like
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
        i.e., shape=x.shape[:axis] + x.shape[axis+1:], and `mom_shape` is the shape of the
        moment part, either (mom+1,) or (mom0+1, mom1+1).  Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment.
    """

    if isinstance(mom, int):
        mom = (mom,)

    kws = dict(
        x=x, mom=mom, w=w, axis=axis, last=last, dtype=dtype, order=order, out=out
    )
    if len(mom) == 1:
        func = _central_moments
    else:
        func = _central_comoments
        kws["broadcast"] = broadcast

    return func(**kws)


###############################################################################
# Classes
###############################################################################
class StatsAccum(object):
    """
    Base class for moments accumulation
    """

    __slots__ = (
        "_mom_len",
        "_cache",
        "_data",
        "_data_flat",
        "_push",
    )

    def __init__(self, data, mom_len=1, **kws):
        """
        Parameters
        ----------
        data : numpy data array
            data should have the shape `shape_val + shape_mom`
            where `shape_val` is the shape of a single observation,
            and shape_mom is the shape of the moments
        **kws : dict
            optional arguments to be used in subclasses
        """

        if mom_len not in (1, 2):
            raise ValueError(
                "mom_len must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )
        self._mom_len = mom_len

        if data.ndim < self.mom_len:
            raise ValueError("not enough dimensions in data")

        self._data = data

        # check moments make sense
        if not all([x > 0 for x in self.mom]):
            raise ValueError("All moments must positive")

        self._data_flat = self._data.reshape(self.shape_tot_flat)

        # setup pushers
        vec = len(self.shape) > 0
        cov = self.mom_len == 2
        self._push = factory_pushers(cov=cov, vec=vec)

    @property
    def values(self):
        """accessor to underlying central moments"""
        return self._data

    @property
    def data(self):
        """accessor to numpy/dask underlying data

        By convention data has the following meaning for the moments indexes

        * `data[i_0=0,... i_n=0]`, if all moment indices are zero, this is the sum of weights
        * `data[i_0=0,... i_k=1, ... i_n=0]`, if only one moment indice is one and all
        others zero, then this is the average value of the variable with unit index.

        * all other cases, the central moments `<(x0-<x0>)**i0 * (x1 - <x1>)**i1 * ...>`
        """
        return self._data

    @property
    def mom_len(self):
        """length of moment part
        if `mom_len` == 1, then single variable moments
        if `mom_len` == 2, then co-moments
        """
        return self._mom_len

    @property
    def shape(self):
        """shape, less moments dimensions"""
        return self._data.shape[: -self.mom_len]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape_val(self):
        """shape of values part"""
        return self.shape

    @property
    def shape_mom(self):
        """shape of moments part"""
        return self._data.shape[-self.mom_len :]

    @property
    def shape_val_flat(self):
        """shape of values part flattened"""
        if self.shape == ():
            return ()
        else:
            return (np.prod(self.shape),)

    @property
    def shape_tot(self):
        """Shape of vals + moment axis.
        That is, shape of self.data
        """
        return self._data.shape

    @property
    def shape_tot_flat(self):
        return self.shape_val_flat + self.shape_mom

    @property
    def shape_mom_var(self):
        """shape of moment part of variance"""
        return tuple(x - 1 for x in self.mom)

    # variance shape
    @property
    def shape_tot_var(self):
        """total variance shape"""
        return self.shape + self.shape_mom_var

    @property
    def shape_tot_flat_var(self):
        return self.shape_val_flat + self.shape_mom_var

    @property
    def mom(self):
        """number of moments for each variable"""
        return tuple(x - 1 for x in self.shape_mom)

    @property
    def dtype(self):
        return self._data.dtype

    # I think this is for pickling
    # probably don't need it anymore
    # def __setstate__(self, state):
    #     self.__dict__ = state
    #     # make sure datar points to data
    #     self._data_flat = self._data.reshape(self.shape_tot_flat)

    # not sure I want to actually implements this
    # could lead to all sorts of issues applying
    # ufuncs to underlying data
    # def __array_wrap__(self, obj, context=None):
    #     return self, obj, context

    def __repr__(self):
        s = "<StatsAccum(shape={}, mom={})>\n".format(self.shape, self.mom)
        return s + repr(self.values)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    ###########################################################################
    # SECTION: top level creation/copy/new
    ###########################################################################
    def new_like(
        self, data=None, verify=False, check=False, copy=False, copy_kws=None, **kws
    ):
        """create new object like self, with new data

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
            arguments to `type(self).__init__`
        """

        if data is None:
            data = np.zeros_like(self._data)
        else:
            if verify:
                data = np.asarray(data, dtype=self.dtype)

            if check:
                assert data.shape == self.shape_tot

            if copy:
                if copy_kws is None:
                    copy_kws = {}
                data = data.copy(**copy_kws)
        return type(self)(data=data, mom_len=self.mom_len, **kws)

    def zeros_like(self, zeros_kws=None, **kws):
        """create new object empty object like self"""
        if zeros_kws is None:
            zeros_kws = {}
        return self.new_like(data=np.zeros_like(self._data, **zeros_kws), **kws)

    def copy(self, **copy_kws):
        """create a new object with copy of data"""
        return self.new_like(
            data=self.values, verify=False, check=False, copy=True, copy_kws=copy_kws
        )

    @classmethod
    def zeros(
        cls,
        mom=None,
        shape=None,
        mom_len=None,
        shape_tot=None,
        dtype=None,
        zeros_kws=None,
        **kws,
    ):
        """create a new base object

        Parameters
        ----------
        mom : int or tuple
            moments.
            if integer, or length one tuple, then moments of single variable.
            if tuple of length 2, then comoments of two variables.
        shape : tuple, optional
            shape of values, excluding moments.  For example, if considering the average of
            observations `x`, then shape = x.shape.  
            if not passed, then assume shape = ()
        shape_tot : tuple, optional
            if passed, create object with this total shape
        mom_len : int {1, 2}, optional
            number of variables.  
            if pass `shape_tot`, then must pass mom_len
        dtype : nunpy dtype, default=float
        zeros_kws : dict
        extra arguments to `np.zeros`
        kws : dict
            extra arguments to `__init__`

        Returns
        -------
        object : instance of class `cls`

        Notes
        -----
        the resulting total shape of data is shape + (mom + 1)
        """

        if zeros_kws is None:
            zeros_kws = {}

        if shape_tot is None:
            assert mom is not None
            if isinstance(mom, int):
                mom = (mom,)
            if mom_len is None:
                mom_len = len(mom)
            assert len(mom) == mom_len

            if shape is None:
                shape = ()
            elif isinstance(shape, int):
                shape = (shape,)
            shape_tot = shape + tuple(x + 1 for x in mom)

        else:
            assert mom_len is not None

        if dtype is None:
            dtype = np.float

        if zeros_kws is None:
            zeros_kws = {}
        data = np.zeros(shape=shape_tot, dtype=dtype, **zeros_kws)

        return cls(data=data, mom_len=mom_len, **kws)

    ###########################################################################
    # SECTION: Access to underlying statistics
    ###########################################################################
    @gcached()
    def _weight_index(self):
        index = [0] * len(self.mom)
        if self.ndim > 0:
            index = [...] + index
        return tuple(index)

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
        return np.sqrt(self.var())

    def cmom(self):
        """central moments
        `cmom[i0, i1, ...] = < (x0 - <x0>)**i0 * (x1 - <x1>)**i1 ...>`
        """
        out = self.data.copy()
        # zeroth central moment
        out[self._weight_index] = 1
        # first central moment
        out[self._single_index(1)] = 0
        return out

    # convert to/from raw moments
    def to_raw(self):
        """convert central moments to raw moments

        out[...,0,0] = weight
        out[...,i0,i1] =  <x0**i0 * x1**i1 * ...>
        """
        if self.mom_len == 1:
            func = convert.to_raw_moments
        elif self.mom_len == 2:
            func = convert.to_raw_comoments
        return func(self.data)

    def rmom(self):
        """raw moments"""
        out = self.to_raw()
        out[self._weight_index] = 1
        return out

    ###########################################################################
    # SECTION: pushing routines
    ###########################################################################
    def _asarray(self, val):
        return np.asarray(val, dtype=self.dtype)

    # @property
    # def _unit_weight(self):
    #     """internally cached unit weight"""
    #     return _cached_ones(self.shape, dtype=self.dtype)

    def _verify_value(
        self,
        x,
        target=None,
        axis=None,
        broadcast=False,
        expand=False,
        shape_flat=None,
        other=None,
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

        x = self._asarray(x)

        if isinstance(target, str):
            if target == "val":
                target = self.shape
            elif target == "vals":
                target = _shape_insert_axis(self.shape, axis, x.shape[axis])
            elif target == "data":
                target = self.shape_tot
            elif target == "datas":
                # make sure axis in limits
                axis = normalize_axis_index(axis, self.ndim+1)
                # if axis < 0:
                #     axis += self.ndim - self.mom_len
                target = _shape_insert_axis(self.shape_tot, axis, x.shape[axis])
            elif target == "var":
                target = self.shape_tot_var
            elif target == "vars":
                target = _shape_insert_axis(self.shape_tot_var, axis, other.shape[axis])

        if isinstance(target, tuple):
            target_shape = target
            target_output = x

        else:
            target_shape = target.shape
            target_output = None

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
            nrec = (x.shape[0],)
        else:
            nrec = ()

        if shape_flat is not None:
            x = x.reshape(nrec + shape_flat)

        if x.ndim == 0:
            x = x[()]

        if target_output is None:
            return x
        else:
            return x, target_output

    def check_weight(self, w, target):
        if w is None:
            w = 1.0
        return self._verify_value(
            w,
            target=target,
            axis=None,
            broadcast=True,
            expand=True,
            shape_flat=self.shape_val_flat,
        )

    def check_weights(self, w, target, axis=0):
        if w is None:
            w = 1.0
        return self._verify_value(
            w,
            target=target,
            axis=axis,
            broadcast=True,
            expand=True,
            shape_flat=self.shape_val_flat,
        )

    def check_val(self, x, target, broadcast=False):
        return self._verify_value(
            x,
            target=target,
            broadcast=broadcast,
            expand=False,
            shape_flat=self.shape_val_flat,
        )

    def check_vals(self, x, target, axis=0, broadcast=False):
        return self._verify_value(
            x,
            target=target,
            axis=axis,
            broadcast=broadcast,
            expand=broadcast,
            shape_flat=self.shape_val_flat,
        )

    def check_var(self, v, broadcast=False):
        return self._verify_value(
            v,
            target="var",  # self.shape_tot_var,
            broadcast=broadcast,
            expand=False,
            shape_flat=self.shape_tot_flat_var,
        )[0]

    def check_vars(self, v, target, axis=0, broadcast=False):
        return self._verify_value(
            v,
            target="vars",
            axis=axis,
            broadcast=broadcast,
            expand=broadcast,
            shape_flat=self.shape_tot_flat_var,
            other=target,
        )[0]

    def check_data(self, data):
        return self._verify_value(
            data, target="data", shape_flat=self.shape_tot_flat)[0]

    def check_datas(self, datas, axis=0):
        return self._verify_value(
            datas, target="datas", axis=axis, shape_flat=self.shape_tot_flat
        )[0]

    def fill(self, value=0):
        """fill data with value"""
        self._data.fill(value)
        return self

    def zero(self):
        """zero out underlying data"""
        return self.fill(value=0)

    def push_data(self, data):
        """push `data` to moments

        Parameters
        ----------
        data : array-like, shape=self.shape_tot
            array storing moment information
        Returns
        -------
        self

        See Also
        --------
        `self.data`
        """
        data = self.check_data(data)
        self._push.data(self._data_flat, data)
        return self

    def push_datas(self, datas, axis=0):
        """push and reduce multiple average central moments

        Parameters
        ----------
        datas : array-like
            this should have shape like `(nrec,) + self.shape_tot`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        axis : int, default=0
            axis to reduce along

        Returns
        -------
        self
        """
        datas = self.check_datas(datas, axis)
        self._push.datas(self._data_flat, datas)
        return self

    def push_val(self, x, w=None, broadcast=False):
        """dd single sample to central moments

        Parameters
        ----------
        x : array-like or tuple of arrays
            if `self.mom_len` == 1, then this is the value to consider
            if `self.mom_len` == 2, then x = (x0, x1)
            `x.shape == self.shape`

        w : int, float, array-like, optional
            optional weight of each sample
        broadcast : bool, default = False
            If true, do smart broadcasting for `x[1:]`

        Returns
        -------
        self
        """

        if self.mom_len == 1:
            ys = ()
        else:
            assert len(x) == self.mom_len
            x, *ys = x

        xr, target = self.check_val(x, "val")
        yr = tuple(self.check_val(y, target=target, broadcast=broadcast) for y in ys)
        wr = self.check_weight(w, target)
        self._push.val(self._data_flat, *((wr, xr) + yr))
        return self

    def push_vals(self, x, w=None, axis=0, broadcast=False):
        """
        add multiple samples to central moments

        Parameters
        ----------
        x : array-like or tuple of arrays
            if `self.mom_len` == 1, then this is the value to consider
            if `self.mom_len` == 2, then x = (x0, x1)
            `x.shape[:axis] + x.shape[axis+1:] == self.shape`

        w : int, float, array-like, optional
            optional weight of each sample
        axis : int, default=0
            axis to reduce along
        broadcast : bool, default = False
            If true, do smart broadcasting for `x[1:]`
        """
        if self.mom_len == 1:
            ys = ()
        else:
            assert len(x) == self.mom_len
            x, *ys = x

        xr, target = self.check_vals(x, axis=axis, target="vals")
        yr = tuple(
            self.check_vals(y, target=target, axis=axis, broadcast=broadcast)
            for y in ys
        )
        wr = self.check_weights(w, target=target, axis=axis)
        self._push.vals(self._data_flat, *((wr, xr) + yr))
        return self

    ###########################################################################
    # SECTION: Operators
    ###########################################################################
    def _check_other(self, b):
        """check other object"""
        assert type(self) == type(b)
        assert self.mom_len == b.mom_len
        assert self.shape_tot == b.shape_tot

    def __iadd__(self, b):
        self._check_other(b)
        # self.push_data(b.data)
        # return self
        return self.push_data(b.data)


    def __add__(self, b):
        self._check_other(b)
        # new = self.copy()
        # new.push_data(b.data)
        # return new
        return self.copy().push_data(b.data)

    def __isub__(self, b):
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        data = b.data.copy()
        data[self._weight_index] *= -1
        # self.push_data(data)
        # return self
        return self.push_data(data)

    def __sub__(self, b):
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        new = b.copy()
        new._data[self._weight_index] *= -1
        # new.push_data(self.data)
        # return new
        return new.push_data(self.data)

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

    ###########################################################################
    # SECTION: Constructors
    ###########################################################################
    @classmethod
    def _check_mom(cls, moments, mom_len, shape=None):
        """check moments for correct shape
        if moments is None, infer from shape[-mom_len:]
        if integer, convert to tuple
        """

        if moments is None:
            if shape is not None:
                if mom_len is None:
                    raise ValueError("must speficy either moments or shape and mom_len")
                moments = tuple(x - 1 for x in shape[-mom_len:])
            else:
                raise ValueError("must specify moments")

        if isinstance(moments, int):
            if mom_len is None:
                mom_len = 1
            moments = (moments,) * mom_len

        else:
            moments = tuple(moments)
            if mom_len is None:
                mom_len = len(moments)

        assert len(moments) == mom_len
        return moments

    @staticmethod
    def _datas_axis_to_first(datas, axis, mom_len):
        """move axis to first first position"""
        # NOTE: removinvg this. should be handles elsewhere
        # datas = np.asarray(datas)
        # ndim = datas.ndim - mom_len
        # if axis < 0:
        #     axis += ndim
        # assert 0 <= axis < ndim
        axis = normalize_axis_index(axis, datas.ndim - mom_len)
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        return datas, axis

    def _wrap_axis(self, axis, default=0, ndim=None):
        """wrap axis to positive value and check"""
        if axis is None:
            axis = default
        if ndim is None:
            ndim = self.ndim

        axis = normalize_axis_index(axis, ndim)
        # if axis < 0:
        #     axis += ndim
        # assert 0 <= axis < ndim
        return axis

    @classmethod
    def _mom_len_from_mom(cls, mom):
        if isinstance(mom, int):
            return 1
        elif isinstance(mom, tuple):
            return len(mom)
        else:
            raise ValueError("mom must be int or tuple")

    @classmethod
    def _choose_mom_len(cls, mom, mom_len):
        if mom is not None:
            mom_len = cls._mom_len_from_mom(mom)

        if mom_len is None:
            raise ValueError("must specify mom_len or mom")

        return mom_len

    @classmethod
    def from_data(
        cls,
        data,
        mom_len=None,
        mom=None,
        shape=None,
        copy=True,
        copy_kws=None,
        verify=True,
        dtype=None,
        **kws,
    ):
        """
        create new object with additional checks

        If pass `mom` and `shape`, make sure data conforms to this

        must pass either mom_len or mom
        """

        mom_len = cls._choose_mom_len(mom, mom_len)

        if verify:
            data = np.asarray(data, dtype=dtype)

        if shape is None:
            shape = data.shape[:-mom_len]
        mom = cls._check_mom(mom, mom_len, data.shape)

        if data.shape != shape + tuple(x + 1 for x in mom):
            raise ValueError(f"{data.shape} does not conform to {shape} and {moments}")

        if copy:
            if copy_kws is None:
                copy_kws = {}
            data = data.copy(**copy_kws)

        return cls(data=data, mom_len=mom_len, **kws)

    @classmethod
    def from_datas(
        cls,
        datas,
        mom_len=None,
        axis=0,
        mom=None,
        shape=None,
        dtype=None,
        verify=True,
        **kws,
    ):
        """
        Data should have shape

        [..., moments] (axis!= -1)

        [..., moment, axis] (axis == -1)
        """

        mom_len = cls._choose_mom_len(mom, mom_len)

        if verify:
            datas = np.asarray(datas, dtype=dtype)
        datas, axis = cls._datas_axis_to_first(datas, axis, mom_len)

        if shape is None:
            shape = datas.shape[1:-mom_len]

        mom = cls._check_mom(mom, mom_len, datas.shape)
        assert datas.shape[1:] == shape + tuple(x + 1 for x in mom)

        if dtype is None:
            dtype = datas.dtype

        # TODO : inline
        # new = cls.zeros(shape_tot=datas.shape[1:], mom_len=mom_len, dtype=dtype, **kws)
        # new.push_datas(datas=datas, axis=0)
        # return new
        return (
            cls.zeros(shape_tot=datas.shape[1:], mom_len=mom_len, dtype=dtype, **kws)
            .push_datas(datas=datas, axis=0)
        )

    @classmethod
    def from_vals(
        cls, x, w=None, axis=0, mom=2, shape=None, dtype=None, broadcast=False, **kws
    ):

        mom_len = cls._mom_len_from_mom(mom)
        x0 = x if mom_len == 1 else x[0]
        if shape is None:
            shape = list(x0.shape)
            shape.pop(axis)
            shape = tuple(shape)
        if dtype is None:
            dtype = x0.dtype

        # TODO: inline this
        # new = cls.zeros(shape=shape, mom=mom, dtype=dtype, **kws)
        # new.push_vals(x=x, axis=axis, w=w, broadcast=broadcast)
        # return new
        return (
            cls.zeros(shape=shape, mom=mom, dtype=dtype, **kws)
            .push_vals(x=x, axis=axis, w=w, broadcast=broadcast)
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
        dtype=None,
        broadcast=False,
        resample_kws=None,
        **kws,
    ):

        mom_len = cls._mom_len_from_mom(mom)

        x0 = x if mom_len == 1 else x[0]
        freq = randsamp_freq(
            nrep=nrep, freq=freq, indices=indices, size=x0.shape[axis], check=True,
        )

        if resample_kws is None:
            resample_kws = {}

        data = resample_vals(
            x,
            freq=freq,
            mom=mom,
            axis=axis,
            w=w,
            mom_len=mom_len,
            **resample_kws,
            broadcast=broadcast,
        )
        return cls.from_data(data, mom_len=mom_len, copy=False, **kws)

    @classmethod
    def from_raw(cls, raw, mom_len=None, mom=None, shape=None, dtype=None, **kws):
        """create object from raw

        must specify either `mom_len` or `mom`
        """

        mom_len = cls._choose_mom_len(mom, mom_len)

        if mom_len == 1:
            func = convert.to_central_moments
        elif mom_len == 2:
            func = convert.to_central_comoments
        data = func(raw)

        return cls.from_data(
            data, mom_len=mom_len, mom=mom, shape=shape, dtype=dtype, copy=False, **kws
        )

    @classmethod
    def from_raws(
        cls, raws, mom_len=None, mom=None, axis=0, shape=None, dtype=None, **kws
    ):
        mom_len = cls._choose_mom_len(mom, mom_len)

        if mom_len == 1:
            func = convert.to_central_moments
        elif mom_len == 2:
            func = convert.to_central_comoments
        datas = func(raws)
        return cls.from_datas(
            datas=datas,
            axis=axis,
            mom_len=mom_len,
            mom=mom,
            shape=shape,
            dtype=dtype,
            copy=False,
            **kws,
        )

    ###########################################################################
    # SECTION: Manipulation
    ###########################################################################
    @property
    def _is_vector(self):
        return self.ndim > 0

    def _raise_if_scalar(self, message=None):
        if not self._is_vector:
            if message is None:
                message = "not implemented for scalar"
            raise ValueError(message)

    # Universal reducers
    def resample_and_reduce(
        self, freq=None, indices=None, nrep=None, axis=None, resample_kws=None, **kws,
    ):
        """
        bootstrap resample and reduce

        Parameter
        ----------
        freq : array-like, shape=(nrep, nrec), optional
            frequence table.  freq[i, j] is the weight of the jth record to the ith replicate
        indices : array-like, shape=(nrep, nrec), optional
            resampling array.  idx[i, j] is the record index of the original array to place in new sample[i, j].
            if specified, create freq array from idx
        nrep : int, optional
            if specified, create idx array with this number of replicates 
        axis : int, Default=0
            axis to resample and reduce along
        resample_kws : dict
            extra arguments to resample.resample_and_reduce
        args : tuple
            extra positional arguments to from_data method
        kwargs : dict
            extra key-word arguments to from_data method
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        if resample_kws is None:
            resample_kws = {}

        freq = randsamp_freq(
            nrep=nrep, indices=indices, freq=freq, size=self.shape[axis], check=True
        )
        data = resample_data(self.data, freq, mom=self.mom, axis=axis, **resample_kws)
        return type(self).from_data(data, mom_len=self.mom_len, copy=False, **kws)

    def resample(self, indices, axis=0, first=True, **kws):
        """
        create a new object sampled from index

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
        return type(self)(data=out, mom_len=self.mom_len, **kws)

    def reduce(self, axis=0, **kws):
        """
        create new object reducealong axis
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        return type(self).from_datas(
            self.values, mom_len=self.mom_len, axis=axis, **kws
        )

    def block(self, block_size=None, axis=None, **kws):
        """
        block average reduction

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

        new_shape = (nblock, block_size) + data.shape[1:]
        datas = data[: (nblock * block_size), ...].reshape(
            (nblock, block_size) + data.shape[1:]
        )

        return type(self).from_datas(datas=datas, mom_len=self.mom_len, axis=1, **kws)

    def reshape(self, shape, copy=True, copy_kws=None, **kws):
        """
        create a new object with reshaped data

        Parameters
        ---------
        shape : tuple
            shape of values part of data
        """
        self._raise_if_scalar()
        new_shape = shape + self.shape_mom
        data = self._data.reshape(new_shape)
        return self.new_like(
            data=data, verify=False, check=False, copy=copy, copy_kws=copy_kws, **kws
        )

    def moveaxis(self, source, destination, copy=True, copy_kws=None, **kws):
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

        # use from data for extra checks
        # return self.new_like(data=data, copy=copy, *args, **kwargs)
        return type(self).from_data(
            data,
            mom=self.mom,
            mom_len=self.mom_len,
            shape=data.shape[: -self.mom_len],
            copy=copy,
            copy_kws=copy_kws,
            **kws,
        )

    # --------------------------------------------------
    # mom_len == 1 specific
    # --------------------------------------------------

    @staticmethod
    def _raise_if_not_1d(mom_len):
        if mom_len != 1:
            raise NotImplementedError("only available for mom_len == 1")

    # special, 1d only methods
    def push_stat(self, a, v=0.0, w=None, broadcast=True):
        self._raise_if_not_1d(self.mom_len)

        ar, target = self.check_val(a, target="val")
        vr = self.check_var(v, broadcast=broadcast)
        wr = self.check_weight(w, target=target)
        self._push.stat(self._data_flat, wr, ar, vr)
        return self

    def push_stats(self, a, v=0.0, w=None, axis=0, broadcast=True):
        self._raise_if_not_1d(self.mom_len)

        ar, target = self.check_vals(a, target="vals", axis=axis)
        vr = self.check_vars(v, target=target, axis=axis, broadcast=broadcast)
        wr = self.check_weights(w, target=target, axis=axis)
        self._push.stats(self._data_flat, wr, ar, vr)
        return self

    @classmethod
    def from_stat(cls, a, v=0.0, w=None, mom=2, shape=None, dtype=None, **kws):
        """
        object from single weight, average, variance/covariance
        """
        mom_len = cls._mom_len_from_mom(mom)
        cls._raise_if_not_1d(mom_len)

        if shape is None:
            shape = a.shape
        if dtype is None:
            dtype = a.dtype

        # TODO: inline this
        new = cls.zeros(shape=shape, mom=mom, dtype=dtype, **kws)
        new.push_stat(w=w, a=a, v=v)
        return new

    @classmethod
    def from_stats(
        cls, a, v=0.0, w=None, axis=0, mom=2, shape=None, dtype=None, **kws,
    ):
        """
        object from several weights, averages, variances/covarainces along axis
        """

        mom_len = cls._mom_len_from_mom(mom)
        cls._raise_if_not_1d(mom_len)

        # get shape
        if shape is None:
            shape = list(A.shape)
            shape.pop(axis)
            shape = tuple(shape)

        # TODO : inline this
        new = cls.zeros(shape=shape, dtype=dtype, mom=mom, **kws)
        new.push_stats(a=a, v=v, w=w, axis=axis)
        return new


# class StatsAccumCov(StatsAccumBase):
#     _mom_len = 2

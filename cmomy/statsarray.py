"""
Container for multiple (similar) CentralMoments objects
"""
from __future__ import absolute_import

import numpy as np

from .cached_decorators import cached_clear, gcached

# from .central import CentralMoments, CentralMomentsCov
# from .resample import randsamp_freq, resample_data, resample_vals
# from .utils import (
#     _axis_expand_broadcast,
#     _cached_ones,
#     _my_broadcast,
#     _shape_insert_axis,
# )


def weighted_var(x, w, axis=None, axis_sum=None, unbiased=True, **kwargs):
    """
    return the weighted variance over x with weight w

    v = sum(w)**2/(sum(w)**2 - sum(w**2)) * sum(w * (x-mu)**2 )

    Parameters
    ----------
    x : array
        values to consider

    w : array
        weights

    axis : axis to average over

    axis_sum : axis to sum over for  w,w**2

    unbiased : bool (default True)
    if True, then apply unbiased norm (like ddof=1)
    else, apply biased norm (like ddof=0)


    **kwargs : arguments to np.average

    Returns
    -------
    Ave : weighted average
        shape x with `axis` removed

    Var : weighted variance
        shape x with `axis` removed
    """

    if axis_sum is None:
        axis_sum = axis

    m1 = np.average(x, weights=w, axis=axis, **kwargs)
    m2 = np.average((x - m1) ** 2, weights=w, axis=axis, **kwargs)

    if unbiased:
        w1 = w.sum(axis=axis_sum)
        w2 = (w * w).sum(axis=axis_sum)
        m2 *= w1 * w1 / (w1 * w1 - w2)
    return m1, m2


class StatsArray(object):
    """
    Collection of Accumulator objects
    """

    def __init__(self, obj):
        """
        Parameters
        ----------
        obj : CentralMoments
        """
        self._accum = obj
        self.zero()

        # if isinstance(moments, int):
        #     moments = (moments,)

        # assert isinstance(moments, tuple)

        # if child is None:
        #     if len(moments) == 1:
        #         child = CentralMoments
        #     else:
        #         child = CentralMomentsCov

        # self._child = child
        # self._accum = child(shape=shape, moments=moments, dtype=dtype)
        # self.zero()

    @classmethod
    def zeros(cls, *args, **kwargs):
        """create empty array object with baseclass CentralMoments"""
        pass

    @classmethod
    def xzeros(cls, *args, **kwargs):
        """create empty array object with baseclass xCentralMoments"""
        pass

    @property
    def accum(self):
        return self._accum

    @property
    def mom(self):
        return self._accum.moments

    @property
    def dtype(self):
        return self._accum.dtype

    @property
    def values(self):
        return self._values

    @values.setter
    @cached_clear()
    def values(self, values):
        if not isinstance(values, list):
            raise ValueError("trying to set list to non-list value")
        self._values = values

    @gcached()
    def data(self):
        return np.array(self._values)

    def new_like(self):
        return self.__class__(
            shape=self.accum.shape,
            child=self._child,
            dtype=self.dtype,
            moments=self.accum.moments,
        )

    def __len__(self):
        return len(self._values)

    def __getitem__(self, idx):
        new = self.new_like()
        try:
            y = self._values[idx]
        except Exception:
            y = list(self.data[idx])
        if not isinstance(y, list):
            y = [y]

        new._values = y
        return new

    def to_stats(self, indices=None, copy=True):
        data = self.data
        if indices is not None:
            data = np.take(data, indices, axis=0)
        elif copy:
            data = data.copy()
        return self._child.from_data(data, moments=self.moments, copy=False)

    def resample(self, indices):
        """
        axis = axis of indices to average over
        """
        return self.to_stats(indices)

    def resample_and_reduce(self, freq, **kwargs):
        """
        for bootstrapping
        """
        data = self.data
        data_new = data.resample_data(data, freq, moments=self.moments, **kwargs)
        return self.__class__.from_datas(
            data_new, shape=self._accum.shape, child=self._child, moments=self.moments
        )

    def zero(self):
        self.values = []
        self.accum.zero()

    @cached_clear()
    def append(self, data):
        assert data.shape == self._accum.shape_data
        self._values.append(data)

    @cached_clear()
    def push_stat(self, a, v=0.0, w=1.0):
        s = self._child.from_stat(a=a, v=v, w=w)
        self._values.append(s.data)

    @cached_clear()
    def push_stats(self, a, v=None, w=None):
        if v is None:
            v = np.zeros_like(a)
        if w is None:
            w = np.ones_like(a)
        for (ww, aa, vv) in zip(w, a, v):
            self.push_stat(a=aa, v=vv, w=ww)

    @cached_clear()
    def push_data(self, data, copy=False):
        assert data.shape == self._accum.shape_data
        if copy:
            data = data.copy()
        self._values.append(data)

    @cached_clear()
    def push_datas(self, datas, axis=0):
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        assert datas.shape[1:] == self._accum.shape_data
        for data in datas:
            self._values.append(data)

    @gcached()
    def _weight_index(self):
        return (slice(None),) + self._accum._weight_index

    @gcached(prop=False)
    def _single_index(self, val):
        return (slice(None),) + self._accum._single_index(val)

    def weight(self):
        return self.data[self._weight_index]

    def mean(self):
        return self.data[self._single_index(1)]

    def var(self):
        return self.data[self._single_index(2)]

    @property
    def data_last(self):
        return self._values[-1]

    def mean_last(self):
        return self.data_last[self.accum._single_index(1)]

    def var_last(self):
        return self.data_last[self.accum._single_index(2)]

    def std_last(self):
        return np.sqrt(self.var_last())

    def weight_last(self):
        return self.data_last[self.accum._weight_index]

    def get_stat(self, stat_name="mean", *args, **kwargs):
        return getattr(self, stat_name)(*args, **kwargs)

    @classmethod
    def from_datas(cls, datas, moments, axis=0, shape=None, child=None, dtype=np.float):
        if isinstance(moments, int):
            moments = (moments,)

        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)

            shape = datas.shape[1 : -len(moments)]

        new = cls(child=child, shape=shape, moments=moments, dtype=dtype)

        new.values = list(datas)
        return new

    @classmethod
    def from_accum(cls, accum, axis=None):
        """
        create StatsArray from CentralMoments object

        if accum object is a scalar object, or a vector object with no specified axis,
        then create a StatsArray object with accum as the sole elements

        if accum object is a vector object with a specified axis, create a
        StatsArray object
        with elements along this axis
        """

        ndim = len(accum.moments)
        if ndim == 0:
            axis = None
        if axis is None:
            # create single object
            new = cls(
                moments=accum.moments,
                shape=accum.shape,
                child=type(accum),
                dtype=accum.dtype,
            )

        else:
            new = cls.from_datas(
                accum.data,
                moments=accum.moments,
                axis=axis,
                shape=None,
                child=type(accum),
                dtype=accum.dtype,
            )

        return new

    @gcached()
    def cumdata(self):
        cumdata = np.zeros((len(self),) + self.accum.shape_data)
        self._accum.zero()
        for i, data in enumerate(self.values):
            self._accum.push_data(data)
            cumdata[i, ...] = self._accum.data
        return cumdata

    def cummean(self):
        return self.cumdata[self._single_index(1)]

    def cumvar(self):
        return self.cumdata[self._single_index(2)]

    def cumstd(self):
        return np.sqrt(self.cumvar())

    def cumweight(self):
        return self.cumdata[self._weight_index]

    @property
    def cumdata_last(self):
        return self.cumdata[-1, ...]

    def cummean_last(self):
        return self.cumdata_last[self.accum._single_index(1)]

    def cumvar_last(self):
        return self.cumdata_last[self.accum._single_index(2)]

    def cumstd_last(self):
        return np.sqrt(self.cumvar_last())

    def cumweight_last(self):
        return self.cumdata_last[self.accum._weight_index]

    @gcached()
    def stats_list(self):
        """
        list of stats objects
        """
        return [
            self._child.from_data(
                data=data,
                shape=self.accum.shape,
                moments=self.moments,
                dtype=self.dtype,
                copy=False,
            )
            for data in self.values
        ]

    def block(self, block_size=None):
        """
        create a new stats array object from block averaging this one
        """
        new = self.new_like()
        new.values = self.blockdata(block_size)
        return new

    @gcached(prop=False)
    def blockdata(self, block_size):
        blockdata = []

        n = len(self)
        if block_size is None:
            block_size = n
        if block_size > n:
            block_size = n

        for lb in range(0, len(self), block_size):
            ub = lb + block_size
            if ub > n:
                break
            self._accum.zero()
            datas = self.data[lb:ub, ...]
            self._accum.push_datas(datas)
            blockdata.append(self._accum.data.copy())
        return blockdata

    def blockweight(self, block_size=None):
        return self.blockdata(block_size)[self._weight_index]

    def blockmean(self, block_size=None):
        return self.blockdata(block_size)[self._single_index(1)]

    def blockvar(self, block_size=None):
        return self.blockdata(block_size)[self._single_index(2)]

    def val_SEM(self, x, weighted, unbiased, norm):
        """
        find the standard error in the mean (SEM) of a value

        Parameters
        ----------
        x : array
            array (self.mean(), etc) to consider

        weighted : bool
            if True, use `weighted_var`
            if False, use `np.var`

        unbiased : bool
            if True, use unbiased stats (e.g., ddof=1 for np.var)
            if False, use biased stats (e.g., ddof=0 for np.var)

        norm : bool
            if True, scale var by x.shape[0], i.e., number of samples

        Returns
        -------
        sem : standard error in mean
        """
        if weighted:
            v = weighted_var(x, w=self.weight(), axis=0, unbiased=unbiased)[-1]
        else:
            if unbiased:
                ddof = 1
            else:
                ddof = 0

            v = np.var(x, ddof=ddof, axis=0)
        if norm:
            v = v / x.shape[0]

        return np.sqrt(v)

    def mean_SEM(self, weighted=True, unbiased=True, norm=True):
        """self.val_SEM with x=self.mean()"""
        return self.val_SEM(self.mean(), weighted, unbiased, norm)

    def __repr__(self):
        return "nsample: {}".format(len(self))

    def to_xarray(
        self,
        rec_dim="rec",
        meta_dims=None,
        mom_dims=None,
        rec_coords=None,
        meta_coords=None,
        mom_coords=None,
        **kwargs,
    ):
        import xarray as xr

        if meta_dims is None:
            meta_dims = ["dim_{}".format(i) for i in range(len(self.accum.shape))]
        else:
            meta_dims = list(meta_dims)
        assert len(meta_dims) == len(self.accum.shape)

        if mom_dims is None:
            mom_dims = ["mom_{}".format(i) for i in range(len(self.moments))]

        if isinstance(mom_dims, str):
            mom_dims = [mom_dims]

        assert len(mom_dims) == len(self.accum.moments)

        dims = [rec_dim] + meta_dims + mom_dims

        coords = {}
        coords.update(rec_coords or {})
        coords.update(meta_coords or {})
        coords.update(mom_coords or {})
        return xr.DataArray(self.data, dims=dims, coords=coords, **kwargs)

    @classmethod
    def from_xarray(
        cls,
        data,
        rec_dim="rec",
        meta_dims=None,
        mom_dims=None,
        shape=None,
        moments=None,
        child=None,
        dtype=None,
    ):
        pass

        if mom_dims is None:
            # try to infer moment dimensions
            mom_dims = []
            for k in sorted(data.dims):
                if "mom_" in k:
                    mom_dims.append(k)

        if isinstance(mom_dims, str):
            mom_dims = [mom_dims]

        if moments is None:
            # infer moments
            moments = []
            for k in mom_dims:
                moments.append(len(data[k]) - 1)
            moments = tuple(moments)

        assert len(moments) == len(mom_dims)

        order = [rec_dim]
        if meta_dims is not None:
            if isinstance(meta_dims, str):
                meta_dims = [meta_dims]
            assert data.ndim == 1 + len(mom_dims) + len(meta_dims)
            order += meta_dims
        else:
            order += [...]

        order += mom_dims

        data = data.transpose(*order)

        return cls.from_datas(
            datas=data, moments=moments, axis=0, shape=shape, child=child, dtype=dtype
        )

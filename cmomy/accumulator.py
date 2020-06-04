from __future__ import division, absolute_import, print_function

import numpy as np
from numba import njit, prange

@njit
def _push_val(data, w, x):
    if w == 0.0:
        return
    weight = data[0] + w
    ave = data[1]
    var = data[2]

    alpha = w / weight
    delta = (x - ave)
    incr = delta * alpha

    data[0] = weight
    data[1] = ave + incr
    data[2] = (1.0 - alpha) * (var + delta * incr)


@njit
def _push_vals(data, W, X):
    ns = X.shape[0]
    for s in range(ns):
        _push_val(data, W[s], X[s])


@njit(parallel=True)
def _push_vals_parallel(data, W, X):
    ns = X.shape[0]
    for s in range(ns):
        w = W[s]
        if w == 0.0:
            continue
        x = X[s]

        data[0] += w

        weight = data[0]
        ave = data[1]
        var = data[2]

        alpha = w / weight
        delta = (x - ave)
        incr = delta * alpha

        data[1] += incr
        data[2] += (1.0 - alpha) * (delta * incr) - alpha * var



@njit
def _push_stat(data, w, a, v):
    if w == 0.0:
        return
    weight = data[0] + w
    ave = data[1]
    var = data[2]

    alpha = w / weight
    delta = (a - ave)
    incr = delta * alpha

    data[0] = weight
    data[1] = ave + incr
    data[2] = v * alpha + (1.0 - alpha) * (var + delta * incr)





@njit
def _push_stats(data, W, A, V):
    ns = A.shape[0]
    for s in range(ns):
        _push_stat(data, W[s], A[s], V[s])



@njit(parallel=True)
def _push_stats_parallel(data, W, A, V):
    ns = A.shape[0]
    for s in range(ns):
        w = W[s]
        if w == 0.0:
            continue
        a = A[s]
        v = V[s]

        data[0] += w

        weight = data[0]
        ave = data[1]
        var = data[2]

        alpha = w / weight
        delta = (a - ave)
        incr = delta * alpha

        data[1] += incr
        data[2] += (v-var) * alpha + (1.0 - alpha) * (delta * incr) 


# TODO : _push_stat_data, for pushing data directly, etc
@njit
def _push_stat_data(data, data_in):
    w = data_in[0]
    if w == 0.0:
        return
    a = data_in[1]
    v = data_in[2]

    weight = data[0] + w
    ave = data[1]
    var = data[2]

    alpha = w / weight
    delta = (a - ave)
    incr = delta * alpha

    data[0] = weight
    data[1] = ave + incr
    data[2] = v * alpha + (1.0 - alpha) * (var + delta * incr)


@njit
def _push_stats_data(data, data_in):
    ns = data_in.shape[0]
    for s in range(ns):
        _push_stat_data(data, data_in[s, :])


@njit(parallel=True)
def _push_stats_data_parallel(data, data_in):
    ns = A.shape[0]
    for s in range(ns):
        w = data_in[s, 0]
        if w == 0.0:
            continue
        a = data_in[s, 1]
        v = data_in[s, 2]

        data[0] += w

        weight = data[0]
        ave = data[1]
        var = data[2]

        alpha = w / weight
        delta = (a - ave)
        incr = delta * alpha

        data[1] += incr
        data[2] += (v-var) * alpha + (1.0 - alpha) * (delta * incr) 


# Vector
@njit
def _push_val_vec(data, w, x):
    n = data.shape[0]
    for i in range(n):
        _push_val(data[i, :], w[i], x[i])

@njit
def _push_vals_vec(data, W, X):
    ns = X.shape[0]
    n = data.shape[0]
    for s in range(ns):
        for i in range(n):
            _push_val(data[i, :], W[s, i], X[s, i])




@njit
def _push_stat_vec(data, w, a, v):
    n = data.shape[0]
    for i in range(n):
        _push_stat(data[i, :], w[i], a[i], v[i])


@njit
def _push_stats_vec(data, W, A, V):
    ns = A.shape[0]
    n = data.shape[0]
    for s in range(ns):
        for i in range(n):
            _push_stat(data[i, :], W[s, i], A[s, i], V[s, i])


@njit
def _push_stat_data_vec(data, data_in):
    n = data.shape[0]
    for i in range(n):
        _push_stat_data(data[i, :], data_in[i, :])


@njit
def _push_stats_data_vec(data, Data_in):
    ns = Data_in.shape[0]
    n = data.shape[0]
    for s in range(ns):
        for i in range(n):
            _push_stat_data(data[i, :], Data_in[s, i, :])


# Covariance
@njit
def _push_val_cov(data, w, x):
    # data[i,j,0] = weight[i,j]
    # data[i,j,1] = ave[i,j]
    # data[i,j,2] = cov_prime[i,j]
    # cov[i,j] = cov_prime[i, j] + (ave[i,i] - ave[i,j]) * (ave[j,j] - ave[j,i])

    n = x.shape[0]
    for i in range(n):
        xi = x[i]
        for j in range(i, n):
            wij = w[i, j]
            if wij == 0:
                continue

            weight = data[i, j, 0] + wij
            alpha = wij / weight

            avei = data[i, j, 1]
            deltai = xi - avei
            incri = deltai * alpha
            data[i, j, 0] = weight
            data[i, j, 1] = avei + incri

            if j == i:
                # same
                data[i, j, 2] = (1.0 - alpha) * (
                    data[i, j, 2] + deltai * incri)
            else:
                avej = data[j, i, 1]
                deltaj = x[j] - avej
                incrj = deltaj * alpha

                data[j, i, 0] = weight
                data[j, i, 1] = avej + incrj

                # cov_prime
                v = (1.0 - alpha) * (data[i, j, 2] + incri * deltaj)

                data[i, j, 2] = v
                data[j, i, 2] = v


@njit
def _push_vals_cov(data, W, X):
    ns = X.shape[0]
    for s in range(ns):
        _push_val_cov(data, W[s, ...], X[s, ...])


@njit
def _push_stat_cov(data, w, a, v):
    n = a.shape[0]
    for i in range(n):
        for j in range(i, n):
            wij = w[i, j]
            if wij == 0:
                continue

            weight = data[i, j, 0] + wij
            alpha = wij / weight

            avei = data[i, j, 1]
            deltai = a[i, j] - avei
            incri = deltai * alpha
            data[i, j, 0] = weight
            data[i, j, 1] = avei + incri

            if j == i:
                # same
                data[i, j, 2] = v[i, j] * alpha + (1.0 - alpha) * (
                    data[i, j, 2] + deltai * incri)
            else:
                avej = data[j, i, 1]
                deltaj = a[j, i] - avej
                incrj = deltaj * alpha

                data[j, i, 0] = weight
                data[j, i, 1] = avej + incrj

                # cov_prime
                c = v[i, j] * alpha + (1.0 - alpha) * (
                    data[i, j, 2] + incri * deltaj)

                data[i, j, 2] = c
                data[j, i, 2] = c


@njit
def _push_stats_cov(data, W, A, V):
    ns = X.shape[0]
    for s in range(ns):
        _push_stat_cov(data, W[s, ...], A[s, ...], V[s, ...])


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
    m2 = np.average((x - m1)**2, weights=w, axis=axis, **kwargs)

    if unbiased:
        w1 = w.sum(axis=axis_sum)
        w2 = (w * w).sum(axis=axis_sum)
        m2 *= w1 * w1 / (w1 * w1 - w2)
    return m1, m2


class _StatsAccum(object):

    def __init__(self, shape, dtype=np.float, nmom=2):
        self._shape = shape
        self._shape_var = self._shape
        self._dtype = dtype
        self.nmom = nmom

        self._init_subclass()

        self._data = np.empty(self.shape_var + (1+self.nmom, ), dtype=self._dtype)

        if getattr(self, '_shape_r', None) is None:
            if self.shape is ():
                self._shape_r = ()
            else:
                self._shape_r = (np.prod(self.shape), )

        if getattr(self, '_shape_var_r', None) is None:
            if self.shape_var is ():
                self._shape_var_r = ()
            else:
                self._shape_var_r = (np.prod(self.shape_var), )

        self._datar = self._data.reshape(self._shape_var_r + (1 + self.nmom, ))
        self.zero()


    # when unpickling, make sure self._datar points to same
    # underlieing array as self._data
    def __setstate__(self, state):
        self.__dict__ = state
        # make sure datar points to data
        self._datar = self._data.reshape(self._shape_var_r + (1+self.nmom,))


    def _init_subclass(self):
        """any special subclass stuff here"""
        pass

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        """shape of input values"""
        return self._shape

    @property
    def shape_var(self):
        return self._shape_var

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def ndim_var(self):
        return len(self.shape_var)

    @property
    def dtype(self):
        return self._dtype

    @property
    def _unit_weight(self):
        if not hasattr(self, '_unit_weight_val'):
            self._unit_weight_val = np.ones(self.shape_var)
        return self._unit_weight_val

    def _check_weight(self, w):
        if w is None:
            w = self._unit_weight
        if np.shape(w) != self.shape_var:
            w = np.broadcast_to(w, self.shape_var)
        return np.reshape(w, self._shape_var_r)

    def _check_weights(self, W, X, axis=0):
        shape = list(self.shape_var)
        shape.insert(axis, X.shape[axis])

        if W is None:
            W = self._unit_weight
        if np.shape(W) != shape:
            W = np.broadcast_to(W, shape)
        if axis != 0:
            W = np.rollaxis(W, axis, 0)
        # broadcast to shape_var:
        shaper = W.shape[:1] + self._shape_var_r
        return np.reshape(W, shaper)

    def _check_val(self, x):
        assert np.shape(x) == self.shape
        return np.reshape(x, self._shape_r)

    def _check_vals(self, X, axis=0):
        if axis != 0:
            X = np.rollaxis(X, axis, 0)
        assert np.shape(X)[1:] == self.shape
        shaper = X.shape[:1] + self._shape_r
        return np.reshape(X, shaper)

    def _check_ave(self, a):
        #assert np.shape(a) == self.shape_var
        if a.shape != self.shape_var:
            a = np.broadcast_to(a, self.shape_var)
        return np.reshape(a, self._shape_var_r)

    def _check_aves(self, A, axis=0):
        if axis != 0:
            A = np.rollaxis(A, axis, 0)
        assert np.shape(A)[1:] == self.shape_var, '{}, {}'.format(
            np.shape(A), self.shape_var)
        # if np.shape(A)[1:] != self.shape_var:
        #     new_shape = (A.shape[0], ) + self.shape_var
        #     A = np.broadcast_to(A, new_shape)
        shaper = A.shape[:1] + self._shape_var_r
        return np.reshape(A, shaper)

    def _check_var(self, v):
        if np.shape(v) != self.shape_var:
            v = np.broadcast_to(v, self.shape_var)
        return np.reshape(v, self._shape_var_r)

    def _check_vars(self, V, X, axis=0):
        shape = list(self.shape_var)
        shape.insert(axis, X.shape[axis])
        if np.shape(V) != shape:
            V = np.broadcast_to(V, shape)
        if axis != 0:
            V = np.rollaxis(V, axis, 0)
        shaper = V.shape[:1] + self._shape_var_r
        return np.reshape(V, shaper)

    def _check_data(self, data):
        shape = self.shape_var + (self.nmom+1, )
        if np.shape(data) != shape:
            raise ValueError('data must of have shape {}'.format(shape))
        return np.reshape(data, self._datar.shape)

    def _check_datas(self, datas, axis=0):
        shape = self.shape_var + (self.nmom+1, )
        if axis != 0:
            datas = np.rollaxis(datas, axis, 0)
        if np.shape(datas)[1:] != shape:
            raise ValueError('bad shape {} != {}, axis={}'.format(
                datas.shape, shape, axis))
        shaper = datas.shape[:1] + self._datar.shape
        return np.reshape(datas, shaper)

    def zero(self):
        self._data.fill(0.0)

    def zeros_like(self):
        """create zero object like self"""
        return self.__class__(shape=self.shape, dtype=self.dtype, nmom=self.nmom)

    def copy(self):
        new = self.__class__(shape=self.shape, dtype=self.dtype, nmom=self.nmom)
        new._data[...] = self._data[...]
        return new

    def push_val(self, x, w=None):
        xr = self._check_val(x)
        wr = self._check_weight(w)
        self._push_val(self._datar, wr, xr)

    def push_vals(self, X, W=None, axis=0):
        Xr = self._check_vals(X, axis)
        Wr = self._check_weights(W, X, axis)
        self._push_vals(self._datar, Wr, Xr)

    def push_stat(self, a, v=0.0, w=None):
        ar = self._check_ave(a)
        vr = self._check_var(v)
        wr = self._check_weight(w)
        self._push_stat(self._datar, wr, ar, vr)

    def push_stats(self, A, V=0.0, W=None, axis=0):
        Ar = self._check_aves(A, axis)
        Vr = self._check_vars(V, A, axis)
        Wr = self._check_weights(W, A, axis)
        self._push_stats(self._datar, Wr, Ar, Vr)

    def push_stat_data(self, data):
        data = self._check_data(data)
        self._push_stat_data(self._datar, data)

    def push_stats_data(self, Data, axis=0):
        Data = self._check_datas(Data, axis)
        self._push_stats_data(self._datar, Data)

    def _check_other(self, b):
        assert type(self) == type(b)
        assert self.shape == b.shape

    def __iadd__(self, b):
        self._check_other(b)
        self.push_stat(w=b._data[..., 0], a=b._data[..., 1], v=b._data[..., 2])
        return self

    def __add__(self, b):
        self._check_other(b)
        new = self.copy()
        new.push_stat(w=b._data[..., 0], a=b._data[..., 1], v=b._data[..., 2])
        return new

    def __isub__(self, b):
        self._check_other(b)
        assert np.all(self._data[..., 0] >= b._data[..., 0])
        self.push_stat(
            w=-b._data[..., 0], a=b._data[..., 1], v=b._data[..., 2])

    def __sub__(self, b):
        assert (type(self) == type(b))
        assert np.all(self._data[..., 0] > b._data[..., 0])
        new = self.copy()
        new.push_stat(w=-b._data[..., 0], a=b._data[..., 1], v=b._data[..., 2])
        return new

    def weight(self):
        return self._data[..., 0]

    def mean(self):
        return self._data[..., 1]

    def var(self):
        return self._data[..., 2]

    def std(self):
        return np.sqrt(self._data[..., 2])

    def _diag(self, x):
        return (x.reshape(self._shape_var_r).diagonal().reshape(self.shape))

    def weight_diag(self):
        return self._diag(self.weight())

    def mean_diag(self):
        return self._diag(self.mean())

    def var_diag(self):
        return self._diag(self.var())

    def std_diag(self):
        return self._diag(self.std())

    # --------------------------------------------------
    # constructors
    # --------------------------------------------------
    @classmethod
    def from_stat(cls, a=None, v=0.0, w=None, data=None, shape=None, nmom=2):
        """
        object from single weight, average, variance/covariance
        """

        if data is not None:
            w = data[..., 0]
            a = data[..., 1]
            if nmom == 2:
                v = data[..., 2]
            else:
                v = data[..., 2:]
        else:
            assert a is not None

        if shape is None:
            shape = a.shape
        new = cls(shape=shape, dtype=a.dtype, nmom=nmom)
        new.push_stat(w=w, a=a, v=v)
        return new

    @classmethod
    def from_stats(cls, A=None, V=0.0, W=None, Data=None, axis=0, shape=None, nmom=2):
        """
        object from several weights, averages, variances/covarainces along axis
        """
        if Data is not None:
            W = Data[..., 0]
            A = Data[..., 1]
            if nmom == 2:
                V = Data[..., 2]
            else:
                V = Data[..., 2:]

        else:
            assert A is not None

        #get shape
        if shape is None:
            shape = list(A.shape)
            shape.pop(axis)
            shape = tuple(shape)

        new = cls(shape=shape, dtype=A.dtype, nmom=nmom)
        new.push_stats(W=W, A=A, V=V, axis=axis)
        return new

    @classmethod
    def from_data(cls, data, shape=None, nmom=2):
        assert data.shape[-1] == nmom + 1
        if shape is None:
            shape = data.shape[:-1]
        new = cls(shape=shape, dtype=data.dtype, nmom=nmom)

        # new.push_stat_data(data=data)
        # below is much faster
        datar = new._check_data(data)
        new._datar[...] = datar
        return new

    @classmethod
    def from_datas(cls, Data, shape=None, axis=0, nmom=2):
        assert Data.shape[-1] == nmom + 1
        if shape is None:
            shape = list(Data.shape[:-1])
            shape.pop(axis)
            shape = tuple(shape)

        new = cls(shape=shape, dtype=Data.dtype, nmom=nmom)
        new.push_stats_data(Data=Data, axis=axis)
        return new

    @classmethod
    def from_vals(cls, X, W=None, axis=0, dtype=None, shape=None, nmom=2):

        #get shape
        if shape is None:
            shape = list(X.shape)
            shape.pop(axis)
            shape = tuple(shape)

        if dtype is None:
            dtype = X.dtype
        new = cls(shape=shape, dtype=dtype, nmom=nmom)
        new.push_vals(X, axis=axis, W=W)
        return new

    def reduce(self, axis=0):
        """
        create new object reduced along axis
        """
        ndim = len(self.shape)
        if axis < 0:
            axis = ndim - axis
        assert axis >= 0 and axis <= ndim

        shape = list(self.shape)
        shape.pop(axis)
        shape = tuple(shape)

        Data = self.data
        if Data.ndim == 2:
            Data = Data[:, None, :]

        new = self.__class__.from_datas(Data, axis=axis, nmom=nmom)
        return new


class StatsAccumVec(_StatsAccum):
    def _init_subclass(self):
        self._push_val = _push_val_vec
        self._push_vals = _push_vals_vec
        self._push_stat = _push_stat_vec
        self._push_stats = _push_stats_vec

        self._push_stat_data = _push_stat_data_vec
        self._push_stats_data = _push_stats_data_vec

    def to_array(self, axis=0):

        if axis < 0:
            axis += self.data.ndim - 1

        data = self.data
        if axis != 0:
            data = np.rollaxis(data, axis, 0)

        if data.ndim == 2:
            # expand
            data = data[:, None, :]

        shape = data.shape[1:-1]
        return StatsArray.from_datas(
            Data=data, child=StatsAccumVec, shape=shape, nmom=self.nmom)


class StatsAccum(_StatsAccum):
    def __init__(self, shape=(), dtype=np.float, nmom=2):
        super(StatsAccum, self).__init__(shape=(), dtype=dtype, nmom=nmom)

    def _init_subclass(self):
        self._push_val = _push_val
        self._push_vals = _push_vals
        self._push_stat = _push_stat
        self._push_stats = _push_stats

        self._push_stat_data = _push_stat_data
        self._push_stats_data = _push_stats_data


class StatsAccumCov(_StatsAccum):
    def _init_subclass(self):
        self._push_val = _push_val_cov
        self._push_vals = _push_vals_cov
        self._push_stat = _push_stat_cov
        self._push_stats = _push_stats_cov

        self._shape_var = self._shape * 2
        s = np.prod(self.shape)
        self._shape_var_r = (s, s)

    def cov_corrected(self):
        ar = self.mean().reshape(self._shape_var_r)
        cc = ar.diagonal()[:, None] - ar
        cc = (cc * cc.T).reshape(self.shape_var)
        return self.var() + cc


class StatsArray(object):
    def __init__(self, shape=(), child=None, dtype=np.float, nmom=2):
        if shape == ():
            child = StatsAccum
        else:
            assert child is not None, 'with shape, must specify child object'
        self._child = child
        self._accum = child(shape=shape, dtype=dtype, nmom=nmom)
        self.zero()

    @property
    def nmom(self):
        return self._accum.nmom


    def to_stats(self, indices=None):
        # if subset is None:
        #     data = self.data
        # else:
        #     data = self.data[subset]
        # return StatsAccumVec.from_data(data=data)

        # this is faster
        data = self.data
        if indices is None:
            new = StatsAccumVec.from_data(data=data)
        else:
            shape = indices.shape + data.shape[1:-1]
            new = StatsAccumVec(shape=shape)
            np.take(self.data, indices, axis=0, out=new._data)
        return new

    def resample(self, indices, axis=0):
        data = self.data.take(indices, axis=0)
        return StatsAccumVec.from_datas(data, axis=0)

    def resample_and_reduce(self, freq, **kwargs):
        """
        for bootstrapping
        """
        data = self.data

        data_new = resample_data(data, freq, **kwargs)

        return self.__class__.from_datas(
            data_new, shape=self._accum.shape, child=self._child, nmom=self.nmom)

    @property
    def accum(self):
        return self._accum

    def _zero_cache(self):
        self._data = None
        self._cumdata = None
        self._blockdata = {}
        self._stats_list = None

    def zero(self):
        self._list = []
        self._accum.zero()
        self._zero_cache()

    def push_stat(self, a=None, v=0.0, w=1.0, data=None):
        if data is None:
            assert a is not None
            data = np.stack([w, a, v], axis=-1)
        self._list.append(data)
        self._zero_cache()

    def push_stats(self, A=None, V=None, W=None, Data=None):
        if Data is None:
            assert A is not None
            if V is None:
                V = np.zeros_like(A)
            if W is None:
                W = np.ones_like(A)
            Data = np.stack([W, A, V], axis=-1)
        for data in Data:
            self.push_stat(data=data)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        new = StatsArray(shape=self._accum.shape, child=self._child)
        try:
            y = self._list[idx]
        except:
            y = list(self.data[idx])
        if not isinstance(y, list):
            y = [y]

        new._list = y
        return new

    @property
    def data(self):
        if self._data is None:
            self._data = np.array(self._list)
        return self._data

    def weight(self):
        return self.data[..., 0]

    def mean(self):
        return self.data[..., 1]

    def var(self):
        return self.data[..., 2]

    @property
    def data_last(self):
        return self._list[-1]

    def mean_last(self):
        return self.data_last[..., 1]

    def var_last(self):
        return self.data_last[..., 2]

    def std_last(self):
        return np.sqrt(self.var_last())

    def weight_last(self):
        return self.data_last[..., 0]

    def get_stat(self, stat_name='mean', *args, **kwargs):
        return getattr(self, stat_name)(*args, **kwargs)

    @classmethod
    def from_stats(cls,
                   A=None,
                   V=None,
                   W=None,
                   Data=None,
                   child=None,
                   shape=(), nmom=2):
        new = cls(child=child, shape=shape, nmom=nmom)
        new.push_stats(A=A, V=V, W=W, Data=Data)
        return new

    @classmethod
    def from_datas(cls, Data, shape=(), child=None, nmom=2):
        new = cls(child=child, shape=shape, nmom=nmom)
        new._list = list(Data)
        return new

    @property
    def cumdata(self):
        if self._cumdata is None:
            self._cumdata = np.zeros_like(self.data)
            self._accum.zero()
            for i, data in enumerate(self.data):
                self._accum.push_stat(
                    w=data[..., 0], a=data[..., 1], v=data[..., 2])
                self._cumdata[i, ...] = self._accum.data
        return self._cumdata

    def cummean(self):
        return self.cumdata[..., 1]

    def cumvar(self):
        return self.cumdata[..., 2]

    def cumstd(self):
        return np.sqrt(self.cumvar())

    def cumweight(self):
        return self.cumdata[..., 0]

    @property
    def cumdata_last(self):
        return self.cumdata[-1, ...]

    def cummean_last(self):
        return self.cumdata_last[..., 1]

    def cumvar_last(self):
        return self.cumdata_last[..., 2]

    def cumstd_last(self):
        return np.sqrt(self.cumvar_last())

    def cumweight_last(self):
        return self.cumdata_last[..., 0]

    @property
    def stats_list(self):
        """
        list of stats objects
        """
        if self._stats_list is None:
            self._stats_list = [
                # self._child.from_stat(
                #     w=data[..., 0],
                #     a=data[..., 1],
                #     v=data[..., 2],
                #     shape=self.accum.shape)
                self._child.from_data(data=data,
                                      shape=self.accum.shape,
                                      nmom=self.nmom)
                for data in self._list
            ]
        return self._stats_list

    def block(self, block_size=None):
        """
        create a new stats array object from block averaging this one
        """
        new = StatsArray(shape=self.accum.shape, child=self._child)
        new._list = list(self.blockdata(block_size))
        return new

    def blockdata(self, block_size=None):
        if block_size not in self._blockdata:
            blockdata = []

            n = len(self)

            if block_size == None:
                block_size = n
            if block_size > n:
                block_size = n

            for lb in range(0, len(self), block_size):
                ub = lb + block_size
                if ub > n:
                    break
                self._accum.zero()
                data = self.data[lb:ub, ...]
                self._accum.push_stats(
                    W=data[..., 0], A=data[..., 1], V=data[..., 2])

                blockdata.append(self._accum.data.copy())

            self._blockdata[block_size] = np.array(blockdata)
        return self._blockdata[block_size]

    def blockweight(self, block_size=None):
        return self.blockdata(block_size)[..., 0]

    def blockmean(self, block_size=None):
        return self.blockdata(block_size)[..., 1]

    def blockvar(self, block_size=None):
        return self.blockdata(block_size)[..., 2]

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
        return 'nsample: {}'.format(len(self))

    def to_xarray(self,
                  rec_dim='rec',
                  meta_dims=None,
                  val_dim='val',
                  rec_coords=None,
                  meta_coords=None,
                  val_coords=None,
                  **kwargs):
        import xarray as xr

        if meta_dims is None:
            meta_dims = [
                'dim_{}'.format(i) for i in range(self.accum.ndim_var)
            ]
        else:
            meta_dims = list(meta_dims)

        assert len(meta_dims) == self.accum.ndim_var

        dims = [rec_dim] + meta_dims + [val_dim]

        coords = {}
        coords.update(rec_coords or {})
        coords.update(meta_coords or {})
        coords.update(val_coords or {val_dim: ['cnt', 'ave', 'var']})
        return xr.DataArray(self.data, dims=dims, coords=coords, **kwargs)

    @classmethod
    def from_xarray(cls,
                    data,
                    rec_dim='rec',
                    meta_dims=None,
                    val_dim='val',
                    val_names=['cnt', 'ave', 'var'],
                    shape=(),
                    child=None):
        #dim='val', names=['cnt','ave','var',]):
        import xarray as xr
        if isinstance(data, xr.DataArray):
            s = data.to_dataset(dim=val_dim)
        elif isinstance(data, xr.Dataset):
            s = data
        else:
            raise ValueError('must pass data as dataset or dataframe')
        new = cls.from_stats(
            W=s[val_names[0]].values,
            A=s[val_names[1]].values,
            V=s[val_names[2]].values,
            shape=shape,
            child=child)
        return new


### building random sample with replacement
@njit
def _randsamp_freq_out(freq):
    nrep = freq.shape[0]
    ndat = freq.shape[1]
    for i in range(nrep):
        for j in range(ndat):
            index = np.random.randint(0, ndat)
            freq[i, index] += 1


def _randsamp_freq(ndat, nrep):
    """
    instead of building an index for samples build a frequency table
    freq[irep, jdat] = {# of sample of data at (j,..) for sample irep}
    note that this is the transpose of np.random.choice.  but this is faster to construct
    """
    freq = np.zeros((nrep, ndat), dtype=np.int64)
    _randsamp_freq_out(freq)
    return freq


def _randsamp_freq_threading(ndat, nrep, nthread):
    import threading
    out = np.zeros((nrep, ndat), dtype=np.int64)
    chunk_size = (nrep + nthread - 1) // nthread

    args_list = [(out[chunk_size * i:chunk_size * (i + 1), :], )
                 for i in range(nthread)]
    threads = [
        threading.Thread(target=_randsamp_freq_out, args=args)
        for args in args_list
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    return out


def _randsamp_freq_mult(ndat, nrep, nthread):
    import multiprocessing

    if nthread < 0:
        nthread = multiprocessing.cpu_count()

    chunk_size = (nrep + nthread - 1) // nthread

    args_list = []
    count = 0
    for i in range(nthread):
        count += chunk_size
        if count > nrep:
            args_list.append((ndat, count - nrep))
        else:
            args_list.append((ndat, chunk_size))

    pool = multiprocessing.Pool(processes=nthread)
    pools = [pool.apply_async(_randsamp_freq, args=args) for args in args_list]
    outputs = [p.get() for p in pools]
    return np.concatenate(outputs, axis=0)


def randsamp_freq(ndat, nrep, nthread=None, nproc=None, transpose=True):
    if nthread is not None:
        out = _randsamp_freq_threading(ndat, nrep, nthread)
    elif nproc is not None:
        out = _randsamp_freq_mult(ndat, nrep, nproc)
    else:
        out = _randsamp_freq(ndat, nrep)
    if transpose:
        out = out.T
    return out


def randsamp_numpy(ndat, nrep):
    index = np.random.choice(ndat, (ndat, nrep))
    freq = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=ndat), axis=0, arr=index)

    return index, freq


@njit
def _resample(data, freq, out):
    ndata = data.shape[0]
    nmeta = data.shape[1]

    ndat = freq.shape[0]
    nrep = freq.shape[1]
    assert ndat == ndata

    for idat in range(ndat):
        for imeta in range(nmeta):
            w = data[idat, imeta, 0]
            a = data[idat, imeta, 1]
            v = data[idat, imeta, 2]
            for irep in range(nrep):
                f = freq[idat, irep]
                if f > 0:
                    _push_stat(out[irep, imeta], f * w, a, v)


def _resample_data(data, freq, out=None):
    """
    reduce data along axis=0 from freq table
    """
    data_shape = data.shape
    assert data_shape[-1] == 3

    ndim = data.ndim
    assert ndim > 1

    assert data_shape[0] == freq.shape[0]
    nrep = freq.shape[-1]

    out_shape = (nrep, ) + data_shape[1:]

    if out is not None:
        assert out.shape == out_shape
    else:
        out = np.zeros(out_shape)

    if ndim == 2:
        datar_shape = (data_shape[0], 1, data_shape[-1])
    else:
        datar_shape = (data_shape[0], np.prod(data_shape[1:-1], dtype=np.int),
                       data_shape[-1])

    outr_shape = (nrep, ) + datar_shape[1:]

    #print(ndim, data.shape, datar_shape)

    datar = data.reshape(datar_shape)
    outr = out.reshape(outr_shape)

    _resample(datar, freq, outr)

    return out


def _resample_data_thread(data, freq, nthread):
    import threading
    nrep = freq.shape[1]
    out = np.zeros((nrep, ) + data.shape[1:], dtype=data.dtype)

    chunk_size = (nrep + nthread - 1) // nthread

    #args_list = [(data,_freq) for _freq in np.split(freq, nthread, axis=1)]
    args_list = [(data, freq[:, chunk_size * i:chunk_size * (i + 1)],
                  out[chunk_size * i:chunk_size * (i + 1), ...])
                 for i in range(nthread)]

    threads = [
        threading.Thread(target=_resample_data, args=args)
        for args in args_list
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    return out


def _resample_data_mult(data, freq, nthread):
    import multiprocessing
    if nthread < 0:
        nthread = multiprocessing.cpu_count()

    nrep = freq.shape[1]
    chunk_size = (nrep + nthread - 1) // nthread

    #args_list = [(data,_freq) for _freq in np.split(freq, nthread, axis=1)]
    args_list = [(data, freq[:, chunk_size * i:chunk_size * (i + 1)])
                 for i in range(nthread)]

    pool = multiprocessing.Pool(processes=nthread)
    pools = [pool.apply_async(_resample_data, args=args) for args in args_list]

    outputs = [p.get() for p in pools]
    return np.concatenate(outputs, axis=0)


# not totally sure why this is so much faster, but it is
from numba import prange


@njit(parallel=True)
def _resample_data_parallel_numba(data, freq):
    ndata = data.shape[0]
    nmeta = data.shape[1]

    ndat = freq.shape[0]
    nrep = freq.shape[1]
    assert ndat == ndata

    out = np.zeros((nrep, nmeta, 3))
    for irep in prange(nrep):
        for idat in range(ndat):
            f = freq[idat, irep]
            if f == 0.0:
                continue
            for imeta in range(nmeta):
                _w = data[idat, imeta, 0]
                if _w == 0.0:
                    continue
                a = data[idat, imeta, 1]
                v = data[idat, imeta, 2]

                w = _w * f

                weight = out[irep, imeta, 0] + w
                ave = out[irep, imeta, 1]
                var = out[irep, imeta, 2]

                alpha = w / weight
                delta = (a - ave)
                incr = delta * alpha

                out[irep, imeta, 0] += w
                out[irep, imeta, 1] += incr
                out[irep, imeta, 2] += (v - var) * alpha + (1.0 - alpha) * delta * incr

    return out

def _resample_data_parallel(data, freq):
    data_shape = data.shape
    assert data_shape[-1] == 3

    ndim = data.ndim
    assert ndim > 1

    assert data_shape[0] == freq.shape[0]
    nrep = freq.shape[-1]

    out_shape = (nrep, ) + data_shape[1:]

    if ndim == 2:
        datar_shape = (data_shape[0], 1, data_shape[-1])
    else:
        datar_shape = (data_shape[0], np.prod(data_shape[1:-1], dtype=np.int),
                       data_shape[-1])

    #print(ndim, data.shape, datar_shape)

    datar = data.reshape(datar_shape)
    outr = _resample_data_parallel_numba(datar, freq)

    out = outr.reshape(out_shape)
    return out



def resample_data(data, freq, parallel=True, nthread=None, nproc=None):

    if parallel:
        out = _resample_data_parallel(data, freq)

    elif nthread is not None:
        out = _resample_data_thread(data, freq, nthread)

    elif nproc is not None:
        out = _resample_data_mult(data, freq, nproc)

    else:
        out = _resample_data(data, freq)
    return out


# class _Stats(object):
#     """
#     Calculate mean/variance

#     Four versions to consider.

#     means: mean for each update

#     cummeans : cummulative mean for each update

#     mean : mean of last step (means[-1])

#     cummean : cummulative mean up to now (cummeans[-1])
#     """

#     def __init__(self, dtype=np.float):
#         # fundamental data object
#         # data = [mean, var/weight, weight]
#         # this is for stats in single update cycle
#         self.data = np.zeros(3, dtype=np.float)
#         self.zero()

#     def zero(self):
#         """zero everything"""
#         # stats now
#         self.data[:] = 0.0
#         # list of stats per store

#         self._list = []
#         self._datas = None
#         self._cumdatas = None

#     def push_stats(self, a, v=0.0, w=1.0):
#         """push values"""
#         _running_accum(self.data, a=a, v=v, w=w)

#     def push_val(self, x, w=1.0):
#         """push a single value"""
#         _running_accum(self.data, a=x, v=0.0, w=w)

#     def update(self):
#         # accumulate stats
#         self._list.append(self.data.copy())

#         # zero data
#         self.data[:] = 0.0

#         # zeros caches
#         self._datas = None
#         self._cumdatas = None

#     def __len__(self):
#         return len(self._list)

#     @property
#     def datas(self):
#         """[mean, var/W, W] of all updates"""
#         if self._datas is None:
#             self._datas = np.array(self._list)
#         return self._datas

#     @property
#     def means(self):
#         return self.datas[:, 0]

#     @property
#     def variances(self):
#         return self.datas[:, 1] / self.datas[:, 2]

#     @property
#     def weights(self):
#         return self.datas[:, 2]

#     @property
#     def data_last(self):
#         return self._list[-1]

#     @property
#     def mean(self):
#         return self.data_last[0]

#     @property
#     def variance(self):
#         return self.data_last[1] / self.data_last[2]

#     @property
#     def weight(self):
#         return self.data_last[2]

#     @property
#     def cumdatas(self):
#         if self._cumdatas is None:
#             # setup cumdatas
#             cum = np.zeros_like(self.data)

#             self._cumdatas = np.zeros_like(self.datas)
#             for i, data in enumerate(self.datas):
#                 _running_accum(cum, a=data[0], v=data[1] / data[2], w=data[2])
#                 self._cumdatas[i, :] = cum
#         return self._cumdatas

#     @property
#     def cummeans(self):
#         return self.cumdatas[:, 0]

#     @property
#     def cumvariances(self):
#         return self.cumdatas[:, 1] / self.cumdatas[:, 2]

#     @property
#     def cumweights(self):
#         return self.cumdatas[:, 2]

#     @property
#     def cummean(self):
#         return self.cumdatas[-1, 0]

#     @property
#     def cumvariance(self):
#         return self.cumdatas[-1, 1] / self.cumdatas[-1, 2]

#     @property
#     def cumweight(self):
#         return self.cumdatas[-1, 2]

# class Running_Accumulator(object):
#     """
#     Calculate running mean/variance
#     """

#     def __init__(self, dtype=np.float):
#         # fundamental data object
#         # data = [mean, var/weight, weight]
#         # this is for running stats.
#         self.data = np.zeros(3, dtype=np.float)
#         # this is for average per cycle
#         # calculated on demand
#         self.zero()

#     def zero(self):
#         """zero everything"""
#         self.data[:] = 0.0
#         self._list = []

#         self._data_cum = None
#         self._data_per = None

#     def push_stats(self, a, v=0.0, w=1.0):
#         """push values"""
#         _running_accum(self.data, a=a, v=v, w=w)

#     def update(self):
#         # accumulate stats
#         self._list.append(self.data.copy())
#         # zeros caches
#         self._data_cum = None
#         self._data_per = None

#     def mean(self):
#         return self.data[0]

#     def var(self):
#         return self.data[1] / self.data[2]

#     def weight(self):
#         return self.data[2]

#     # accumulated stats
#     @property
#     def data_cum(self):
#         if self._data_cum is None:
#             self._data_cum = np.array(self._list)
#         return self._data_cum

#     def mean_cum(self):
#         return self.data_cum[:, 0]

#     def var_cum(self):
#         return self.data_cum[:, 1] / self.data_cum[:, 2]

#     def weight_cum(self):
#         return self.data_cum[:, 2]

#     # per cycle stats
#     @property
#     def data_per(self):
#         if self._data_per is None:
#             # update data/cycle
#             data_cum = self.data_cum
#             self._data_per = data_cum.copy()

#             for i in range(1, data_cum.shape[0]):
#                 _running_accum(
#                     data=self._data_per[i, ...],
#                     a=data_cum[i - 1, 0],
#                     v=data_cum[i - 1, 1] / data_cum[i - 1, 2],
#                     w=-data_cum[i - 1, 2])
#         return self._data_per

#     def mean_per(self):
#         return self.data_per[:, 0]

#     def var_per(self):
#         return self.data_per[:, 1] / self.data_per[:, 2]

#     def weight_per(self):
#         return self.data_per[:, 2]

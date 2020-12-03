from __future__ import absolute_import
from functools import lru_cache

import numpy as np
from numba import njit, prange
from scipy.special import binom

from .cached_decorators import cached_clear, gcached


def myjit(func):
    """
    "my" jit function
    uses option inline='always', fastmath=True
    """
    return njit(inline="always", fastmath=True)(func)

def factory_binomial(order):
    irange = np.arange(order + 1)
    bfac = np.array([binom(i, irange) for i in irange])
    return bfac


def _my_broadcast(x, shape, dtype=None, order=None):
    x = np.asarray(x, dtype=dtype, order=order)
    if x.shape != shape:
        x = np.broadcast(x, shape)
    return x


def _shape_insert_axis(shape, axis, new_size):
    """
    given shape, get new shape with size put in position axis
    """
    n = len(shape)

    axis = np.core.numeric.normalize_axis_index(axis, n+1)
    # assert -(n+1) <= axis <= n
    # if axis < 0: 
    #     axis = axis + n + 1

    # if axis < 0:
    #     axis += len(shape) + 1
    shape = list(shape)
    shape.insert(axis, new_size)
    return tuple(shape)


def _shape_reduce(shape, axis):
    """given input shape, give shape after reducing along axis"""
    shape = list(shape)
    shape.pop(axis)
    return tuple(shape)


def _axis_expand_broadcast(x, shape, axis,
                           verify=True,
                           expand=True,
                           broadcast=True,
                           roll=True,
                           dtype=None, order=None):
    """
    broadcast x to shape.  If x is 1d, and shape is n-d, but len(x) is same
    as shape[axis], broadcast x across all dimensions
    """

    if verify is True:
        x = np.asarray(x, dtype=dtype, order=order)


    # if array, and 1d with size same as shape[axis]
    # broadcast from here
    if expand:
        if x.ndim == 1 and x.ndim != len(shape) and len(x) == shape[axis]:
            # reshape for broadcasting
            reshape = [1] * (len(shape) - 1)
            reshape = _shape_insert_axis(reshape, axis, -1)
            x = x.reshape(*reshape)

    if broadcast and x.shape != shape:
        x = np.broadcast_to(x, shape)
    if roll and axis != 0:
        x = np.moveaxis(x, axis, 0)
    return x



@lru_cache(maxsize=5)
def _cached_ones(shape, dtype=None):
    return np.ones(shape, dtype=dtype)






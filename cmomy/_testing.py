"""Common routings used in testing."""

import numpy as np


# Dumb calculations
def _get_cmom(w, x, moments, axis=0, last=True):
    if w is None:
        w = np.array(1.0)

    if w.ndim == 1 and w.ndim != x.ndim and len(w) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        w = w.reshape(*shape)

    if w.shape != x.shape:
        w = np.broadcast_to(w, x.shape)

    wsum_keep = w.sum(axis, keepdims=True)
    wsum_keep_inv = 1.0 / wsum_keep

    wsum = w.sum(axis)
    wsum_inv = 1.0 / wsum

    # get moments
    xave = (w * x).sum(axis, keepdims=True) * wsum_keep_inv
    dx = x - xave

    xmean = (w * x).sum(axis) * wsum_inv
    weight = wsum
    data = [weight, xmean]

    for n in range(2, moments + 1):
        y = (w * dx**n).sum(axis) * wsum_inv
        data.append(y)

    data = np.array(data)
    if last:
        data = np.moveaxis(data, 0, -1)
    return data


def _get_comom(w, x, y, moments, axis=0, broadcast=True):
    if w is None:
        w = np.array(1.0)

    if w.ndim == 1 and w.ndim != x.ndim and len(w) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        w = w.reshape(*shape)

    if w.shape != x.shape:
        w = np.broadcast_to(w, x.shape)

    if y.ndim != x.ndim and y.ndim == 1 and len(y) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        y = y.reshape(*shape)

    if broadcast and y.shape != x.shape:
        y = np.broadcast_to(y, x.shape)

    assert w.shape == x.shape
    assert y.shape == x.shape

    shape = list(x.shape)
    shape.pop(axis)
    shape = tuple(shape) + tuple(x + 1 for x in moments)

    out = np.zeros(shape)
    wsum = w.sum(axis)
    wsum_inv = 1.0 / wsum

    wsum_keep = w.sum(axis, keepdims=True)
    wsum_keep_inv = 1.0 / wsum_keep

    xave = (w * x).sum(axis, keepdims=True) * wsum_keep_inv
    dx = x - xave

    yave = (w * y).sum(axis, keepdims=True) * wsum_keep_inv
    dy = y - yave

    for i in range(moments[0] + 1):
        for j in range(moments[1] + 1):
            if i == 0 and j == 0:
                val = wsum

            elif i + j == 1:
                val = (w * x**i * y**j).sum(axis) * wsum_inv
            else:
                val = (w * dx**i * dy**j).sum(axis) * wsum_inv

            out[..., i, j] = val
    return out

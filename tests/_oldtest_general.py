import cmomy.central as central


import numpy as np
import pytest


# central moments single variable
def _get_data(shape, style=None, axis=None):
    if isinstance(shape, int):
        shape = (shape,)

    if style is None or style == "total":
        return np.random.rand(*shape)
    elif style == "broadcast":
        return np.random.rand(shape[axis])
    else:
        raise ValueError('unknown style')


def _get_weight(shape, style, axis=0):
    if isinstance(shape, int):
        shape = (shape,)

    if style is None:
        w = np.array(1)
    elif style == "total":
        w = np.random.rand(*shape)
    elif style == "broadcast":
        w = np.random.rand(shape[axis])
    else:
        raise ValueError("bad style")

    return w


def _get_cmom(w, x, moments, axis=0, last=True):
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
        y = (w * dx ** n).sum(axis) * wsum_inv
        data.append(y)

    data = np.array(data)
    if last:
        data = np.moveaxis(data, 0, -1)
    return data


def _get_comom(w, x, y, moments, axis=0, broadcast=True):
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
                val = (w * x ** i * y ** j).sum(axis) * wsum_inv

            else:
                val = (w * dx ** i * dy ** j).sum(axis) * wsum_inv

            out[..., i, j] = val

    return out




def setup_test_data(shape, axis, style, mom, nsplit=3):

    cov = isinstance(mom, tuple) and len(mom) == 2


    x = _get_data(shape)
    w = _get_weight(shape=shape, style=style, axis=axis)
    if cov:
        y = _get_data(shape=shape, style=style, axis=axis)


    # shape of data
    if isinstance(shape, int):
        shape = (shape,)
    dshape = list(shape)
    dshape.pop(axis)
    dshape = tuple(dshape)

    # split data for multiple calcalations
    v = x.shape[axis] // nsplit
    splits = [v * i for i in range(1, nsplit)]
    X = np.split(x, splits, axis=axis)

    if style == 'total':
        W = np.split(w, splits, axis=axis)
    elif style == 'broadcast':
        W = np.split(w, splits)
    else:
        W = [w for xx in X]

    if cov:
        if style == 'broadcast':
            Y = np.split(y, splits)
        else:
            Y = np.split(y, splits, axis=axis)

        # pack X, Y
        X = list(zip(X, Y))
        x = (x, y)

    return dshape, w, x, W, X





@pytest.mark.parametrize(
    "shape, axis", [(100, 0), ((50, 9, 10), 0), ((8, 50, 10), 1), ((8, 9, 50), 2)]
)
@pytest.mark.parametrize("style", [None, "total", "broadcast"])
@pytest.mark.parametrize("mom", [5, (5,5)])
def test_push_moments(shape, axis, style, mom):


    dshape, w, x, W, X = setup_test_data(shape, axis, style, mom)
    broadcast = style == 'broadcast'


    if isinstance(mom, int):
        cls = central.StatsAccum

        data_test = central.central_moments(x=x, mom=mom, w=w, axis=axis, last=True)
        np.testing.assert_allclose(data_test,
                                   _get_cmom(w=w, x=x, moments=mom, axis=axis, last=True))
    else:
        cls = central.StatsAccumCov
        data_test = central.central_comoments(x=x[0], y=x[1], mom=mom, w=w,
                                              axis=axis, last=True, broadcast=broadcast)

        np.testing.assert_allclose(data_test,
                                   _get_comom(w=w, x=x[0], y=x[1], moments=mom, axis=axis, broadcast=True)
        )


    S = []
    for ww, xx in zip(W, X):
        _s = cls.from_vals(x=xx, w=ww, axis=axis, mom=mom, broadcast=broadcast)
        S.append(_s)

    s = cls.zeros(mom=mom, shape=dshape)


    # push vals
    s.push_vals(x, w=w, axis=axis, broadcast=broadcast)
    np.testing.assert_allclose(s.values, data_test)

    # from_vals
    t = cls.from_vals(x=x, w=w, axis=axis, mom=mom, broadcast=broadcast)
    np.testing.assert_allclose(t.values, data_test)

    # push_val
    t = s.zeros_like()
    if axis == 0 and style == 'total' and hasattr(cls, 'push_stat'):
        for ww, xx in zip(w, x):
            t.push_val(x=xx, w=ww, broadcast=broadcast)
        np.testing.assert_allclose(t.values, data_test)


    # combining data
    t = s.zeros_like()
    for _s in S:
        t.push_data(_s.values)
    np.testing.assert_allclose(t.values, data_test)


    # push datas
    datas = np.array([_s.values for _s in S])
    t.zero()
    t.push_datas(datas, axis=0)
    np.testing.assert_allclose(t.values, data_test)


    # from datas
    t = cls.from_datas(datas, mom=mom)
    np.testing.assert_allclose(t.values, data_test)

    # from data
    t = cls.from_data(s.data)
    np.testing.assert_allclose(t.data, s.data)

    # multiples push_vals
    t.zero()
    for ww, xx in zip(W, X):
        t.push_vals(x=xx, w=ww, axis=axis, broadcast=broadcast)
    np.testing.assert_allclose(t.values, data_test)

    if hasattr(cls, 'push_stat'):
        # stats
        t = cls.from_stat(a=s.mean(), v=s.values[...,2:], w=s.weight(), mom=mom)
        np.testing.assert_allclose(t.values, data_test)

        t.zero()
        for _s in S:
            t.push_stat(_s.mean(), v=_s.values[..., 2:], w=_s.weight())
        np.testing.assert_allclose(t.values, data_test)


        t.zero()
        t.push_stats(
            a=np.array([_s.mean() for _s in S]),
            v=np.array([_s.values[..., 2:] for _s in S]),
            w=np.array([_s.weight() for _s in S]),
            axis=0
        )
        np.testing.assert_allclose(t.values, data_test)


    # addition
    t.zero()
    for _s in S:
        t = t + _s
    np.testing.assert_allclose(t.values, data_test)

    # summing
    t = sum(S, s.zeros_like())
    np.testing.assert_allclose(t.values, data_test)


    # inplace
    t = s.zeros_like()
    for _s in S:
        t += _s
    np.testing.assert_allclose(t.values, data_test)


    # subtraction
    t = s - sum(S[1:], s.zeros_like())
    np.testing.assert_allclose(t.values, S[0].values)

    # inplace
    t = s.copy()
    for _s in S[1:]:
        t -= _s
    np.testing.assert_allclose(t.values, S[0].values)

    # multiplication
    np.testing.assert_allclose(
        (s * 2).values,
        (s + s).values
    )

    t = s.copy()
    t *= 2
    np.testing.assert_allclose(t, (s+s).values)



# @pytest.mark.parametrize(
#     "shape, axis", [(100, 0), ((40, 30, 20), 0), ((40, 30, 20), 1), ((40, 30, 20), 2),]
# )

# @pytest.mark.parametrize("style", [None, "total", "broadcast"])
# @pytest.mark.parametrize('style_y', ['total', 'broadcast'])
# @pytest.mark.parametrize("mom", [(4,1), (1, 4), (4,4)])
# def test_push_comoments(shape, axis, style, style_y, mom):
#     x = _get_data(shape=shape)
#     y = _get_data(shape=shape, style=style_y, axis=axis)
#     w = _get_weight(shape=shape, style=style, axis=axis)
#     broadcast = style_y == 'broadcast'

#     data_test = central.central_comoments(x=x, y=y, broadcast=broadcast, mom=mom, w=w, axis=axis, last=True)

#     # blank create of object
#     if isinstance(shape, int):
#         shape = (shape,)

#     # shape of data
#     dshape = list(shape)
#     dshape.pop(axis)
#     dshape = tuple(dshape)

#     cls = central.StatsAccumCov
#     s = cls.from_vals((x,y), w=w, axis=axis, broadcast=broadcast, mom=mom)
#     np.testing.assert_allclose(s.data, data_test)

#     # split data for further analysis
#     v = x.shape[axis] // 3
#     splits = [v, v*2]

#     X = np.split(x, splits, axis=axis)
#     if style == 'total':
#         W = np.split(w, splits, axis=axis)
#     elif style == 'broadcast':
#         W = np.split(w, splits)
#     else:
#         W = [w for xx in X]


#     if style_y == 'total':
#         ys = np.split(y, splits, axis=axis)
#     else:
#         ys = np.split(y, splits)

#     # pack into tuples
#     X = [(xx, yy) for xx, yy in zip(X, ys)]


#     # print('W', [xx.shape for xx  in W])
#     # print('ys', [xx.shape for xx  in ys])
#     # print('X', [xx.shape for xx in  X])

#     S = []
#     for ww, xx in zip(W, X):
#         _test = central.central_comoments(x=xx[0], y=xx[1], mom=mom, w=ww, axis=axis, broadcast=broadcast)
#         _s = cls.from_vals(x=xx, w=ww, axis=axis, mom=mom, broadcast=broadcast)
#         np.testing.assert_allclose(_s.values, _test)
#         S.append(_s)

#     # StatsAccum
#     s = cls.zeros(mom=mom, shape=dshape)

#     _basic_push_tests(cls=cls, w=w, x=(x, y), data_test=data_test, dshape=dshape, W=W, X=X,
#                       s=s, S=S, axis=axis, mom=mom, style=style, broadcast=broadcast)

import numpy as np
import pytest
from numpy.core.multiarray import normalize_axis_index

from cmomy import CentralMoments, central_moments
from cmomy.central import _central_comoments


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


@pytest.fixture(
    scope="module",
    params=[
        (10, 0),
        ((10,), 0),
        ((1, 2, 3), 0),
        ((5, 6, 7), 0),
        ((5, 6, 7), 1),
        ((5, 6, 7), 2),
        ((5, 6, 7), -1),
        ((5, 6, 7), -2),
    ],
)
def shape_axis(request):
    return request.param


@pytest.fixture(scope="module")
def shape(shape_axis):
    return shape_axis[0]


@pytest.fixture(scope="module")
def shape_tuple(shape):
    if isinstance(shape, int):
        return (shape,)
    else:
        return shape


@pytest.fixture(scope="module")
def axis(shape_axis):
    return shape_axis[1]


@pytest.fixture(scope="module", params=[None, "total", "broadcast"])
def style(request):
    return request.param


@pytest.fixture(scope="module")
def xdata(shape_tuple):
    return np.random.rand(*shape_tuple)


@pytest.fixture(scope="module")
def ydata(shape_tuple, style, axis):
    shape = shape_tuple
    if style is None or style == "total":
        return np.random.rand(*shape)
    elif style == "broadcast":
        return np.random.rand(shape[axis])
    else:
        raise ValueError


@pytest.fixture(scope="module")
def wdata(shape_tuple, style, axis):
    shape = shape_tuple
    if style is None:
        return None
    elif style == "total":
        return np.random.rand(*shape)
    elif style == "broadcast":
        return np.random.rand(shape[axis])


@pytest.fixture(scope="module", params=[3, (3, 3)])
def mom(request):
    return request.param


@pytest.fixture(scope="module")
def mom_tuple(mom):
    if isinstance(mom, int):
        return (mom,)
    else:
        return mom


@pytest.fixture(scope="module")
def cov(mom_tuple):
    return len(mom_tuple) == 2


@pytest.fixture(scope="module")
def xydata(xdata, ydata, cov):
    if cov:
        return (xdata, ydata)
    else:
        return xdata


@pytest.fixture(scope="module")
def val_shape(shape_tuple, axis):
    shape = shape_tuple
    axis = normalize_axis_index(axis, len(shape))
    return shape[:axis] + shape[axis + 1 :]


@pytest.fixture(scope="module")
def broadcast(style):
    return style == "broadcast"


@pytest.fixture(scope="module")
def expected(xdata, ydata, wdata, mom_tuple, broadcast, axis):

    if len(mom_tuple) == 1:
        return _get_cmom(w=wdata, x=xdata, moments=mom_tuple[0], axis=axis, last=True)

    else:
        return _get_comom(
            w=wdata, x=xdata, y=ydata, axis=axis, moments=mom_tuple, broadcast=broadcast
        )


def test_simple(expected):
    assert isinstance(expected, np.ndarray)


def test_central_moments(xydata, wdata, mom, broadcast, axis, expected):
    out = central_moments(
        x=xydata, mom=mom, w=wdata, axis=axis, last=True, broadcast=broadcast
    )
    np.testing.assert_allclose(out, expected)
    # out = central_moments(

    # test using data
    out = np.zeros_like(expected)
    _ = central_moments(
        x=xydata, mom=mom, w=wdata, axis=axis, last=True, broadcast=broadcast, out=out
    )
    np.testing.assert_allclose(out, expected)


def test_mom_ndim():
    with pytest.raises(ValueError):
        CentralMoments(np.zeros((4, 4)), mom_ndim=0)

    with pytest.raises(ValueError):
        CentralMoments(np.zeros((4, 4)), mom_ndim=3)


@pytest.fixture
def c_obj(xydata, wdata, mom, broadcast, axis):
    return CentralMoments.from_vals(
        x=xydata, w=wdata, axis=axis, mom=mom, broadcast=broadcast
    )


def test_c_obj(c_obj, expected):
    np.testing.assert_allclose(c_obj.data, expected, rtol=1e-10, atol=1e-10)


def test_propeties(c_obj, shape_tuple, val_shape, mom_tuple):
    assert val_shape == c_obj.val_shape

    assert mom_tuple == c_obj.mom
    assert len(mom_tuple) == c_obj.mom_ndim


def test_push_vals(xydata, wdata, mom, mom_tuple, broadcast, axis, c_obj, val_shape):

    assert val_shape == c_obj.val_shape

    # create new
    new = c_obj.zeros_like()
    new.push_vals(x=xydata, w=wdata, axis=axis, broadcast=broadcast)
    np.testing.assert_allclose(new.data, c_obj.data)

    # create new
    new = CentralMoments.zeros(mom=mom, val_shape=val_shape)
    new.push_vals(x=xydata, w=wdata, axis=axis, broadcast=broadcast)
    np.testing.assert_allclose(new.data, c_obj.data)


def test_push(xydata, wdata, mom, mom_tuple, broadcast, style, axis, c_obj, val_shape):

    if len(mom_tuple) == 1:
        x = xydata
        y = None
    else:
        x, y = xydata

    x = np.moveaxis(x, axis, 0)

    # w
    if style is None:
        w_ = (None for _ in x)

    elif style == "total":
        w_ = (_ for _ in np.moveaxis(wdata, axis, 0))

    else:
        w_ = (_ for _ in wdata)

    if y is None:
        xy_ = (_ for _ in x)

    else:
        if style is None or style == "total":
            y = np.moveaxis(y, axis, 0)

        xy_ = zip(x, y)

    new = CentralMoments.zeros(val_shape=val_shape, mom=mom)

    for xy, w in zip(xy_, w_):
        new.push_val(x=xy, w=w, broadcast=broadcast)

    np.testing.assert_allclose(c_obj.data, new.data)


def split_data(xdata, ydata, wdata, axis, style, nsplit):

    v = xdata.shape[axis] // nsplit
    splits = [v * i for i in range(1, nsplit)]
    X = np.split(xdata, splits, axis=axis)

    if style == "total":
        W = np.split(wdata, splits, axis=axis)
        Y = np.split(ydata, splits, axis=axis)
    elif style == "broadcast":
        W = np.split(wdata, splits)
        Y = np.split(ydata, splits)
    elif style is None:
        W = [wdata for _ in X]
        Y = np.split(ydata, splits, axis=axis)

    # Stopping here for now.  Will continue down  the road
    pass

    # # test from vals
    # c = CentralMoments.from_vals(x=xydata, w=wdata, axis=axis, mom=mom, broadcast=broadcast)
    # np.testing.assert_allclose(c.data, expected, rtol=1e-10, atol=1e-10)

    # # test push


# class TestCentral:
#     @pytest.fixture(autouse=True)
#     def setup(self, shape, axis):
#         self.x = np.random.rand(*shape)

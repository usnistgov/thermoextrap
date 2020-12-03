import numpy as np
import cmomy.convert as convert
import cmomy.central as central
import pytest


@pytest.mark.parametrize('shape,axis', [
    ((100,), 0),
    ((100, 10), 0),
    ((10, 100), 1),
    ((100, 5, 5), 0),
    ((5, 100, 5), 1)])
@pytest.mark.parametrize('moments', [2, 5])
@pytest.mark.parametrize('weighted', [True, False])
def test_to_raw_moments(shape, axis, moments, weighted):
    x = np.random.rand(*shape)

    if weighted:
        weights = np.random.rand(*shape)

    else:
        weights = np.ones_like(x)

    raw = np.array([np.average(x**i, weights=weights, axis=axis) for i in range(moments+1)])

    raw[0,...] = weights.sum(axis)

    # central moments
    cen = central.central_moments(x, moments, weights, axis=axis, last=False)

    r = convert.to_raw_moments(cen, axis=0)
    np.testing.assert_allclose(raw, r)


    s = central.CentralMoments.from_vals(x, mom=moments, w=weights, axis=axis)

    r2 = np.moveaxis(s.to_raw().data, -1, 0)
    np.testing.assert_allclose(raw, r2)


    if len(cen.shape) > 1:
        for i in range(1, len(cen.shape)):
            c = np.moveaxis(cen, 0, i)
            r = convert.to_raw_moments(c, axis=i)
            np.testing.assert_allclose(r, np.moveaxis(raw, 0, i))



@pytest.mark.parametrize('shape,axis', [
    ((100,), 0),
    ((100, 10), 0),
    ((10, 100), 1),
    ((100, 5, 5), 0),
    ((5, 100, 5), 1)])
@pytest.mark.parametrize('moments', [2, 5])
@pytest.mark.parametrize('weighted', [True, False])
def test_to_central_moments(shape, axis, moments, weighted):
    x = np.random.rand(*shape)

    if weighted:
        weights = np.random.rand(*shape)

    else:
        weights = np.ones_like(x)

    raw = np.array([np.average(x**i, weights=weights, axis=axis) for i in range(moments+1)])

    raw[0,...] = weights.sum(axis)

    # central moments
    cen = central.central_moments(x, moments, weights, axis=axis, last=False)


    c = convert.to_central_moments(raw, axis=0)
    np.testing.assert_allclose(c, cen)


    _raw = np.moveaxis(raw, 0, -1)
    s = central.CentralMoments.from_raw(_raw, mom=moments)
    c2 = np.moveaxis(s.data, -1, 0)
    np.testing.assert_allclose(cen, c2)


    if len(cen.shape) > 1:
        for i in range(1, len(cen.shape)):
            r = np.moveaxis(raw, 0, i)
            c = convert.to_central_moments(r, axis=i)
            np.testing.assert_allclose(c, np.moveaxis(cen, 0, i))




@pytest.mark.parametrize('shape,axis', [
    ((100,), 0),
    ((100, 10), 0),
    ((10, 100), 1),
    ((100, 5, 5), 0),
    ((5, 100, 5), 1)])
@pytest.mark.parametrize('moments', [(2,2), (3,3)])
@pytest.mark.parametrize('weighted', [True, False])
def test_to_raw_comoments(shape, axis, moments, weighted):
    x = np.random.rand(*shape)
    x1 = np.random.rand(*shape)

    if weighted:
        weights = np.random.rand(*shape)

    else:
        weights = np.ones_like(x)

    # central moments
    cen = central.central_comoments(x, x1, moments, weights, axis=axis, last=False)

    raw = np.zeros_like(cen)
    for i in range(moments[0]+1):
        for j in range(moments[1]+1):
            raw[i, j] = np.average(x**i * x1**j, weights=weights, axis=axis)

    raw[0, 0] = weights.sum(axis)


    r = convert.to_raw_comoments(cen, axis=[0, 1])
    np.testing.assert_allclose(raw, r)


    s = central.CentralMomentsCov.from_vals(x=(x, x1), mom=moments, w=weights, axis=axis)
    r2 = np.moveaxis(s.to_raw().data, [-2, -1], [0, 1])
    np.testing.assert_allclose(raw, r2)


    if len(cen.shape) > 2:
        for i in range(1, len(cen.shape)-1):
            c = np.moveaxis(cen, [0,1], [i, i+1])
            r = convert.to_raw_comoments(c, axis=[i, i+1])
            np.testing.assert_allclose(r, np.moveaxis(raw, [0,1], [i,i+1]))



@pytest.mark.parametrize('shape,axis', [
    ((100,), 0),
    ((100, 10), 0),
    ((10, 100), 1),
    ((100, 5, 5), 0),
    ((5, 100, 5), 1)])
@pytest.mark.parametrize('moments', [(2,2), (5,5)])
@pytest.mark.parametrize('weighted', [True, False])
def test_to_central_comoments(shape, axis, moments, weighted):
    x = np.random.rand(*shape)
    x1 = np.random.rand(*shape)
    if weighted:
        weights = np.random.rand(*shape)

    else:
        weights = np.ones_like(x)

    # central moments
    cen = central.central_comoments(x, x1, moments, weights, axis=axis, last=False)

    raw = np.zeros_like(cen)
    for i in range(moments[0]+1):
        for j in range(moments[1]+1):
            raw[i, j] = np.average(x**i * x1**j, weights=weights, axis=axis)

    raw[0, 0] = weights.sum(axis)


    c = convert.to_central_comoments(raw, axis=[0,1])
    np.testing.assert_allclose(c, cen)


    _raw = np.moveaxis(raw, [0,1], [-2, -1])
    s = central.CentralMomentsCov.from_raw(_raw, mom=moments)
    c2 = np.moveaxis(s.data, [-2, -1], [0,1])
    np.testing.assert_allclose(cen, c2)



    if len(cen.shape) > 2:
        for i in range(1, len(cen.shape)-1):
            r = np.moveaxis(raw, [0,1], [i,i+1])
            c = convert.to_central_comoments(r, axis=[i,i+1])
            np.testing.assert_allclose(c, np.moveaxis(cen, [0,1], [i,i+1]))

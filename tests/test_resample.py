import numpy as np

import cmomy.central as central
import cmomy.accumulator as accumulator

import pytest


@pytest.mark.parametrize('shape,axis,nrep', [
    ((100,), 0, 20),
    ((100, 5), 0, 20),
    ((5, 100), 1, 20)])
@pytest.mark.parametrize('moments', [2, 5])
@pytest.mark.parametrize('weighted',[False, True])
def test_resample_vals(shape, axis, nrep, moments, weighted):
    """
    test resampling
    """

    x = np.random.rand(*shape)
    if weighted:
        weights = np.random.rand(*shape)
    else:
        weights = None

    ndat = shape[axis]
    shape = (nrep, ndat)
    idx = np.random.choice(ndat, shape, replace=True)

    # direct sampling
    xx = np.take(x, idx, axis=axis)
    if weighted:
        ww = np.take(weights, idx, axis=axis)
    else:
        ww = None
    data = central.central_moments(xx, moments, weights=ww, axis=axis+1)
    # could have rep in weird place so move it
    data = np.rollaxis(data, axis, 0)

    # frequnecy samplign
    freq = central.randsamp_freq(nrep, ndat, indices=idx)
    out = central.resample_vals(x, freq, moments, axis=axis, weights=weights)
    np.testing.assert_allclose(data, out)

    # factory
    s = central.StatsAccum.from_resample_vals(x=x, freq=freq, moments=moments, axis=axis, w=weights)
    np.testing.assert_allclose(s.data, out)



@pytest.mark.parametrize('shape,axis,nrep', [
    ((100, 10), 0, 20),
    ((100, 10, 5), 0, 20),
    ((100, 5, 10), 1, 20)])
@pytest.mark.parametrize('moments', [2, 5])
@pytest.mark.parametrize('weighted',[False, True])
def test_resample_data(shape, axis, nrep, moments, weighted):
    """
    test resampling
    """
    x = np.random.rand(*shape)
    if weighted:
        weights = np.random.rand(*shape)
    else:
        weights = None

    data = central.central_moments(x, moments, weights=weights, axis=0)

    ndat = data.shape[axis]
    shape = (nrep, ndat)

    idx = np.random.choice(ndat, shape, replace=True)
    freq = central.randsamp_freq(nrep, ndat, idx)

    if axis != 0:
        d = np.rollaxis(data, axis, 0)
    else:
        d = data
    ref = accumulator.resample_data(d[..., :3], freq.T, parallel=False)


    out = central.resample_data(data, freq, moments,
                                parallel=False, axis=axis)
    # only test against ref up to 3rd order
    np.testing.assert_allclose(ref, out[..., :3])


    # resample_and_reduce
    s = central.StatsAccum.from_vals(x, moments=moments, w=weights, axis=0)
    sr = s.resample_and_reduce(freq, axis=axis)
    np.testing.assert_allclose(out, sr.data)

    # resample/reduce
    sr = s.resample(idx, axis=axis).reduce(1)
    np.testing.assert_allclose(out, sr.data)





@pytest.mark.parametrize('shape,axis,nrep', [
    ((10,), 0, 20),
    ((10, 5), 0, 20),
    ((5, 10), 1, 20)])
@pytest.mark.parametrize('moments', [2, 5])
@pytest.mark.parametrize('weighted',[False, True])
def test_resample_data_against_vals(shape, axis, nrep, moments, weighted):
    """
    test resampling
    """
    x = np.random.rand(*shape)

    xx = x[None, ...]
    if weighted:
        weights = np.random.rand(*shape)
        ww = weights[None, ...]
    else:
        weights = None
        ww = None

    # first do a resampling of values
    ndat = x.shape[axis]

    idx = np.random.choice(ndat, (nrep, ndat), True)

    freq = central.randsamp_freq(nrep, ndat, idx)
    ref = central.resample_vals(x, freq, moments, weights=weights, axis=axis)

    # create singleton dataset
    s = central.StatsAccum.from_vals(x=xx, w=ww, axis=0, moments=moments)

    # resample
    out = central.resample_data(s.data, freq, moments, axis=axis)
    np.testing.assert_allclose(out, ref)

    sr = s.resample_and_reduce(freq, axis=axis)
    np.testing.assert_allclose(ref, sr.data)


    sr = s.resample(idx, axis=axis).reduce(1)
    np.testing.assert_allclose(ref, sr.data)




@pytest.mark.parametrize('shape,axis,nrep', [
    ((100,), 0, 20),
    ((100, 5), 0, 20),
    ((5, 100), 1, 20)])
@pytest.mark.parametrize('moments', [(2,2), (5,5)])
@pytest.mark.parametrize('weighted',[False, True])
def test_resample_vals_cov(shape, axis, nrep, moments, weighted):
    """
    test resampling
    """

    x = np.random.rand(*shape)
    x1 = np.random.rand(*shape)

    if weighted:
        weights = np.random.rand(*shape)
    else:
        weights = None

    ndat = shape[axis]
    shape = (nrep, ndat)
    idx = np.random.choice(ndat, shape, replace=True)

    # direct sampling
    xx = np.rollaxis(x, axis, 0)
    xx1 = np.rollaxis(x1, axis, 0)

    xx = np.take(xx, idx, axis=0)
    xx1 = np.take(xx1, idx, axis=0)

    if weighted:
        ww = np.rollaxis(weights, axis, 0)
        ww = np.take(ww, idx, axis=0)
    else:
        ww = None

    data = central.central_comoments(xx, xx1, moments, weights=ww, axis=1)

    # frequnecy sampling
    freq = central.randsamp_freq(nrep, ndat, indices=idx)
    out = central.resample_vals(x=x, x1=x1, freq=freq, moments=moments, axis=axis, weights=weights, parallel=False)
    np.testing.assert_allclose(data, out)

    # factory
    s = central.StatsAccumCov.from_resample_vals(x0=x, x1=x1, freq=freq, w=weights, moments=moments, axis=axis, resample_kws=dict(parallel=False))
    np.testing.assert_allclose(s.data, out)



@pytest.mark.parametrize('shape,axis,nrep', [
    ((100, 10), 0, 20),
    ((100, 10, 5), 0, 20),
    ((100, 5, 10), 1, 20)])
@pytest.mark.parametrize('moments', [(2,2), (5,5)])
@pytest.mark.parametrize('weighted',[False, True])
def test_resample_data_cov(shape, axis, nrep, moments, weighted):
    """
    test resampling
    """
    x = np.random.rand(*shape)
    x1 = np.random.rand(*shape)
    if weighted:
        weights = np.random.rand(*shape)
    else:
        weights = None

    data = central.central_comoments(x, x1, moments, weights=weights, axis=0)

    ndat = data.shape[axis]
    shape = (nrep, ndat)

    idx = np.random.choice(ndat, shape, replace=True)
    freq = central.randsamp_freq(nrep, ndat, idx)

    if axis != 0:
        d = np.rollaxis(data, axis, 0)
    else:
        d = data

    ref0 = accumulator.resample_data(d[..., :3, 0], freq.T, parallel=False)
    ref1 = accumulator.resample_data(d[..., 0, :3], freq.T, parallel=False)

    out = central.resample_data(data, freq, moments, parallel=False, axis=axis)

    # can only test up to third order
    np.testing.assert_allclose(ref0, out[..., :3, 0])
    np.testing.assert_allclose(ref1, out[..., 0, :3])

    s = central.StatsAccumCov.from_vals(x, x1, moments=moments, w=weights, axis=0)
    sr = s.resample_and_reduce(freq, axis=axis)
    np.testing.assert_allclose(out, sr.data)


    sr = s.resample(idx, axis=axis).reduce(1)
    np.testing.assert_allclose(out, sr.data)





@pytest.mark.parametrize('shape,axis,nrep', [
    ((10,), 0, 20),
    ((10, 5), 0, 20),
    ((5, 10), 1, 20)])
@pytest.mark.parametrize('moments', [(2,1), (5,5)])
@pytest.mark.parametrize('weighted',[False, True])
def test_resample_data_against_vals_cov(shape, axis, nrep, moments, weighted):
    """
    test resampling
    """
    x = np.random.rand(*shape)
    x1 = np.random.rand(*shape)

    xx = x[None, ...]
    xx1 = x1[None, ...]

    if weighted:
        weights = np.random.rand(*shape)
        ww = weights[None, ...]
    else:
        weights = None
        ww = None

    # first do a resampling of values
    ndat = x.shape[axis]
    freq = central.randsamp_freq(nrep, ndat)
    ref = central.resample_vals(x, freq, moments, x1=x1, weights=weights, axis=axis)

    # create singleton dataset
    s = central.StatsAccumCov.from_vals(x0=xx, x1=xx1, w=ww, axis=0, moments=moments)

    # resample
    out = central.resample_data(s.data, freq, moments, axis=axis)
    np.testing.assert_allclose(out, ref)

    sr = s.resample_and_reduce(freq, axis=axis)
    np.testing.assert_allclose(ref, sr.data)



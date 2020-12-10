import numpy as np

import cmomy.central as central


import pytest


def test_resample_vals(other):
    # test basic resampling
    if other.style == 'total':
        datar = central.resample_vals(
            x=other.x,
            mom=other.mom,
            freq=other.freq,
            axis=other.axis,
            w=other.w,
            mom_ndim=other.s._mom_ndim,
            broadcast=other.broadcast,
        )

        np.testing.assert_allclose(datar, other.data_test_resamp)



def test_stats_resample_vals(other):

    if other.style == 'total':
        t = other.cls.from_resample_vals(
            x=other.x,
            w=other.w,
            mom=other.mom,
            freq=other.freq,
            axis=other.axis,
            broadcast=other.broadcast,
        )
        np.testing.assert_allclose(t.data, other.data_test_resamp)

        # test based on indices
        t = other.cls.from_resample_vals(
            x=other.x,
            w=other.w,
            mom=other.mom,
            indices=other.indices,
            axis=other.axis,
            broadcast=other.broadcast,
        )
        np.testing.assert_allclose(t.data, other.data_test_resamp)



def test_resample_data(other):

    nrep = 10

    if len(other.val_shape) > 0:
        for axis in range(other.s.val_ndim):

            data = other.data_test

            ndat = data.shape[axis]

            idx = np.random.choice(ndat, (nrep, ndat), replace=True)
            freq = central.randsamp_freq(indices=idx)

            if axis != 0:
                data = np.rollaxis(data, axis, 0)
            data = np.take(data, idx, axis=0)
            data_ref = other.cls.from_datas(data, mom_ndim=other.mom_ndim, axis=1)


            t = other.s.resample_and_reduce(freq=freq, axis=axis)
            np.testing.assert_allclose(data_ref, t.data)



def test_resample_against_vals(other):

    nrep = 10

    if len(other.val_shape) > 0:
        s = other.s

        for axis in range(s.val_ndim):
            ndat = s.val_shape[axis]
            idx = np.random.choice(ndat, (nrep, ndat), replace=True)

            t0 = s.resample_and_reduce(indices=idx, axis=axis)

            t1 = s.resample(idx, axis=axis).reduce(1)

            np.testing.assert_allclose(t0.values, t1.values)



from cmomy.resample import bootstrap_confidence_interval, xbootstrap_confidence_interval


def test_bootstrap_stats(other):

    x = other.xdata
    axis = other.axis
    alpha = 0.05

    # test styles
    test = bootstrap_confidence_interval(x,
                                         stats_val=None,
                                         axis=axis,
                                         alpha=alpha)


    p_low = 100 * (alpha / 2.)
    p_mid = 50
    p_high = 100 - p_low

    expected = np.percentile(x, [p_mid, p_low, p_high], axis=axis)
    np.testing.assert_allclose(test, expected)

    # 'mean'
    test = bootstrap_confidence_interval(x,
                                         stats_val='mean',
                                         axis=axis,
                                         alpha=alpha)

    q_high = 100 * (alpha / 2.0)
    q_low = 100 - q_high
    stats_val = x.mean(axis=axis)
    val = stats_val
    low = 2 * stats_val - np.percentile(a=x, q=q_low, axis=axis)
    high = 2 * stats_val - np.percentile(a=x, q=q_high, axis=axis)


    expected = np.array([val, low, high])
    np.testing.assert_allclose(test, expected)


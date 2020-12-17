import numpy as np

# import cmomy.central as central
import cmomy.convert as convert

# import pytest


def test_to_raw_moments(other):
    raw = other.raw
    if raw is not None:
        # straight convert

        if not other.cov:
            r = convert.to_raw_moments(other.values, axis=-1)

        else:
            r = convert.to_raw_comoments(other.values, axis=(-2, -1))

        np.testing.assert_allclose(raw, r)
        np.testing.assert_allclose(raw, other.s.to_raw())


def test_to_central_moments(other):
    raw = other.s.to_raw()
    if not other.cov:
        cen = convert.to_central_moments(raw)
    else:
        cen = convert.to_central_comoments(raw)
    np.testing.assert_allclose(cen, other.values)

    # also test from raw method
    t = other.cls.from_raw(raw, mom=other.mom)
    np.testing.assert_allclose(t.values, other.values, rtol=1e-6, atol=1e-14)


def test_from_raws(other):
    raws = np.array([s.to_raw() for s in other.S])
    t = other.cls.from_raws(raws, mom_ndim=other.mom_ndim, axis=0)
    np.testing.assert_allclose(t.values, other.values)

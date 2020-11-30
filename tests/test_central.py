
import numpy as np
import pytest

from cmomy.cached_decorators import gcached
import cmomy.central as central



# Tests
def test_fix_test(other):
    np.testing.assert_allclose(other.data_fix, other.data_test)



def test_s(other):
    other.test_values(other.s.values)

def test_push(other):
    s, x, w, axis, broadcast = other.unpack("s", "x", "w", "axis", "broadcast",)
    t = s.zeros_like()
    t.push_vals(x, w=w, axis=axis, broadcast=broadcast)
    other.test_values(t.values)




def test_create(other):
    t = other.cls.zeros(mom=other.mom, shape=other.shape_val)
    t.push_vals(other.x, w=other.w, axis=other.axis, broadcast=other.broadcast)
    other.test_values(t.values)


def test_from_vals(other):
    t = other.cls.from_vals(
        x=other.x, w=other.w, axis=other.axis, mom=other.mom, broadcast=other.broadcast
    )
    other.test_values(t.values)


def test_push_val(other):
    if other.axis == 0 and other.style == "total":
        t = other.s.zeros_like()
        if other.s._mom_len == 1:
            print("do_push_val")
            for ww, xx in zip(other.w, other.x):
                t.push_val(x=xx, w=ww, broadcast=other.broadcast)
            other.test_values(t.values)


def test_push_vals_mult(other):
    t = other.s.zeros_like()
    for ww, xx, s in zip(other.W, other.X, other.S):
        t.push_vals(x=xx, w=ww, axis=other.axis, broadcast=other.broadcast)
    other.test_values(t.values)


def test_combine(other):
    t = other.s.zeros_like()
    for s in other.S:
        t.push_data(s.values)
    other.test_values(t.values)


def test_from_datas(other):
    datas = np.array([s.values for s in other.S])
    t = other.cls.from_datas(datas, mom=other.mom)
    other.test_values(t.values)


def test_push_datas(other):
    datas = np.array([s.values for s in other.S])
    t = other.s.zeros_like()
    t.push_datas(datas)
    other.test_values(t.values)


def test_push_stat(other):
    if other.s._mom_len == 1:

        t = other.s.zeros_like()
        for s in other.S:
            t.push_stat(s.mean(), v=s.values[..., 2:], w=s.weight())
        other.test_values(t.values)


def test_from_stat(other):
    if other.s._mom_len == 1:
        t = other.cls.from_stat(
            a=other.s.mean(),
            v=other.s.values[..., 2:],
            w=other.s.weight(),
            mom=other.mom,
        )
        other.test_values(t.values)


def test_from_stats(other):
    if other.s._mom_len == 1:
        t = other.s.zeros_like()
        t.push_stats(
            a=np.array([s.mean() for s in other.S]),
            v=np.array([s.values[..., 2:] for s in other.S]),
            w=np.array([s.weight() for s in other.S]),
            axis=0,
        )
        other.test_values(t.values)


def test_add(other):
    t = other.s.zeros_like()
    for s in other.S:
        t = t + s
    other.test_values(t.values)


def test_sum(other):
    t = sum(other.S, other.s.zeros_like())
    other.test_values(t.values)


def test_iadd(other):
    t = other.s.zeros_like()
    for s in other.S:
        t += s
    other.test_values(t.values)


def test_sub(other):
    t = other.s - sum(other.S[1:], other.s.zeros_like())
    np.testing.assert_allclose(t.values, other.S[0].values)


def test_isub(other):
    t = other.s.copy()
    for s in other.S[1:]:
        t -= s
    np.testing.assert_allclose(t.values, other.S[0].values)


def test_mult(other):
    s = other.s

    np.testing.assert_allclose(
        (s * 2).values,
        (s + s).values
    )

    t = s.copy()
    t *= 2
    np.testing.assert_allclose(t.values, (s+s).values)




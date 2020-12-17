"""
routines to convert central (co)moments to raw (co)moments
"""
from __future__ import absolute_import

import numpy as np

from .options import OPTIONS
from .utils import factory_binomial, myjit

_bfac = factory_binomial(OPTIONS["nmax"])


@myjit
def _central_to_raw_moments(central, raw):
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        ave = c[1]

        r[0] = c[0]
        r[1] = ave

        for n in range(2, order + 1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(0, n - 1):
                tmp += c[n - i] * ave_i * _bfac[n, i]
                ave_i *= ave

            # last two
            # <dx> = 0 so skip i = n-1
            # i = n
            tmp += ave_i * ave
            r[n] = tmp


@myjit
def _raw_to_central_moments(raw, central):
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        ave = r[1]

        c[0] = r[0]
        c[1] = ave

        for n in range(2, order + 1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(0, n - 1):
                tmp += r[n - i] * ave_i * _bfac[n, i]
                ave_i *= -ave

            # last two
            # right now, ave_i = (-ave)**(n-1)
            # i = n-1
            # ave * ave_i * n
            # i = n
            # 1 * (-ave) * ave_i
            tmp += ave * ave_i * (n - 1)
            c[n] = tmp


# comoments
@myjit
def _central_to_raw_comoments(central, raw):
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        ave0 = c[1, 0]
        ave1 = c[0, 1]

        for n in range(0, order0 + 1):
            for m in range(0, order1 + 1):
                nm = n + m
                if nm <= 1:
                    r[n, m] = c[n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n + 1):
                        ave_j = 1.0
                        for j in range(m + 1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp += ave_i * ave_j
                            elif nm_ij == 1:
                                # <dx**0 * dy**1> = 0
                                pass
                            else:
                                tmp += (
                                    c[n - i, m - j]
                                    * ave_i
                                    * ave_j
                                    * _bfac[n, i]
                                    * _bfac[m, j]
                                )
                            ave_j *= ave1
                        ave_i *= ave0
                    r[n, m] = tmp


@myjit
def _raw_to_central_comoments(raw, central):
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        c = central[v]
        r = raw[v]

        ave0 = r[1, 0]
        ave1 = r[0, 1]

        for n in range(0, order0 + 1):
            for m in range(0, order1 + 1):
                nm = n + m
                if nm <= 1:
                    c[n, m] = r[n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n + 1):
                        ave_j = 1.0
                        for j in range(m + 1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp += ave_i * ave_j
                            else:
                                tmp += (
                                    r[n - i, m - j]
                                    * ave_i
                                    * ave_j
                                    * _bfac[n, i]
                                    * _bfac[m, j]
                                )
                            ave_j *= -ave1
                        ave_i *= -ave0
                    c[n, m] = tmp


def _convert_moments(data, axis, target_axis, func, dtype=None, order=None, out=None):
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(target_axis, int):
        target_axis = (target_axis,)

    axis = tuple(axis)
    target_axis = tuple(target_axis)

    assert len(axis) == len(target_axis)

    data = np.asarray(data, dtype=dtype, order=order)
    if out is None:
        out = np.zeros_like(data)
    else:
        assert out.shape == data.shape
        out[...] = 0.0

    if axis != target_axis:
        data_r = np.moveaxis(data, axis, target_axis)
        out_r = np.moveaxis(out, axis, target_axis)
    else:
        data_r = data
        out_r = out

    shape = data_r.shape[: -len(axis)]
    if shape == ():
        reshape = (1,) + data_r.shape[-len(axis) :]
    else:
        reshape = (np.prod(shape),) + data_r.shape[-len(axis) :]

    data_r = data_r.reshape(reshape)
    out_r = out_r.reshape(reshape)

    func(data_r, out_r)
    return out


def to_raw_moments(x, axis=-1, dtype=None, order=None, out=None):
    """
    convert central moments to raw moments
    """
    if axis is None:
        axis = -1

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=-1,
        func=_central_to_raw_moments,
        dtype=dtype,
        order=order,
        out=out,
    )


def to_raw_comoments(x, axis=(-2, -1), dtype=None, order=None, out=None):
    """
    convert central moments to raw moments
    """

    if axis is None:
        axis = (-2, -1)

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=(-2, -1),
        func=_central_to_raw_comoments,
        dtype=dtype,
        order=order,
        out=out,
    )


def to_central_moments(x, axis=-1, dtype=None, order=None, out=None):
    """
    convert central moments to raw moments
    """

    if axis is None:
        axis = -1

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=-1,
        func=_raw_to_central_moments,
        dtype=dtype,
        order=order,
        out=out,
    )


def to_central_comoments(x, axis=(-2, -1), dtype=None, order=None, out=None):
    """
    convert raw comoments to central comoments
    """

    if axis is None:
        axis = (-2, -1)

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=(-2, -1),
        func=_raw_to_central_comoments,
        dtype=dtype,
        order=order,
        out=out,
    )

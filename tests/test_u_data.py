from collections import namedtuple

import numpy as np
import pytest

import thermoextrap.xtrapy.data as xdata
import thermoextrap.xtrapy.xpan_beta as xpan_beta
from thermoextrap.xtrapy import xpan_vol


# testing routines
def do_testing(attrs, obj0, obj1, **kwargs):
    for attr in attrs:
        val0, val1 = [getattr(obj, attr) for obj in (obj0, obj1)]
        val1 = val1.transpose(*val0.dims)
        np.testing.assert_allclose(val0, val1, **kwargs)


def do_testing_r(*args, **kwargs):
    return do_testing(["u", "xu"], *args, **kwargs)


def do_testing_c(*args, **kwargs):
    return do_testing(["du", "dxdu", "xave"], *args, **kwargs)


def do_testing_gen(central, *args, **kwargs):
    if central:
        f = do_testing_c
    else:
        f = do_testing_r
    f(*args, **kwargs)


@pytest.fixture(params=[1_000, 2_000])
def nsamp(request):
    return request.param


@pytest.fixture(params=[5])
def order(request):
    return request.param


@pytest.fixture(params=[True, False])
def central(request):
    return request.param


@pytest.fixture
def data(nsamp):
    x = np.random.rand(nsamp)
    u = np.random.rand(nsamp)
    Data = namedtuple("data", ["x", "u", "n"])

    data = Data(**{"x": x, "u": u, "n": nsamp})
    return data


@pytest.fixture
def data_x(request, data, order, central):
    return xpan_beta.factory_data(xv=data.u, uv=data.u, order=order, central=central)


@pytest.fixture(params=[True, False])
def xv_fixture(request, data):
    if request:
        return data.u
    else:
        return None


@pytest.fixture(params=["factory", "cmom", "cmom_vals"])
def data_other(request, data, xv_fixture, order, central):
    style = request.param

    if style == "factory":
        factory = xpan_beta.factory_data
    elif style == "cmom":
        factory = xdata.DataCentralMoments.from_vals
    elif style == "cmom_vals":
        factory = xdata.DataCentralMomentsVals.from_vals
    return factory(xv=xv_fixture, uv=data.u, order=order, central=central)


def test_factory_0(data_x, data_other, order, central):
    do_testing_gen(central, data_x, data_other)


# test extrap models
@pytest.fixture(params=[1.0])
def beta(request):
    return request.param


@pytest.fixture(params=[[0.2, 0.5, 0.8]])
def betas_extrap(request):
    return request.param


@pytest.fixture
def em_x(data_x, central, beta):
    return xpan_beta.factory_extrapmodel(beta=beta, data=data_x, central=central)


@pytest.fixture
def em_x_out(em_x, betas_extrap):
    return em_x.predict(betas_extrap, cumsum=True)


@pytest.fixture
def em_other(data_other, central, beta):
    return xpan_beta.factory_extrapmodel(beta=beta, data=data_other, central=central)


@pytest.fixture
def em_other_out(em_other, betas_extrap):
    return em_other.predict(betas_extrap, cumsum=True)


def test_em_other(em_x_out, em_other_out):
    np.testing.assert_allclose(em_x_out, em_other_out)


# x = u
@pytest.fixture(params=["factory", "cmom", "cmom_vals"])
def data_x_is_u(request, data, order, central):
    style = request.param

    if style == "factory":
        factory = xpan_beta.factory_data
    elif style == "cmom":
        factory = xdata.DataCentralMoments.from_vals
    elif style == "cmom_vals":
        factory = xdata.DataCentralMomentsVals.from_vals
    return factory(xv=None, uv=data.u, order=order, central=central, x_is_u=True)


@pytest.fixture
def em_x_is_u(data_x_is_u, central, beta):
    return xpan_beta.factory_extrapmodel(
        beta=beta, data=data_x_is_u, central=central, name="u_ave"
    )


@pytest.fixture
def em_x_is_u_out(em_x_is_u, betas_extrap):
    return em_x_is_u.predict(betas_extrap, cumsum=True)


def test_em_x_is_u(em_x_out, em_x_is_u_out):
    np.testing.assert_allclose(em_x_out, em_x_is_u_out)


# # test a higher moment?
# @pytest.fixture
# def em_x_i

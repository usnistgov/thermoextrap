from collections import namedtuple

import numpy as np
import pytest

import thermoextrap as xtrap


# testing routines
def do_testing(attrs, obj0, obj1, **kwargs):
    for attr in attrs:
        val0, val1 = (getattr(obj, attr) for obj in (obj0, obj1))
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


@pytest.fixture
def data(nsamp):
    x = np.random.rand(nsamp)
    u = np.random.rand(nsamp)
    Data = namedtuple("data", ["x", "u", "n"])

    data = Data(**{"x": x, "u": u, "n": nsamp})
    return data


@pytest.fixture(params=[True, False])
def central(request):
    return request.param


# test all other data constructors
@pytest.fixture
def data_x(data, order, central):
    return xtrap.factory_data_values(xv=data.u, uv=data.u, order=order, central=central)


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
        factory = xtrap.factory_data_values
    elif style == "cmom":
        factory = xtrap.DataCentralMoments.from_vals
    elif style == "cmom_vals":
        factory = xtrap.DataCentralMomentsVals.from_vals
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
    return xtrap.beta.factory_extrapmodel(beta=beta, data=data_x, central=central)


@pytest.fixture
def em_x_out(em_x, betas_extrap):
    return em_x.predict(betas_extrap, cumsum=True)


@pytest.fixture
def em_other(data_other, central, beta):
    return xtrap.beta.factory_extrapmodel(beta=beta, data=data_other, central=central)


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
        factory = xtrap.factory_data_values
    elif style == "cmom":
        factory = xtrap.DataCentralMoments.from_vals
    elif style == "cmom_vals":
        factory = xtrap.DataCentralMomentsVals.from_vals
    return factory(xv=None, uv=data.u, order=order, central=central, x_is_u=True)


@pytest.fixture
def em_x_is_u(data_x_is_u, central, beta):
    return xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_x_is_u, central=central, name="u_ave"
    )


@pytest.fixture
def em_x_is_u_out(em_x_is_u, betas_extrap):
    return em_x_is_u.predict(betas_extrap, cumsum=True)


def test_em_x_is_u(em_x_out, em_x_is_u_out):
    np.testing.assert_allclose(em_x_out, em_x_is_u_out)


# # test a higher moment?
# <u**2>, <du**2>
# <du**2> = <u**2> - <u>**2


@pytest.fixture
def data_u(data, order, central):
    return xtrap.factory_data_values(
        xv=None, uv=data.u, order=order, central=central, x_is_u=True
    )


@pytest.fixture
def em_u(data_u, central, beta):
    return xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_u, central=central, name="u_ave"
    )


@pytest.fixture
def em_u_out(em_u, betas_extrap):
    return em_u.predict(betas_extrap, cumsum=True)


def test_data_u(em_u_out, em_x_is_u_out):
    np.testing.assert_allclose(em_u_out, em_x_is_u_out)


@pytest.fixture
def data_x2(data, order, central):
    return xtrap.factory_data_values(
        xv=data.u**2, uv=data.u, order=order, central=central
    )


@pytest.fixture
def em_x2(beta, order, central, data_x2):
    return xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_x2, order=order - 1, central=central, name="x_ave"
    )


@pytest.fixture
def em_x2_out(em_x2, betas_extrap):
    return em_x2.predict(betas_extrap, cumsum=True)


@pytest.fixture
def em_u2(beta, data_u, order, central):
    if not central:
        return xtrap.beta.factory_extrapmodel(
            beta=beta,
            data=data_u,
            order=order - 1,
            central=central,
            name="un_ave",
            n=2,
        )

    else:
        return None


@pytest.fixture
def em_u2_out(em_u2, betas_extrap, central):
    if not central:
        return em_u2.predict(betas_extrap, cumsum=True)
    else:
        return None


def test_x2_u2(em_x2_out, em_u2_out, central):
    if not central:
        np.testing.assert_allclose(em_x2_out, em_u2_out)


def test_du2_3(beta, order, data, betas_extrap):
    data_u = xtrap.factory_data_values(
        uv=data.u, xv=None, x_is_u=True, order=order, central=True
    )
    data_u_r = xtrap.factory_data_values(
        uv=data.u, xv=None, x_is_u=True, order=order, central=False
    )

    em_du2 = xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_u, central=True, name="dun_ave", n=2, order=order - 1
    )
    em_du3 = xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_u, central=True, name="dun_ave", n=3, order=order - 2
    )

    em_u1 = xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_u_r, central=False, name="u_ave", order=order - 1
    )
    em_u1_sq = xtrap.beta.factory_extrapmodel(
        beta=beta,
        data=data_u_r,
        central=False,
        name="u_ave",
        order=order - 1,
        post_func="pow_2",
    )
    em_u1_cube = xtrap.beta.factory_extrapmodel(
        beta=beta,
        data=data_u_r,
        central=False,
        name="u_ave",
        order=order - 1,
        post_func="pow_3",
    )

    em_u2 = xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_u_r, central=False, name="un_ave", n=2, order=order - 1
    )
    em_u3 = xtrap.beta.factory_extrapmodel(
        beta=beta, data=data_u_r, central=False, name="un_ave", n=3, order=order - 2
    )

    # <du**2> = <u**2> - <u>**2
    a = em_du2.predict(betas_extrap, cumsum=True)
    b = em_u2.predict(betas_extrap, cumsum=True) - em_u1_sq.predict(
        betas_extrap, cumsum=True
    )

    np.testing.assert_allclose(a, b)

    # <du**3> = <u**3> - 3 * <u**2><u> + 2<u>**3
    # need to be carful with product <u**2> * <u>
    # <u**2> * <u>
    o = order - (3 - 1)
    kws = {"alpha": betas_extrap, "no_sum": True, "order": o}

    t_u3 = em_u3.predict(**kws)
    t_u2 = em_u2.predict(**kws)
    t_u1 = em_u1.predict(**kws)
    t_u1_cube = em_u1_cube.predict(**kws)

    t_u2_u1 = (
        (t_u2.rename(order="order_a") * t_u1.rename(order="order_b"))
        .assign_coords(order=lambda x: x["order_a"] + x["order_b"])
        .groupby("order")
        .sum()
        .reindex(order=t_u3.order)
    )

    b = (t_u3 - 3 * t_u2_u1 + 2 * t_u1_cube).cumsum("order")
    a = em_du3.predict(betas_extrap, cumsum=True)
    np.testing.assert_allclose(a, b)

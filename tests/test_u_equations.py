from collections import namedtuple

import pytest

import thermoextrap
import thermoextrap as xtrap
from thermoextrap.core.sputils import get_default_indexed

n_list = [6]


@pytest.fixture(params=n_list)
def data(request):
    data = namedtuple("data", ["n", "u", "x1", "du", "dxdu", "xu", "ui" "subs"])

    n = request.param
    data.n = n
    data.u, data.x1 = xtrap.core.models.get_default_symbol("u", "x1")
    data.du, data.dxdu = get_default_indexed("du", "dxdu")
    data.xu, data.ui = get_default_indexed("xu", "u")

    subs_central = {data.dxdu[i]: data.du[i + 1] for i in range(1, 2 * n)}
    subs_central[data.x1] = data.u

    subs_raw = {data.xu[i]: data.ui[i + 1] for i in range(0, 2 * n)}

    data.subs = {True: subs_central, False: subs_raw}

    return data


@pytest.fixture(params=[None, "minus_log"])
def post_func(request):
    return request.param


@pytest.fixture(params=[False, True])
def central(request):
    return request.param


def test_x_ave(data, central, post_func):
    n, subs = data.n, data.subs[central]

    f0 = xtrap.beta.factory_derivatives(
        name="x_ave", central=central, post_func=post_func
    )
    f1 = xtrap.beta.factory_derivatives(
        name="u_ave", central=central, post_func=post_func
    )

    for i in range(0, n + 1):
        assert f0.exprs[i].subs(subs) - f1.exprs[i] == 0


def test_central_dx(data):
    n, subs = data.n, data.subs[True]

    for m in range(1, n):
        f0 = xtrap.beta.factory_derivatives(name="dxdun_ave", n=m, central=True)
        f1 = xtrap.beta.factory_derivatives(name="dun_ave", n=m + 1, central=True)

        for i in range(0, n + 1):
            assert f0.exprs[i].subs(subs) - f1.exprs[i] == 0


def test_raw_un(data):
    n, subs = data.n, data.subs[False]

    for m in range(1, n):
        f0 = xtrap.beta.factory_derivatives(name="xun_ave", n=m, central=False)
        f1 = xtrap.beta.factory_derivatives(name="un_ave", n=m + 1, central=False)

        for i in range(0, n + 1):
            assert f0.exprs[i].subs(subs) - f1.exprs[i] == 0

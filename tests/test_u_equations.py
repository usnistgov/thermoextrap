from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest

import thermoextrap as xtrap
from thermoextrap.core.sputils import get_default_indexed

if TYPE_CHECKING:
    from typing import Any

    from sympy.core.symbol import Symbol
    from sympy.tensor.indexed import IndexedBase

n_list = [6]


class DataNamedTuple(NamedTuple):
    n: int
    u: Symbol
    x1: Symbol
    du: IndexedBase
    dxdu: IndexedBase
    xu: IndexedBase
    subs: dict[bool, Any]


@pytest.fixture(params=n_list)
def data(request) -> DataNamedTuple:
    n = request.param
    u, x1 = xtrap.models.get_default_symbol("u", "x1")
    du, dxdu = get_default_indexed("du", "dxdu")
    xu, ui = get_default_indexed("xu", "u")

    subs_central = {dxdu[i]: du[i + 1] for i in range(1, 2 * n)}
    subs_central[x1] = u
    subs_raw = {xu[i]: ui[i + 1] for i in range(2 * n)}

    subs = {True: subs_central, False: subs_raw}

    return DataNamedTuple(n=n, u=u, x1=x1, du=du, dxdu=dxdu, xu=xu, subs=subs)


@pytest.fixture(params=[None, "minus_log"])
def post_func(request):
    return request.param


@pytest.fixture(params=[False, True])
def central(request):
    return request.param


def test_x_ave(data, central, post_func) -> None:
    n, subs = data.n, data.subs[central]

    f0 = xtrap.beta.factory_derivatives(
        name="x_ave", central=central, post_func=post_func
    )
    f1 = xtrap.beta.factory_derivatives(
        name="u_ave", central=central, post_func=post_func
    )

    for i in range(n + 1):
        assert f0.exprs[i].subs(subs) - f1.exprs[i] == 0


def test_central_dx(data) -> None:
    n, subs = data.n, data.subs[True]

    for m in range(1, n):
        f0 = xtrap.beta.factory_derivatives(name="dxdun_ave", n=m, central=True)
        f1 = xtrap.beta.factory_derivatives(name="dun_ave", n=m + 1, central=True)

        for i in range(n + 1):
            assert f0.exprs[i].subs(subs) - f1.exprs[i] == 0


def test_raw_un(data) -> None:
    n, subs = data.n, data.subs[False]

    for m in range(1, n):
        f0 = xtrap.beta.factory_derivatives(name="xun_ave", n=m, central=False)
        f1 = xtrap.beta.factory_derivatives(name="un_ave", n=m + 1, central=False)

        for i in range(n + 1):
            assert f0.exprs[i].subs(subs) - f1.exprs[i] == 0

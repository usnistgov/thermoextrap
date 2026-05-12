from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr
from module_utilities import cached

import thermoextrap as xtrap
from thermoextrap.core.xrutils import xrwrap_uv, xrwrap_xv

from . import legacy


class FixtureData:  # pylint: disable=missing-class-docstring
    def __init__(self, n, nv, order=5, uoff=0.0, xoff=0.0, seed=0) -> None:
        self.order = order

        self.rng = np.random.default_rng(seed)
        self.uoff = uoff
        self.xoff = xoff

        self.u = xrwrap_uv(self.rng.random(n) + uoff)
        self.x = xrwrap_xv(self.rng.random((n, nv)) + xoff)

        self.ub = xrwrap_uv(self.rng.random(n) + uoff)
        self.xb = xrwrap_xv(self.rng.random((n, nv)) + xoff)
        self._cache: dict[str, Any] = {}

    # bunch of things to test
    @cached.prop
    def rdata(self):
        return legacy.data.factory_data_values(
            uv=self.u, xv=self.x, order=self.order, central=False
        )

    @cached.prop
    def cdata(self):
        return legacy.data.factory_data_values(
            uv=self.u, xv=self.x, order=self.order, central=True
        )

    @cached.prop
    def xdata(self):
        return xtrap.DataCentralMoments.from_vals(
            xv=self.x,
            uv=self.u,
            order=self.order,
            central=True,
            axis=0,
        )

    @cached.prop
    def xdata_val(self):
        return xtrap.DataCentralMomentsVals.from_vals(
            xv=self.x, uv=self.u, order=self.order, central=True
        )

    @cached.prop
    def xrdata(self):
        return xtrap.DataCentralMoments.from_vals(
            xv=self.x,
            uv=self.u,
            order=self.order,
            central=False,
            axis=0,
        )

    @cached.prop
    def xrdata_val(self):
        return xtrap.DataCentralMomentsVals.from_vals(
            xv=self.x,
            uv=self.u,
            order=self.order,
            central=False,
        )

    @property
    def beta0(self) -> float:
        return 0.5

    @property
    def betas(self):
        return [0.3, 0.4]

    @cached.prop
    def em(self):
        """Extrapolation model fixture"""
        em = legacy.ExtrapModel(maxOrder=self.order)
        em.train(self.beta0, xData=self.x, uData=self.u, saveParams=True)

        return em

    @cached.prop
    def u_xu_funcs(self):
        ufunc, xufunc = legacy.buildAvgFuncs(self.x, self.u, self.order)  # type: ignore[attr-defined]
        return ufunc, xufunc

    @cached.prop
    def derivs_list(self):
        fs = [legacy.symDerivAvgX(i) for i in range(self.order + 1)]  # type: ignore[attr-defined]
        ufunc, xufunc = self.u_xu_funcs  # pylint: disable=unpacking-non-sequence

        return [fs[i](ufunc, xufunc) for i in range(self.order + 1)]

    def xr_test_raw(self, b, a: Any = None) -> None:
        if a is None:
            a = self.rdata

        self.xr_test(a.u, b.u.sel(val=0))
        self.xr_test(a.xu, b.xu)

        for i in range(self.order):
            self.xr_test(a.u_selector[i], b.u_selector[i].sel(val=0))
            self.xr_test(a.xu_selector[i], b.xu_selector[i])

    def xr_test_central(self, b, a: Any = None) -> None:
        if a is None:
            a = self.cdata

        self.xr_test(a.du, b.du.sel(val=0))
        self.xr_test(a.dxdu, b.dxdu)
        self.xr_test(a.xave, b.xave)
        self.xr_test(a.xave_selector, b.xave_selector)

        for i in range(self.order):
            self.xr_test(a.du_selector[i], b.du_selector[i].sel(val=0))
            self.xr_test(a.dxdu_selector[i], b.dxdu_selector[i])

    @staticmethod
    def xr_test(a, b, **kwargs) -> None:
        xr.testing.assert_allclose(a, b.transpose(*a.dims), **kwargs)


@pytest.fixture(params=[(100, 5)])  # , scope="module")
def fixture(request):
    return FixtureData(*request.param)

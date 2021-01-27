import numpy as np
import pytest
import xarray as xr

import thermoextrap
import thermoextrap.xtrapy.data as xtrapy_data
import thermoextrap.xtrapy.models as xtrapy_models
import thermoextrap.xtrapy.xpan_beta as xpan_beta
from thermoextrap.xtrapy.cached_decorators import gcached


def test_rdata(fix_old, fix_rdata):

    ufunc, xufunc = fix_old.u_xu_funcs
    data = fix_rdata.data

    np.testing.assert_allclose(data.u, [ufunc(i) for i in range(fix_old.order + 1)])
    np.testing.assert_allclose(data.xu, [xufunc(i) for i in range(fix_old.order + 1)])


def test_xrdata(fix_rdata, fix_xrdata):
    print(type(fix_xrdata.data))
    print(fix_xrdata.data.u)
    fix_rdata.xr_test_raw(fix_xrdata.data)


def test_xrdata_val(fix_rdata, fix_xdata_val):
    fix_rdata.xr_test_raw(fix_xdata_val.data)


def test_xdata(fix_cdata, fix_xdata):
    fix_cdata.xr_test_central(fix_xdata.data)


def test_xdata_val(fix_cdata, fix_xdata_val):
    fix_cdata.xr_test_central(fix_xdata_val.data)


def test_rdata_derivs(fix_old, fix_rdata):

    a = fix_old
    b = fix_rdata

    vala = a.em.params
    valb = b.xem.derivatives.derivs(data=b.data)

    np.testing.assert_allclose(vala, valb)

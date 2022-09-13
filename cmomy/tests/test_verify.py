import numpy as np
import pytest
import xarray as xr

import cmomy


@pytest.fixture(scope="module")
def data():
    out = np.random.rand(1, 2, 3, 4)
    out = np.moveaxis(out, [2, 1], [1, 2])
    return out


@pytest.fixture(scope="module")
def xdata(data):
    return xr.DataArray(data)


def get_c_contig(data):
    return data.flags["C_CONTIGUOUS"]


def test_data(data, xdata):
    assert get_c_contig(data) is False
    assert xdata.values is data


@pytest.fixture(params=[1, 2])
def mom_ndim(request):
    return request.param


@pytest.mark.parametrize(
    "copy, verify, copy_order, same_data, c_contig",
    [
        (False, False, "C", True, False),
        (True, False, "K", False, False),
        (True, False, "C", False, True),
        (False, True, None, False, True),
    ],
)
def test_Central(data, mom_ndim, copy, verify, copy_order, same_data, c_contig):

    # no copy no verify should yield same thing this
    c = cmomy.CentralMoments.from_data(
        data,
        mom_ndim=mom_ndim,
        copy=copy,
        copy_kws={"order": copy_order},
        verify=verify,
    )
    assert (c.data is data) is same_data
    assert get_c_contig(c.data) is c_contig


@pytest.mark.parametrize(
    "copy, verify, copy_order, same_data, c_contig",
    [
        (False, False, "C", True, False),
        (True, False, "K", False, False),
        (True, False, "C", False, True),
        (False, True, None, False, True),
    ],
)
def test_xCentral(data, mom_ndim, copy, verify, copy_order, same_data, c_contig):

    # no copy no verify should yield same thing this
    c = cmomy.xCentralMoments.from_data(
        data,
        mom_ndim=mom_ndim,
        copy=copy,
        copy_kws={"order": copy_order},
        verify=verify,
    )
    assert c.data is c.values.data
    assert (c.data is data) is same_data
    assert get_c_contig(c.data) is c_contig


@pytest.mark.parametrize(
    "copy, verify, copy_order, same_data, c_contig, same_xdata",
    [
        (False, False, "C", True, False, True),
        (True, False, "K", False, False, False),
        (True, False, "C", False, True, False),
        (False, True, None, False, True, False),
    ],
)
def test_xCentral_xdata(
    xdata, mom_ndim, copy, verify, copy_order, same_data, c_contig, same_xdata
):

    # no copy no verify should yield same thing this
    c = cmomy.xCentralMoments.from_data(
        xdata,
        mom_ndim=mom_ndim,
        copy=copy,
        copy_kws={"order": copy_order},
        verify=verify,
    )

    assert c.data is c.values.data

    assert (c.data is xdata.values) is same_data
    assert (c.values is xdata) is same_xdata
    assert get_c_contig(c.data) is c_contig

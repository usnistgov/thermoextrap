"""test lnPi stuff"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import thermoextrap as xtrap


# Equations
@pytest.mark.parametrize("order", [6])
def test_equations(central, order):
    f0 = xtrap.beta.factory_derivatives("u_ave", central=central)
    f1 = xtrap.lnpi.factory_derivatives(central=central)

    for i in range(2, order):
        assert f0.exprs[i] + f1.exprs[i + 1] == 0


@pytest.fixture(params=[True, False])
def central(request):
    return request.param


data_path = Path(__file__).parent / "lnpi_data"


def load_data():
    with open(data_path / "sample_data.json") as f:
        d = json.load(f)

    ref, samples = d["ref"], d["samples"]
    for x in [ref] + samples:
        # cleanup data
        x["lnPi"] = np.array(x["lnPi"])
        x["energy"] = np.array(x["energy"])

    return ref, samples


def prepare_data(lnPi, energy, mu, temp, order, beta):
    beta = 1.0 / temp
    mu = xr.DataArray(np.atleast_1d(mu), dims=["comp"])
    lnPi = xr.DataArray(lnPi, dims=["n"])

    # adjust lnPi to have lnPi[n=0] = 0
    lnPi = lnPi - lnPi.sel(n=0)

    # have to include mom = 0
    a = np.ones_like(lnPi)
    energy = np.concatenate((a[:, None], energy), axis=-1)
    energy = xr.DataArray(energy, dims=["n", "umom"])

    return {
        "lnPi": lnPi,
        "energy": energy,
        "mu": mu,
        "beta": beta,
        "order": order,
        "temp": temp,
    }


@pytest.fixture
def sample_data():
    ref, samples = load_data()
    ref = prepare_data(**ref)
    samples = [prepare_data(**d) for d in samples]
    return ref, samples


@pytest.fixture
def ref(sample_data):
    return sample_data[0]


@pytest.fixture
def samples(sample_data):
    return sample_data[1]


@pytest.fixture
def betas(samples):
    return np.unique([s["beta"] for s in samples])


@pytest.fixture
def temps(betas):
    return np.round(1.0 / betas, 3)


def test_data(ref, samples, betas, temps):
    assert isinstance(ref, dict)
    assert isinstance(samples, list)
    assert isinstance(betas, np.ndarray)


@pytest.fixture
def data_u(ref, central):
    return xtrap.DataCentralMoments.from_ave_raw(
        u=ref["energy"], xu=None, x_is_u=True, central=central, meta=None
    )


@pytest.fixture
def em_u(data_u, ref):
    return xtrap.beta.factory_extrapmodel(beta=ref["beta"], data=data_u, name="u_ave")


@pytest.fixture
def out_u(em_u, betas):
    return em_u.predict(betas, cumsum=True)


def test_out_u(samples, out_u):
    for s in samples:
        a = s["energy"].sel(umom=1)
        b = out_u.sel(beta=s["beta"], order=s["order"])
        np.testing.assert_allclose(a, b, rtol=1e-5)


@pytest.fixture
def meta_lnpi(ref):
    return xtrap.lnpi.lnPiDataCallback(
        ref["lnPi"], ref["mu"], dims_n=["n"], dims_comp="comp"
    )


@pytest.fixture
def data_lnpi(data_u, meta_lnpi):
    return data_u.new_like(meta=meta_lnpi)


@pytest.fixture
def em_lnpi(data_lnpi, ref):
    return xtrap.lnpi.factory_extrapmodel_lnPi(beta=ref["beta"], data=data_lnpi)


@pytest.fixture
def out_lnpi(em_lnpi, betas):
    return (
        em_lnpi.predict(betas, cumsum=True)
        .pipe(lambda x: x - x.sel(n=0))
        .assign_coords(temp=lambda x: np.round(1.0 / x["beta"], 3))
    )


def test_out_lnpi(samples, out_lnpi):
    for s in samples:
        a = s["lnPi"]
        b = out_lnpi.sel(beta=s["beta"], order=s["order"])
        np.testing.assert_allclose(a, b)

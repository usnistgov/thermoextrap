import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.core import multiarray

import thermoextrap.xtrapy.core as xtrapy_core
import thermoextrap.xtrapy.xpan_beta as xpan_beta
import thermoextrap.xtrapy.xstack as xstack


@pytest.fixture
def states():
    shape = (3, 2, 4)
    dims = ["rec", "pair", "position"]

    coords = {"position": np.linspace(0, 2, shape[-1])}

    xems = []
    for beta in [0.1, 10.0]:
        x = xr.DataArray(np.random.rand(*shape), dims=dims, coords=coords)
        u = xr.DataArray(np.random.rand(shape[0]), dims=dims[0])
        data = xpan_beta.DataCentralMomentsVals.from_vals(x, u, order=3, central=True)
        xems.append(xpan_beta.factory_extrapmodel(beta, data))
    s = xtrapy_core.StateCollection(xems)

    return s.resample(nrep=3)


def test_mean_var(states):

    x = states[0].xcoefs(norm=False)

    out = xstack.to_mean_var(x, dim="rep")

    xr.testing.assert_allclose(out.sel(variable="mean", drop=True), x.mean("rep"))
    xr.testing.assert_allclose(out.sel(variable="var", drop=True), x.var("rep"))

    # test concat_dim
    out = xstack.to_mean_var(x, dim="rep", concat_dim="var")

    xr.testing.assert_allclose(out.sel(var=0, drop=True), x.mean("rep"))
    xr.testing.assert_allclose(out.sel(var=1, drop=True), x.var("rep"))


def test_xcoefs_concat(states):

    a = xr.concat(
        (s.xcoefs(norm=False) for s in states),
        dim=pd.Index(states.alpha0, name=states.alpha_name),
    )
    b = xstack.states_xcoefs_concat(states)
    xr.testing.assert_allclose(a, b)


def test_stack(states):

    Y_unstack = xstack.states_xcoefs_concat(states).pipe(xstack.to_mean_var, "rep")
    Y = xstack.stack_dataarray(Y_unstack, xdims=["beta", "order"], vdim="variable")

    X = xstack.multiindex_to_array(Y.indexes["xstack"])

    ij = 0
    for i, beta in enumerate(Y_unstack.beta):
        for j, order in enumerate(Y_unstack.order):
            np.testing.assert_allclose((beta, order), X[ij, :])
            ij += 1

    Ytest = Y_unstack.transpose("beta", "order", ..., "variable")
    newshape = (Ytest.sizes["beta"] * Ytest.sizes["order"], -1, Ytest.sizes["variable"])
    np.testing.assert_allclose(Ytest.values.reshape(newshape), Y.values)

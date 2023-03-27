import numpy as np
import pandas as pd
import pytest
import xarray as xr

import thermoextrap as xtrap
from thermoextrap.core import stack


@pytest.fixture
def states():
    shape = (3, 2, 4)
    dims = ["rec", "pair", "position"]

    coords = {"position": np.linspace(0, 2, shape[-1])}

    xems = []
    for beta in [0.1, 10.0]:
        x = xr.DataArray(np.random.rand(*shape), dims=dims, coords=coords)
        u = xr.DataArray(np.random.rand(shape[0]), dims=dims[0])
        data = xtrap.DataCentralMomentsVals.from_vals(x, u, order=3, central=True)
        xems.append(xtrap.beta.factory_extrapmodel(beta, data))
    s = xtrap.StateCollection(xems)

    return s.resample(nrep=3)


def test_mean_var(states):
    x = states[0].derivs(norm=False)

    out = stack.to_mean_var(x, dim="rep")

    xr.testing.assert_allclose(out.sel(stats="mean", drop=True), x.mean("rep"))
    xr.testing.assert_allclose(out.sel(stats="var", drop=True), x.var("rep"))

    # test concat_dim
    out = stack.to_mean_var(x, dim="rep", concat_dim="var")

    xr.testing.assert_allclose(out.sel(var=0, drop=True), x.mean("rep"))
    xr.testing.assert_allclose(out.sel(var=1, drop=True), x.var("rep"))


def test_derivs_concat(states):
    a = xr.concat(
        (s.derivs(norm=False) for s in states),
        dim=pd.Index(states.alpha0, name=states.alpha_name),
    )
    b = stack.states_derivs_concat(states)
    xr.testing.assert_allclose(a, b)


def test_stack(states):
    Y_unstack = stack.states_derivs_concat(states).pipe(stack.to_mean_var, "rep")
    Y = stack.stack_dataarray(Y_unstack, x_dims=["beta", "order"], stats_dim="stats")

    X = stack.multiindex_to_array(Y.indexes["xstack"])

    ij = 0
    for i, beta in enumerate(Y_unstack.beta):
        for j, order in enumerate(Y_unstack.order):
            np.testing.assert_allclose((beta, order), X[ij, :])
            ij += 1

    Ytest = Y_unstack.transpose("beta", "order", ..., "stats")
    newshape = (Ytest.sizes["beta"] * Ytest.sizes["order"], -1, Ytest.sizes["stats"])
    np.testing.assert_allclose(Ytest.values.reshape(newshape), Y.values)

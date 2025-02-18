import cmomy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import thermoextrap as xtrap
from thermoextrap import stack


@pytest.fixture
def states():
    shape = (3, 2, 4)
    dims = ["rec", "pair", "position"]
    coords = {"position": np.linspace(0, 2, shape[-1])}

    rng = cmomy.random.default_rng()

    xems = []
    for beta in [0.1, 10.0]:
        x = xr.DataArray(rng.random(shape), dims=dims, coords=coords)
        u = xr.DataArray(rng.random(shape[0]), dims=dims[0])
        data = xtrap.DataCentralMomentsVals.from_vals(x, u, order=3, central=True)
        xems.append(xtrap.beta.factory_extrapmodel(beta, data))
    s = xtrap.StateCollection(xems)

    return s.resample(sampler={"nrep": 3})


def test_mean_var(states) -> None:
    x = states[0].derivs(norm=False)

    out = stack.to_mean_var(x, dim="rep")

    xr.testing.assert_allclose(out.sel(stats="mean", drop=True), x.mean("rep"))
    xr.testing.assert_allclose(out.sel(stats="var", drop=True), x.var("rep"))

    # test concat_dim
    out = stack.to_mean_var(x, dim="rep", concat_dim="var")

    xr.testing.assert_allclose(out.sel(var=0, drop=True), x.mean("rep"))
    xr.testing.assert_allclose(out.sel(var=1, drop=True), x.var("rep"))


def test_derivs_concat(states) -> None:
    a = xr.concat(
        (s.derivs(norm=False) for s in states),
        dim=pd.Index(states.alpha0, name=states.alpha_name),
    )
    b = stack.states_derivs_concat(states)
    xr.testing.assert_allclose(a, b)


def test_stack(states) -> None:
    y_unstack = stack.states_derivs_concat(states).pipe(stack.to_mean_var, "rep")
    y_data = stack.stack_dataarray(
        y_unstack, x_dims=["beta", "order"], stats_dim="stats"
    )

    x_data = stack.multiindex_to_array(y_data.indexes["xstack"])

    ij = 0
    for _i, beta in enumerate(y_unstack.beta):
        for _j, order in enumerate(y_unstack.order):
            np.testing.assert_allclose((beta, order), x_data[ij, :])
            ij += 1

    y_test = y_unstack.transpose("beta", "order", ..., "stats")
    newshape = (y_test.sizes["beta"] * y_test.sizes["order"], -1, y_test.sizes["stats"])
    np.testing.assert_allclose(y_test.values.reshape(newshape), y_data.values)

"""
A set of routines to stack data for gpflow analysis
"""


import numpy as np
import pandas as pd
import xarray as xr

from .cached_decorators import gcached
from .core import StateCollection


def stack_dataarray(
    da,
    x_dims,
    y_dims=None,
    xstack_dim="xstack",
    ystack_dim="ystack",
    stats_dim=None,
    policy="infer",
):
    """
    Given an xarray.DataArray, stack for gpflow analysis

    Parameters
    ----------
    da : xarray.DataArray
        input DataArray
    x_dims : str or sequence of str
        dimensions stacked under `xstack_dim`
    y_dims : str or sequance of str, optional
        dimensions stacked under `ystack_dim`.
        Defaults to all dimenions not specified by `x_dims` or `stats_dim`.
    xstack_dim, ystack_dim : str
        name of new stacked dimension from stacking `x_dims`, `y_dims`
    stats_dim : str or tuple of strings, optional
        Use this to indicate a dimension that contains statistics (mean and variance).
        This dimenension is moved to the last position.
    policy : {'infer', 'raise'}
        policy if coordinates not available

    Returns
    -------
    da_stacked : xarray.DataArray
        stacked DataArray
    """
    dims = da.dims
    for name in [xstack_dim, ystack_dim]:
        if name in dims:
            raise ValueError("{} conficts with existing {}".format(xstack_dim, dims))

    if isinstance(x_dims, str):
        x_dims = (x_dims,)

    stacker = {xstack_dim: x_dims}
    if isinstance(y_dims, str):
        y_dims = (y_dims,)
    elif y_dims is None:
        # could use set, but would rather keep order
        y_dims = [x for x in dims if x not in x_dims]
        if stats_dim is not None:
            y_dims.remove(stats_dim)
        y_dims = tuple(y_dims)

    if len(y_dims) > 0:
        stacker[ystack_dim] = y_dims

    if policy == "raise":
        for dim in x_dims:
            if dim not in da.coords:
                raise ValueError("da.coords[{}] not set".format(dim))

    out = da.stack(**stacker)

    if stats_dim is not None:
        if isinstance(stats_dim, str):
            stats_dim = (stats_dim,)
        out = out.transpose(..., *stats_dim)

    return out


def wrap_like_dataarray(x, da):
    """
    wrap an array x with properties of da
    """
    return xr.DataArray(
        x,
        dims=da.dims,
        coords=da.coords,
        indexes=da.indexes,
        attrs=da.attrs,
        name=da.name,
    )


def multiindex_to_array(idx):
    """
    turn xarray multiindex to numpy arrayj
    """
    return np.array(list(idx.values))


def to_mean_var(da, dim, concat_dim=None, concat_kws=None, **kws):
    """
    for a dataarray apply mean/variance along a dimension

    Parameters
    ----------
    da : DataArray
        DataArray to be analyzed
    dim : str
        dimension to reduce along
    kws : dict
        Reduction arguments to `xarray.DataArray.mean` and `xarray.DataArray.var`
    concat_dim : str or DataArray or pandas.Index, optional
        dimension parameter to `xarray.concat`.
        Defaults to creating a dimension `stats` with names `['mean', 'var']`
    concat_kws : dict
        key-word arguments to `xarray.concat`
    """
    if concat_kws is None:
        concat_kws = {}

    if concat_dim is None:
        concat_dim = pd.Index(["mean", "var"], name="stats")

    return xr.concat(
        (da.mean(dim, **kws), da.var(dim, **kws)), dim=concat_dim, **concat_kws
    )


def states_xcoefs_concat(states, dim=None, concat_kws=None, **kws):
    """
    concatanate [s.xcoefs(norm=False) for s in states]

    Parameters
    ----------
    states : StateCollection
        states to consider
    dim : str or DataArray or pandas.Index, optional
        dimension to concat along.
        Defaults to `pandas.Index(states.alpha0, name=states.alpha_name)`
    concat_kws : dict
        extra arguments to xarray.concat

    kws : dict
        extra arguments to `states[i].xcoefs` method
        Note, default is `norm = False`
    Returns
    -------
    out : DataArray
    """
    if dim is None:
        dim = pd.Index(states.alpha0, name=states.alpha_name)

    if concat_kws is None:
        concat_kws = {}

    kws.setdefault("norm", False)

    return xr.concat((s.xcoefs(**kws) for s in states), dim=dim, **concat_kws)


class GPRData(StateCollection):
    """
    Statecollection for GPFlow analysis

    Parameters
    ----------
    collection : StateCollection object
    x_dims : sequence of str
        dimensions for X.   The last element should correspond to the dimension
        which specifies the order of the derivative (eg, 'order').
        If not specified, then `x_dims = [collections.alpha_name, 'order']`
    y_dims : str or sequence of str
        dimensions for Y
    reduce_dim : str, default='rep'
        name of dimensions to calculate mean/variance along
    stats_dim : str, default='stats'
        name of mean/variance dimension
    xstack_dim, ystack_dim : str
        name of new stacked dimensions
    order_dim : str, default='order'
        name of derivative order dimension
    deriv_kws : dict, optional
        optional arguments to be passed to `collection[i].xcoefs`
    """

    # reduce_dim -> dimension to reduce alo

    def __init__(
        self,
        states,
        x_dims=None,
        y_dims=None,
        xstack_dim="xstack",
        ystack_dim="ystack",
        stats_dim="stats",
        reduce_dim="rep",
        deriv_kws=None,
    ):
        if x_dims is None:
            x_dims = [states[0].alpha_name, "order"]
        if deriv_kws is None:
            deriv_kws = {}

        self.x_dims = x_dims
        self.y_dims = y_dims
        self.xstack_dim = xstack_dim
        self.ystack_dim = ystack_dim
        self.stats_dim = stats_dim
        self.reduce_dim = reduce_dim
        self.deriv_kws = deriv_kws

        super().__init__(
            states,
            x_dims=self.x_dims,
            y_dims=self.y_dims,
            xstack_dim=self.xstack_dim,
            ystack_dim=self.ystack_dim,
            stats_dim=self.stats_dim,
            reduce_dim=self.reduce_dim,
            deriv_kws=self.deriv_kws,
        )

    @property
    def order_dim(self):
        return self.x_dims[-1]

    @gcached(prop=False)
    def _stacked(self, order):
        """
        Get stacked data representation

        Parameters
        ----------
        order : int
            order of derivatives to consider
        kws : dict
            extra arguments to `self.xcoefs`

        Returns
        -------
        stacked : DataArray
            this will be in a stacked representation

        See Also
        --------
        states_xcoefs_concat, to_mean_var, stack_dataarray

        """

        kws = dict(self.deriv_kws, order_name=self.order_dim)
        return (
            states_xcoefs_concat(self, order=order, **kws)
            .pipe(
                to_mean_var,
                dim=self.reduce_dim,
                concat_dim=pd.Index(["mean", "var"], name=self.stats_dim),
            )
            .pipe(
                stack_dataarray,
                x_dims=self.x_dims,
                y_dims=self.y_dims,
                xstack_dim=self.xstack_dim,
                ystack_dim=self.ystack_dim,
                stats_dim=self.stats_dim,
                policy="infer",
            )
        )

    def stacked(self, order=None):
        if order is None:
            order = self.order
        return self._stacked(order)

    def array_data(self, order=None):
        """
        get X and Y data for gpflow analysis
        """
        stacked = self.stacked(order=order)
        xdata = multiindex_to_array(stacked.indexes[self.xstack_dim])

        ydata = [g.values for _, g in stacked.groupby(self.ystack_dim)]

        return xdata, ydata

    def xindexer_from_arrays(self, **kwargs):
        """
        create indexer for indexing into gpflow trained object by name

        Parameters
        ----------
        kwargs : dict
            should include all names in `self.x_dims[:-1]`
            sets self.x_dims[-1] (the order dimension) to 0
        """
        return self.xindexer_from_dataframe(pd.DataFrame(kwargs))

    def xindexer_from_dataframe(self, df):
        """
        create indexer from frame

        Example
        -------
        x_dims = ['beta', 'order']

        df = pd.DataFrame([{'beta': 1}, {'beta': 2}, ...])
        """

        assert set(df.columns) == set(self.x_dims[:-1])

        index = df.assign(**{self.order_dim: 0}).set_index(self.x_dims).index
        return index

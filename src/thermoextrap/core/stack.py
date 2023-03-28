"""A set of routines to stack data for gpflow analysis."""


import numpy as np
import pandas as pd
import xarray as xr

from .cached_decorators import gcached
from .models import StateCollection


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
    Given an xarray.DataArray, stack for gpflow analysis.

    Parameters
    ----------
    da : xarray.DataArray
        input DataArray
    x_dims : str or sequence of str
        dimensions stacked under `xstack_dim`
    y_dims : str or sequance of str, optional
        dimensions stacked under `ystack_dim`.
        Defaults to all dimensions not specified by `x_dims` or `stats_dim`.
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
            raise ValueError(f"{xstack_dim} conflicts with existing {dims}")

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
                raise ValueError(f"da.coords[{dim}] not set")

    out = da.stack(**stacker)

    if stats_dim is not None:
        if isinstance(stats_dim, str):
            stats_dim = (stats_dim,)
        out = out.transpose(..., *stats_dim)

    return out


def wrap_like_dataarray(x, da):
    """Wrap an array x with properties of da."""
    return xr.DataArray(
        x,
        dims=da.dims,
        coords=da.coords,
        indexes=da.indexes,
        attrs=da.attrs,
        name=da.name,
    )


def multiindex_to_array(idx):
    """Turn xarray multiindex to numpy array."""
    return np.array(list(idx.values))


def apply_reduction(
    da, dim, funcs, concat=True, concat_dim=None, concat_kws=None, **kws
):
    """
    Apply multiple reductions to DataArray.


    Parameters
    ----------
    da : DataArray
    dim : str
        dimension to reduce along
    funcs : callable or string or sequence of callables or strings
        If callable, funcs(da, dim=dim, **kws)
        If str, then da.funcs(dim=dim, **kws)
        If sequence, then perform reductions sequentially
    concat_dim : str, optional
        if not `None`, and multiple funcs, call xr.concat(out, dim=concat_dim, **concat_kws)
    concat_kws : dict, optional
    kws : dict
        optional keyword arguments to func.  Note that this is passed to all reduction functions

    Returns
    -------
    out : DataArray or list of DataArray
        if concat_dim is None and multiple funcs, then list of DataArrays corresponding to each reduction.  Otherwise, single DataArray
    """

    if not isinstance(funcs, (tuple, list)):
        funcs = [funcs]

    out = []
    for func in funcs:
        if callable(func):
            y = func(da, dim=dim, **kws)
        else:
            y = getattr(da, func)(dim=dim, **kws)
        out.append(y)

    if len(out) == 1:
        out = out[0]
    elif concat_dim is not None:
        if concat_kws is None:
            concat_kws = {}
        out = xr.concat(out, dim=concat_dim, **concat_kws)
    return out


def to_mean_var(da, dim, concat_dim=None, concat_kws=None, **kws):
    """
    For a dataarray apply mean/variance along a dimension.

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


def states_derivs_concat(states, dim=None, concat_kws=None, **kws):
    """
    Concatanate [s.derivs(norm=False) for s in states].

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
        extra arguments to `states[i].derivs` method
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

    return xr.concat((s.derivs(**kws) for s in states), dim=dim, **concat_kws)


class StackedDerivatives:
    """
    Data object for gpflow analysis.

    Parameters
    ----------
    da : DataArray
        DataArray containing mean and variance of derivatives.
        This array should have dimensions indicated in `x_dims`, `y_dims`,
        and `stats_dim`
    x_dims : str or sequence of str
        Dimensions which make up the "x" part of the data.
        For example, this would be things like "beta" or "volume".
        x_dims[-1] should be the order of the derivative
        x_dims[0] should be `alpha_name`.  That is, variable taking the derivative of.
    y_dims : str or sequence of str, optional
        Dimensions making up the "y" part of the data.  Defaults to all dimensions not specified
        by `x_dims` or `stats_dim`
    stats_dim : str, default='stats'
        Name of dimensions with mean and variance.  It is assumed that
        `mean_var.isel(**{stats_dim: 0})` is the mean and
        `mean_var.isel(**{stats_dim: 1})` is the variance
    xstack_dim, ystack_dim : str
        name of new stacked dimensions
    policy : str, default='infer'
        policy for coordinates.  See stack_dataarray

    """

    def __init__(
        self,
        da,
        x_dims,
        y_dims=None,
        xstack_dim="xstack",
        ystack_dim="ystack",
        stats_dim="stats",
        policy="infer",
    ):
        if isinstance(x_dims, str):
            x_dims = [x_dims]
        if isinstance(y_dims, str):
            y_dims = [y_dims]

        self.da = da
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.xstack_dim = xstack_dim
        self.ystack_dim = ystack_dim
        self.stats_dim = stats_dim
        self.policy = policy

    @property
    def order_dim(self):
        return self.x_dims[-1]

    @property
    def order(self):
        """Maximum order available."""
        return self.da.sizes[self.order_dim] - 1

    @property
    def alpha_name(self):
        return self.x_dims[0]

    @gcached(prop=False)
    def _stacked(self, order):
        da = self.da
        if order is not None:
            # select orders up to and including order
            da = da.isel(**{self.order_dim: slice(None, order + 1)})

        return stack_dataarray(
            da,
            x_dims=self.x_dims,
            y_dims=self.y_dims,
            xstack_dim=self.xstack_dim,
            ystack_dim=self.ystack_dim,
            stats_dim=self.stats_dim,
            policy=self.policy,
        )

    def stacked(self, order=None):
        if order is None:
            order = self.order
        return self._stacked(order)

    def array_data(self, order=None):
        """Get X and Y data for gpflow analysis."""
        stacked = self.stacked(order=order)
        xdata = multiindex_to_array(stacked.indexes[self.xstack_dim])

        ydata = [g.values for _, g in stacked.groupby(self.ystack_dim)]

        return xdata, ydata

    def xindexer_from_arrays(self, **kwargs):
        """
        Create indexer for indexing into gpflow trained object by name.

        Parameters
        ----------
        kwargs : dict
            should include all names in `self.x_dims[:-1]`
            sets self.x_dims[-1] (the order dimension) to 0
        """
        return self.xindexer_from_dataframe(pd.DataFrame(kwargs))

    def xindexer_from_dataframe(self, df):
        """
        Create indexer from frame.

        Example:
        -------
        x_dims = ['beta', 'order']

        df = pd.DataFrame([{'beta': 1}, {'beta': 2}, ...])
        """

        assert set(df.columns) == set(self.x_dims[:-1])

        index = df.assign(**{self.order_dim: 0}).set_index(self.x_dims).index
        return index

    @classmethod
    def from_mean_var(
        cls,
        mean,
        var,
        x_dims,
        y_dims=None,
        xstack_dim="xstack",
        ystack_dim="ystack",
        policy="infer",
        concat_dim=None,
        concat_kws=None,
    ):
        """Create object from mean and variance."""

        if concat_dim is None:
            concat_dim = pd.Index(["mean", "var"], name="stats")

        if isinstance(concat_dim, str):
            stats_dim = concat_dim
        elif hasattr(concat_dim, "name"):
            stats_dim = concat_dim.name
        else:
            raise ValueError("concat_dim must be string, pandas.Index, order DataArray")

        if concat_kws is None:
            concat_kws = {}

        da = xr.concat((mean, var), dim=concat_dim, **concat_kws)
        return cls(
            da=da,
            x_dims=x_dims,
            y_dims=y_dims,
            xstack_dim=xstack_dim,
            ystack_dim=ystack_dim,
            stats_dim=stats_dim,
            policy=policy,
        )

    @classmethod
    def from_derivs(
        cls,
        derivs,
        x_dims,
        reduce_dim="rep",
        reduce_funcs=None,
        reduce_kws=None,
        concat_dim=None,
        concat_kws=None,
        # alpha_name="alpha",
        y_dims=None,
        xstack_dim="xstack",
        ystack_dim="ystack",
        policy="infer",
    ):
        """
        Create object from DataArray of derivatives.

        Parameters
        ----------
        derivs : DataArray
            DataArray containing derivative information
        reduce_dim : str, default='rep'
            dimension to reduce along
        reduce_funcs : list of callables or str
            Functions applied to `derivs` to calculate statistics.
            See apply_reduction.
            Defaults to ['mean', 'var']
        concat_dim : str, optional
            dimension of concatenated results.
            That is out.isel(**{concat_dim : i}) = reduce_funcs[i](derivs, **reduce_kws)
            Default is pd.Index(['mean','var'], name='stats)
        reduce_kws : dict, optional
            optional arguments to `reduce_funcs`
        """
        if reduce_funcs is None:
            reduce_funcs = ["mean", "var"]
        if concat_dim is None:
            concat_dim = pd.Index(["mean", "var"], name="stats")

        if isinstance(concat_dim, str):
            stats_dim = concat_dim
        else:
            stats_dim = concat_dim.name

        if reduce_kws is None:
            reduce_kws = {}

        da = apply_reduction(
            derivs,
            dim=reduce_dim,
            funcs=reduce_funcs,
            concat=True,
            concat_dim=concat_dim,
            concat_kws=concat_kws,
            **reduce_kws,
        )

        return cls(
            da,
            x_dims=x_dims,
            y_dims=y_dims,
            xstack_dim=xstack_dim,
            ystack_dim=ystack_dim,
            stats_dim=stats_dim,
            policy=policy,
        )

    @classmethod
    def from_states(
        cls,
        states,
        x_dims,
        resample=True,
        resample_kws=None,
        map_func="derivs",
        map_kws=None,
        reduce_dim="rep",
        reduce_funcs=None,
        reduce_kws=None,
        concat_dim=None,
        concat_kws=None,
        # alpha_name=None,
        y_dims=None,
        xstack_dim="xstack",
        ystack_dim="ystack",
        policy="infer",
    ):
        """
        Create data object for StateCollection or list of states.

        Parameter
        ---------
        states : StateCollection of sequence of states
            If sequence of states, `states = StateCollection(states)`
        x_dims : sequence of strings
            Note that if `resample` is `True`, then `x_dims` should reflect the dimensions
            of the resampled data
        resample : bool, default=False
            If `True`, then `states = states.resample(**resample_kws)`
        resample_kws : dict, optional
            key word arguments to `states.resample`
        map_func : callable or str, default='deriv
            Set derivatives by `derivs = states.map_concat(map_func, **map_kws)`
        map_kws : dict, optional
            keyword arguments to `states.map_concat`


        See Also
        --------
        `StackedDerivatives.from_derivs`
        """

        from .models import StateCollection

        if not isinstance(states, StateCollection):
            states = StateCollection(states)

        if resample:
            if resample_kws is None:
                resample_kws = {}
            states = states.resample(**resample_kws)

        if map_kws is None:
            map_kws = {}
        derivs = states.map_concat(map_func, **map_kws)

        return cls.from_derivs(
            derivs=derivs,
            x_dims=x_dims,
            reduce_dim=reduce_dim,
            reduce_funcs=reduce_funcs,
            reduce_kws=reduce_kws,
            concat_dim=concat_dim,
            concat_kws=concat_kws,
            y_dims=y_dims,
            xstack_dim=xstack_dim,
            ystack_dim=ystack_dim,
            policy=policy,
        )


class GPRData(StateCollection):
    """
    Statecollection for GPFlow analysis.

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
        optional arguments to be passed to `collection[i].derivs`
    """

    # reduce_dim -> dimension to reduce along

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
        Get stacked data representation.

        Parameters
        ----------
        order : int
            order of derivatives to consider
        kws : dict
            extra arguments to `self.derivs`

        Returns
        -------
        stacked : DataArray
            this will be in a stacked representation

        See Also
        --------
        states_derivs_concat, to_mean_var, stack_dataarray

        """

        kws = dict(self.deriv_kws, order_dim=self.order_dim)
        return (
            states_derivs_concat(self, order=order, **kws)
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
        """Get X and Y data for gpflow analysis."""
        stacked = self.stacked(order=order)
        xdata = multiindex_to_array(stacked.indexes[self.xstack_dim])

        ydata = [g.values for _, g in stacked.groupby(self.ystack_dim)]

        return xdata, ydata

    def xindexer_from_arrays(self, **kwargs):
        """
        Create indexer for indexing into gpflow trained object by name.

        Parameters
        ----------
        kwargs : dict
            should include all names in `self.x_dims[:-1]`
            sets self.x_dims[-1] (the order dimension) to 0
        """
        return self.xindexer_from_dataframe(pd.DataFrame(kwargs))

    def xindexer_from_dataframe(self, df):
        """
        Create indexer from frame.

        Example:
        -------
        x_dims = ['beta', 'order']

        df = pd.DataFrame([{'beta': 1}, {'beta': 2}, ...])
        """

        assert set(df.columns) == set(self.x_dims[:-1])

        index = df.assign(**{self.order_dim: 0}).set_index(self.x_dims).index
        return index

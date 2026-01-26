# pyright: reportUnknownMemberType=false
"""A set of routines to stack data for gpflow analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xarray as xr
from module_utilities import cached

from .models import StateCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from typing import Any

    from numpy.typing import NDArray

    from .core.typing import (
        ApplyReduceFuncs,
        OptionalKwsAny,
        SingleDim,
        StackPolicy,
        SupportsModel,
        SupportsModelDerivs,
    )
    from .core.typing_compat import Self


def stack_dataarray(  # noqa: C901
    da: xr.DataArray,
    x_dims: str | Sequence[Hashable],
    y_dims: str | Sequence[Hashable] | None = None,
    xstack_dim: str = "xstack",
    ystack_dim: str = "ystack",
    stats_dim: str | Sequence[str] | None = None,
    policy: StackPolicy = "infer",
) -> xr.DataArray:
    """
    Given an xarray.DataArray, stack for gpflow analysis.

    Parameters
    ----------
    da : xarray.DataArray
        input DataArray
    x_dims : str or sequence of str
        dimensions stacked under `xstack_dim`
    y_dims : str or sequence of str, optional
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
    for name in (xstack_dim, ystack_dim):
        if name in dims:
            msg = f"{xstack_dim} conflicts with existing {dims}"
            raise ValueError(msg)

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
                msg = f"da.coords[{dim}] not set"
                raise ValueError(msg)

    out = da.stack(stacker)

    if stats_dim is not None:
        if isinstance(stats_dim, str):
            stats_dim = (stats_dim,)
        out = out.transpose(..., *stats_dim)

    return out


def multiindex_to_array(idx: pd.MultiIndex) -> NDArray[Any]:
    """Turn xarray multiindex to numpy array."""
    return np.array(idx.tolist())


def apply_reduction(
    da: xr.DataArray,
    dim: SingleDim,
    funcs: ApplyReduceFuncs,
    concat_dim: Any = None,
    concat_kws: OptionalKwsAny = None,
    **kws: Any,
) -> xr.DataArray | list[xr.DataArray]:
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
    if isinstance(funcs, str) or callable(funcs):
        funcs = [funcs]

    out: list[xr.DataArray] = [
        (
            func(da, dim=dim, **kws)
            if callable(func)
            else getattr(da, func)(dim=dim, **kws)
        )
        for func in funcs
    ]

    if len(out) == 1:
        return out[0]
    if concat_dim is not None:
        return xr.concat(
            out, dim=concat_dim, **({} if concat_kws is None else concat_kws)
        )
    return out


def to_mean_var(
    da: xr.DataArray,
    dim: str,
    concat_dim: Any = None,
    concat_kws: OptionalKwsAny = None,
    **kws: Any,
) -> xr.DataArray:
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


def states_derivs_concat(
    states: StateCollection[xr.DataArray, SupportsModelDerivs[xr.DataArray]],
    dim: Any = None,
    concat_kws: OptionalKwsAny = None,
    **kws: Any,
) -> xr.DataArray:
    """
    Concatenate [s.derivs(norm=False) for s in states].

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


def _convert_dims(dims: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(dims, str):
        return (dims,)
    return tuple(dims)


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
        da: xr.DataArray,
        x_dims: str | Sequence[str],
        y_dims: str | Sequence[str] | None = None,
        xstack_dim: str = "xstack",
        ystack_dim: str = "ystack",
        stats_dim: str = "stats",
        policy: StackPolicy = "infer",
    ) -> None:
        x_dims = _convert_dims(x_dims)
        if y_dims is not None:
            y_dims = _convert_dims(y_dims)

        self.da = da
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.xstack_dim = xstack_dim
        self.ystack_dim = ystack_dim
        self.stats_dim = stats_dim
        self.policy: StackPolicy = policy
        self._cache: dict[str, Any] = {}

    @property
    def order_dim(self) -> str:
        return self.x_dims[-1]

    @property
    def order(self) -> int:
        """Maximum order available."""
        return self.da.sizes[self.order_dim] - 1

    @property
    def alpha_name(self) -> str:
        return self.x_dims[0]

    @cached.meth
    def _stacked(self, order: int | None) -> xr.DataArray:
        da = self.da
        if order is not None:
            # select orders up to and including order
            da = da.isel({self.order_dim: slice(None, order + 1)})

        return stack_dataarray(
            da,
            x_dims=self.x_dims,
            y_dims=self.y_dims,
            xstack_dim=self.xstack_dim,
            ystack_dim=self.ystack_dim,
            stats_dim=self.stats_dim,
            policy=self.policy,
        )

    def stacked(self, order: int | None = None) -> xr.DataArray:
        if order is None:
            order = self.order
        return self._stacked(order)

    def array_data(
        self, order: int | None = None
    ) -> tuple[NDArray[Any], list[NDArray[Any]]]:
        """Get X and Y data for gpflow analysis."""
        stacked = self.stacked(order=order)
        xdata = multiindex_to_array(stacked.indexes[self.xstack_dim])  # pyright: ignore[reportUnknownArgumentType]

        ydata = [g.to_numpy() for _, g in stacked.groupby(self.ystack_dim)]  # pyright: ignore[reportUnknownVariableType]

        return xdata, ydata  # pyright: ignore[reportUnknownVariableType]

    def xindexer_from_arrays(self, **kwargs: Any) -> pd.Index[Any]:
        """
        Create indexer for indexing into gpflow trained object by name.

        Parameters
        ----------
        kwargs : dict
            should include all names in `self.x_dims[:-1]`
            sets self.x_dims[-1] (the order dimension) to 0
        """
        return self.xindexer_from_dataframe(pd.DataFrame(kwargs))

    def xindexer_from_dataframe(self, df: pd.DataFrame) -> pd.Index[Any]:
        """
        Create indexer from frame.

        Example:
        -------
        x_dims = ['beta', 'order']

        df = pd.DataFrame([{'beta': 1}, {'beta': 2}, ...])
        """
        if set(df.columns) != set(self.x_dims[:-1]):
            raise ValueError

        return df.assign(**{self.order_dim: 0}).set_index(self.x_dims).index

    @classmethod
    def from_mean_var(
        cls,
        mean: xr.DataArray,
        var: xr.DataArray,
        x_dims: str | Sequence[str],
        y_dims: str | Sequence[str] | None = None,
        xstack_dim: str = "xstack",
        ystack_dim: str = "ystack",
        policy: StackPolicy = "infer",
        concat_dim: Any = None,
        concat_kws: OptionalKwsAny = None,
    ) -> Self:
        """Create object from mean and variance."""
        if concat_dim is None:
            concat_dim = pd.Index(["mean", "var"], name="stats")

        if isinstance(concat_dim, str):
            stats_dim = concat_dim
        elif hasattr(concat_dim, "name"):
            stats_dim = concat_dim.name
        else:
            msg = "concat_dim must be string, pandas.Index, order DataArray"
            raise ValueError(msg)

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
        derivs: xr.DataArray,
        x_dims: str | Sequence[str],
        reduce_dim: str = "rep",
        reduce_funcs: ApplyReduceFuncs | None = None,
        reduce_kws: OptionalKwsAny = None,
        concat_dim: Any = None,
        concat_kws: OptionalKwsAny = None,
        y_dims: str | Sequence[str] | None = None,
        xstack_dim: str = "xstack",
        ystack_dim: str = "ystack",
        policy: StackPolicy = "infer",
    ) -> Self:
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

        stats_dim = concat_dim if isinstance(concat_dim, str) else concat_dim.name

        if reduce_kws is None:
            reduce_kws = {}

        da: xr.DataArray = apply_reduction(  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
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
        states: StateCollection[xr.DataArray] | Sequence[SupportsModel[xr.DataArray]],
        x_dims: str | Sequence[str],
        resample: bool = True,
        resample_kws: OptionalKwsAny = None,
        map_func: str | Callable[..., Any] = "derivs",
        map_kws: OptionalKwsAny = None,
        reduce_dim: SingleDim = "rep",
        reduce_funcs: ApplyReduceFuncs | None = None,
        reduce_kws: OptionalKwsAny = None,
        concat_dim: Any = None,
        concat_kws: OptionalKwsAny = None,
        y_dims: str | Sequence[str] | None = None,
        xstack_dim: str = "xstack",
        ystack_dim: str = "ystack",
        policy: StackPolicy = "infer",
    ) -> Self:
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
        states = cast(
            "StateCollection[xr.DataArray]",
            states if isinstance(states, StateCollection) else StateCollection(states),
        )

        if resample:
            if resample_kws is None:
                resample_kws = {}
            states = states.resample(**resample_kws)

        if map_kws is None:
            map_kws = {}
        if isinstance(map_func, str):
            from operator import methodcaller

            map_func = methodcaller(map_func, **map_kws)
        else:
            from functools import partial

            map_func = partial(map_func, **map_kws)
        derivs = states.map_concat(map_func)

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


# class GPRData(
#     StateCollection[SupportsModelDataT, xr.DataArray],
#     Generic[SupportsModelDataT],
# ):
#     """
#     Statecollection for GPFlow analysis.

#     Parameters
#     ----------
#     collection : StateCollection object
#     x_dims : sequence of str
#         dimensions for X.   The last element should correspond to the dimension
#         which specifies the order of the derivative (eg, 'order').
#         If not specified, then `x_dims = [collections.alpha_name, 'order']`
#     y_dims : str or sequence of str
#         dimensions for Y
#     reduce_dim : str, default='rep'
#         name of dimensions to calculate mean/variance along
#     stats_dim : str, default='stats'
#         name of mean/variance dimension
#     xstack_dim, ystack_dim : str
#         name of new stacked dimensions
#     order_dim : str, default='order'
#         name of derivative order dimension
#     deriv_kws : dict, optional
#         optional arguments to be passed to `collection[i].derivs`
#     """

#     # reduce_dim -> dimension to reduce along

#     def __init__(
#         self,
#         states,
#         x_dims=None,
#         y_dims=None,
#         xstack_dim="xstack",
#         ystack_dim="ystack",
#         stats_dim="stats",
#         reduce_dim="rep",
#         deriv_kws=None,
#     ) -> None:
#         if x_dims is None:
#             x_dims = [states[0].alpha_name, "order"]
#         if deriv_kws is None:
#             deriv_kws = {}

#         self.x_dims = x_dims
#         self.y_dims = y_dims
#         self.xstack_dim = xstack_dim
#         self.ystack_dim = ystack_dim
#         self.stats_dim = stats_dim
#         self.reduce_dim = reduce_dim
#         self.deriv_kws = deriv_kws

#         super().__init__(
#             states,
#             x_dims=self.x_dims,
#             y_dims=self.y_dims,
#             xstack_dim=self.xstack_dim,
#             ystack_dim=self.ystack_dim,
#             stats_dim=self.stats_dim,
#             reduce_dim=self.reduce_dim,
#             deriv_kws=self.deriv_kws,
#         )

#     @property
#     def order_dim(self):
#         return self.x_dims[-1]

#     @cached.meth
#     def _stacked(self, order):
#         """
#         Get stacked data representation.

#         Parameters
#         ----------
#         order : int
#             order of derivatives to consider
#         kws : dict
#             extra arguments to `self.derivs`

#         Returns
#         -------
#         stacked : DataArray
#             this will be in a stacked representation

#         See Also
#         --------
#         states_derivs_concat, to_mean_var, stack_dataarray

#         """
#         kws = dict(self.deriv_kws, order_dim=self.order_dim)
#         return (
#             states_derivs_concat(self, order=order, **kws)
#             .pipe(
#                 to_mean_var,
#                 dim=self.reduce_dim,
#                 concat_dim=pd.Index(["mean", "var"], name=self.stats_dim),
#             )
#             .pipe(
#                 stack_dataarray,
#                 x_dims=self.x_dims,
#                 y_dims=self.y_dims,
#                 xstack_dim=self.xstack_dim,
#                 ystack_dim=self.ystack_dim,
#                 stats_dim=self.stats_dim,
#                 policy="infer",
#             )
#         )

#     def stacked(self, order=None):
#         if order is None:
#             order = self.order
#         return self._stacked(order)

#     def array_data(self, order=None):
#         """Get X and Y data for gpflow analysis."""
#         stacked = self.stacked(order=order)
#         xdata = multiindex_to_array(stacked.indexes[self.xstack_dim])

#         ydata = [g.values for _, g in stacked.groupby(self.ystack_dim)]

#         return xdata, ydata

#     def xindexer_from_arrays(self, **kwargs):
#         """
#         Create indexer for indexing into gpflow trained object by name.

#         Parameters
#         ----------
#         kwargs : dict
#             should include all names in `self.x_dims[:-1]`
#             sets self.x_dims[-1] (the order dimension) to 0
#         """
#         return self.xindexer_from_dataframe(pd.DataFrame(kwargs))

#     def xindexer_from_dataframe(self, df):
#         """
#         Create indexer from frame.

#         Example:
#         -------
#         x_dims = ['beta', 'order']

#         df = pd.DataFrame([{'beta': 1}, {'beta': 2}, ...])
#         """
#         if set(df.columns) != set(self.x_dims[:-1]):
#             raise ValueError

#         return df.assign(**{self.order_dim: 0}).set_index(self.x_dims).index

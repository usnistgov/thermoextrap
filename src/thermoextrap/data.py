"""
Data handlers (:mod:`~thermoextrap.data`)
=========================================

The general scheme is to use the following:

* uv, xv -> samples (values) for u, x
* u, xu -> averages of u and x*u
* u[i] = <u**i>
* xu[i] = <x * u**i>
* xu[i, j] = <d^i x/d beta^i * u**j
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast

import attrs
import cmomy
import numpy as np
import xarray as xr
from attrs import field
from attrs import validators as attv
from cmomy.core.missing import MISSING
from cmomy.core.validate import is_dataarray, is_xarray
from module_utilities import cached

from .core._attrs_utils import (
    MyAttrsMixin,
    convert_dims_to_tuple,
    convert_mapping_or_none_to_dict,
)
from .core.xrutils import xrwrap_uv, xrwrap_xv
from .docstrings import DOCFILLER_SHARED

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping
    from typing import Any, ClassVar

    from cmomy.core.typing import (
        AxisReduce,
        DataT,
        DimsReduce,
        MissingType,
        Sampler,
    )

    from thermoextrap.core.typing import MetaKws, MultDims, SingleDim, XArrayObj
    from thermoextrap.core.typing_compat import Self, TypeVar

    _T = TypeVar("_T")


docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap")

__all__ = [
    "DataCallback",
    "DataCallbackABC",
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "DataValues",
    "DataValuesCentral",
    "factory_data_values",
]


# TODO(wpk): rename order to something like mom_order or expansion_order just umom...


# * Utilities
def _raise_if_not_dataarray(x: object, name: str | None = None) -> None:
    if not is_dataarray(x):
        msg = f"type({name})={type(x)} must be a DataArray."
        raise TypeError(msg)


def _raise_if_not_xarray(x: object, name: str | None = None) -> None:
    if not is_xarray(x):
        msg = f"type({name})={type(x)} must be a DataArray or Dataset."
        raise TypeError(msg)


def _validate_dims(self: Any, attribute: attrs.Attribute, dims: Any) -> None:  # noqa: ARG001
    for d in dims:
        if d not in self.data.dims:
            msg = f"{d} not in data.dimensions {self.data.dims}"
            raise ValueError(msg)


@attrs.define(frozen=True)
class DataSelector(MyAttrsMixin):
    """
    Wrap xarray object so can index like ds[i, j].

    Parameters
    ----------
    data : DataArray or Dataset
        Object to index into.
    dims : Hashable or sequence of hashables.
        Name of dimensions to be indexed.

    Examples
    --------
    >>> x = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=["x", "y"])
    >>> s = DataSelector(data=x, dims=["y", "x"])
    >>> s[0, 1]
    <xarray.DataArray ()> Size: 8B
    array(4)
    """

    #: Data to index
    data: XArrayObj = field(validator=attv.instance_of((xr.DataArray, xr.Dataset)))  # pyright: ignore[reportAssignmentType]
    #: Dims to index along
    dims: tuple[Hashable, ...] = field(
        converter=convert_dims_to_tuple, validator=_validate_dims
    )

    @classmethod
    def from_defaults(
        cls,
        data: XArrayObj,
        *,
        dims: MultDims | None = None,
        mom_dim: SingleDim = "moment",
        deriv_dim: SingleDim | None = None,
    ) -> Self:
        """
        Create DataSelector object with default values for dims.

        Parameters
        ----------
        data : DataArray or Dataset
            object to index into.
        dims : str or sequence of hashable.
            Name of dimensions to be indexed.
            If dims is None, default to either
            ``dims=(mom_dim,)`` if ``deriv_dim is None``.
            Otherwise ``dims=(mom_dim, deriv_dim)``.
        mom_dim : str, default='moment'
        deriv_dim : str, optional
            If passed and `dims` is None, set ``dims=(mom_dim, deriv_dim)``

        Returns
        -------
        out : DataSelector
        """
        if dims is None:
            dims = (mom_dim, deriv_dim) if deriv_dim is not None else (mom_dim,)
        return cls(data=data, dims=dims)

    def __getitem__(self, idx: int | tuple[int, ...]) -> XArrayObj:
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != len(self.dims):
            msg = f"bad idx {idx}, vs dims {self.dims}"
            raise ValueError(msg)
        selector = dict(zip(self.dims, idx))
        return self.data.isel(selector, drop=True)

    def __repr__(self) -> str:
        return repr(self.data)


@attrs.define
class DataCallbackABC(
    MyAttrsMixin,
):
    """
    Base class for handling callbacks to adjust data.

    For some cases, the default Data classes don't quite cut it.
    For example, for volume extrapolation, extrap parameters need to
    be included in the derivatives.  To handle this generally,
    the Data class include `self.meta` which performs these actions.

    DataCallback can be subclassed to fine tune things.
    """

    @abstractmethod
    def check(self, data: AbstractData) -> None:
        """Perform any consistency checks between self and data."""

    @abstractmethod
    def derivs_args(
        self, data: AbstractData, *, derivs_args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        """
        Adjust derivs args from data class.

        should return a tuple
        """
        return derivs_args

    # define these to raise error instead
    # of forcing usage.
    def resample(
        self,
        data: AbstractData,
        *,
        meta_kws: MetaKws,
        sampler: cmomy.IndexSampler[Any],
        **kws: Any,
    ) -> Self:
        """
        Adjust create new object.

        Should return new instance of class or self no change
        """
        raise NotImplementedError

    def reduce(self, data: AbstractData, *, meta_kws: MetaKws, **kws: Any) -> Self:
        """Reduce along dimension."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


@attrs.define
class DataCallback(DataCallbackABC):
    """
    Basic version of DataCallbackABC.

    Implemented to pass things through unchanged.  Will be used for default construction
    """

    def check(self, data: AbstractData) -> None:
        pass

    def derivs_args(  # noqa: PLR6301
        self,
        data: AbstractData,  # noqa: ARG002
        *,
        derivs_args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        return derivs_args

    def resample(
        self,
        data: AbstractData,  # noqa: ARG002
        *,
        meta_kws: MetaKws,  # noqa: ARG002
        sampler: cmomy.IndexSampler[Any],  # noqa: ARG002
        **kws: Any,  # noqa: ARG002
    ) -> Self:
        return self

    def reduce(self, data: AbstractData, *, meta_kws: MetaKws, **kws: Any) -> Self:  # noqa: ARG002
        return self


def _meta_converter(meta: DataCallbackABC | None) -> DataCallbackABC:
    if meta is None:
        meta = DataCallback()
    return meta


def _meta_validator(self: Any, attribute: attrs.Attribute, meta: Any) -> None:  # noqa: ARG001
    if not isinstance(meta, DataCallbackABC):
        msg = "meta must be None or subclass of DataCallbackABC"
        raise TypeError(msg)
    meta.check(data=self)


@attrs.define
class AbstractData(
    MyAttrsMixin,
):
    """Abstract class for data."""

    #: Callback
    meta: DataCallbackABC = field(
        kw_only=True,
        converter=_meta_converter,
        validator=_meta_validator,
    )
    #: Energy moments dimension
    umom_dim: SingleDim = field(kw_only=True, default="umom")
    #: Derivative dimension
    deriv_dim: SingleDim | None = field(kw_only=True, default=None)
    #: Whether the observable `x` is the same as energy `u`
    x_is_u: bool = field(kw_only=True, default=False)
    # cache field
    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict)

    @property
    @abstractmethod
    def central(self) -> bool:
        """Whether central (True) or raw (False) moments are used."""

    @property
    @abstractmethod
    def derivs_args(self) -> tuple[Any, ...]:
        """Sequence of arguments to derivative calculation function."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def resample(self, sampler: Sampler) -> Self:
        pass

    @property
    def xalpha(self) -> bool:
        """
        Whether X has explicit dependence on `alpha`.

        That is, if `self.deriv_dim` is not `None`
        """
        return self.deriv_dim is not None

    def pipe(self, func: Callable[..., _T], *args, **kwargs) -> _T:
        return func(self, *args, **kwargs)


# def _convert_xv(
#     xv: xr.DataArray | xr.Dataset | None, self_: DataValuesBase
# ) -> xr.DataArray | xr.Dataset:
#     if xv is None:
#         return self_.uv
#     return xv


@attrs.define
@docfiller_shared.decorate
class DataValuesBase(AbstractData):
    """
    Base class to work with data based on values (non-cmomy).

    Parameters
    ----------
    {uv}
    {xv}
    {order}
    {rec_dim}
    {umom_dim}
    {x_is_u}
    {deriv_dim}
    {meta}
    """

    #: Energy values
    uv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))  # type: ignore[misc]
    #: Obervable values
    xv: XArrayObj = field(  # type: ignore[misc]
        # converter=attrs.Converter(_convert_xv, takes_self=True),  # pyright: ignore[reportCallIssue, reportArgumentType]
        validator=attv.instance_of((xr.DataArray, xr.Dataset)),
    )
    #: Expansion order
    order: int = field()  # type: ignore[misc]
    #: Records dimension
    rec_dim: SingleDim = field(kw_only=True, default="rec")

    _CENTRAL: ClassVar[bool] = False

    @classmethod
    @docfiller_shared.decorate
    def from_vals(
        cls,
        uv: xr.DataArray,
        xv: XArrayObj | None,
        *,
        order: int,
        rec_dim: SingleDim = "rec",
        umom_dim: SingleDim = "umom",
        deriv_dim: SingleDim | None = None,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
    ) -> Self:
        """
        Constructor from arrays.

        Parameters
        ----------
        {uv_xv_array}
        {order}
        {rec_dim}
        {umom_dim}
        {deriv_dim}
        {meta}
        {x_is_u}
        """
        return cls(
            uv=uv,
            xv=uv if xv is None else xv,
            order=order,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            deriv_dim=deriv_dim,
            meta=meta,
            x_is_u=x_is_u,
        )

    @property
    def central(self) -> bool:
        return self._CENTRAL

    def __len__(self) -> int:
        return len(self.uv[self.rec_dim])

    @docfiller_shared.decorate
    def resample(
        self,
        sampler: Sampler,
        *,
        rep_dim: SingleDim = "rep",
        meta_kws: MetaKws = None,
    ) -> Self:
        """
        Resample object.

        Parameters
        ----------
        {sampler}
        {rep_dim}
        {meta_kws}
        """
        sampler = cmomy.factory_sampler(sampler, data=self.xv, dim=self.rec_dim)
        indices = sampler.indices
        if not isinstance(indices, xr.DataArray):
            indices = xr.DataArray(indices, dims=(rep_dim, self.rec_dim))

        # assert indices.sizes[self.rec_dim] == len(self)
        if indices.sizes[self.rec_dim] != len(self):
            msg = f"{indices.sizes[self.rec_dim]=} must equal {len(self)=}"
            raise ValueError(msg)

        uv = self.uv.isel({self.rec_dim: indices})
        xv = uv if self.x_is_u else self.xv.isel({self.rec_dim: indices})

        meta = self.meta.resample(
            data=self,
            meta_kws={} if meta_kws is None else meta_kws,
            sampler=sampler,
            rep_dim=rep_dim,
        )

        return type(self)(
            uv=uv,
            xv=xv,  # pyright: ignore[reportArgumentType]
            order=self.order,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            meta=meta,
            x_is_u=self.x_is_u,
        )


###############################################################################
# Data
###############################################################################
@docfiller_shared.decorate
def build_aves_xu(
    uv: xr.DataArray,
    xv: XArrayObj,
    *,
    order: int,
    dim: DimsReduce | MissingType = MISSING,
    umom_dim: SingleDim = "umom",
) -> tuple[xr.DataArray, XArrayObj]:
    """
    Build averages from values uv, xv up to order `order`.

    Parameters
    ----------
    {uv}
    {xv}
    {order}
    {dim}
    {umom_dim}

    Returns
    -------
    u : xr.DataArray
        Energy moments
    xu : xr.DataArray or xr.Dataset
        Same type as ``xv``.  Moments of :math:` x u^k`
    """
    _raise_if_not_dataarray(uv, "uv")
    _raise_if_not_xarray(xv, "xv")

    u = cmomy.wrap_reduce_vals(uv, mom=order, dim=dim, mom_dims=umom_dim).rmom()
    xu = cmomy.select_moment(
        cmomy.wrap_reduce_vals(
            xv, uv, mom=(1, order), dim=dim, mom_dims=("_xmom", umom_dim)
        ).rmom(),
        "xmom_1",
        mom_ndim=2,
    )
    return u, xu


@docfiller_shared.decorate
def build_aves_dxdu(
    uv: xr.DataArray,
    xv: XArrayObj,
    *,
    order: int,
    dim: DimsReduce | MissingType = MISSING,
    umom_dim: SingleDim = "umom",
) -> tuple[XArrayObj, xr.DataArray, XArrayObj]:
    """
    Build central moments from values uv, xv up to order `order`.

    Parameters
    ----------
    {uv}
    {xv}
    {order}
    {dim}
    {umom_dim}

    Returns
    -------
    xave : xr.DataArray or xr.Dataset
        Average of ``xv``. Same type as ``xv``.
    duave : xr.DataArray
        Energy central moments.
    dxduave : xr.DataArray or xr.Dataset
        Central comoments of ``xv`` and ``uv``.
    """
    _raise_if_not_dataarray(uv, "uv")
    _raise_if_not_xarray(xv, "xv")

    duave = cmomy.wrap_reduce_vals(uv, mom=order, dim=dim, mom_dims=umom_dim).cmom()

    c = cmomy.wrap_reduce_vals(
        xv, uv, mom=(1, order), dim=dim, mom_dims=("_xmom", umom_dim)
    )
    xave = c.select_moment("xave")
    dxduave = cmomy.select_moment(c.cmom(), "xmom_1", mom_ndim=2)

    return xave, duave, dxduave


def _xu_to_u(xu: XArrayObj, dim: str = "umom") -> XArrayObj:
    """For case where x = u, shift umom and add umom=0."""
    n = xu.sizes[dim]
    out = xu.assign_coords({dim: lambda x: x[dim] + 1}).reindex({dim: range(n + 1)})

    # add 0th element
    out.loc[{dim: 0}] = 1.0
    return out.drop_vars(dim)


@attrs.define
@docfiller_shared.inherit(DataValuesBase)
class DataValues(DataValuesBase):
    """Class to hold uv/xv data."""

    _CENTRAL: ClassVar[bool] = False

    @cached.meth
    def _mean(self) -> XArrayObj:
        return build_aves_xu(
            uv=self.uv,
            xv=self.xv,  # pyright: ignore[reportArgumentType]
            order=self.order,
            dim=self.rec_dim,
            umom_dim=self.umom_dim,
        )

    @cached.prop
    def xu(self) -> XArrayObj:
        """Average of `x * u ** n`."""
        return self._mean()[1]

    @cached.prop
    def u(self) -> XArrayObj:  # Could make this DataArray
        """Average of `u ** n`."""
        if self.x_is_u:
            return _xu_to_u(self.xu, self.umom_dim)
        return self._mean()[0]

    @cached.prop
    def u_selector(self) -> DataSelector:
        """Indexer for `self.u`."""
        return DataSelector.from_defaults(self.u, deriv_dim=None, mom_dim=self.umom_dim)

    @cached.prop
    def xu_selector(self) -> DataSelector:
        """Indexer for `self.xu`."""
        return DataSelector.from_defaults(
            self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    def derivs_args(self) -> tuple[Any, ...]:
        out = (self.u_selector,) if self.x_is_u else (self.u_selector, self.xu_selector)
        return self.meta.derivs_args(data=self, derivs_args=out)


@attrs.define
@docfiller_shared.inherit(DataValuesBase)
class DataValuesCentral(DataValuesBase):
    """Data class using values and central moments."""

    _CENTRAL: ClassVar[bool] = True

    @cached.meth
    def _mean(self) -> XArrayObj:
        return build_aves_dxdu(
            uv=self.uv,
            xv=self.xv,  # pyright: ignore[reportArgumentType]
            order=self.order,
            dim=self.rec_dim,
            umom_dim=self.umom_dim,
        )

    @cached.prop
    def xave(self) -> XArrayObj:
        """Averages of `x`."""
        return self._mean()[0]

    @cached.prop
    def dxdu(self) -> XArrayObj:
        """Averages of `dx * du ** n`."""
        return self._mean()[2]

    @cached.prop
    def du(self) -> XArrayObj:
        """Averages of `du ** n`."""
        if self.x_is_u:
            return _xu_to_u(self.dxdu, dim=self.umom_dim)
        return self._mean()[1]

    @cached.prop
    def du_selector(self) -> DataSelector:
        return DataSelector.from_defaults(
            self.du, deriv_dim=None, mom_dim=self.umom_dim
        )

    @cached.prop
    def dxdu_selector(self) -> DataSelector:
        return DataSelector.from_defaults(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @cached.prop
    def xave_selector(self) -> XArrayObj | DataSelector:
        if self.deriv_dim is None:
            return self.xave
        return DataSelector.from_defaults(self.xave, dims=[self.deriv_dim])

    @property
    def derivs_args(self) -> tuple[Any, ...]:
        out = (
            (self.xave_selector, self.du_selector)
            if self.x_is_u
            else (self.xave_selector, self.du_selector, self.dxdu_selector)
        )

        return self.meta.derivs_args(data=self, derivs_args=out)


@docfiller_shared.decorate
def factory_data_values(
    order,
    uv,
    xv,
    central=False,
    xalpha=False,
    rec_dim="rec",
    umom_dim="umom",
    val_dims="val",
    rep_dim="rep",
    deriv_dim=None,
    x_is_u=False,
    **kws,
):
    """
    Factory function to produce a DataValues object.

    Parameters
    ----------
    order : int
        Highest moment <x * u ** order>.
        For the case `x_is_u`, highest order is <u ** (order+1)>
    {uv_xv_array}
    {central}
    {xalpha}
    {rec_dim}
    {umom_dim}
    {val_dims}
    {rep_dim}
    {deriv_dim}
    {x_is_u}
    **kws :
        Extra arguments passed to constructor

    Returns
    -------
    output : DataValues or DataValuesCentral


    See Also
    --------
    DataValuesCentral
    DataValues
    """
    cls = DataValuesCentral if central else DataValues

    if xalpha and deriv_dim is None:
        msg = "if xalpha, must pass string name of derivative"
        raise ValueError(msg)

    uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)

    if xv is not None:
        xv = xrwrap_xv(
            xv,
            rec_dim=rec_dim,
            rep_dim=rep_dim,
            deriv_dim=deriv_dim,
            val_dims=val_dims,
        )

    return cls.from_vals(
        uv=uv,
        xv=xv,
        order=order,
        rec_dim=rec_dim,
        umom_dim=umom_dim,
        deriv_dim=deriv_dim,
        x_is_u=x_is_u,
        **kws,
    )


# @docfiller_shared.decorate
# def factory_data_values(
#     order,
#     uv,
#     xv,
#     weight=None,
#     central=False,
#     xalpha=False,
#     rec_dim="rec",
#     umom_dim="umom",
#     xmom_dim="xmom",
#     val_dims="val",
#     rep_dim="rep",
#     deriv_dim=None,
#     from_vals_kws=None,
#     meta=None,
#     x_is_u=False,
# ):
#     """
#     Factory function to produce a DataValues object.

#     Parameters
#     ----------
#     order : int
#         Highest moment <x * u ** order>.
#         For the case `x_is_u`, highest order is <u ** (order+1)>
#     {uv_xv_array}
#     {central}
#     {xalpha}
#     {rec_dim}
#     {umom_dim}
#     {val_dims}
#     {rep_dim}
#     {deriv_dim}
#     {meta}
#     {x_is_u}

#     Returns
#     -------
#     output : DataCentralMomentsVals


#     See Also
#     --------
#     DataValuesCentral
#     DataValues
#     """

#     if xalpha and deriv_dim is None:
#         msg = "if xalpha, must pass string name of derivative"
#         raise ValueError(msg)

#     return DataCentralMomentsVals.from_vals(xv=xv, uv=uv, order=order, weight=weight, rec_dim=rec_dim, umom_dim=umom_dim, xmom_dim=xmom_dim, rep_dim=rep_dim, deriv_dim=deriv_dim, val_dims=val_dims, central=central, from_vals_kws=from_vals_kws, meta=meta, x_is_u=x_is_u,)


################################################################################
# StatsCov objects
################################################################################
@attrs.define
@docfiller_shared.decorate
class DataCentralMomentsBase(AbstractData):
    """
    Data object based on central co-moments array.

    Parameters
    ----------
    {dxduave}
    {rec_dim}
    {umom_dim}
    {xmom_dim}
    {deriv_dim}
    {central}
    {meta}
    {x_is_u}
    use_cache : bool
        If ``True`` (default), cache intermediate result.  Speeds up calculations,
        but can lead to large objects.
    """

    #: :class:`cmomy.CentralMomentsData` object
    dxduave: cmomy.CentralMomentsData = field(
        validator=attv.instance_of(cmomy.CentralMomentsData)
    )
    #: Overvable moment dimension
    xmom_dim: SingleDim = field(kw_only=True, default="xmom")
    #: Records dimension
    rec_dim: SingleDim = field(kw_only=True, default="rec")
    #: Whether central or raw moments are used
    central: bool = field(kw_only=True, default=False)
    #: Whether observable `x` is same as energy `u`
    x_is_u: bool = field(kw_only=True, default=None)

    _use_cache: bool = field(kw_only=True, default=True)

    @property
    def order(self) -> int:
        """Order of expansion."""
        return self.dxduave.sizes[self.umom_dim] - 1

    @property
    def values(self) -> XArrayObj:
        """
        Data underlying :attr:`dxduave`.

        See Also
        --------
        cmomy.CentralMomentsData.obj

        """
        return self.dxduave.obj

    @cached.meth(check_use_cache=True)
    def rmom(self) -> XArrayObj:
        """Raw co-moments."""
        return self.dxduave.rmom()

    @cached.meth(check_use_cache=True)
    def cmom(self) -> XArrayObj:
        """Central co-moments."""
        return self.dxduave.cmom()

    @cached.prop(check_use_cache=True)
    def xu(self) -> XArrayObj:
        """Averages of form ``x * u ** n``."""
        return cmomy.select_moment(
            self.rmom(),
            "xmom_1",
            mom_ndim=2,
            mom_dims=self.dxduave.mom_dims,
        )

    @cached.prop(check_use_cache=True)
    def u(self) -> XArrayObj:
        """Averages of form ``u ** n``."""
        if self.x_is_u:
            return cmomy.convert.comoments_to_moments(
                self.rmom(),
                mom_dims=self.dxduave.mom_dims,
                mom_dims_out=self.umom_dim,
            )

        out = cmomy.select_moment(
            self.rmom(),
            "xmom_0",
            mom_ndim=2,
            mom_dims=self.dxduave.mom_dims,
        )
        if self.xalpha:
            out = out.sel({self.deriv_dim: 0}, drop=True)
        return out

    @cached.prop(check_use_cache=True)
    def xave(self) -> XArrayObj:
        """Averages of form observable ``x``."""
        return self.dxduave.select_moment("xave")

    @cached.prop(check_use_cache=True)
    def dxdu(self) -> XArrayObj:
        """Averages of form ``dx * dx ** n``."""
        return cmomy.select_moment(
            self.cmom(), "xmom_1", mom_ndim=2, mom_dims=self.dxduave.mom_dims
        )

    @cached.prop(check_use_cache=True)
    def du(self) -> XArrayObj:
        """Averages of ``du ** n``."""
        if self.x_is_u:
            return cmomy.convert.comoments_to_moments(
                self.cmom(), mom_dims=self.dxduave.mom_dims, mom_dims_out=self.umom_dim
            )

        out = cmomy.select_moment(
            self.cmom(), "xmom_0", mom_ndim=2, mom_dims=self.dxduave.mom_dims
        )
        if self.xalpha:
            out = out.sel({self.deriv_dim: 0}, drop=True)
        return out

    @cached.prop(check_use_cache=True)
    def u_selector(self) -> DataSelector:
        """Indexer for ``u_selector[n] = u ** n``."""
        return DataSelector.from_defaults(self.u, deriv_dim=None, mom_dim=self.umom_dim)

    @cached.prop(check_use_cache=True)
    def xu_selector(self) -> DataSelector:
        """Indexer for ``xu_select[n] = x * u ** n``."""
        return DataSelector.from_defaults(
            self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @cached.prop(check_use_cache=True)
    def xave_selector(self) -> XArrayObj | DataSelector:
        """Selector for ``xave``."""
        if self.deriv_dim is None:
            return self.xave
        return DataSelector(self.xave, dims=[self.deriv_dim])

    @cached.prop(check_use_cache=True)
    def du_selector(self) -> DataSelector:
        """Selector for ``du_selector[n] = du ** n``."""
        return DataSelector.from_defaults(
            self.du, deriv_dim=None, mom_dim=self.umom_dim
        )

    @cached.prop(check_use_cache=True)
    def dxdu_selector(self) -> DataSelector:
        """Selector for ``dxdu_selector[n] = dx * du ** n``."""
        return DataSelector.from_defaults(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    def derivs_args(self) -> tuple[Any, ...]:
        """
        Arguments to be passed to derivative function.

        For example, ``derivs(*self.derivs_args)``.
        """
        if not self.x_is_u:
            if self.central:
                out = (self.xave_selector, self.du_selector, self.dxdu_selector)

            else:
                out = (self.u_selector, self.xu_selector)
        elif self.central:
            out = (self.xave_selector, self.du_selector)
        else:
            out = (self.u_selector,)

        return self.meta.derivs_args(data=self, derivs_args=out)


@attrs.define(slots=True)
@docfiller_shared.inherit(DataCentralMomentsBase)
class DataCentralMoments(DataCentralMomentsBase):
    """Data class using :class:`cmomy.CentralMomentsData` to handle central moments."""

    def __len__(self) -> int:
        return self.values.sizes[self.rec_dim]

    @docfiller_shared.decorate
    def reduce(
        self,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        meta_kws: MetaKws = None,
        **kwargs: Any,
    ) -> Self:
        """
        Reduce along axis.

        Parameters
        ----------
        {dim}
        {axis}
        {meta_kws}
        **kwargs
            Keyword arguments to :meth:`cmomy.CentralMomentsData.reduce`
        """
        if dim is MISSING and axis is MISSING:
            dim = self.rec_dim
        kws = dict(dim=dim, axis=axis, **kwargs)
        return self.new_like(
            dxduave=self.dxduave.reduce(**kws),  # pyright: ignore[reportArgumentType]
            meta=self.meta.reduce(data=self, meta_kws=meta_kws, **kws),
        )

    @docfiller_shared.decorate
    def resample(
        self,
        sampler: Sampler,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        rep_dim: SingleDim = "rep",
        parallel: bool | None = None,
        meta_kws: MetaKws = None,
        **kwargs: Any,
    ) -> Self:
        """
        Resample data.

        Parameters
        ----------
        {sampler}
        {dim}
        {axis}
        {rep_dim}
        {parallel}
        meta_kws : mapping, optional
            Parameters to `self.meta.resample`
        """
        if dim is MISSING and axis is MISSING:
            dim = self.rec_dim

        # go ahead and get sampler now in case need for meta..
        sampler = cmomy.factory_sampler(
            sampler,
            data=self.dxduave.obj,
            dim=dim,
            axis=axis,
            mom_ndim=self.dxduave.mom_ndim,
            mom_dims=self.dxduave.mom_dims,
            rep_dim=rep_dim,
            parallel=parallel,
        )

        kws = dict(
            sampler=sampler,
            dim=dim,
            axis=axis,
            rep_dim=rep_dim,
            parallel=parallel,
            **kwargs,
        )

        dxdu_new = (
            self.dxduave.resample_and_reduce(**kws)
            # TODO(wpk): remove this if possible...
            .transpose(rep_dim, ...)
        )

        meta = self.meta.resample(data=self, meta_kws=meta_kws, **kws)
        return self.new_like(dxduave=dxdu_new, rec_dim=rep_dim, meta=meta)

    # TODO(wpk): update from_raw from_data to
    # include a mom_dims arguments
    # that defaults to (xmom_dim, umom_dim)
    # so if things are in wrong order, stuff still works out

    @classmethod
    @docfiller_shared.decorate
    def from_raw(
        cls,
        raw: XArrayObj,
        rec_dim: SingleDim = "rec",
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        x_is_u: bool = False,
        meta: DataCallbackABC | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Convert raw moments to data object.

        The raw moments have the form ``raw[..., i, j] = weight`` if ``i = j = 0``.  Otherwise,
        ``raw[..., i, j] = <x ** i * u ** j>``.

        Parameters
        ----------
        raw : array-like
            raw moments.  The form of this array is such that
            The shape should be ``(..., 2, order+1)``
        {rec_dim}
        {xmom_dim}
        {umom_dim}
        {deriv_dim}
        {central}
        {xr_params}
        {meta}
        {x_is_u}
        **kwargs
            Extra arguments to :func:`cmomy.wrap_raw`

        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        cmomy.wrap_raw
        from_data

        """
        if x_is_u:
            data = cmomy.convert.moments_type(
                raw, mom_ndim=1, mom_dims=umom_dim, to="central", **kwargs
            )
        else:
            data = cmomy.convert.moments_type(
                raw, mom_ndim=2, mom_dims=(xmom_dim, umom_dim), to="central", **kwargs
            )

        return cls.from_data(
            data,
            rec_dim=rec_dim,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    @docfiller_shared.decorate
    def from_vals(
        cls,
        uv: xr.DataArray,
        xv: XArrayObj | None,
        order: int,
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        rec_dim: SingleDim = "rec",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        weight: XArrayObj | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
        **kwargs: Any,
    ):
        """
        Create DataCentralMoments object from individual (unaveraged) samples.

        Parameters
        ----------
        {xv}
        {uv}
        {order}
        {xmom_dim}
        {umom_dim}
        {rec_dim}
        {deriv_dim}
        {central}
        {weight}
        {dim}
        {axis}
        {dtype}
        {meta}
        {x_is_u}
        **kwargs
            Extra arguments to :meth:`cmomy.wrap_reduce_vals`


        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        :meth:`cmomy.CentralMomentsData.from_vals`
        """
        _raise_if_not_dataarray(uv)
        if axis is MISSING and dim is MISSING:
            axis = 0

        if xv is None or x_is_u:
            dxduave = cmomy.wrap_reduce_vals(
                uv,
                weight=weight,
                axis=axis,
                dim=dim,
                mom=order + 1,
                mom_dims=umom_dim,
                **kwargs,
            ).moments_to_comoments(mom_dims_out=(xmom_dim, umom_dim), mom=(1, order))
        else:
            _raise_if_not_xarray(xv)
            dxduave = cmomy.wrap_reduce_vals(
                xv,
                uv,
                weight=weight,
                axis=axis,
                dim=dim,
                mom=(1, order),
                mom_dims=(xmom_dim, umom_dim),
                **kwargs,
            )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    @docfiller_shared.decorate
    def from_data(
        cls,
        data: XArrayObj,
        rec_dim: SingleDim = "rec",
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
        **kwargs: Any,
    ):
        """
        Create DataCentralMoments object from data.

        data[..., i, j] = weight                          i = j = 0
                        = < x >                           i = 1 and j = 0
                        = < u >                           i = 0 and j = 1
                        = <(x - <x>)**i * (u - <u>)**j >  otherwise

        If pass in ``x_is_u = True``, then treat ``data`` as a moments array for `energy` (i.e., using ``umom_dim``).
        This is then converted to a comoments array using :func:`cmomy.convert.moments_to_comoments`.

        Parameters
        ----------
        data : DataArray
        {rec_dim}
        {xmom_dim}
        {umom_dim}
        {deriv_dim}
        {central}
        {dtype}
        {meta}
        {x_is_u}
        **kwargs
            Extra arguments to :meth:`cmomy.wrap`

        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        :meth:`cmomy.CentralMomentsData.from_data`
        """
        _raise_if_not_dataarray(data)

        if x_is_u:
            dxduave = cmomy.wrap(
                data, mom_ndim=1, mom_dims=umom_dim, **kwargs
            ).moments_to_comoments(mom_dims_out=(xmom_dim, umom_dim), mom=(1, -1))
        else:
            dxduave = cmomy.wrap(
                data, mom_ndim=2, mom_dims=(xmom_dim, umom_dim), **kwargs
            )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    @docfiller_shared.decorate
    def from_resample_vals(  # noqa: PLR0913,PLR0917
        cls,
        xv: XArrayObj | None,
        uv: xr.DataArray,
        order: int,
        sampler: Sampler,
        weight: XArrayObj | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        rep_dim: SingleDim = "rep",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        meta: DataCallbackABC | None = None,
        meta_kws: MetaKws = None,
        x_is_u: bool = False,
        parallel: bool | None = None,
        **kwargs,
    ):
        """
        Create DataCentralMoments object from unaveraged samples with resampling.

        Parameters
        ----------
        {xv}
        {uv}
        {order}
        {weight}
        {axis}
        {dim}
        {xmom_dim}
        {umom_dim}
        {rep_dim}
        {deriv_dim}
        {central}
        {dtype}
        {meta}
        {meta_kws}
        {x_is_u}
        **kwargs
            Extra arguments to :meth:`cmomy.wrap_resample_vals`

        See Also
        --------
        cmomy.wrap_resample_vals
        cmomy.resample.factory_sampler
        cmomy.resmaple.IndexSampler
        """
        if xv is None or x_is_u:
            xv = uv

        _raise_if_not_dataarray(xv)
        _raise_if_not_dataarray(uv)

        mom_dims = (xmom_dim, umom_dim)

        sampler = cmomy.factory_sampler(
            sampler,
            data=xv,
            dim=dim,
            axis=axis,
            mom_dims=mom_dims,
            rep_dim=rep_dim,
            parallel=parallel,
        )

        dxduave = cmomy.wrap_resample_vals(
            xv,
            uv,
            weight=weight,
            sampler=sampler,
            mom=(1, order),
            axis=axis,
            dim=dim,
            mom_dims=mom_dims,
            rep_dim=rep_dim,
            parallel=parallel,
            **kwargs,
        )

        out = cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rep_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

        return out.set_params(
            meta=out.meta.resample(
                data=out,
                meta_kws=meta_kws,
                sampler=sampler,
                weight=weight,
                mom=(1, order),
                axis=axis,
                dim=dim,
                mom_dims=mom_dims,
                rep_dim=rep_dim,
                **kwargs,
            )
        )

    @classmethod
    @docfiller_shared.decorate
    def from_ave_raw(
        cls,
        u: DataT,
        xu: DataT | None,
        weight: DataT | None = None,
        rec_dim: SingleDim = "rec",
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
    ) -> Self:
        """
        Create object with <u**n>, <x * u**n> arrays.

        Parameters
        ----------
        u : array-like
            u[n] = <u**n>.
        xu : array_like
            xu[n] = <x * u**n>.
        w : array-like, optional
            sample weights
        umom_axis : int, optional
            axis of `u` array corresponding to moments.
            Ignored if xu is an xarray.DataArray object
        xumom_axis : int, optional
            axis of `xu` array corresponding to moments
            Ignored if xu is an xarray.DataArray object
        axis : int, default=-1
            if `umom_axis` or `xumom_axis` is None, set to axis
            Ignored if xu is an xarray.DataArray object
        {rec_dim}
        {xmom_dim}
        {umom_dim}
        {deriv_dim}
        {central}
        {mom}
        {dtype}
        {xr_params}
        {meta}
        {x_is_u}

        See Also
        --------
        :meth:`cmomy.CentralMomentsData.from_raw`
        """
        _raise_if_not_dataarray(u)
        raw: DataT
        if xu is None or x_is_u:
            raw = u
            if weight is not None:
                raw = cmomy.assign_moment(
                    raw, weight=weight, mom_dims=umom_dim, copy=False
                )
            raw = raw.transpose(..., umom_dim)

        else:
            _raise_if_not_dataarray(xu)
            raw = cast("DataT", xr.concat((u, xu), dim=xmom_dim))  # pyright: ignore[reportCallIssue, reportArgumentType]
            if weight is not None:
                raw = cmomy.assign_moment(
                    raw, weight=weight, mom_dims=(xmom_dim, umom_dim), copy=False
                )
            # make sure in correct order
            raw = raw.transpose(..., xmom_dim, umom_dim)

        return cls.from_raw(
            raw=raw,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            deriv_dim=deriv_dim,
            rec_dim=rec_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    @docfiller_shared.decorate
    def from_ave_central(  # noqa: C901,PLR0912,PLR0913,PLR0917
        cls,
        du,
        dxdu,
        weight=None,
        xave=None,
        uave=None,
        axis=-1,
        umom_axis=None,
        xumom_axis=None,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Constructor from central moments, with reduction along axis.

        Parameters
        ----------
        du : array-like
            du[0] = 1 or weight,
            du[1] = <u> or uave
            du[n] = <(u-<u>)**n>, n >= 2
        dxdu : array-like
            dxdu[0] = <x> or xave,
            dxdu[n] = <(x-<x>) * (u - <u>)**n>, n >= 1
        weight : array-like, optional
            sample weights
        xave : array-like, optional
            if present, set dxdu[0] to xave
        uave : array-like, optional
            if present, set du[0] to uave
        umom_axis : int, optional
            axis of `u` array corresponding to moments.
            Ignored if xu is an xarray.DataArray object
        xumom_axis : int, optional
            axis of `xu` array corresponding to moments
            Ignored if xu is an xarray.DataArray object
        axis : int, default=-1
            if `umom_axis` or `xumom_axis` is None, set to axis
            Ignored if xu is an xarray.DataArray object
        {rec_dim}
        {xmom_dim}
        {umom_dim}
        {deriv_dim}
        {central}
        {mom}
        {val_shape}
        {dtype}
        {xr_params}
        {meta}
        {x_is_u}

        See Also
        --------
        :meth:`cmomy.CentralMomentsData.from_data`

        """
        if dxdu is None or x_is_u:
            dxdu, du = (
                (
                    du.sel(**{umom_dim: s}).assign_coords(
                        **{umom_dim: lambda x: range(x.sizes[umom_dim])}
                    )
                )
                for s in [slice(1, None), slice(None, -1)]
            )

        if (xave is None or x_is_u) and uave is not None:
            xave = uave

        if isinstance(dxdu, xr.DataArray):
            data = xr.concat((du, dxdu), dim=xmom_dim)
            if weight is not None:
                data.loc[{umom_dim: 0, xmom_dim: 0}] = weight
            if xave is not None:
                data.loc[{umom_dim: 0, xmom_dim: 1}] = xave
            if uave is not None:
                data.loc[{umom_dim: 1, xmom_dim: 0}] = uave
            data = data.transpose(..., xmom_dim, umom_dim)
            dxduave = cmomy.CentralMomentsData(
                data,
                mom_ndim=2,
                mom_dims=(xmom_dim, umom_dim),
            )

        else:
            if axis is None:
                axis = -1
            if umom_axis is None:
                umom_axis = axis
            if xumom_axis is None:
                xumom_axis = axis

            du = np.swapaxes(du, umom_axis, -1)
            dxdu = np.swapaxes(dxdu, xumom_axis, -1)

            shape = dxdu.shape[:-1]
            shape_moments = (2, min(du.shape[-1], dxdu.shape[-1]))
            shape_out = shape + shape_moments

            if dtype is None:
                dtype = dxdu.dtype

            data = np.empty(shape_out, dtype=dtype)
            data[..., 0, :] = du
            data[..., 1, :] = dxdu
            if weight is not None:
                data[..., 0, 0] = weight
            if xave is not None:
                data[..., 1, 0] = xave
            if uave is not None:
                data[..., 0, 1] = uave

            dxduave = cmomy.CentralMomentsArray(data, mom_ndim=2).to_x(
                mom_dims=(xmom_dim, umom_dim),
                dims=dims,
                attrs=attrs,
                coords=coords,
                name=name,
            )

        return cls(
            dxduave=dxduave,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rec_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )


def _convert_dxduave(
    dxduave: cmomy.CentralMomentsData[Any] | None, self_: DataCentralMomentsVals
) -> cmomy.CentralMomentsData[Any]:
    if dxduave is not None:
        return dxduave

    if self_.order is None or self_.order <= 0:
        msg = "must pass order if calculating dxduave"
        raise ValueError(msg)

    # if self_.order > 0:
    return cmomy.wrap_reduce_vals(
        self_.xv,
        self_.uv,
        weight=self_.weight,
        dim=self_.rec_dim,
        mom=(1, self_.order),
        mom_dims=(self_.xmom_dim, self_.umom_dim),
        **self_.from_vals_kws,
    )


@attrs.define
@docfiller_shared.inherit(DataCentralMomentsBase)
class DataCentralMomentsVals(DataCentralMomentsBase):
    """
    Parameters
    ----------
    uv : xarray.DataArray
        raw values of u (energy)
    {xv}
    {order}
    from_vals_kws : dict, optional
        extra arguments passed to :meth:`cmomy.CentralMomentsData.from_vals`.
    {dxduave}
    """

    #: Stored energy values
    uv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Stored observable values
    xv: XArrayObj = field(validator=attv.instance_of((xr.DataArray, xr.Dataset)))
    #: Expansion order
    order: int | None = field(
        kw_only=True,
        default=None,
        validator=attv.optional(attv.instance_of(int)),
    )
    #: Stored weights
    weight: XArrayObj | None = field(
        kw_only=True,
        validator=attv.optional(attv.instance_of((xr.DataArray, xr.Dataset))),
        default=None,
    )  # pyright: ignore[reportArgumentType]
    #: Optional parameters to :meth:`cmomy.CentralMomentsData.from_vals`
    from_vals_kws: dict[str, Any] = field(
        kw_only=True, converter=convert_mapping_or_none_to_dict, default=None
    )
    #: :class:`cmomy.CentralMomentsData` object
    dxduave: cmomy.CentralMomentsData = field(
        kw_only=True,
        converter=attrs.Converter(_convert_dxduave, takes_self=True),  # pyright: ignore[reportCallIssue, reportArgumentType]
        validator=attv.instance_of(cmomy.CentralMomentsData),
        default=None,
    )

    @classmethod
    @docfiller_shared.decorate
    def from_vals(
        cls,
        xv: XArrayObj | None,
        uv: xr.DataArray,
        order: int,
        weight: XArrayObj | None = None,
        rec_dim: SingleDim = "rec",
        umom_dim: SingleDim = "umom",
        xmom_dim: SingleDim = "xmom",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        from_vals_kws: Mapping[str, Any] | None = None,
        meta=None,
        x_is_u=False,
    ):
        """
        Constructor from arrays.

        Parameters
        ----------
        {xv}
        {uv}
        {order}
        {xmom_dim}
        {umom_dim}
        {rec_dim}
        {deriv_dim}
        {central}
        {weight}
        {val_shape}
        {dtype}
        {xr_params}
        {meta}
        {x_is_u}


        Returns
        -------
        output : DataCentralMomentsVals

        See Also
        --------
        :meth:`cmomy.CentralMomentsData.from_vals`
        """
        return cls(
            uv=uv,
            xv=uv if xv is None else xv,
            order=order,
            weight=weight,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            xmom_dim=xmom_dim,
            deriv_dim=deriv_dim,
            central=central,
            from_vals_kws=from_vals_kws,
            meta=meta,
            x_is_u=x_is_u,
        )

    def __len__(self) -> int:
        return len(self.uv[self.rec_dim])

    @docfiller_shared.inherit(DataCentralMoments.resample)
    def resample(
        self,
        sampler: Sampler,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        rep_dim: SingleDim = "rep",
        parallel: bool | None = None,
        meta_kws: MetaKws = None,
        **kwargs,
    ) -> Self:
        """
        Resample data.

        Parameters
        ----------
        {sampler}
        {dim}
        {axis}
        {rep_dim}
        {parallel}
        {meta_kws}
        **kwargs
            Keyword arguments to :meth:`cmomy.wrap_resample_vals`

        See Also
        --------
        :meth:`cmomy.CentralMomentsData.from_resample_vals`
        """
        if dim is MISSING and axis is MISSING:
            dim = self.rec_dim

        sampler = cmomy.factory_sampler(
            sampler,
            data=self.xv,
            dim=dim,
            axis=axis,
            rep_dim=rep_dim,
            parallel=parallel,
        )

        kws = {
            "sampler": sampler,
            "parallel": parallel,
            "axis": axis,
            "dim": dim,
            "rep_dim": rep_dim,
            **kwargs,
        }
        # back to indices for meta analysis...
        # Not sure if I like this.  In general, don't have metadata resampling...
        meta = self.meta.resample(data=self, meta_kws=meta_kws, **kws)

        dxduave = cmomy.wrap_resample_vals(
            self.xv,
            self.uv,
            weight=self.weight,
            mom=(1, self.order),
            mom_dims=(self.xmom_dim, self.umom_dim),
            **kws,
        )

        dxduave = dxduave.transpose(rep_dim, ...)
        return self.new_like(dxduave=dxduave, rec_dim=rep_dim, meta=meta)

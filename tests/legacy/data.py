from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import attrs
import cmomy
import xarray as xr
from attrs import field
from attrs import validators as attv
from cmomy.core.missing import MISSING
from module_utilities import cached

from thermoextrap.core.docstrings import DOCFILLER_SHARED
from thermoextrap.core.typing import DataT
from thermoextrap.core.validate import validator_xarray_typevar
from thermoextrap.core.xrutils import xrwrap_uv, xrwrap_xv
from thermoextrap.data import (
    AbstractData,
    DataSelector,
    _raise_if_not_dataarray,
    _raise_if_not_xarray,
)

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from cmomy.core.typing import (
        DimsReduce,
        MissingType,
    )
    from cmomy.resample.typing import SamplerType
    from numpy.typing import ArrayLike

    from thermoextrap.core.typing import (
        DataDerivArgs,
        OptionalKwsAny,
        SingleDim,
    )
    from thermoextrap.core.typing_compat import Self


docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap")


# * DataValues ----------------------------------------------------------------
@attrs.frozen
@docfiller_shared.decorate
class DataValuesBase(AbstractData, Generic[DataT]):
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
    uv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Obervable values
    xv: DataT = field(validator=validator_xarray_typevar)
    #: Expansion order
    _order: int = field(alias="order")
    #: Records dimension
    rec_dim: SingleDim = field(kw_only=True, default="rec")

    _CENTRAL: ClassVar[bool] = False

    @property
    def order(self) -> int:
        return self._order

    @property
    def central(self) -> bool:
        return self._CENTRAL

    def __len__(self) -> int:
        return len(self.uv[self.rec_dim])

    @docfiller_shared.decorate
    def resample(
        self,
        sampler: SamplerType,
        *,
        rep_dim: SingleDim = "rep",
        meta_kws: OptionalKwsAny = None,
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
            indices = xr.DataArray(indices, dims=(rep_dim, self.rec_dim))  # pylint: disable=redefined-variable-type

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
            xv=xv,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
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
def _build_aves_xu(
    *,
    xv: DataT,
    uv: xr.DataArray,
    order: int,
    dim: DimsReduce | MissingType = MISSING,
    umom_dim: SingleDim = "umom",
) -> tuple[xr.DataArray, DataT]:
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
def _build_aves_dxdu(
    *,
    xv: DataT,
    uv: xr.DataArray,
    order: int,
    dim: DimsReduce | MissingType = MISSING,
    umom_dim: SingleDim = "umom",
) -> tuple[DataT, xr.DataArray, DataT]:
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


def _xu_to_u(xu: DataT, dim: str = "umom") -> DataT:
    """For case where x = u, shift umom and add umom=0."""
    n = xu.sizes[dim]
    out = xu.assign_coords({dim: lambda x: x[dim] + 1}).reindex({dim: range(n + 1)})  # pyright: ignore[reportUnknownMemberType,reportUnknownLambdaType]

    # add 0th element
    out.loc[{dim: 0}] = 1.0  # pyright: ignore[reportUnknownMemberType]
    return out.drop_vars(dim)


@docfiller_shared.inherit(DataValuesBase)
@attrs.frozen
class DataValues(DataValuesBase[DataT]):
    """Class to hold uv/xv data."""

    _CENTRAL: ClassVar[bool] = False

    @cached.meth
    def _mean(self) -> tuple[xr.DataArray, DataT]:
        return _build_aves_xu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            dim=self.rec_dim,
            umom_dim=self.umom_dim,
        )

    @cached.prop
    def xu(self) -> DataT:
        """Average of `x * u ** n`."""
        return self._mean()[1]

    @cached.prop
    def u(self) -> xr.DataArray:
        """Average of `u ** n`."""
        if self.x_is_u:
            return _xu_to_u(self.xu, self.umom_dim)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
        return self._mean()[0]

    @cached.prop
    def u_selector(self) -> DataSelector[xr.DataArray]:
        """Indexer for `self.u`."""
        return DataSelector[xr.DataArray].from_defaults(
            self.u, deriv_dim=None, mom_dim=self.umom_dim
        )

    @cached.prop
    def xu_selector(self) -> DataSelector[DataT]:
        """Indexer for `self.xu`."""
        return DataSelector[DataT].from_defaults(
            self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    def deriv_args(self) -> DataDerivArgs:
        out = (self.u_selector,) if self.x_is_u else (self.u_selector, self.xu_selector)
        return self.meta.deriv_args(data=self, deriv_args=out)


@attrs.frozen
@docfiller_shared.inherit(DataValuesBase)
class DataValuesCentral(DataValuesBase[DataT]):
    """Data class using values and central moments."""

    _CENTRAL: ClassVar[bool] = True

    @cached.meth
    def _mean(self) -> tuple[DataT, xr.DataArray, DataT]:
        return _build_aves_dxdu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            dim=self.rec_dim,
            umom_dim=self.umom_dim,
        )

    @cached.prop
    def xave(self) -> DataT:
        """Averages of `x`."""
        return self._mean()[0]

    @cached.prop
    def dxdu(self) -> DataT:
        """Averages of `dx * du ** n`."""
        return self._mean()[2]

    @cached.prop
    def du(self) -> xr.DataArray:
        """Averages of `du ** n`."""
        if self.x_is_u:
            return _xu_to_u(self.dxdu, dim=self.umom_dim)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
        return self._mean()[1]

    @cached.prop
    def du_selector(self) -> DataSelector[xr.DataArray]:
        return DataSelector[xr.DataArray].from_defaults(
            self.du, deriv_dim=None, mom_dim=self.umom_dim
        )

    @cached.prop
    def dxdu_selector(self) -> DataSelector[DataT]:
        return DataSelector[DataT].from_defaults(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @cached.prop
    def xave_selector(self) -> DataT | DataSelector[DataT]:
        if self.deriv_dim is None:
            return self.xave
        return DataSelector[DataT].from_defaults(self.xave, dims=[self.deriv_dim])

    @property
    def deriv_args(self) -> DataDerivArgs:
        out = (
            (self.xave_selector, self.du_selector)
            if self.x_is_u
            else (self.xave_selector, self.du_selector, self.dxdu_selector)
        )

        return self.meta.deriv_args(data=self, deriv_args=out)


# TODO(wpk): overload on central = True/False and overload for arraylike
@docfiller_shared.decorate
def factory_data_values(
    order: int,
    uv: ArrayLike | xr.DataArray,
    xv: ArrayLike | DataT,
    central: bool = False,
    xalpha: bool = False,
    rec_dim: str = "rec",
    umom_dim: str = "umom",
    val_dims: str = "val",
    rep_dim: str = "rep",
    deriv_dim: str | None = None,
    x_is_u: bool = False,
    **kws: Any,
) -> DataValues[Any] | DataValuesCentral[Any]:
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

    xv = xrwrap_xv(
        xv,
        rec_dim=rec_dim,
        rep_dim=rep_dim,
        deriv_dim=deriv_dim,
        val_dims=val_dims,
    )

    return cls(
        uv=uv,
        xv=xv,  # pyright: ignore[reportArgumentType]
        order=order,
        rec_dim=rec_dim,
        umom_dim=umom_dim,
        deriv_dim=deriv_dim,
        x_is_u=x_is_u,
        **kws,
    )

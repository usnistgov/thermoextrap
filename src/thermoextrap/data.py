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
from typing import TYPE_CHECKING, Generic, cast, overload

import attrs
import cmomy
import numpy as np
import xarray as xr
from attrs import field
from attrs import validators as attv
from cmomy.core.missing import MISSING
from cmomy.core.validate import is_dataarray, is_dataset, is_xarray
from module_utilities import cached

from .core._attrs_utils import (
    MyAttrsMixin,
    convert_dims_to_tuple,
    convert_mapping_or_none_to_dict,
)
from .core.docstrings import DOCFILLER_SHARED
from .core.typing import DataT
from .core.typing_compat import override
from .core.validate import validator_dims, validator_xarray_typevar
from .core.xrutils import xrwrap_uv, xrwrap_xv

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping
    from typing import Any

    from cmomy.core.typing import (
        AxisReduce,
        DimsReduce,
        MissingType,
        SelectMoment,
    )
    from cmomy.resample.typing import SamplerType
    from numpy.typing import ArrayLike

    from .core.typing import (
        DataDerivArgs,
        MultDims,
        NDArrayAny,
        OptionalKwsAny,
        SingleDim,
        SupportsData,
    )
    from .core.typing_compat import Self, TypeVar

    _T = TypeVar("_T")


docfiller_shared = DOCFILLER_SHARED.levels_to_top("cmomy", "xtrap")

__all__ = [
    "DataCallback",
    "DataCallbackABC",
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "factory_data_values",
]


# * Utilities -----------------------------------------------------------------
def _raise_if_not_dataarray(x: object, name: str | None = None) -> None:
    if not is_dataarray(x):
        msg = f"type({name})={type(x)} must be a DataArray."
        raise TypeError(msg)


def _raise_if_not_xarray(x: object, name: str | None = None) -> None:
    if not is_xarray(x):
        msg = f"type({name})={type(x)} must be a DataArray or Dataset."
        raise TypeError(msg)


def _meta_converter(meta: DataCallbackABC | None) -> DataCallbackABC:
    if meta is None:
        meta = DataCallback()
    return meta


def _meta_validator(self: Any, attribute: attrs.Attribute[Any], meta: Any) -> None:  # noqa: ARG001
    if not isinstance(meta, DataCallbackABC):
        msg = "meta must be None or subclass of DataCallbackABC"
        raise TypeError(msg)
    meta.check(data=self)


def _validate_weight(
    instance: DataCentralMomentsVals[Any],
    attribute: Any,  # noqa: ARG001
    value: object,
) -> None:
    if value is None or isinstance(value, (np.ndarray, xr.DataArray)):
        return

    if is_dataset(instance.xv) and is_dataset(value):
        return

    msg = "weight can be None, ndarray, DataArray, or Dataset (if xv is also a Dataset)"
    raise TypeError(msg)


def _validate_dxduave(
    instance: DataCentralMomentsVals[DataT],
    attribute: Any,  # noqa: ARG001
    value: object,
) -> None:
    if value is None:
        if instance._order is None:  # pyright: ignore[reportPrivateUsage] # noqa: SLF001  # pylint: disable=protected-access
            msg = "Must pass order if calculating dxduave"
            raise ValueError(msg)
        return

    if not isinstance(value, cmomy.CentralMomentsData):
        msg = "Must pass None or `cmomy.CentralMomentsData` instance for dxduave"
        raise TypeError(msg)


# * Selector ------------------------------------------------------------------
@attrs.frozen
class DataSelector(MyAttrsMixin, Generic[DataT]):
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
    data: DataT = field(validator=validator_xarray_typevar)
    #: Dims to index along
    dims: tuple[Hashable, ...] = field(
        converter=convert_dims_to_tuple, validator=validator_dims
    )

    @classmethod
    def from_defaults(
        cls,
        data: DataT,
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

    def __getitem__(self, idx: int | tuple[int, ...]) -> DataT:
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != len(self.dims):
            msg = f"bad idx {idx}, vs dims {self.dims}"
            raise ValueError(msg)
        selector = dict(zip(self.dims, idx, strict=True))
        return self.data.isel(selector, drop=True)

    @override
    def __repr__(self) -> str:
        return repr(self.data)


# * Callback -----------------------------------------------------------------
@attrs.frozen
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
    def check(self, data: SupportsData[Any]) -> None:
        """Perform any consistency checks between self and data."""

    @abstractmethod
    def deriv_args(
        self, data: SupportsData[Any], *, deriv_args: DataDerivArgs
    ) -> DataDerivArgs:
        """
        Adjust derivs args from data class.

        should return a tuple
        """
        return deriv_args

    # define these to raise error instead
    # of forcing usage.
    def resample(
        self,
        data: SupportsData[Any],
        *,
        meta_kws: OptionalKwsAny,
        sampler: cmomy.IndexSampler[Any],
        **kws: Any,
    ) -> Self:
        """
        Adjust create new object.

        Should return new instance of class or self no change
        """
        raise NotImplementedError

    def reduce(
        self, data: SupportsData[Any], *, meta_kws: OptionalKwsAny, **kws: Any
    ) -> Self:
        """Reduce along dimension."""
        raise NotImplementedError

    @override
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


@attrs.frozen
class DataCallback(DataCallbackABC):
    """
    Basic version of DataCallbackABC.

    Implemented to pass things through unchanged.  Will be used for default construction
    """

    @override
    def check(self, data: SupportsData[Any]) -> None:
        pass

    @override
    def deriv_args(  # noqa: PLR6301
        self,
        data: SupportsData[Any],  # noqa: ARG002
        *,
        deriv_args: DataDerivArgs,
    ) -> DataDerivArgs:
        return deriv_args

    @override
    def resample(
        self,
        data: SupportsData[Any],  # noqa: ARG002
        *,
        meta_kws: OptionalKwsAny,  # noqa: ARG002
        sampler: cmomy.IndexSampler[Any],  # noqa: ARG002
        **kws: Any,  # noqa: ARG002
    ) -> Self:
        return self

    @override
    def reduce(
        self,
        data: SupportsData[Any],  # noqa: ARG002
        *,
        meta_kws: OptionalKwsAny,  # noqa: ARG002
        **kws: Any,  # noqa: ARG002
    ) -> Self:
        return self


# * Abstract Data -------------------------------------------------------------
@attrs.frozen
class AbstractData(
    MyAttrsMixin,
):
    """Abstract class for data."""

    #: Callback
    meta: DataCallbackABC = field(
        kw_only=True,
        converter=_meta_converter,
        validator=_meta_validator,
        default=None,
    )
    #: Energy moments dimension
    umom_dim: SingleDim = field(kw_only=True, default="umom")
    #: Derivative dimension
    deriv_dim: SingleDim | None = field(kw_only=True, default=None)
    #: Whether the observable `x` is the same as energy `u`
    x_is_u: bool = field(kw_only=True, default=False)
    # cache field
    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict[str, "Any"])

    @property
    @abstractmethod
    def order(self) -> int:
        """Expansion order."""

    @property
    @abstractmethod
    def central(self) -> bool:
        """Whether central (True) or raw (False) moments are used."""

    @property
    @abstractmethod
    def deriv_args(self) -> DataDerivArgs:
        """Sequence of arguments to derivative calculation function."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def resample(self, sampler: SamplerType) -> Self:
        pass

    @property
    def xalpha(self) -> bool:
        """
        Whether X has explicit dependence on `alpha`.

        That is, if `self.deriv_dim` is not `None`
        """
        return self.deriv_dim is not None

    def pipe(self, func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        return func(self, *args, **kwargs)


# * DataCentralMoments --------------------------------------------------------
@attrs.frozen
@docfiller_shared.decorate
class DataCentralMomentsBase(AbstractData, Generic[DataT]):
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

    #: Overvable moment dimension
    xmom_dim: SingleDim = field(kw_only=True, default="xmom")
    #: Records dimension
    rec_dim: SingleDim = field(kw_only=True, default="rec")
    #: Whether central or raw moments are used
    central: bool = field(kw_only=True, default=False)  # pyright: ignore[reportIncompatibleMethodOverride]
    #: Whether observable `x` is same as energy `u`
    x_is_u: bool = field(kw_only=True, default=None)

    _use_cache: bool = field(kw_only=True, default=True)

    @property
    def dxduave(self) -> cmomy.CentralMomentsData[DataT]:
        """Wrapped :class:`cmomy.CentralMomentsData` object."""
        raise NotImplementedError

    @property
    @override
    def order(self) -> int:
        """Order of expansion."""
        return self.dxduave.sizes[self.umom_dim] - 1

    @property
    def values(self) -> xr.DataArray | xr.Dataset:
        """
        Data underlying :attr:`dxduave`.

        See Also
        --------
        cmomy.CentralMomentsData.obj

        """
        return self.dxduave.obj

    @cached.meth(check_use_cache=True)
    def rmom(self) -> DataT:
        """Raw co-moments."""
        return self.dxduave.rmom()

    @cached.meth(check_use_cache=True)
    def cmom(self) -> DataT:
        """Central co-moments."""
        return self.dxduave.cmom()

    @cached.prop(check_use_cache=True)
    def xu(self) -> DataT:
        """Averages of form ``x * u ** n``."""
        return cmomy.select_moment(
            self.rmom(),
            "xmom_1",
            mom_ndim=2,
            mom_dims=self.dxduave.mom_dims,
        )

    @cached.prop(check_use_cache=True)
    def u(self) -> DataT:
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
            out = out.sel({self.deriv_dim: 0}, drop=True)  # pyright: ignore[reportUnknownMemberType]
        return out

    @cached.prop(check_use_cache=True)
    def xave(self) -> DataT:
        """Averages of form observable ``x``."""
        return self.dxduave.select_moment("xave")

    @cached.prop(check_use_cache=True)
    def dxdu(self) -> DataT:
        """Averages of form ``dx * dx ** n``."""
        return cmomy.select_moment(
            self.cmom(), "xmom_1", mom_ndim=2, mom_dims=self.dxduave.mom_dims
        )

    @cached.prop(check_use_cache=True)
    def du(self) -> DataT:
        """Averages of ``du ** n``."""
        if self.x_is_u:
            return cmomy.convert.comoments_to_moments(
                self.cmom(), mom_dims=self.dxduave.mom_dims, mom_dims_out=self.umom_dim
            )

        out = cmomy.select_moment(
            self.cmom(), "xmom_0", mom_ndim=2, mom_dims=self.dxduave.mom_dims
        )
        if self.xalpha:
            out = out.sel({self.deriv_dim: 0}, drop=True)  # pyright: ignore[reportUnknownMemberType]
        return out

    @cached.prop(check_use_cache=True)
    def u_selector(self) -> DataSelector[DataT]:
        """Indexer for ``u_selector[n] = u ** n``."""
        return DataSelector[DataT].from_defaults(
            self.u, deriv_dim=None, mom_dim=self.umom_dim
        )

    @cached.prop(check_use_cache=True)
    def xu_selector(self) -> DataSelector[DataT]:
        """Indexer for ``xu_select[n] = x * u ** n``."""
        return DataSelector[DataT].from_defaults(
            self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @cached.prop(check_use_cache=True)
    def xave_selector(self) -> DataT | DataSelector[DataT]:
        """Selector for ``xave``."""
        if self.deriv_dim is None:
            return self.xave
        return DataSelector(self.xave, dims=[self.deriv_dim])

    @cached.prop(check_use_cache=True)
    def du_selector(self) -> DataSelector[DataT]:
        """Selector for ``du_selector[n] = du ** n``."""
        return DataSelector[DataT].from_defaults(
            self.du, deriv_dim=None, mom_dim=self.umom_dim
        )

    @cached.prop(check_use_cache=True)
    def dxdu_selector(self) -> DataSelector[DataT]:
        """Selector for ``dxdu_selector[n] = dx * du ** n``."""
        return DataSelector[DataT].from_defaults(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    @override
    def deriv_args(self) -> DataDerivArgs:
        """
        Arguments to be passed to derivative function.

        For example, ``derivs(*self.deriv_args)``.
        """
        out: DataDerivArgs
        if not self.x_is_u:
            if self.central:
                out = (self.xave_selector, self.du_selector, self.dxdu_selector)

            else:
                out = (self.u_selector, self.xu_selector)
        elif self.central:
            out = (self.xave_selector, self.du_selector)
        else:
            out = (self.u_selector,)

        return self.meta.deriv_args(data=self, deriv_args=out)


@attrs.frozen
@docfiller_shared.inherit(DataCentralMomentsBase)
class DataCentralMoments(DataCentralMomentsBase[DataT]):
    """Data class using :class:`cmomy.CentralMomentsData` to handle central moments."""

    #: :class:`cmomy.CentralMomentsData` object
    _dxduave: cmomy.CentralMomentsData[DataT] = field(
        validator=attv.instance_of(cmomy.CentralMomentsData), alias="dxduave"
    )

    @property
    @override
    def dxduave(self) -> cmomy.CentralMomentsData[DataT]:
        return self._dxduave

    @override
    def __len__(self) -> int:
        return self.values.sizes[self.rec_dim]

    @docfiller_shared.decorate
    def reduce(
        self,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        meta_kws: OptionalKwsAny = None,
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
        kws = {"dim": dim, "axis": axis, **kwargs}
        return self.new_like(
            dxduave=self.dxduave.reduce(**kws),  # pyright: ignore[reportArgumentType]
            meta=self.meta.reduce(data=self, meta_kws=meta_kws, **kws),
        )

    @docfiller_shared.decorate
    @override
    def resample(
        self,
        sampler: SamplerType,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        rep_dim: SingleDim = "rep",
        parallel: bool | None = None,
        meta_kws: OptionalKwsAny = None,
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

        kws = {
            "sampler": sampler,
            "dim": dim,
            "axis": axis,
            "rep_dim": rep_dim,
            "parallel": parallel,
            **kwargs,
        }

        dxdu_new = (
            self.dxduave.resample_and_reduce(**kws)  # pyright: ignore[reportArgumentType]
            # TODO(wpk): remove this if possible...
            .transpose(rep_dim, ...)
        )

        meta = self.meta.resample(data=self, meta_kws=meta_kws, **kws)  # pyright: ignore[reportArgumentType]
        return self.new_like(dxduave=dxdu_new, rec_dim=rep_dim, meta=meta)

    # TODO(wpk): update from

    @classmethod
    @docfiller_shared.decorate
    def from_raw(
        cls,
        raw: DataT,
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
        data = (
            cmomy.convert.moments_type(
                raw, mom_ndim=1, mom_dims=umom_dim, to="central", **kwargs
            )
            if x_is_u
            else cmomy.convert.moments_type(
                raw, mom_ndim=2, mom_dims=(xmom_dim, umom_dim), to="central", **kwargs
            )
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
        xv: DataT,
        order: int,
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        rec_dim: SingleDim = "rec",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        weight: NDArrayAny | xr.DataArray | DataT | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
        **kwargs: Any,
    ) -> Self:
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
            Extra arguments to :func:`cmomy.wrap_reduce_vals`


        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        cmomy.wrap_reduce_vals
        """
        _raise_if_not_dataarray(uv)
        if axis is MISSING and dim is MISSING:
            axis = 0

        dxduave: cmomy.CentralMomentsData[DataT]
        if x_is_u:
            if not is_dataarray(xv):
                msg = f"{type(xv)=} must be DataArray if x_is_ug"
                raise TypeError(msg)

            dxduave = cmomy.wrap_reduce_vals(
                cast("DataT", uv),  # type: ignore[redundant-cast]
                weight=cast("NDArrayAny | DataT", weight),
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
        data: DataT,
        rec_dim: SingleDim = "rec",
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
        **kwargs: Any,
    ) -> Self:
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
            Extra arguments to :func:`cmomy.wrap`

        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        :class:`cmomy.CentralMomentsData`
        """
        _raise_if_not_xarray(data)

        dxduave = (
            cmomy.wrap(
                data, mom_ndim=1, mom_dims=umom_dim, **kwargs
            ).moments_to_comoments(mom_dims_out=(xmom_dim, umom_dim), mom=(1, -1))
            if x_is_u
            else cmomy.wrap(data, mom_ndim=2, mom_dims=(xmom_dim, umom_dim), **kwargs)
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
        xv: DataT,
        uv: xr.DataArray,
        order: int,
        sampler: SamplerType,
        weight: NDArrayAny | xr.DataArray | DataT | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        xmom_dim: SingleDim = "xmom",
        umom_dim: SingleDim = "umom",
        rep_dim: SingleDim = "rep",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        meta: DataCallbackABC | None = None,
        meta_kws: OptionalKwsAny = None,
        x_is_u: bool = False,
        parallel: bool | None = None,
        **kwargs: Any,
    ) -> Self:
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
            Extra arguments to :func:`cmomy.wrap_resample_vals`

        See Also
        --------
        cmomy.wrap_resample_vals
        cmomy.resample.factory_sampler
        cmomy.resample.IndexSampler
        """
        if x_is_u:
            xv = uv  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]

        _raise_if_not_xarray(xv)
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
            dxduave=dxduave,  # pyright: ignore[reportArgumentType]
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            rec_dim=rep_dim,
            deriv_dim=deriv_dim,
            central=central,
            meta=meta,
            x_is_u=x_is_u,
        )

        return out.new_like(
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
        weight: NDArrayAny | xr.DataArray | DataT | None = None,
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
        weight : array-like, optional
            sample weights
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
        cmomy.wrap_raw
        """
        _raise_if_not_xarray(u)
        raw: DataT
        if xu is None or x_is_u:
            raw = u
            if weight is not None:
                raw = cmomy.assign_moment(
                    raw, weight=weight, mom_dims=umom_dim, copy=False
                )
            raw = raw.transpose(..., umom_dim)

        else:
            _raise_if_not_xarray(xu)
            raw = xr.concat((u, xu), dim=xmom_dim)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]
            if weight is not None:
                raw = cmomy.assign_moment(  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
                    raw,  # pyright: ignore[reportUnknownArgumentType]
                    weight=weight,
                    mom_dims=(xmom_dim, umom_dim),
                    copy=False,
                )
            # make sure in correct order
            raw = raw.transpose(..., xmom_dim, umom_dim)  # pyright: ignore[reportAssignmentType, reportUnknownVariableType, reportUnknownMemberType]

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
    def from_ave_central(
        cls,
        du: DataT,
        dxdu: DataT | None,
        weight: ArrayLike | xr.DataArray | DataT | None = None,
        xave: ArrayLike | xr.DataArray | DataT | None = None,
        uave: ArrayLike | xr.DataArray | DataT | None = None,
        rec_dim: str = "rec",
        xmom_dim: str = "xmom",
        umom_dim: str = "umom",
        deriv_dim: str | None = None,
        central: bool = False,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
    ) -> Self:
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
        {rec_dim}
        {xmom_dim}
        {umom_dim}
        {deriv_dim}
        {central}
        {mom}
        {meta}
        {x_is_u}

        See Also
        --------
        :class:`cmomy.CentralMomentsData`


        """
        if dxdu is None or x_is_u:
            dxdu, du = (
                (
                    du.sel(**{umom_dim: s}).assign_coords(  # type: ignore[arg-type]  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
                        **{umom_dim: lambda x: range(x.sizes[umom_dim])}  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportUnknownMemberType, reportUnknownLambdaType, reportUnknownArgumentType]
                    )
                )
                for s in (slice(1, None), slice(None, -1))
            )

        if (xave is None or x_is_u) and uave is not None:
            xave = uave

        if is_xarray(dxdu):
            data = cast("DataT", xr.concat((du, dxdu), dim=xmom_dim))  # type: ignore[redundant-cast]  # pyright: ignore[reportCallIssue, reportArgumentType]

            # using cmomy.assign_moment
            mapper: Mapping[SelectMoment, ArrayLike | xr.DataArray | DataT] = {}
            if weight is not None:
                mapper["weight"] = weight  # type: ignore[index]
            if xave is not None:
                mapper["xave"] = xave  # type: ignore[index]
            if uave is not None:
                mapper["yave"] = uave  # type: ignore[index]

            if mapper:
                data = cmomy.assign_moment(
                    data, moment=mapper, mom_dims=(xmom_dim, umom_dim)
                )

            # direct assign
            # if weight is not None:
            #     data.loc[{umom_dim: 0, xmom_dim: 0}] = weight  # type: ignore[union-attr]
            # if xave is not None:
            #     data.loc[{umom_dim: 0, xmom_dim: 1}] = xave  # type: ignore[union-attr]
            # if uave is not None:
            #     data.loc[{umom_dim: 1, xmom_dim: 0}] = uave  # type: ignore[union-attr]
            dxduave: cmomy.CentralMomentsData[DataT] = cmomy.CentralMomentsData(
                data.transpose(..., xmom_dim, umom_dim),
                mom_ndim=2,
                mom_dims=(xmom_dim, umom_dim),
            )

        else:
            msg = "Only supports xarray objects."  # pyright: ignore[reportUnreachable]
            raise ValueError(msg)

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


@attrs.frozen
@docfiller_shared.inherit(DataCentralMomentsBase)
class DataCentralMomentsVals(DataCentralMomentsBase[DataT]):
    """
    Parameters
    ----------
    uv : xarray.DataArray
        raw values of u (energy)
    {xv}
    {order}
    from_vals_kws : dict, optional
        extra arguments passed to :func:`cmomy.wrap_reduce_vals`.
    {resample_values}
    {dxduave}
    """

    #: Stored energy values
    uv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Stored observable values
    xv: DataT = field(validator=validator_xarray_typevar)
    #: Expansion order input.  Only needed if not passing dxduave
    _order: int | None = field(
        kw_only=True,
        default=None,
        validator=attv.instance_of(int),
        alias="order",
    )
    #: Stored weights
    weight: NDArrayAny | xr.DataArray | DataT | None = field(
        kw_only=True,
        validator=_validate_weight,
        default=None,
    )
    #: Optional parameters to :func:`cmomy.wrap_reduce_vals`
    from_vals_kws: dict[str, Any] = field(
        kw_only=True,
        factory=dict[str, "Any"],
        converter=convert_mapping_or_none_to_dict,
    )
    #: If ``True``, resample ``uv`` and ``xv``. Otherwise resample during construction of ``dxduave``.
    resample_values: bool = field(default=False)
    #: :class:`cmomy.CentralMomentsData` object
    _dxduave: cmomy.CentralMomentsData[DataT] | None = field(
        kw_only=True,
        validator=_validate_dxduave,
        default=None,
        alias="dxduave",
    )

    @property
    @cached.meth
    def dxduave(self) -> cmomy.CentralMomentsData[DataT]:
        if self._dxduave is None:
            return cmomy.wrap_reduce_vals(  # type: ignore[no-any-return]
                self.xv,
                self.uv,
                weight=self.weight,
                dim=self.rec_dim,
                mom=(1, self.order),
                mom_dims=(self.xmom_dim, self.umom_dim),
                **self.from_vals_kws,
            )
        return self._dxduave

    @property
    @override
    def order(self) -> int:
        if self._order is None:
            return super().order
        return self._order

    # TODO(wpk): pointless classmethod
    @classmethod
    @docfiller_shared.decorate
    def from_vals(
        cls,
        xv: DataT,
        uv: xr.DataArray,
        order: int,
        weight: NDArrayAny | xr.DataArray | DataT | None = None,
        rec_dim: SingleDim = "rec",
        umom_dim: SingleDim = "umom",
        xmom_dim: SingleDim = "xmom",
        deriv_dim: SingleDim | None = None,
        central: bool = False,
        from_vals_kws: OptionalKwsAny = None,
        resample_values: bool = False,
        meta: DataCallbackABC | None = None,
        x_is_u: bool = False,
    ) -> Self:
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
        cmomy.wrap_reduce_vals
        """
        return cls(
            uv=uv,
            xv=xv,
            order=order,
            weight=weight,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            xmom_dim=xmom_dim,
            deriv_dim=deriv_dim,
            central=central,
            from_vals_kws=from_vals_kws,
            resample_values=resample_values,
            meta=meta,
            x_is_u=x_is_u,
        )

    @override
    def __len__(self) -> int:
        return len(self.uv[self.rec_dim])

    @docfiller_shared.inherit(DataCentralMoments.resample)
    @override
    def resample(
        self,
        sampler: SamplerType,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        rep_dim: SingleDim = "rep",
        parallel: bool | None = None,
        meta_kws: OptionalKwsAny = None,
        resample_values: bool | None = None,
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
        {meta_kws}
        {resample_values}
        **kwargs
            Keyword arguments to :func:`cmomy.wrap_resample_vals`

        See Also
        --------
        :func:`cmomy.wrap_resample_vals`

        Notes
        -----
        ``resample_values`` defaults to ``self.resample_values``.
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

        if resample_values is None:
            resample_values = self.resample_values

        if resample_values:
            # resample xv/uv
            indices = sampler.indices

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

            return self.new_like(
                uv=uv,
                xv=xv,
                resample_values=resample_values,
                meta=meta,
                dxduave=None,
            )

        # Resample dxduave
        kws = {
            "parallel": parallel,
            "axis": axis,
            "dim": dim,
            "rep_dim": rep_dim,
            **kwargs,
        }

        meta = self.meta.resample(data=self, meta_kws=meta_kws, sampler=sampler, **kws)
        dxduave: cmomy.CentralMomentsData[DataT] = cmomy.wrap_resample_vals(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            self.xv,
            self.uv,
            weight=self.weight,
            sampler=sampler,
            mom=(1, self.order),
            mom_dims=(self.xmom_dim, self.umom_dim),
            **kws,  # pyright: ignore[reportArgumentType]
        )

        dxduave = dxduave.transpose(rep_dim, ...)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return self.new_like(
            dxduave=dxduave, rec_dim=rep_dim, meta=meta, resample_values=resample_values
        )


@overload
def factory_data_values(
    uv: ArrayLike | xr.DataArray,
    xv: DataT,
    *,
    order: int,
    central: bool = ...,
    xalpha: bool = ...,
    rec_dim: str = ...,
    umom_dim: str = ...,
    xmom_dim: str = ...,
    val_dims: str = ...,
    rep_dim: str = ...,
    deriv_dim: str | None = ...,
    x_is_u: bool = ...,
    resample_values: bool = ...,
    **kws: Any,
) -> DataCentralMomentsVals[DataT]: ...
@overload
def factory_data_values(
    uv: ArrayLike | xr.DataArray,
    xv: ArrayLike,
    *,
    order: int,
    central: bool = ...,
    xalpha: bool = ...,
    rec_dim: str = ...,
    umom_dim: str = ...,
    xmom_dim: str = ...,
    val_dims: str = ...,
    rep_dim: str = ...,
    deriv_dim: str | None = ...,
    x_is_u: bool = ...,
    resample_values: bool = ...,
    **kws: Any,
) -> DataCentralMomentsVals[xr.DataArray]: ...
@overload
def factory_data_values(
    uv: ArrayLike | xr.DataArray,
    xv: ArrayLike | DataT,
    *,
    order: int,
    central: bool = ...,
    xalpha: bool = ...,
    rec_dim: str = ...,
    umom_dim: str = ...,
    xmom_dim: str = ...,
    val_dims: str = ...,
    rep_dim: str = ...,
    deriv_dim: str | None = ...,
    x_is_u: bool = ...,
    resample_values: bool = ...,
    **kws: Any,
) -> DataCentralMomentsVals[Any]: ...


@docfiller_shared.decorate
def factory_data_values(
    uv: ArrayLike | xr.DataArray,
    xv: ArrayLike | DataT,
    *,
    order: int,
    central: bool = False,
    xalpha: bool = False,
    rec_dim: str = "rec",
    umom_dim: str = "umom",
    xmom_dim: str = "xmom",
    val_dims: str = "val",
    rep_dim: str = "rep",
    deriv_dim: str | None = None,
    x_is_u: bool = False,
    resample_values: bool = True,
    **kws: Any,
) -> DataCentralMomentsVals[Any]:
    """
    Factory function to produce a DataCentralMomentsVals object.

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
    {resample_values}
    **kws :
        Extra arguments passed to constructor

    Returns
    -------
    output : DataCentralMomentsVals


    See Also
    --------
    DataCentralMomentsVals
    """
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

    return DataCentralMomentsVals(
        uv=uv,
        xv=xv,
        order=order,
        rec_dim=rec_dim,
        umom_dim=umom_dim,
        xmom_dim=xmom_dim,
        central=central,
        deriv_dim=deriv_dim,
        x_is_u=x_is_u,
        resample_values=resample_values,
        **kws,
    )

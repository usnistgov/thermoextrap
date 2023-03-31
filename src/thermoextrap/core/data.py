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
from typing import Hashable, Mapping, Sequence

import attrs
import numpy as np
import xarray as xr
from attrs import converters as attc
from attrs import field
from attrs import validators as attv
from custom_inherit import DocInheritMeta

from ._attrs_utils import (
    MyAttrsMixin,
    _cache_field,
    convert_dims_to_tuple,
    kw_only_field,
)
from ._docstrings import factory_docfiller_shared
from .cached_decorators import gcached
from .xrutils import xrwrap_uv, xrwrap_xv

try:
    from cmomy import xCentralMoments

    _HAS_CMOMY = True
except ImportError:
    _HAS_CMOMY = False


docfiller_shared = factory_docfiller_shared(
    names=("default",),
)

__all__ = [
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "DataValues",
    "DataValuesCentral",
    "DataCallbackABC",
    "AbstractData",
    "factory_data_values",
    "resample_indices",
]


@docfiller_shared
def resample_indices(size, nrep, rec_dim="rec", rep_dim="rep", replace=True):
    """
    Get indexing DataArray.

    Parameters
    ----------
    size : int
        size of axis to bootstrap along
    {nrep}
    {rec_dim}
    {rep_dim}
    replace : bool, default=True
        If True, sample with replacement.
    transpose : bool, default=False
        Output format.

    Returns
    -------
    indices : DataArray
        if transpose, shape=(size, nrep)
        else, shape=(nrep, size)
    """
    indices = xr.DataArray(
        data=np.random.choice(size, size=(nrep, size), replace=replace),
        dims=[rep_dim, rec_dim],
    )

    # if transpose:
    #     indices = indices.transpose(rec_dim, rep_dim)
    return indices


@attrs.frozen
class DatasetSelector(MyAttrsMixin, metaclass=DocInheritMeta(style="numpy_with_merge")):
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
    >>> s = DatasetSelector(data=x, dims=["y", "x"])
    >>> s[0, 1]
    4
    """

    #: Data to index
    data: xr.DataArray | xr.Dataset = field(
        validator=attv.instance_of((xr.DataArray, xr.Dataset))
    )
    #: Dims to index along
    dims: Hashable | Sequence[Hashable] = field(converter=convert_dims_to_tuple)

    @dims.validator
    def _validate_dims(self, attribute, dims):
        for d in dims:
            if d not in self.data.dims:
                raise ValueError(f"{d} not in data.dimensions {self.data.dims}")

    @classmethod
    def from_defaults(cls, data, dims=None, mom_dim="moment", deriv_dim=None):
        """
        Create DataSelector object with default values for dims.

        Parameters
        ----------
        data : DataArray or Dataset
            object to index into.
        dims : hashable or sequence of hashable.
            Name of dimensions to be indexed.
            If dims is None, default to either
            ``dims=(mom_dim,)`` if ``deriv_dim is None``.
            Otherwise ``dims=(mom_dim, deriv_dim)``.
        mom_dim : hashable, default='moment'
        deriv_dim, hashable, optional
            If passed and `dims` is None, set ``dims=(mom_dim, deriv_dim)``

        Returns
        -------
        out : DatasetSelector
        """

        if dims is None:
            if deriv_dim is not None:
                dims = (mom_dim, deriv_dim)
            else:
                dims = (mom_dim,)

        return cls(data=data, dims=dims)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) != len(self.dims):
            raise ValueError(f"bad idx {idx}, vs dims {self.dims}")
        selector = dict(zip(self.dims, idx))
        return self.data.isel(**selector, drop=True)

    def __repr__(self):
        return repr(self.data)


@attrs.define
class DataCallbackABC(
    MyAttrsMixin,
    metaclass=DocInheritMeta(style="numpy_with_merge", abstract_base_class=True),
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
    def check(self, data):
        """Perform any consistency checks between self and data."""
        pass

    @abstractmethod
    def derivs_args(self, data, derivs_args):
        """
        Adjust derivs args from data class.

        should return a tuple
        """
        return derivs_args

    # define these to raise error instead
    # of forcing usage.
    def resample(self, data, meta_kws, **kws):
        """
        Adjust create new object.

        Should return new instance of class or self no change
        """
        raise NotImplementedError

    def block(self, data, meta_kws, **kws):
        raise NotImplementedError

    def reduce(self, data, meta_kws, **kws):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


@attrs.define
class DataCallback(DataCallbackABC):
    """
    Basic version of DataCallbackABC.

    Implemented to pass things through unchanged.  Will be used for default construction
    """

    def check(self, data):
        """Perform any consistency checks between self and data."""
        pass

    def derivs_args(self, data, derivs_args):
        """
        Adjust derivs args from data class.

        should return a tuple
        """
        return derivs_args

    def resample(self, data, meta_kws, **kws):
        """
        Adjust create new object.

        Should return new instance of class or self no change
        """
        return self

    def block(self, data, meta_kws, **kws):
        return self

    def reduce(self, data, meta_kws, **kws):
        return self


def _meta_converter(meta):
    if meta is None:
        meta = DataCallback()
    return meta


@attrs.define
class AbstractData(
    MyAttrsMixin,
    metaclass=DocInheritMeta(style="numpy_with_merge", abstract_base_class=True),
):
    """Abstract class for data."""

    #: Callback
    meta: DataCallbackABC | None = field(
        kw_only=True,
        converter=_meta_converter,
    )
    _cache: dict = _cache_field()

    @meta.validator
    def _meta_validate(self, attribute, meta):
        if not isinstance(meta, DataCallbackABC):
            raise ValueError("meta must be None or subclass of DataCallbackABC")
        meta.check(data=self)

    @property
    @abstractmethod
    def central(self):
        """Whether central (True) or raw (False) moments are used."""
        pass

    @property
    @abstractmethod
    def derivs_args(self):
        """Sequence of arguments to derivative calculation function."""
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def resample(self, indices=None, nrep=None, **kwargs):
        pass

    @property
    def xalpha(self):
        """
        Whether X has explicit dependence on `alpha`.

        That is, if `self.deriv_dim` is not `None`
        """
        return self.deriv_dim is not None

    @property
    def x_isnot_u(self):
        return not self.x_is_u

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)


# NOTE : custom_inherit doesn't play super will with docfiller.
# the first argument to a class with custom_inherit must be an explicit parameter


@attrs.define
@docfiller_shared
class DataValuesBase(AbstractData):
    """
    Base class to work with data based on values (non-cmomy).

    Parameters
    ----------
    uv : xarray.DataArray
        raw values of u (energy)
    {xv}
    {order}
    {rec_dim}
    {umom_dim}
    {deriv_dim}
    {skipna}
    {chunk}
    {compute}
    {meta}
    {x_is_u}
    """

    #: Energy values
    uv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Obervable values
    xv: xr.DataArray | None = field(
        validator=attv.optional(attv.instance_of((xr.DataArray, xr.Dataset)))
    )
    #: Expansion order
    order: int = field()
    #: Records dimension
    rec_dim: Hashable | None = kw_only_field(default="rec")
    #: Energy moments dimension
    umom_dim: Hashable | None = kw_only_field(default="umom")
    #: Derivative dimension
    deriv_dim: Hashable | None = kw_only_field(default=None)
    #: Whether to skip NAN values
    skipna: bool = kw_only_field(default=False)
    #: Whether to chunk the xarray objects
    chunk: int | Mapping | None = kw_only_field(default=None)
    #: Whether to compute the chunked data
    compute: bool | None = kw_only_field(default=None)
    #: Arguments to building the averages
    build_aves_kws: Mapping | None = kw_only_field(
        default=None, converter=attc.default_if_none(factory=dict)
    )
    #: Whether the observable `x` is the same as energy `u`
    x_is_u: bool = kw_only_field(default=False)

    def __attrs_post_init__(self):
        assert isinstance(self.uv, xr.DataArray)
        if self.xv is not None:
            assert isinstance(self.xv, (xr.DataArray, xr.Dataset))

        if self.chunk is not None:
            if isinstance(self.chunk, int):
                self.chunk = {self.rec_dim: self.chunk}

            self.uv = self.uv.chunk(self.chunk)
            if self.xv is not None or self.x_isnot_u:
                self.xv = self.xv.chunk(self.chunk)

        if self.compute is None:
            if self.chunk is None:
                self.compute = False
            else:
                self.compute = True

        if self.xv is None or self.x_is_u:
            self.xv = self.uv

    @classmethod
    @docfiller_shared
    def from_vals(
        cls,
        xv,
        uv,
        order,
        rec_dim="rec",
        umom_dim="umom",
        rep_dim="rep",
        deriv_dim=None,
        val_dims="val",
        skipna=False,
        chunk=None,
        compute=None,
        build_aves_kws=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Constructor from arrays.

        Parameters
        ----------
        {uv_xv_array}
        {order}
        {rec_dim}
        {umom_dim}
        {val_dims}
        {deriv_dim}
        {skipna}
        {chunk}
        {meta}
        {x_is_u}
        """

        # make sure "val" is a list
        if isinstance(val_dims, str):
            val_dims = [val_dims]
        elif not isinstance(val_dims, list):
            val_dims = list(val_dims)

        uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)

        if xv is not None:
            xv = xrwrap_xv(
                xv,
                rec_dim=rec_dim,
                rep_dim=rep_dim,
                deriv_dim=deriv_dim,
                val_dims=val_dims,
            )

        return cls(
            uv=uv,
            xv=xv,
            order=order,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            deriv_dim=deriv_dim,
            skipna=skipna,
            chunk=chunk,
            compute=compute,
            build_aves_kws=build_aves_kws,
            meta=meta,
            x_is_u=x_is_u,
        )

    @property
    def central(self):
        return self._CENTRAL

    def __len__(self):
        return len(self.uv[self.rec_dim])

    @docfiller_shared
    def resample(
        self,
        indices=None,
        nrep=None,
        rep_dim="rep",
        chunk=None,
        compute="None",
        meta_kws=None,
    ):
        """
        Resample object.

        Parameters
        ----------
        {indices}
        {nrep}
        {rep_dim}
        {chunk}
        {compute}
        """

        if chunk is None:
            chunk = self.chunk

        if compute == "None":
            compute = None
        elif compute is None:
            compute = self.compute

        if rep_dim is None:
            rep_dim = self.rep_dim

        if indices is None:
            assert nrep is not None
            indices = resample_indices(
                len(self), nrep, rec_dim=self.rec_dim, rep_dim=rep_dim
            )
        elif not isinstance(indices, xr.DataArray):
            indices = xr.DataArray(indices, dims=(rep_dim, self.rec_dim))

        assert indices.sizes[self.rec_dim] == len(self)

        uv = self.uv.compute()[indices]
        if self.x_isnot_u:
            xv = self.xv.compute().isel(**{self.rec_dim: indices})
        else:
            xv = None

        meta = self.meta.resample(
            data=self,
            meta_kws=meta_kws,
            indices=indices,
            nrep=nrep,
            rep_dim=rep_dim,
            chunk=chunk,
            compute=compute,
        )

        return self.__class__(
            uv=uv,
            xv=xv,
            order=self.order,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            skipna=self.skipna,
            chunk=chunk,
            compute=compute,
            build_aves_kws=self.build_aves_kws,
            meta=meta,
            x_is_u=self.x_is_u,
        )


###############################################################################
# Data
###############################################################################
@docfiller_shared
def build_aves_xu(
    uv,
    xv,
    order,
    rec_dim="rec",
    umom_dim="umom",
    deriv_dim=None,
    skipna=False,
    u_name=None,
    xu_name=None,
    merge=False,
    transpose=False,
):
    """
    Build averages from values uv, xv up to order `order`.

    Parameters
    ----------
    {uv}
    {xv}
    {order}
    {rec_dim}
    {umom_dim}
    {deriv_dim}
    {skipna}
    u_name : str, optional
        Name to add to energy output.
    xu_name
        Name to add to ``x * u`` output.
    merge : bool, default=False
        Merge output `u` and `xu`.
    transpose : bool, default=False
        If True, transpose results such that `umom_dim` is *first*.

    Returns
    -------
    output :
        If merge is False, return ``u`` and ``xu``, xarray objects for average energy and observable times energy.
        If merge is True, then return an :class:`xr.Dataset` containing ``u`` and ``xu``.
    """

    assert isinstance(uv, xr.DataArray)
    assert isinstance(xv, (xr.DataArray, xr.Dataset))

    u = []
    xu = []

    uave = uv.mean(rec_dim, skipna=skipna)
    xave = xv.mean(rec_dim, skipna=skipna)
    for i in range(order + 1):
        if i == 0:
            # <u**0>
            u.append(xr.ones_like(uave))
            # <x * u**0> = <x>
            xu.append(xave)

        elif i == 1:
            u_n = uv.copy()
            xu_n = xv * uv

            u.append(uave)
            xu.append(xu_n.mean(rec_dim, skipna=skipna))
        else:
            u_n *= uv
            xu_n *= uv
            u.append(u_n.mean(rec_dim, skipna=skipna))
            xu.append(xu_n.mean(rec_dim, skipna=skipna))

    u = xr.concat(u, dim=umom_dim)
    xu = xr.concat(xu, dim=umom_dim)

    # simple, but sometimes slow....
    # nvals = xr.DataArray(np.arange(order + 1), dims=[umom_dim])
    # un = uv**nvals
    # u = (un).mean(rec_dim, skipna=skipna)
    # xu = (un * xv).mean(rec_dim, skipna=skipna)

    if transpose:
        u_order = (umom_dim, ...)
        if deriv_dim is not None:
            x_order = (umom_dim, deriv_dim, ...)
        else:
            x_order = u_order
        u = u.trapsose(*u_order)
        xu = xu.transpose(*x_order)

    if u_name is not None:
        u = u.rename(u_name)

    if xu_name is not None and isinstance(xu, xr.DataArray):
        xu = xu.rename(xu_name)

    if merge:
        return xr.merge((u, xu))
    else:
        return u, xu


@docfiller_shared
def build_aves_dxdu(
    uv,
    xv,
    order,
    rec_dim="rec",
    umom_dim="umom",
    deriv_dim=None,
    skipna=False,
    du_name=None,
    dxdu_name=None,
    xave_name=None,
    merge=False,
    transpose=False,
):
    """
    Build central moments from values uv, xv up to order `order`.

    Parameters
    ----------
    {uv}
    {xv}
    {order}
    {rec_dim}
    {umom_dim}
    {deriv_dim}
    {skipna}
    du_name : str, optional
        Name for output ``du``
    dxdu_name : str, optional
        Name for output ``dxdu``.
    xave_name : str, optional
        Name for output ``xave``
    merge : bool, default=False
    transpose : bool, default=False

    Returns
    -------
    xave, duave, dxduave :
        xarray objects with averaged data.  If merge is True, merge output
        into single :class:`xr.Dataset` object.
    """

    assert isinstance(uv, xr.DataArray)
    assert isinstance(xv, (xr.DataArray, xr.Dataset))

    xave = xv.mean(rec_dim, skipna=skipna)
    uave = uv.mean(rec_dim, skipna=skipna)

    # i=0
    # <du**0> = 1
    # <dx * du**0> = 0
    duave = []
    dxduave = []
    du = uv - uave

    for i in range(order + 1):
        if i == 0:
            # <du ** 0> = 1
            # <dx * du**0> = 0
            duave.append(xr.ones_like(uave))
            dxduave.append(xr.zeros_like(xave))

        elif i == 1:
            # <du**1> = 0
            # (dx * du**1> = ...

            du_n = du.copy()
            dxdu_n = (xv - xave) * du
            duave.append(xr.zeros_like(uave))
            dxduave.append(dxdu_n.mean(rec_dim, skipna=skipna))

        else:
            du_n *= du
            dxdu_n *= du
            duave.append(du_n.mean(rec_dim, skipna=skipna))
            dxduave.append(dxdu_n.mean(rec_dim, skipna=skipna))

    duave = xr.concat(duave, dim=umom_dim)
    dxduave = xr.concat(dxduave, dim=umom_dim)

    if transpose:
        u_order = (umom_dim, ...)
        if deriv_dim is not None:
            x_order = (deriv_dim,) + u_order
        else:
            x_order = u_order

        duave = duave.transpose(*u_order)
        dxduave = dxduave.transpoe(*x_order)

    if du_name is not None:
        duave = duave.rename(du_name)
    if dxdu_name is not None and isinstance(dxduave, xr.DataArray):
        dxduave = dxduave.rename(dxdu_name)
    if xave_name is not None and isinstance(xave, xr.DataArray):
        xave = xave.renamae(xave_name)

    if merge:
        return xr.merge((xave, duave, dxduave))
    else:
        return xave, duave, dxduave


def _xu_to_u(xu, dim="umom"):
    """For case where x = u, shift umom and add umom=0."""

    n = xu.sizes[dim]

    out = xu.assign_coords(**{dim: lambda x: x[dim] + 1}).reindex(**{dim: range(n + 1)})

    # add 0th element
    out.loc[{dim: 0}] = 1.0
    return out


@attrs.define
class DataValues(DataValuesBase):
    """Class to hold uv/xv data."""

    _CENTRAL = False

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves_xu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            **self.build_aves_kws,
        )

    @gcached()
    def xu(self):
        """Average of `x * u ** n`."""
        out = self._mean()[1]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def u(self):
        """Average of `u ** n`."""
        if self.x_isnot_u:
            out = self._mean()[0]
            if self.compute:
                out = out.compute()
        else:
            out = _xu_to_u(self.xu, self.umom_dim)

        return out

    @gcached()
    def u_selector(self):
        """Indexer for `self.u`."""
        return DatasetSelector.from_defaults(
            self.u, deriv_dim=None, mom_dim=self.umom_dim
        )

    @gcached()
    def xu_selector(self):
        """Indexer for `self.xu`."""
        return DatasetSelector.from_defaults(
            self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    def derivs_args(self):
        if self.x_isnot_u:
            out = (self.u_selector, self.xu_selector)
        else:
            out = (self.u_selector,)
        return self.meta.derivs_args(data=self, derivs_args=out)


@attrs.define
class DataValuesCentral(DataValuesBase):
    """Data class using values and central moments."""

    _CENTRAL = True

    @gcached(prop=False)
    def _mean(self, skipna=None):
        if skipna is None:
            skipna = self.skipna

        return build_aves_dxdu(
            uv=self.uv,
            xv=self.xv,
            order=self.order,
            skipna=skipna,
            rec_dim=self.rec_dim,
            umom_dim=self.umom_dim,
            deriv_dim=self.deriv_dim,
            **self.build_aves_kws,
        )

    @gcached()
    def xave(self):
        """Averages of `x`."""
        out = self._mean()[0]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def dxdu(self):
        """Averages of `dx * du ** n`."""
        out = self._mean()[2]
        if self.compute:
            out = out.compute()
        return out

    @gcached()
    def du(self):
        """Averages of `du ** n`."""
        if self.x_isnot_u:
            out = self._mean()[1]
            if self.compute:
                out = out.compute()
        else:
            out = _xu_to_u(self.dxdu, dim=self.umom_dim)

        return out

    @gcached()
    def du_selector(self):
        return DatasetSelector.from_defaults(
            self.du, deriv_dim=None, mom_dim=self.umom_dim
        )

    @gcached()
    def dxdu_selector(self):
        return DatasetSelector.from_defaults(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @gcached()
    def xave_selector(self):
        if self.deriv_dim is None:
            return self.xave
        else:
            return DatasetSelector.from_defaults(self.xave, dims=[self.deriv_dim])

    @property
    def derivs_args(self):
        if self.x_isnot_u:
            out = (self.xave_selector, self.du_selector, self.dxdu_selector)
        else:
            out = (self.xave_selector, self.du_selector)
        return self.meta.derivs_args(self, out)


@docfiller_shared
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
    skipna=False,
    chunk=None,
    compute=None,
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
    {skipna}
    {chunk}
    {compute}
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

    if central:
        cls = DataValuesCentral
    else:
        cls = DataValues

    if xalpha and deriv_dim is None:
        raise ValueError("if xalpha, must pass string name of derivative")

    return cls.from_vals(
        uv=uv,
        xv=xv,
        order=order,
        skipna=skipna,
        rec_dim=rec_dim,
        umom_dim=umom_dim,
        val_dims=val_dims,
        rep_dim=rep_dim,
        deriv_dim=deriv_dim,
        chunk=chunk,
        compute=compute,
        x_is_u=x_is_u,
        **kws,
    )


################################################################################
# StatsCov objects
################################################################################
@attrs.define
@docfiller_shared
class DataCentralMomentsBase(AbstractData):
    """
    Data object based on central co-moments array.

    Parameters
    ----------
    dxduave : xCentralMoments
        Central moments object.
    {rec_dim}
    {umom_dim}
    {xmom_dim}
    {deriv_dim}
    {central}
    {meta}
    {x_is_u}
    """

    #: :class:`cmomy.xCentralMoments` object
    dxduave: xCentralMoments = field(validator=attv.instance_of(xCentralMoments))
    #: Energy moment dimension
    umom_dim: Hashable = kw_only_field(default="umom")
    #: Overvable moment dimension
    xmom_dim: Hashable = kw_only_field(default="xmom")
    #: Records dimension
    rec_dim: Hashable = kw_only_field(default="rec")
    #: Derivative with respect to alpha dimension
    deriv_dim: Hashable | None = kw_only_field(default=None)
    #: Whether central or raw moments are used
    central: bool = kw_only_field(default=False)
    #: Whether observable `x` is same as energy `u`
    x_is_u: bool = kw_only_field(default=None)

    @property
    def order(self):
        """Order of expansion."""
        return self.dxduave.sizes[self.umom_dim] - 1

    @property
    def values(self):
        """
        Data underlying :attr:`dxduave`.

        See Also
        --------
        cmomy.xCentralMoments.values

        """
        return self.dxduave.values

    @gcached(prop=False)
    def rmom(self):
        """Raw co-moments."""
        return self.dxduave.rmom()

    @gcached(prop=False)
    def cmom(self):
        """Central co-moments."""
        return self.dxduave.cmom()

    @gcached()
    def xu(self):
        """Averages of form ``x * u ** n``."""
        return self.rmom().sel(**{self.xmom_dim: 1}, drop=True)

    @gcached()
    def u(self):
        """Averages of form ``u ** n``."""
        if self.x_isnot_u:
            out = self.rmom().sel(**{self.xmom_dim: 0}, drop=True)
            if self.xalpha:
                out = out.sel(**{self.deriv_dim: 0}, drop=True)
        else:
            out = _xu_to_u(self.xu, self.umom_dim)

        return out

    @gcached()
    def xave(self):
        """Averages of form observable ``x``."""
        return self.dxduave.values.sel(
            **{self.umom_dim: 0, self.xmom_dim: 1}, drop=True
        )

    @gcached()
    def dxdu(self):
        """Averages of form ``dx * dx ** n``."""
        return self.cmom().sel(**{self.xmom_dim: 1}, drop=True)

    @gcached()
    def du(self):
        """Averages of ``du ** n``."""
        if self.x_isnot_u:
            out = self.cmom().sel(**{self.xmom_dim: 0}, drop=True)
            if self.xalpha:
                out = out.sel(**{self.deriv_dim: 0}, drop=True)
        else:
            out = _xu_to_u(self.dxdu, self.umom_dim)

        return out

    @gcached()
    def u_selector(self):
        """Indexer for ``u_selector[n] = u ** n``."""
        return DatasetSelector.from_defaults(
            self.u, deriv_dim=None, mom_dim=self.umom_dim
        )

    @gcached()
    def xu_selector(self):
        """Indexer for ``xu_select[n] = x * u ** n``."""
        return DatasetSelector.from_defaults(
            self.xu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @gcached()
    def xave_selector(self):
        """Selector for ``xave``."""
        if self.deriv_dim is None:
            return self.xave
        else:
            return DatasetSelector(self.xave, dims=[self.deriv_dim])

    @gcached()
    def du_selector(self):
        """Selector for ``du_selector[n] = du ** n``."""
        return DatasetSelector.from_defaults(
            self.du, deriv_dim=None, mom_dim=self.umom_dim
        )

    @gcached()
    def dxdu_selector(self):
        """Selector for ``dxdu_selector[n] = dx * du ** n``."""
        return DatasetSelector.from_defaults(
            self.dxdu, deriv_dim=self.deriv_dim, mom_dim=self.umom_dim
        )

    @property
    def derivs_args(self):
        """
        Arguments to be passed to derivative function.

        For example, ``derivs(*self.derivs_args)``.
        """
        if self.x_isnot_u:
            if self.central:
                out = (self.xave_selector, self.du_selector, self.dxdu_selector)

            else:
                out = (self.u_selector, self.xu_selector)
        else:
            if self.central:
                out = (self.xave_selector, self.du_selector)
            else:
                out = (self.u_selector,)

        return self.meta.derivs_args(data=self, derivs_args=out)


@attrs.define(slots=True)
class DataCentralMoments(DataCentralMomentsBase):
    """Data class using :class:`cmomy.xCentralMoments` to handle central moments."""

    def __len__(self):
        return self.values.sizes[self.rec_dim]

    @docfiller_shared
    def block(self, block_size, dim=None, axis=None, meta_kws=None, **kwargs):
        """
        Block resample along axis.

        Parameters
        ----------
        block_size : int
            number of sample to block average together
        {dim}
        {axis}
        axis : int or str, default=:attr:`rec_dim`
            axis or dimension to block average along
        **kwargs : dict
            extra arguments to :meth:`cmomy.xCentralMoments.block`
        """

        if dim is None and axis is None:
            dim = self.rec_dim

        kws = dict(block_size=block_size, dim=dim, axis=axis, **kwargs)
        return self.new_like(
            dxduave=self.dxduave.block(**kws),
            meta=self.meta.block(data=self, meta_kws=meta_kws, **kws),
        )

    @docfiller_shared
    def reduce(self, dim=None, axis=None, meta_kws=None, **kwargs):
        """
        Reduce along axis.

        Parameters
        ----------
        {dim}
        {axis}
        """
        if dim is None and axis is None:
            dim = self.rec_dim
        kws = dict(dim=dim, axis=axis, **kwargs)

        return self.new_like(
            dxduave=self.dxduave.reduce(**kws),
            meta=self.meta.reduce(data=self, meta_kws=meta_kws, **kws),
        )

    @docfiller_shared
    def resample(
        self,
        freq=None,
        indices=None,
        nrep=None,
        dim=None,
        axis=None,
        rep_dim="rep",
        parallel=True,
        resample_kws=None,
        meta_kws=None,
        **kwargs,
    ):
        """
        Resample data.

        Parameters
        ----------
        {freq}
        {indices}
        {nrep}
        {dim}
        {axis}
        {rep_dim}
        parallel : bool, default=True
            If true, perform resampling in parallel
        meta_kws : mapping, optional
            Parameters to `self.meta.resample`
        resample_kws : mapping, optional
            dictionary of values to pass to self.dxduave.resample_and_reduce
        """

        if dim is None and axis is None:
            dim = self.rec_dim

        kws = dict(
            freq=freq,
            indices=indices,
            nrep=nrep,
            dim=dim,
            axis=axis,
            rep_dim=rep_dim,
            parallel=True,
            resample_kws=resample_kws,
            **kwargs,
        )

        kws["full_output"] = True

        dxdu_new, kws["freq"] = self.dxduave.resample_and_reduce(**kws)

        meta = self.meta.resample(data=self, meta_kws=meta_kws, **kws)
        return self.new_like(dxduave=dxdu_new, rec_dim=rep_dim, meta=meta)

    # TODO : update from_raw from_data to
    # include a mom_dims arguments
    # that defaults to (xmom_dim, umom_dim)
    # so if things are in wrong order, stuff still works out

    @classmethod
    @docfiller_shared
    def from_raw(
        cls,
        raw,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
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
        {mom}
        {val_shape}
        {dtype}
        {xr_params}
        {meta}
        {x_is_u}


        Returns
        -------
        output : DataCentralMoments


        See Also
        --------
        :meth:`cmomy.xCentralMoments.from_raw`
        """

        if x_is_u:
            raw = xr.concat(
                [
                    (
                        raw.sel(**{umom_dim: s}).assign_coords(
                            **{umom_dim: lambda x: range(x.sizes[umom_dim])}
                        )
                    )
                    for s in [slice(None, -1), slice(1, None)]
                ],
                dim=xmom_dim,
            )
            raw = raw.transpose(..., xmom_dim, umom_dim)

        dxduave = xCentralMoments.from_raw(
            raw=raw,
            mom=mom,
            mom_ndim=2,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
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
    @docfiller_shared
    def from_vals(
        cls,
        xv,
        uv,
        order,
        xmom_dim="xmom",
        umom_dim="umom",
        rec_dim="rec",
        deriv_dim=None,
        central=False,
        w=None,
        axis=None,
        dim=None,
        broadcast=True,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
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
        {w}
        {dim}
        {axis}
        {broadcast}
        {val_shape}
        {dtype}
        {xr_params}
        {meta}
        {x_is_u}


        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        :meth:`cmomy.xCentralMoments.from_vals`
        """

        if axis is None and dim is None:
            axis = 0

        if xv is None or x_is_u:
            xv = uv

        dxduave = xCentralMoments.from_vals(
            x=(xv, uv),
            w=w,
            axis=axis,
            dim=dim,
            mom=(1, order),
            broadcast=broadcast,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
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
    @docfiller_shared
    def from_data(
        cls,
        data,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
        """
        Create DataCentralMoments object from data.

        data[..., i, j] = weight                          i = j = 0
                        = < x >                           i = 1 and j = 0
                        = < u >                           i = 0 and j = 1
                        = <(x - <x>)**i * (u - <u>)**j >  otherwise

        Parameters
        ----------
        data : array-like
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

        Returns
        -------
        output : DataCentralMoments

        See Also
        --------
        :meth:`cmomy.xCentralMoments.from_data`
        """

        if x_is_u:
            # convert from central moments to central co-moments
            # out_0 = data.sel(**{umom_dim: slice(None, -1)})
            # out_1 = data.sel(**{umom_dim: slice(1, None)})
            data = xr.concat(
                [
                    (
                        data.sel(**{umom_dim: s}).assign_coords(
                            **{umom_dim: lambda x: range(x.sizes[umom_dim])}
                        )
                    )
                    for s in [slice(None, -1), slice(1, None)]
                ],
                dim=xmom_dim,
            )

        dxduave = xCentralMoments.from_data(
            data=data,
            mom_ndim=2,
            mom=mom,
            val_shape=val_shape,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
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
    def from_resample_vals(
        cls,
        xv,
        uv,
        order,
        freq=None,
        indices=None,
        nrep=None,
        xmom_dim="xmom",
        umom_dim="umom",
        rep_dim="rep",
        deriv_dim=None,
        central=False,
        w=None,
        axis=None,
        dim=None,
        broadcast=True,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        resample_kws=None,
        parallel=True,
        meta=None,
        meta_kws=None,
        x_is_u=False,
    ):
        """
        Create DataCentralMoments object from unaveraged samples with resampling.

        Parameters
        ----------
        {xv}
        {uv}
        {order}
        {freq}
        {indices}
        {nrep}
        {xmom_dim}
        {umom_dim}
        {rep_dim}
        {deriv_dim}
        {central}
        w : array-like, optional
            Weights for each observation.
        {axis}
        {dim}
        {broadcast}
        {dtype}
        {xr_params}
        {resample_kws}
        {meta}
        {meta_kws}
        {x_is_us}

        See Also
        --------
        :meth:`cmomy.xCentralMoments.from_resample_vals`
        """
        if xv is None or x_is_u:
            xv = uv

        kws = dict(
            x=(xv, uv),
            w=w,
            freq=freq,
            indices=indices,
            nrep=nrep,
            dim=dim,
            axis=axis,
            mom=(1, order),
            parallel=parallel,
            resample_kws=resample_kws,
            broadcast=broadcast,
            dtype=dtype,
            dims=dims,
            rep_dim=rep_dim,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
        )

        dxduave = xCentralMoments.from_resample_vals(**kws)

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

        out = out.set_params(meta=out.meta.resample(data=out, meta_kws=meta_kws, **kws))

        return out

    @classmethod
    @docfiller_shared
    def from_ave_raw(
        cls,
        u,
        xu,
        w=None,
        axis=-1,
        umom_axis=None,
        xumom_axis=None,
        rec_dim="rec",
        xmom_dim="xmom",
        umom_dim="umom",
        deriv_dim=None,
        central=False,
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
        name=None,
        meta=None,
        x_is_u=False,
    ):
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
        :meth:`cmomy.xCentralMoments.from_raw`
        """

        if xu is None or x_is_u:
            raw = u

            if w is not None:
                raw.loc[{umom_dim: 0}] = w
            raw = raw.transpose(..., umom_dim)

        elif isinstance(xu, xr.DataArray):
            raw = xr.concat((u, xu), dim=xmom_dim)
            if w is not None:
                raw.loc[{umom_dim: 0, xmom_dim: 0}] = w
            # make sure in correct order
            raw = raw.transpose(..., xmom_dim, umom_dim)
            # return raw
            # raw.data = np.ascontiguousarray(raw.data)
        else:
            if axis is None:
                axis = -1
            if umom_axis is None:
                umom_axis = axis
            if xumom_axis is None:
                xumom_axis = axis

            u = np.swapaxes(u, umom_axis, -1)
            xu = np.swapaxes(xu, xumom_axis, -1)

            shape = xu.shape[:-1]
            shape_moments = (2, min(u.shape[-1], xu.shape[-1]))
            shape_out = shape + shape_moments

            if dtype is None:
                dtype = xu.dtype

            raw = np.empty(shape_out, dtype=dtype)
            raw[..., 0, :] = u
            raw[..., 1, :] = xu
            if w is not None:
                raw[..., 0, 0] = w
            # raw = np.ascontiguousarray(raw)

        return cls.from_raw(
            raw=raw,
            deriv_dim=deriv_dim,
            rec_dim=rec_dim,
            xmom_dim=xmom_dim,
            umom_dim=umom_dim,
            mom=mom,
            central=central,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            meta=meta,
            x_is_u=x_is_u,
        )

    @classmethod
    @docfiller_shared
    def from_ave_central(
        cls,
        du,
        dxdu,
        w=None,
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
        mom=None,
        val_shape=None,
        dtype=None,
        dims=None,
        attrs=None,
        coords=None,
        indexes=None,
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
        w : array-like, optional
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
        :meth:`cmomy.xCentralMoments.from_data`

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
            if w is not None:
                data.loc[{umom_dim: 0, xmom_dim: 0}] = w
            if xave is not None:
                data.loc[{umom_dim: 0, xmom_dim: 1}] = xave
            if uave is not None:
                data.loc[{umom_dim: 1, xmom_dim: 0}] = uave
            data = data.transpose(..., xmom_dim, umom_dim)

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
            if w is not None:
                data[..., 0, 0] = w
            if xave is not None:
                data[..., 1, 0] = xave
            if uave is not None:
                data[..., 0, 1] = uave

        dxduave = xCentralMoments.from_data(
            data=data,
            mom=mom,
            mom_ndim=2,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            mom_dims=(xmom_dim, umom_dim),
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


@attrs.define
@docfiller_shared
class DataCentralMomentsVals(DataCentralMomentsBase):
    """
    Parameters
    ----------
    uv : xarray.DataArray
        raw values of u (energy)
    {xv}
    {order}
    from_vals_kws : dict, optional
        extra arguments passed to xCentralMoments.from_vals
    {dxduave}
    """

    #: Stored energy values
    uv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Stored observable values
    xv: xr.DataArray | None = field(
        validator=attv.optional(attv.instance_of(xr.DataArray))
    )
    #: Expansion order
    order: int | None = kw_only_field(
        default=None, validator=attv.optional(attv.instance_of(int))
    )
    #: Stored weights
    w: Sequence | None = kw_only_field(default=None)
    #: Optional parameteres to :meth:`cmomy.xCentralMoments.from_vals`
    from_vals_kws: Mapping | None = kw_only_field(default=None)
    #: :class:`cmomy.xCentralMoments` object
    dxduave: xCentralMoments | None = kw_only_field(
        default=None,
        validator=attv.optional(attv.instance_of(xCentralMoments)),
    )

    def __attrs_post_init__(self):
        if self.xv is None or self.x_is_u:
            self.xv = self.uv
        if self.from_vals_kws is None:
            self.from_vals_kws = {}

        if self.dxduave is None:
            if self.order is None:
                raise ValueError("must pass order if calculating dxduave")

            self.dxduave = xCentralMoments.from_vals(
                x=(self.xv, self.uv),
                w=self.w,
                dim=self.rec_dim,
                mom=(1, self.order),
                broadcast=True,
                mom_dims=(self.xmom_dim, self.umom_dim),
                **self.from_vals_kws,
            )

    @classmethod
    @docfiller_shared
    def from_vals(
        cls,
        xv,
        uv,
        order,
        w=None,
        rec_dim="rec",
        umom_dim="umom",
        xmom_dim="xmom",
        rep_dim="rep",
        deriv_dim=None,
        val_dims="val",
        central=False,
        from_vals_kws=None,
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
        {w}
        {broadcast}
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
        :meth:`cmomy.xCentralMoments.from_vals`
        """

        # make sure "val" is a list
        if isinstance(val_dims, str):
            val_dims = [val_dims]
        elif not isinstance(val_dims, list):
            val_dims = list(val_dims)

        uv = xrwrap_uv(uv, rec_dim=rec_dim, rep_dim=rep_dim)

        if xv is not None and not x_is_u:
            xv = xrwrap_xv(
                xv,
                rec_dim=rec_dim,
                rep_dim=rep_dim,
                deriv_dim=deriv_dim,
                val_dims=val_dims,
            )

        return cls(
            uv=uv,
            xv=xv,
            order=order,
            w=w,
            rec_dim=rec_dim,
            umom_dim=umom_dim,
            xmom_dim=xmom_dim,
            deriv_dim=deriv_dim,
            central=central,
            from_vals_kws=from_vals_kws,
            meta=meta,
            x_is_u=x_is_u,
        )

    def __len__(self):
        return len(self.uv[self.rec_dim])

    def resample(
        self,
        indices=None,
        nrep=None,
        freq=None,
        resample_kws=None,
        parallel=True,
        axis=None,
        dim=None,
        rep_dim="rep",
        meta_kws=None,
    ):
        """
        Resample data.

        Parameters
        ----------
        {indices}
        {nrep}
        {freq}
        {resample_kws}
        {rep_dim}
        {axis}
        {dim}
        {meta_kws}

        See Also
        --------
        :meth:`cmomy.xCentralMoments.resample`
        """

        if dim is None and axis is None:
            dim = self.rec_dim

        kws = dict(
            indices=indices,
            nrep=nrep,
            freq=freq,
            resample_kws=resample_kws,
            parallel=True,
            axis=axis,
            dim=dim,
            rep_dim=rep_dim,
        )

        dxduave = xCentralMoments.from_resample_vals(
            x=(self.xv, self.uv),
            w=self.w,
            mom=(1, self.order),
            broadcast=True,
            mom_dims=(self.xmom_dim, self.umom_dim),
            **kws,
        )

        meta = self.meta.resample(data=self, meta_kws=meta_kws, **kws)

        return self.new_like(dxduave=dxduave, rec_dim=rep_dim, meta=meta)

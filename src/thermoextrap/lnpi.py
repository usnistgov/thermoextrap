"""
Routinese to temperature expand lnPi
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Hashable, Sequence

import attrs
import numpy as np
import sympy as sp
import xarray as xr

# from attrs import converters as attc
from attrs import field
from attrs import validators as attv
from cmomy import xCentralMoments

from .beta import ExtrapModel, SymDerivBeta
from .beta import factory_derivatives as factory_derivatives_beta
from .beta import u_func, u_func_central
from .core._attrs_utils import (  # MyAttrsMixin,; kw_only_field,
    _cache_field,
    convert_dims_to_tuple,
)
from .core._docstrings import factory_docfiller_shared
from .core.cached_decorators import gcached
from .core.data import (  # DataCentralMome nts,; DataCentralMomentsVals,; DataValues,; DataValuesCentral,
    DataCallbackABC,
)
from .core.models import Derivatives, SymSubs
from .core.sputils import get_default_indexed, get_default_symbol

docfiller_shared = factory_docfiller_shared(names=("default", "beta"))


################################################################################
# lnPi correction stuff
################################################################################
class lnPi_func_central(sp.Function):
    """
    This is a special case of u_func_central.

    For lnPi, have dlnPi/dbeta = mu * N - <u> + <u - mu * N>_GC.
    We ignore the GC average term, as it does not depend on N
    So, the first derivative of this function is u_func_central.
    We consider only a correction of the form
    lnPi_energy = lnPi - beta * mu * N = lnQ - ln XI, where Q, XI are the canonical
    and GC partition functions.

    Then,
    Y' = -U
    Y'' = -U'
    etc
    """

    nargs = 1
    u = get_default_symbol("u")
    lnPi0 = get_default_symbol("lnPi0")
    mudotN = get_default_symbol("mudotN")

    @classmethod
    def deriv_args(cls):
        return u_func_central.deriv_args() + [cls.lnPi0, cls.mudotN]

    def fdiff(self, argindex=1):
        (beta,) = self.args
        return self.mudotN - u_func_central(beta)

    @classmethod
    def eval(cls, beta):
        if beta is None:
            return cls.lnPi0
        else:
            out = None
        return out


class lnPi_func_raw(sp.Function):
    """
    Raw moments version.
    """

    nargs = 1
    u = get_default_indexed("u")
    lnPi0 = get_default_symbol("lnPi0")
    mudotN = get_default_symbol("mudotN")

    @classmethod
    def deriv_args(cls):
        return u_func.deriv_args() + [cls.lnPi0, cls.mudotN]

    def fdiff(self, argindex=1):
        (beta,) = self.args
        return self.mudotN - u_func(beta, 1)

    @classmethod
    def eval(cls, beta):
        if beta is None:
            return cls.lnPi0
        else:
            out = None
        return out


@lru_cache(5)
@docfiller_shared
def factory_derivatives(
    name="lnPi",
    n=None,
    d=None,
    xalpha=False,
    central=False,
    expand=True,
    post_func=None,
):
    """
    Expansion for ln(Pi/Pi_0) (ignore bad parts of stuff)

    Parameters
    ----------
    name : str, default='lnPi'
        If name is `'lnPi'`, then get derivatives of lnPi.
        Otherwise, get derivative object for general `X`.
    {n_order}
    {d_order}
    {xalpha}
    {central}
    {expand}
    {post_func}

    Returns
    -------
    ~thermoextrap.Derivatives

    See Also
    --------
    thermoextrap.beta.factory_derivatives
    """

    if name == "lnPi":
        beta = get_default_symbol("beta")
        if central:
            func = lnPi_func_central(beta)
        else:
            func = lnPi_func_raw(beta)
        derivs = SymDerivBeta(func=func, expand=expand, post_func=post_func)

        exprs = SymSubs(
            derivs, subs_all={derivs.beta: "None"}, expand=False, simplify=False
        )
        return Derivatives.from_sympy(exprs, args=derivs.args)
    else:
        return factory_derivatives_beta(
            name=name,
            n=n,
            d=d,
            xalpha=xalpha,
            central=central,
            post_func=post_func,
            expand=expand,
        )


def _is_xr(name, x):
    if not isinstance(x, xr.DataArray):
        raise ValueError(f"{name} must be an xr.DataArray")
    return x


@attrs.define
class lnPiDataCallback(DataCallbackABC):
    """
    Class to handle metadata callbacks for lnPi data.

    Parameters
    ----------
    lnPi0 : DataArray
        Reference value of lnPi.
    mu : xr.DataArray
        Value of chemical potential.  Must have dimension ``dims_comp``.
    dims_n : hashable or sequence of hashable
        Dimension(s) for number of particle(s).  That is, the dimensions of lnPi0 corresponding to particle number.
    dims_comp : hashable
        Dimension corresponding to components.
    ncoords : DataArray, optional.
        Count of number of particles for given particle number (vector) and component.
        Must have dimensions ``dims_comp`` and ``dims_n``.
    allow_resample : bool, default=False
        If True, allow simplified resampling of ``lnPi0`` data.

    """

    # TODO: rename dims_comp to dim_comp.

    #: lnPi data
    lnPi0: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Chemical potential
    mu: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Dimensions for particle number
    dims_n: Hashable | Sequence[Hashable] = field(converter=convert_dims_to_tuple)
    #: Dimensions for component
    dims_comp: Hashable = field()
    #: Particle number coordinates
    ncoords: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    #: Flag to allow/disallow resampling of ``lnPi0``.
    allow_resample: bool = field(default=False)

    _cache: dict = _cache_field()
    # FIXME: using dims_n, dims_comp naming because this is what is used in lnPi module

    # def __init__(self, lnPi0, mu, dims_n, dims_comp, ncoords=None):

    #     if isinstance(dims_n, str):
    #         dims_n = [dims_n]
    #     self.dims_n = dims_n
    #     self.dims_comp = dims_comp

    #     self.lnPi0 = lnPi0

    #     self.mu = _is_xr("mu", mu)
    #     if ncoords is None:
    #         ncoords = self.get_ncoords()
    #     self.ncoords = _is_xr("ncoords", ncoords)

    def check(self, data):
        pass

    @ncoords.default
    def _set_default_ncoords(self):
        # create grid
        ncoords = np.meshgrid(
            *tuple(self.lnPi0[x].values for x in self.dims_n), indexing="ij"
        )
        ncoords = xr.DataArray(np.array(ncoords), dims=(self.dims_comp,) + self.dims_n)
        return ncoords

    @property
    def lnPi0_ave(self):
        if isinstance(self.lnPi0, xr.DataArray):
            return self.lnPi0
        else:
            # assume lnPi0 is an averaging object ala cmomy
            return self.lnPi0.values

    @gcached()
    def mudotN(self):
        """Dot product of `self.mu` and `self.ncoords`, reduces along `self.dims_comp`"""
        return xr.dot(self.mu, self.ncoords, dims=self.dims_comp)

    def resample(self, data, meta_kws=None, **kws):
        """
        Resample lnPi0 data.

        """

        if not self.allow_resample:
            raise ValueError(
                "Must set `self.allow_resample` to `True` to use resampling. "
                "Resampling here is handled in an ad-hoc way, and should be "
                "used with care."
            )

        warnings.warn(
            "'Correct' resampling of lnPi should be handled externally. "
            "This resamples the average lnPi values.  Instead, it is "
            "recommended to resample based on collection matrices, and "
            "construct lnPi values based on these.",
            category=UserWarning,
            stacklevel=2,
        )

        # wrap in xarray object:
        dc = xCentralMoments.from_vals(
            self.lnPi0.expand_dims(dim="_new", axis=0),
            axis="_new",
            mom_dims="_mom",
            mom=1,
        )
        # resample and reduce
        dc, _ = dc.resample_and_reduce(**kws)
        # return new object
        return self.new_like(lnPi0=dc.values.sel(_mom=1))

    def derivs_args(self, data, derivs_args):
        return tuple(derivs_args) + (self.lnPi0_ave, self.mudotN)


# def factory_data_values(
#     uv,
#     order,
#     lnPi,
#     mu,
#     dims_n,
#     dims_comp,
#     ncoords=None,
#     central=False,
#     skipna=False,
#     rec_dim="rec",
#     umom_dim="umom",
#     xmom_dim="xmom",
#     val_dims="val",
#     rep_dim="rep",
#     deriv_dim=None,
#     chunk=None,
#     compute=None,
#     xv=None,
#     x_is_u=True,
#     **kws,
# ):
#     """
#     Factory function to produce a Data object

#     Parameters
#     ----------
#     uv : array-like
#         energy values.  These are not averaged
#     order : int
#         highest umom_dim to calculate
#     skipna : bool, default=False
#         if True, skip `np.nan` values in creating averages.
#         Can make some "big" calculations slow
#     rec_dim, umom_dim, val_dim, rep_dim, deriv_dim : str
#         names of record (i.e. time), umom_dim, value, replicate,
#         and derivative (with respect to alpha)
#     chunk : int or dict, optional
#         If specified, perform chunking on resulting uv, xv arrays.
#         If integer, chunk with {rec: chunk}
#         otherwise, should be a mapping of form {dim_0: chunk_0, dim_1: chunk_1, ...}
#     compute : bool, optional
#         if compute is True, do compute averages greedily.
#         if compute is False, and have done chunking, then defer calculation of averages (i.e., will be dask future objects).
#         Default is to do greedy calculation

#     constructor : 'val'
#     kws : dict, optional

#         extra arguments
#     """

#     if central:
#         cls = DataValuesCentral
#     else:
#         cls = DataValues

#     meta = lnPiDataCallback(
#         lnPi0=lnPi, mu=mu, dims_n=dims_n, dims_comp=dims_comp, ncoords=ncoords
#     )

#     return cls.from_vals(
#         uv=uv,
#         xv=xv,
#         order=order,
#         skipna=skipna,
#         rec_dim=rec_dim,
#         umom_dim=umom_dim,
#         val_dims=val_dims,
#         rep_dim=rep_dim,
#         deriv_dim=deriv_dim,
#         chunk=chunk,
#         compute=compute,
#         x_is_u=x_is_u,
#         meta=meta,
#         **kws,
#     )


# much more likely to have pre-aves here, but save that for the user
@docfiller_shared
def factory_extrapmodel_lnPi(
    beta,
    data,
    *,
    central=None,
    order=None,
    alpha_name="beta",
    derivatives=None,
    post_func=None,
    derivatives_kws=None,
):
    """
    factory function to create Extrapolation model for beta expansion

    Parameters
    ----------
    {beta}
    data : Data object
        Should include lnPiDataCallback object as well
        See data.AbstractData
    order : int, optional
        maximum order.
        If not specified, default to `data.order + 1`
    {central}
    {post_func}
    {alpha_name}
    derivatives : :class:`thermoextrap.Derivatives`, optional
        Derivates object.  If not passed, construct derivatives using :func:`thermoextrap.lnpi.factory_derivatives`.
    derivates_kws : mapping, optional
        Optional parameters to :func:`thermoextrap.lnpi.factory_derivatives`.

    Returns
    -------
    extrapmodel : ExtrapModel

    See Also
    --------
    thermoextrap.lnpi.factory_derivaties
    ~thermoextrap.ExtrapModel
    """

    if central is None:
        central = data.central
    if order is None:
        order = data.order + 1

    assert central == data.central
    assert order <= data.order + 1
    assert data.x_is_u

    if derivatives is None:
        if derivatives_kws is None:
            derivatives_kws = {}
        derivatives = factory_derivatives(
            name="lnPi", central=central, post_func=post_func, **derivatives_kws
        )
    return ExtrapModel(
        alpha0=beta,
        data=data,
        derivatives=derivatives,
        order=order,
        # minus_log=mineus_log,
        alpha_name=alpha_name,
    )
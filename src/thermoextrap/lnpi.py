r"""
Inverse temperature expansion of macrostate distribution (:mod:`~thermoextrap.lnpi`)
====================================================================================

This is used to extrapolate, in inverse temperature :math:`\beta = (k_{\rm B} T)^{-1}`, the macrostate distribution function :math:`\ln\Pi` from transition matrix Monte Carlo simulations.

See :ref:`notebooks/macrostate_dist_extrap:macrostate distribution extrapolation` for example usage.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Hashable, Sequence

import attrs
import numpy as np
import xarray as xr

# from attrs import converters as attc
from attrs import field
from attrs import validators as attv
from cmomy import xCentralMoments

from . import beta as beta_xpan

# from .beta import ExtrapModel, SymDerivBeta, u_func, u_func_central
# from .beta import factory_derivatives as factory_derivatives_beta
from .core._attrs_utils import _cache_field, convert_dims_to_tuple
from .core._docstrings import factory_docfiller_shared
from .core.cached_decorators import gcached
from .core.data import DataCallbackABC
from .core.models import Derivatives, ExtrapModel, SymFuncBase, SymSubs
from .core.sputils import get_default_indexed, get_default_symbol

docfiller_shared = factory_docfiller_shared(names=("default", "beta"))


################################################################################
# lnPi correction stuff
################################################################################
class lnPi_func_central(SymFuncBase):
    r"""
    Special case of u_func_central.

    For lnPi, have:

    .. math::

        \newcommand{\ave}[1]{\langle #1 \rangle}

        (\ln \Pi)' = \frac{d \ln \Pi}{d \beta} = \mu N - \ave{u} + \ave{u - \mu N}_{\rm GC}

    where :math:`\ave{}` and :math:`\ave{}_{\rm GC}` are the canonical and grand canonical (GC) ensemble averages.
    We ignore the GC average term, as it does not depend on N.  Note that this is not
    necessarily the case for molecular systems.
    So, the first derivative of this function is :func:`thermoextrap.beta.u_func_central`.
    We consider only a correction of the form:

    .. math::

        (\ln\Pi)_{\text{energy}} = \ln\Pi - \beta \mu N = \ln Q - \ln \Xi

    where :math:`Q\text{ and }\Xi` are the canonical and GC partition functions, respectively. thus,

    .. math::

        \begin{align*}
          (\ln\Pi)_{\text{energy}}'  &= - U \\
          (\ln\Pi)_{\text{energy}}'' &=  -U' \\
              &\,\,\vdots
        \end{align*}

    """

    nargs = 1
    u = get_default_symbol("u")
    lnPi0 = get_default_symbol("lnPi0")
    mudotN = get_default_symbol("mudotN")

    @classmethod
    def deriv_args(cls):
        return beta_xpan.u_func_central.deriv_args() + [cls.lnPi0, cls.mudotN]

    def fdiff(self, argindex=1):
        (beta,) = self.args
        return self.mudotN - beta_xpan.u_func_central(beta)

    @classmethod
    def eval(cls, beta):
        if beta is None:
            return cls.lnPi0
        else:
            out = None
        return out


class lnPi_func_raw(SymFuncBase):
    """Raw moments version."""

    nargs = 1
    u = get_default_indexed("u")
    lnPi0 = get_default_symbol("lnPi0")
    mudotN = get_default_symbol("mudotN")

    @classmethod
    def deriv_args(cls):
        return beta_xpan.u_func.deriv_args() + [cls.lnPi0, cls.mudotN]

    def fdiff(self, argindex=1):
        (beta,) = self.args
        return self.mudotN - beta_xpan.u_func(beta, 1)

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
    Expansion for ln(Pi/Pi_0) (ignore bad parts of stuff).

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
    ~thermoextrap.models.Derivatives

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
        derivs = beta_xpan.SymDerivBeta(func=func, expand=expand, post_func=post_func)

        exprs = SymSubs(
            derivs, subs_all={derivs.beta: "None"}, expand=False, simplify=False
        )
        return Derivatives.from_sympy(exprs, args=derivs.args)
    else:
        return beta_xpan.factory_derivatives(
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
    mu : DataArray
        Value of chemical potential.  Must have dimension ``dims_comp``.
    dims_n : hashable or sequence of hashable
        Dimension(s) for number of particle(s).  That is, the dimensions of lnPi0 corresponding to particle number.
    dims_comp : hashable
        Dimension corresponding to components.
    ncoords : DataArray, optional
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
        """Dot product of `self.mu` and `self.ncoords`, reduces along `self.dims_comp`."""
        return xr.dot(self.mu, self.ncoords, dims=self.dims_comp)

    def resample(self, data, meta_kws=None, **kws):
        """Resample lnPi0 data."""

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
    Factory function to create Extrapolation model for beta expansion.

    Parameters
    ----------
    {beta}
    data : object
        Data object.
        Should include :class:`lnPiDataCallback` object as well
    order : int, optional
        maximum order.
        If not specified, default to `data.order + 1`
    {central}
    {post_func}
    {alpha_name}
    derivatives : :class:`thermoextrap.models.Derivatives`, optional
        Derivatives object.  If not passed, construct derivatives using :func:`thermoextrap.lnpi.factory_derivatives`.
    derivates_kws : mapping, optional
        Optional parameters to :func:`thermoextrap.lnpi.factory_derivatives`.

    Returns
    -------
    extrapmodel : :class:`~thermoextrap.models.ExtrapModel`

    See Also
    --------
    thermoextrap.lnpi.factory_derivatives
    ~thermoextrap.models.ExtrapModel
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

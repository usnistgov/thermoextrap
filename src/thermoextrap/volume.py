"""
Volume extrapolation (:mod:`~thermoextrap.volume`)
==================================================

Note: This only handles volume expansion to first order.
Also, Only DataValues like objects are supported.
"""


from functools import lru_cache

import attrs
import xarray as xr
from attrs import field
from attrs import validators as attv

from .core._attrs_utils import _cache_field
from .core._docstrings import factory_docfiller_shared
from .core.cached_decorators import gcached
from .core.data import DataCallbackABC, DataValues
from .core.models import Derivatives, ExtrapModel
from .core.xrutils import xrwrap_xv

docfiller_shared = factory_docfiller_shared(
    names=("default", "beta", "volume"),
)


# Need funcs to pass to Coefs class
# Just needs to be indexable based on order, so...
# d^n X / d V^n = funcs[n](*args)
# Could create list of simple functions for each derivative order
# But here I'm trying something similar to what's in xtrapy already
# For any general observable, might need to modify Data class to also pass full x values
# (or other data, perhaps)
# This is because the last, custom term may need more information than W and x*W moments
# Though ALL of the observables in the paper end up with a unique term that is just
# some constant multiplied by an average of x (same with ideal gas, too).


class VolumeDerivFuncs:
    """
    Calculates specific derivative values at refV with data x and W.
    Only go to first order for volume extrapolation.
    Here W represents the virial instead of the potential energy.
    """

    def __getitem__(self, order):
        # Check to make sure not going past first order
        if order > 1:
            raise ValueError(
                "Volume derivatives cannot go past 1st order"
                + " and received %i" % order
                + "\n(because would need derivatives of forces)"
            )
        else:
            return self.create_deriv_func(order)

    # TODO: move this to just a functions
    @staticmethod
    def create_deriv_func(order):
        # Works only because of local scope
        # Even if order is defined somewhere outside of this class, won't affect returned func
        def func(W, xW, dxdq, volume, ndim=1):
            """
            Calculat function.  dxdq is <sum_{i=1}^N dy/dx_i x_i>.

            for ideal gas
            """
            # NOTE: W here has beta in it:
            # that is W <- beta * virial

            # Zeroth order derivative
            if order == 0:
                deriv_val = xW[0]
            # First order derivative
            else:
                deriv_val = (-xW[0] * W[1] + xW[1] + dxdq) / (volume * ndim)
            return deriv_val

        return func


@lru_cache(5)
def factory_derivatives():
    """Factory function to provide coefficients of expansion."""
    deriv_funcs = VolumeDerivFuncs()
    return Derivatives(deriv_funcs)


@attrs.define
@docfiller_shared
class VolumeDataCallback(DataCallbackABC):
    """
    Object to handle callbacks of metadata.

    Parameters
    ----------
    volume : float
        Reference value of system volume.
    {dxdqv}
    {ndim}

    See Also
    --------
    thermoextrap.data.DataCallbackABC
    """

    volume: float = field(validator=attv.instance_of(float))
    dxdqv: xr.DataArray = field(validator=attv.instance_of(xr.DataArray))
    ndim: int = field(default=3, validator=attv.instance_of(int))

    _cache: dict = _cache_field()

    def check(self, data):
        pass

    @gcached(prop=False)
    def dxdq(self, rec_dim, skipna):
        return self.dxdqv.mean(rec_dim, skipna=skipna)

    def resample(self, data, meta_kws, indices, **kws):
        if not isinstance(data, DataValues):
            raise NotImplementedError("resampling only possible with DataValues style.")
        else:
            return self.new_like(dxdqv=self.dxdqv[indices])

    def derivs_args(self, data, derivs_args):
        return tuple(derivs_args) + (
            self.dxdq(data.rec_dim, data.skipna),
            self.volume,
            self.ndim,
        )


@docfiller_shared
def factory_extrapmodel(
    volume,
    uv,
    xv,
    dxdqv,
    ndim=3,
    order=1,
    alpha_name="volume",
    rec_dim="rec",
    val_dims="val",
    rep_dim="rep",
    **kws,
):
    """
    Factory function to create Extrapolation model for volume expansion.

    Parameters
    ----------
    {volume}
    volume : float
        reference value of volume
    uv, xv : array-like
        values for u and x
        Note that here, uv should be the temperature scaled virial `beta * virial`
    dxdqv : array-like
        values of `sum dx/dq_i q_i` where `q_i` is the ith coordinate
        This array is wrapped with `cmomy.data.xrwrap_xv`
    {ndim}
    order : int, default=1
        maximum order.  Only `order=1` is currently supported
    alpha_name, str, default='volume'
        name of expansion parameter
    {rec_dim}
    {val_dims}
    {rep_dim}
    **kws :
        Extra arguments to :meth:`thermoextrap.data.DataValues.from_vals`

    Returns
    -------
    extrapmodel : :class:`thermoextrap.models.ExtrapModel`

    """

    if order != 1:
        raise ValueError("only order=1 is supported")

    dxdqv = xrwrap_xv(
        dxdqv, rec_dim=rec_dim, rep_dim=rep_dim, deriv_dim=None, val_dims=val_dims
    )

    meta = VolumeDataCallback(volume=volume, dxdqv=dxdqv, ndim=ndim)

    data = DataValues.from_vals(
        uv=uv,
        xv=xv,
        order=order,
        meta=meta,
        rec_dim=rec_dim,
        rep_dim=rep_dim,
        val_dims=val_dims,
        deriv_dim=None,
        **kws,
    )

    derivatives = factory_derivatives()
    return ExtrapModel(
        alpha0=volume,
        data=data,
        derivatives=derivatives,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )

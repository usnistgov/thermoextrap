"""
Routines for volume expansion

This only handles volume expansion to first order.
Also, Only DataValues like objects are supported.
"""

from __future__ import absolute_import

from functools import lru_cache

from .core.cached_decorators import gcached
from .core.data import DataCallbackABC, DataValues
from .core.models import Derivatives, ExtrapModel
from .core.xrutils import xrwrap_xv

# Lazily imported everything above - will trim down later
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


class VolumeDerivFuncs(object):
    """Calculates specific derivative values at refV with data x and W.
    Only go to first order for volume extrapolation.
    Here W represents the virial instead of the potential energy.
    """

    # def __init__(self):
    #     pass

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
            dxdq is <sum_{i=1}^N dy/dx_i x_i>

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
    """
    factory function to provide coefficients of expansion

    Parameters
    ----------
    refV : reference volume (default 1 - if already divided by volume no need to set)

    Returns
    -------
    coefs : Coefs object used to calculate moments
    """
    deriv_funcs = VolumeDerivFuncs()
    return Derivatives(deriv_funcs)


class VolumeDataCallback(DataCallbackABC):
    """
    object to handle callbacks of metadata
    """

    def __init__(self, volume, dxdqv, ndim=3):
        self.volume = volume
        self.dxdqv = dxdqv
        self.ndim = ndim

    def check(self, data):
        pass

    @property
    def param_names(self):
        return ["volume", "dxdqv", "ndim"]

    @gcached(prop=False)
    def dxdq(self, rec_dim, skipna):
        return self.dxdqv.mean(rec_dim, skipna=skipna)

    def resample(self, data, meta_kws, indices, **kws):
        return self.new_like(dxdqv=self.dxdqv[indices])

    def derivs_args(self, data, derivs_args):
        return tuple(derivs_args) + (
            self.dxdq(data.rec_dim, data.skipna),
            self.volume,
            self.ndim,
        )


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
    **kws
):
    """
    factory function to create Extrapolation model for volume expansion

    Parameters
    ----------
    volume : float
        reference value of volume
    uv, xv : array-like
        values for u and x
        Note that here, uv should be the temperature scaled virial `beta * virial`
    dxdqv : array-like
        values of `sum dx/dq_i q_i` where `q_i` is the ith coordinate
        This array is wrapped with `cmomy.data.xrwrap_xv`
    ndim : int, default=3
        number of dimensions
    order : int, default=1
        maximum order.  Only `order=1` is currently supported
    alpha_name, str, default='volume'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_data_values`

    Returns
    -------
    extrapmodel : ExtrapModel object
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
        # meta=dict(
        #     dxdqv=dxdqv,
        #     volume=volume,
        #     ndim=ndim,
        # ),
        rec_dim=rec_dim,
        rep_dim=rep_dim,
        val_dims=val_dims,
        deriv_dim=None,
        **kws
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

"""
Routines for volume expansion

This only handles volume expansion to first order.
Also, Only DataValues like objects are supported.
"""

from __future__ import absolute_import

from functools import lru_cache

from .cached_decorators import gcached
from .core import Coefs, ExtrapModel
from .data import DataValues, xrwrap_xv

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
def factory_coefs():
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
    return Coefs(deriv_funcs)


# make a special data class
class DataValuesVolume(DataValues):
    def __init__(
        self,
        uv,
        xv,
        order,
        dxdqv,
        volume,
        ndim=3,
        rec="rec",
        mom_u="mom_u",
        deriv=None,
        skipna=False,
        chunk=None,
        compute=None,
        build_aves_kws=None,
    ):

        super(DataValuesVolume, self).__init__(
            uv=uv,
            xv=xv,
            order=order,
            skipna=skipna,
            rec=rec,
            mom_u=mom_u,
            deriv=deriv,
            chunk=chunk,
            compute=compute,
            build_aves_kws=build_aves_kws,
            # kws
            volume=volume,
            ndim=ndim,
            dxdqv=dxdqv,
        )

    @property
    def xcoefs_args(self):
        return (
            self.u_selector,
            self.xu_selector,
            self.dxdq,
            self.kws["volume"],
            self.kws["ndim"],
        )

    @gcached()
    def dxdq(self):
        return self.kws["dxdqv"].mean("rec", skipna=self.skipna)

    def resample_other_params(self, indices):
        # resample "other" arrays
        out = self.kws.copy()
        out["dxdqv"] = out["dxdqv"][indices]
        return out


def factory_extrapmodel(
    volume,
    uv,
    xv,
    dxdqv,
    ndim=3,
    order=1,
    alpha_name="volume",
    rec="rec",
    val="val",
    rep="rep",
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
        extra arguments to `factory_data`

    Returns
    -------
    extrapmodel : ExtrapModel object
    """

    if order != 1:
        raise ValueError("only order=1 is supported")

    dxdqv = xrwrap_xv(dxdqv, rec="rec", rep="rep", deriv=None, val=val)
    data = DataValuesVolume.from_vals(
        uv=uv,
        xv=xv,
        order=order,
        dxdqv=dxdqv,
        volume=volume,
        ndim=ndim,
        rec=rec,
        rep=rep,
        val=val,
        deriv=None,
        **kws
    )

    coefs = factory_coefs()
    return ExtrapModel(
        alpha0=volume,
        data=data,
        coefs=coefs,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )

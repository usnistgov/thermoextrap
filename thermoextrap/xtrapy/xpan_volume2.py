
"""
Routines for volume expansion(s)
"""

from __future__ import absolute_import

from functools import lru_cache

import xarray as xr
import sympy as sp

from .cached_decorators import gcached
from .core import _get_default_symbol, _get_default_indexed
from .core import DataTemplateValues, DatasetSelector
from .core import SymSubs, Coefs
from .core import ExtrapModel, PerturbModel

from .xpan_beta import Data

#Lazily imported everything above - will trim down later

#Need funcs to pass to Coefs class
#Just needs to be indexable based on order, so...
# d^n X / d V^n = funcs[n](*args)
#Could create list of simple functions for each derivative order
#But here I'm trying something similar to what's in xtrapy already
#For any general observable, might need to modify Data class to also pass full x values
#(or other data, perhaps)
#This is because the last, custom term may need more information than W and x*W moments
#Though ALL of the observables in the paper end up with a unique term that is just
#some constant multiplied by an average of x (same with ideal gas, too).
class VolumeDerivFuncsIG(object):
    """Calculates specific derivative values at refV with data x and W.
       Only go to first order for volume extrapolation.
       Here W represents the virial instead of the potential energy.
    """

    # args = volume, beta, ndim

    def __init__(self):
        pass

    def __getitem__(self, order):
        #Check to make sure not going past first order
        if order > 1:
            raise ValueError("Volume derivatives cannot go past 1st order"
                             +" and received %i"%order
                             +"\n(because would need derivatives of forces)")
        else:
            return self.create_deriv_func(order)

    def create_deriv_func(self, order):
        #Works only because of local scope
        #Even if order is defined somewhere outside of this class, won't affect returned func

        def func(W, xW, volume, beta, ndim=1):
            #Zeroth order derivative
            if order == 0:
                deriv_val =  xW[0]
            #First order derivative
            else:
                deriv_val = beta / (volume * ndim) * (xW[1] - xW[0]*W[1]) 
                #Term unique to Ideal Gas... <x>/L
                #Replace with whatever is ap propriate to observable of interest
                deriv_val += xW[0] * beta / (volume * ndim)
            return deriv_val

        return func

@lru_cache(5)
def factory_coefs_volume():
    """
    factory function to provide coefficients of expansion

    Parameters
    ----------
    refV : reference volume (default 1 - if already divided by volume no need to set)

    Returns
    -------
    coefs : Coefs object used to calculate moments
    """
    deriv_funcs = VolumeDerivFuncsIG()
    return Coefs(deriv_funcs)


# make a special data class
class DataSpecial(Data):

    def __init__(self, uv, xv, order,
                 volume, beta, ndim=1,
                 skipna=False, xalpha=False, rec='rec', moment='moment', val='val', rep='rep', deriv='deriv', chunk=None, compute=None, **kws):

        super(DataSpecial, self).__init__(
            uv=uv,
            xv=xv,
            order=order, skipna=skipna,
            xalpha=xalpha, rec=rec, moment=moment, val=val, rep=rep, deriv=deriv, chunk=chunk, compute=compute, **kws)

        self.volume = volume
        self.beta = beta
        self.ndim = ndim

    @property
    def xcoefs_args(self):
        return (self.u_selector, self.xu_selector, self.volume, self.beta, self.ndim)





def factory_extrapmodel_volume(
        order, uv, xv,
        volume, beta, ndim=1,
        alpha_name='volume', **kws
):
    """
    factory function to create Extrapolation model for volume expansion

    Parameters
    ----------
    order : int
        maximum order
    volume : float
        reference value of volume
    uv, xv : array-like
        values for u and x
    beta : float
        value of inverse temperature
    ndim : int
        number of dimensions
    alpha_name, str, default='volume'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_data`

    Returns
    -------
    extrapmodel : ExtrapModel object
    """
    data = DataSpecial(uv=uv, xv=xv, order=order, volume=volume, beta=beta, ndim=ndim,
                       xalpha=False, **kws

    )

    coefs = factory_coefs_volume()
    return ExtrapModel(
        alpha0=volume, data=data, coefs=coefs, order=order, minus_log=False,
        alpha_name=alpha_name
    )



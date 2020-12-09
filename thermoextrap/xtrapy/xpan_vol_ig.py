"""
Routines for volume expansion(s) of ideal gas
"""

from __future__ import absolute_import

from functools import lru_cache

import xarray as xr
import sympy as sp

from .cached_decorators import gcached
from .core import _get_default_symbol, _get_default_indexed
# from .core import DataTemplateValues, DatasetSelector
from .core import SymSubs, Coefs
from .core import ExtrapModel, PerturbModel

from .xpan_beta import factory_data
#from .data import DataCentralMoments, DataCentralMomentsVals

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
    def __init__(self, refV=1.0):
        #If do not set refV, assumes virial data is already divided by the reference volume
        #If this is not the case, need to set refV
        #Or if need refV to also compute custom term, need to specify
        self.refV = refV

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

        def func(W, xW):

            if order == 0:
                #Zeroth order derivative
                deriv_val =  xW[0]

            else:
                #First order derivative
                deriv_val = (xW[1] - xW[0]*W[1]) / (self.refV) #No 3 b/c our IG is 1D
                #Term unique to Ideal Gas... <x>/L
                #Replace with whatever is appropriate to observable of interest
                deriv_val += (xW[0] / self.refV)
            return deriv_val

        return func


@lru_cache(5)
def factory_coefs(refV=1.0):
    """
    factory function to provide coefficients of expansion

    Parameters
    ----------
    refV : reference volume (default 1 - if already divided by volume no need to set)

    Returns
    -------
    coefs : Coefs object used to calculate moments
    """
    deriv_funcs = VolumeDerivFuncsIG(refV=refV)
    return Coefs(deriv_funcs)


def factory_extrapmodel(
        order, alpha0, uv, xv, alpha_name='volume', **kws
):
    """
    factory function to create Extrapolation model for volume expansion

    Parameters
    ----------
    order : int
        maximum order
    alpha0 : float
        reference value of volume
    uv, xv : array-like
        values for u and x
    alpha_name, str, default='volume'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_data`

    Returns
    -------
    extrapmodel : ExtrapModel object
    """
    data = factory_data(
        uv=uv, xv=xv, order=order, central=False, xalpha=False, **kws
    )
    coefs = factory_coefs(refV=alpha0)
    return ExtrapModel(
        alpha0=alpha0, data=data, coefs=coefs, order=order, minus_log=False,
        alpha_name=alpha_name
    )



def factory_extrapmodel_data(
        alpha0, data, order=None, alpha_name='volume'
):
    """
    factory function to create Extrapolation model for volume expansion

    Parameters
    ----------
    alpha0 : float
        reference value of volume
    data : data object
        Note that this data object should have central=False, deriv=None
    alpha_name, str, default='volume'
        name of expansion parameter
    Returns
    -------
    extrapmodel : ExtrapModel object
    """

    if order is None:
        order = data.order

    assert not data.central
    assert data.deriv is None

    coefs = factory_coefs_volume(refV=alpha0)
    return ExtrapModel(
        alpha0=alpha0, data=data, coefs=coefs, order=order, minus_log=False,
        alpha_name=alpha_name
    )

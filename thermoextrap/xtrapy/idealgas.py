"""
Analytic ideal gas in 1d
"""

from __future__ import absolute_import
from functools import lru_cache
# import xarray as xr
import numpy as np
import sympy as sp
from .cached_decorators import gcached
from .core import _get_default_symbol, _get_default_indexed

beta_sym, vol_sym = sp.symbols('beta_sym vol_sym')
xave_sym = (
    (1 / beta_sym) -
    vol_sym / (sp.exp(beta_sym * vol_sym) - 1)
)

# def x_ave_sym(beta, vol):
#     return (1 / beta) - vol / (sp.exp(beta * vol) - 1)


def x_ave(beta, vol=1.0):
    """<x> position"""
    return 1. / beta - vol / (np.exp(beta * vol) - 1.)

def x_var(beta, vol=1.0):
    return (
        1. / beta ** 2 -
        (vol ** 2 * np.exp(beta * vol) /
         ((np.exp(beta * vol) - 1)**2)))


def x_prob(x, beta, vol=1.0):
    return (
        (beta * np.exp(-beta * x)) /
        (1.0 - np.exp(-beta * vol))
    )

#from scipy.stats import norm
def u_prob(u, npart, beta, vol=1.0):
    u_ave = npart * x_ave(beta, vol)
    u_std = np.sqrt(npart * x_var(beta, vol))
    return np.exp(-0.5 * ((u - u_ave)/u_std)**2) / (u_std * np.sqrt(2 * np.pi))
#    return norm.pdf(u, u_ave, u_std)


def x_cdf(x, beta, vol=1.0):
    return (
        (1. - np.exp(-beta * x)) /
        (1. - np.exp(-beta * vol))
    )

def x_sample(shape, beta, vol=1.0, r=None):
    if r is None:
        if isinstance(shape, int):
            shape = (shape,)
        r = np.random.rand(*shape)
    return (
        (-1. / beta) *
        np.log(1. - r * (1. - np.exp(-beta * vol)))
    )

def u_sample(shape, beta, vol=1.0, r=None):
    """for sampling multiple configurations, with particles
    shape = (nsamp, npart)
    """
    return x_sample(shape=shape, beta=beta, vol=vol, r=r).sum(axis=-1)



@lru_cache(maxsize=100)
def dbeta_xave(k):
    deriv = sp.diff(xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, 'numpy')

@lru_cache(maxsize=100)
def dvol_xave(k):
    deriv = sp.diff(xave_sym, vol_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, 'numpy')


def x_beta_extrap(order, beta0, beta, vol=1.0):
    """Returns total and unnormalized coefs"""
    dbeta = beta - beta0

    out = []
    tot = 0.0
    for k in range(order + 1):
        val = dbeta_xave(k)(beta0, vol)
        out.append(val)
        tot += val / np.math.factorial(k) * (dbeta ** k)
    return tot, np.array(out)

def x_vol_extrap(order, vol0, beta, vol):

    dvol = vol - vol0

    out = []
    tot = 0.0

    for k in range(order + 1):
        val = dvol_xave(k)(beta, vol0)
        out.append(val)
        tot += val / np.math.factorial(k) * (dvol ** k)
    return tot, np.array(out)


def generate_data(shape, beta, vol=1.0, r=None):
    """shape = (nsamp, npart)"""
    positions = x_sample(shape, beta, vol)
    x = positions.mean(axis=-1)
    u = positions.sum(axis=-1)
    return (x, u)





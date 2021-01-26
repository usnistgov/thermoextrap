"""
Analytic ideal gas in 1d
"""

from __future__ import absolute_import

from functools import lru_cache

# import xarray as xr
import numpy as np
import sympy as sp

# from .cached_decorators import gcached
# from .models import _get_default_indexed, _get_default_symbol

beta_sym, vol_sym = sp.symbols("beta_sym vol_sym")
xave_sym = (1 / beta_sym) - vol_sym / (sp.exp(beta_sym * vol_sym) - 1)

# def x_ave_sym(beta, vol):
#     return (1 / beta) - vol / (sp.exp(beta * vol) - 1)


def x_ave(beta, vol=1.0):
    """<x> position"""
    return 1.0 / beta - vol / (np.exp(beta * vol) - 1.0)


def x_var(beta, vol=1.0):
    return 1.0 / beta ** 2 - (
        vol ** 2 * np.exp(beta * vol) / ((np.exp(beta * vol) - 1) ** 2)
    )


def x_prob(x, beta, vol=1.0):
    return (beta * np.exp(-beta * x)) / (1.0 - np.exp(-beta * vol))


# from scipy.stats import norm
def u_prob(u, npart, beta, vol=1.0):
    u_ave = npart * x_ave(beta, vol)
    u_std = np.sqrt(npart * x_var(beta, vol))
    return np.exp(-0.5 * ((u - u_ave) / u_std) ** 2) / (u_std * np.sqrt(2 * np.pi))


#    return norm.pdf(u, u_ave, u_std)


def x_cdf(x, beta, vol=1.0):
    return (1.0 - np.exp(-beta * x)) / (1.0 - np.exp(-beta * vol))


def x_sample(shape, beta, vol=1.0, r=None):
    if r is None:
        if isinstance(shape, int):
            shape = (shape,)
        r = np.random.rand(*shape)
    return (-1.0 / beta) * np.log(1.0 - r * (1.0 - np.exp(-beta * vol)))


def u_sample(shape, beta, vol=1.0, r=None):
    """for sampling multiple configurations, with particles
    shape = (nsamp, npart)
    """
    return x_sample(shape=shape, beta=beta, vol=vol, r=r).sum(axis=-1)


@lru_cache(maxsize=100)
def dbeta_xave(k):
    deriv = sp.diff(xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_minuslog(k):
    deriv = sp.diff(-sp.log(xave_sym), beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_depend(k):
    deriv = sp.diff(beta_sym * xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_depend_minuslog(k):
    deriv = sp.diff(-sp.log(beta_sym * xave_sym), beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dvol_xave(k):
    deriv = sp.diff(xave_sym, vol_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


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


def x_beta_extrap_minuslog(order, beta0, beta, vol=1.0):
    """Same as x_beta_extrap but with -ln<x>"""
    dbeta = beta - beta0

    out = np.zeros(order + 1)
    tot = 0.0
    for o in range(order + 1):
        out[o] = dbeta_xave_minuslog(o)(beta0, vol)
        # Above does derivative with sympy... slower, but more straight-forward than below
        # if o == 0:
        #     out[o] = -np.log(x_ave(beta0, vol))
        # else:
        #     for k in range(1,o+1):
        #         this_diffs = np.array([dbeta_xave(kk)(beta0, vol) for kk in range(1, o-k+2)])
        #         out[o] += (np.math.factorial(k-1) * (-1/x_ave(beta0, vol))**k) *  sp.bell(o, k, this_diffs)
        tot += out[o] * ((dbeta) ** o) / np.math.factorial(o)

    return tot, out


def x_beta_extrap_depend(order, beta0, beta, vol=1.0):
    """Same as x_beta_extrap but for <beta*x> to provide non-sensical beta dependence"""
    dbeta = beta - beta0

    out = np.zeros(order + 1)
    tot = 0.0
    for o in range(order + 1):
        out[o] = dbeta_xave_depend(o)(beta0, vol)
        # Above does derivative with sympy... slower, but more straight-forward than below
        # if o == 0:
        #     out[o] = beta0*(x_ave(beta0, vol))
        # else:
        #     out[o] = (o * dbeta_xave(o-1)(beta0, vol) + beta0 * dbeta_xave(o)(beta0, vol))
        tot += out[o] * ((dbeta) ** o) / np.math.factorial(o)

    return tot, out


def x_beta_extrap_depend_minuslog(order, beta0, beta, vol=1.0):
    """Same as x_beta_extrap but with -ln<beta*x>"""
    dbeta = beta - beta0

    out = np.zeros(order + 1)
    tot = 0.0
    for o in range(order + 1):
        out[o] = dbeta_xave_depend_minuslog(o)(beta0, vol)
        # Above does derivative with sympy... slower, but more straight-forward than below
        # if o == 0:
        #     out[o] = -np.log(beta0 * x_ave(beta0, vol))
        # else:
        #     out[o] += np.math.factorial(o-1)*((-1.0/beta0)**o)
        #     for k in range(1,o+1):
        #         this_diffs = np.array([dbeta_xave(kk)(beta0, vol) for kk in range(1, o-k+2)])
        #         out[o] += (np.math.factorial(k-1) * (-1/x_ave(beta0, vol))**k) *  sp.bell(o, k, this_diffs)
        tot += out[o] * ((dbeta) ** o) / np.math.factorial(o)

    return tot, out


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

"""
Analytic ideal gas in 1D in an external field.
The position, x, may vary from 0 to L, with the field acting linearly on x, U(x) = a*x, where for simplicity we let a=1.
As a result, the potential energy of a system of N particles with positions x_1, x_2, ... x_N is the sum of the positions, U = Sum(x_i), i from 1 to N.
This is a useful test system with analytical solutions coded alongside the ability to randomly generate data.
"""


from functools import lru_cache

import numpy as np
import sympy as sp

# global variables
beta_sym, vol_sym = sp.symbols("beta_sym vol_sym")
xave_sym = (1 / beta_sym) - vol_sym / (sp.exp(beta_sym * vol_sym) - 1)


def x_ave(beta, vol=1.0):
    """
    Average position x at the inverse temperature beta with L=vol
    """
    return 1.0 / beta - vol / (np.exp(beta * vol) - 1.0)


def x_var(beta, vol=1.0):
    """
    Variance in position, x at the inverse temperature beta with L=vol
    """
    return 1.0 / beta**2 - (
        vol**2 * np.exp(beta * vol) / ((np.exp(beta * vol) - 1) ** 2)
    )


def x_prob(x, beta, vol=1.0):
    """
    Canonical probability of position x for single particle at inverse temperature beta with L=vol
    """
    return (beta * np.exp(-beta * x)) / (1.0 - np.exp(-beta * vol))


# from scipy.stats import norm
def u_prob(u, npart, beta, vol=1.0):
    """
    In the large-N limit, the probability of the potential energy is Normal, so provides that
    """
    u_ave = npart * x_ave(beta, vol)
    u_std = np.sqrt(npart * x_var(beta, vol))
    return np.exp(-0.5 * ((u - u_ave) / u_std) ** 2) / (u_std * np.sqrt(2 * np.pi))


#    return norm.pdf(u, u_ave, u_std)


def x_cdf(x, beta, vol=1.0):
    """
    Cumulative probability density for position x for single particle at inverse temperature B and L=vol
    """
    return (1.0 - np.exp(-beta * x)) / (1.0 - np.exp(-beta * vol))


def x_sample(shape, beta, vol=1.0, r=None):
    """
    Samples x in specified shape from the probability density at inverse temperature B with L=vol
    Does sampling based on inversion of cumulative distribution function

    r may be a specified random number
    """
    if r is None:
        if isinstance(shape, int):
            shape = (shape,)
        r = np.random.rand(*shape)
    return (-1.0 / beta) * np.log(1.0 - r * (1.0 - np.exp(-beta * vol)))


def u_sample(shape, beta, vol=1.0, r=None):
    """
    Samples potential energy values from a system with the first axis in shape being the number of samples and the second being the number of particles.
    Particle positions are randomly sampled with sampleX at the inverse temperature B to generate the configuration of the potential energy.
    """
    return x_sample(shape=shape, beta=beta, vol=vol, r=r).sum(axis=-1)


@lru_cache(maxsize=100)
def dbeta_xave(k):
    """
    Analytical derivative of order k w.r.t. beta for the average of x
    Returns sympy function with expression for derivative
    """
    deriv = sp.diff(xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_minuslog(k):
    """
    Analytical derivative of order k w.r.t. beta for -ln(<x>)
    Returns sympy function with expression for derivative
    """
    deriv = sp.diff(-sp.log(xave_sym), beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_depend(k):
    """
    Analytical derivative of order k w.r.t. beta for the average of beta*x
    Returns sympy function with expression for derivative

    Note that this is also the average dimensionless potential energy for a single particle
    And since particles are independent, can just multiply by N for a system of N particles
    """
    deriv = sp.diff(beta_sym * xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_depend_minuslog(k):
    """
    Analytical derivative of order k w.r.t. beta for -ln(<beta*x>)
    Returns sympy function with expression for derivative
    """
    deriv = sp.diff(-sp.log(beta_sym * xave_sym), beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dvol_xave(k):
    """
    Analytical derivative of order k w.r.t. L for average x
    Returns sympy function with expression for derivative
    """
    deriv = sp.diff(xave_sym, vol_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


def x_beta_extrap(order, beta0, beta, vol=1.0):
    """
    Analytical extrapolation and coefficients from beta0 to beta (at L=vol) using derivatives up to order
    Returns extrapolation as first output and unnormalized coefficients as second
    """
    dbeta = beta - beta0

    out = []
    tot = 0.0
    for k in range(order + 1):
        val = dbeta_xave(k)(beta0, vol)
        out.append(val)
        tot += val / np.math.factorial(k) * (dbeta**k)
    return tot, np.array(out)


def x_beta_extrap_minuslog(order, beta0, beta, vol=1.0):
    """
    Same as x_beta_extrap but with -ln<x>
    """
    dbeta = beta - beta0

    out = np.zeros(order + 1)
    tot = 0.0
    for o in range(order + 1):
        out[o] = dbeta_xave_minuslog(o)(beta0, vol)
        tot += out[o] * ((dbeta) ** o) / np.math.factorial(o)

    return tot, out


def x_beta_extrap_depend(order, beta0, beta, vol=1.0):
    """
    Same as x_beta_extrap but for <beta*x>
    """
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
    """
    Same as x_beta_extrap but with -ln<beta*x>
    """
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


def x_vol_extrap(order, vol0, vol, beta=1.0):
    """
    Analytical extrapolation coefficients from vol0 to vol (at beta) using derivatives up to order
    Returns extrapolation as first output and unnormalized coefficients as second
    """
    dvol = vol - vol0

    out = []
    tot = 0.0

    for k in range(order + 1):
        val = dvol_xave(k)(beta, vol0)
        out.append(val)
        tot += val / np.math.factorial(k) * (dvol**k)
    return tot, np.array(out)


def generate_data(shape, beta, vol=1.0, r=None):
    """
    Generates data points in specified shape, where the first index is the number of samples and the second is the number of independent IG particles
    Sample will be at beta with L=vol
    Returns tuple of the particle positions in each configuration and the potential energy of each sampled configuration

    r may be specified as an array of random numbers instead of shape

    Parameters
    ----------
    shape : tuple
        (nconfig, npart)
    beta : inverse temperature


    """
    positions = x_sample(shape, beta, vol, r=r)
    x = positions.mean(axis=-1)
    u = positions.sum(axis=-1)
    return (x, u)

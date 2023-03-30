r"""
Ideal gas reference (:mod:`~thermoextrap.idealgas`)
===================================================

Analytic ideal gas in 1D in an external field.
The position, :math:`x`, may vary from :math:`0 \leq x \leq L`, with the field acting linearly on :math:`x`, :math:`U(x) = a x`, where for simplicity we let :math:`a=1`.
As a result, the potential energy of a system of :math:`N` particles with positions :math:`x_1, x_2, ... x_N` is the sum of the positions, :math:`U = \sum_{i=1}^N x_i`.
This is a useful test system with analytical solutions coded alongside the ability to randomly generate data.
"""


from functools import lru_cache

import numpy as np
import sympy as sp

from ._docstrings import DocFiller

__all__ = [
    "x_ave",
    "x_var",
    "x_prob",
    "u_prob",
    "x_cdf",
    "x_sample",
    "u_sample",
    "dbeta_xave",
    "dbeta_xave_minuslog",
    "dbeta_xave_depend",
    "dbeta_xave_depend_minuslog",
    "dvol_xave",
    "x_beta_extrap",
    "x_beta_extrap_minuslog",
    "x_beta_extrap_depend",
    "x_beta_extrap_depend_minuslog",
    "x_vol_extrap",
    "generate_data",
]


_shared_docs = """
Parameters
----------
beta : float or ndarray
    Inverse temperature.
vol : float or ndarray, default=1.0
    System volume.
x : float or ndarray
    Position.
u : float or ndarray
    Energy.
npart : int
    Number of particles.
r : float or ndarray, optional
    Random number(s). If passed, use these random numbers to build distribution.
    Useful for reproducibility.
k : int
    Derivative order.
order : int
    Expansion order.
beta0 : float
    Reference inverse temperature.
vol0 : float
    Reference volume.
shape : int or tuple of int
    Shape of output.  Ignored if ``r`` is not ``None``.
"""


docfiller_shared = DocFiller.from_docstring(_shared_docs, combine_keys="parameters")()


# global variables
beta_sym, vol_sym = sp.symbols("beta_sym vol_sym")
xave_sym = (1 / beta_sym) - vol_sym / (sp.exp(beta_sym * vol_sym) - 1)


@docfiller_shared
def x_ave(beta, vol=1.0):
    """
    Average position x at the inverse temperature beta.

    Parameters
    ----------
    {beta}
    {vol}

    """
    return 1.0 / beta - vol / (np.exp(beta * vol) - 1.0)


@docfiller_shared
def x_var(beta, vol=1.0):
    """
    Variance in position, x at the inverse temperature beta.

    Parameters
    ----------
    {beta}
    {vol}
    """
    return 1.0 / beta**2 - (
        vol**2 * np.exp(beta * vol) / ((np.exp(beta * vol) - 1) ** 2)
    )


@docfiller_shared
def x_prob(x, beta, vol=1.0):
    """
    Canonical probability of position x for single article at inverse temperature beta.

    Parameters
    ----------
    {x}
    {beta}
    {vol}
    """
    return (beta * np.exp(-beta * x)) / (1.0 - np.exp(-beta * vol))


# from scipy.stats import norm
@docfiller_shared
def u_prob(u, npart, beta, vol=1.0):
    """
    In the large-N limit, the probability of the potential energy is Normal, so provides that.


    Parameters
    ----------
    {u}
    {npart}
    {beta}
    {vol}
    """
    u_ave = npart * x_ave(beta, vol)
    u_std = np.sqrt(npart * x_var(beta, vol))
    return np.exp(-0.5 * ((u - u_ave) / u_std) ** 2) / (u_std * np.sqrt(2 * np.pi))


#    return norm.pdf(u, u_ave, u_std)


@docfiller_shared
def x_cdf(x, beta, vol=1.0):
    """
    Cumulative probability density for position x for single particle.

    Parameters
    ----------
    {x}
    {beta}
    {vol}
    """
    return (1.0 - np.exp(-beta * x)) / (1.0 - np.exp(-beta * vol))


@docfiller_shared
def x_sample(shape, beta, vol=1.0, r=None):
    """
    Sample positions from distribution at `beta` and `vol`.

    Does sampling based on inversion of cumulative distribution function.

    Parameters
    ----------
    {shape}
    {beta}
    {vol}
    {r}

    Returns
    -------
    output : ndarray
        Random sample from distribution.

    Notes
    -----
    If pass ``r``, then use these to build ``output``.  Otherwise, build random array of shape ``shape``.
    """
    if r is None:
        if isinstance(shape, int):
            shape = (shape,)
        r = np.random.rand(*shape)
    return (-1.0 / beta) * np.log(1.0 - r * (1.0 - np.exp(-beta * vol)))


@docfiller_shared
def u_sample(shape, beta, vol=1.0, r=None):
    """
    Samples potential energy values from a system.

    Note that ``shape = (nsamp, npart)``
    Particle positions are randomly sampled with :func:`x_sample` at `beta` to generate the configuration of the potential energy.


    Parameters
    ----------
    {shape}
    {beta}
    {vol}
    {r}
    """
    return x_sample(shape=shape, beta=beta, vol=vol, r=r).sum(axis=-1)


@lru_cache(maxsize=100)
def dbeta_xave(k):
    """
    Analytical derivative of order k w.r.t. beta for the average of x.

    Returns sympy function with expression for derivative.
    """
    deriv = sp.diff(xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_minuslog(k):
    """
    Analytical derivative of order k w.r.t. beta for -ln(<x>)

    Returns sympy function with expression for derivative.
    """
    deriv = sp.diff(-sp.log(xave_sym), beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_depend(k):
    """
    Analytical derivative of order k w.r.t. beta for the average of beta*x

    Returns sympy function with expression for derivative.

    Note that this is also the average dimensionless potential energy for a single particle
    And since particles are independent, can just multiply by N for a system of N particles
    """
    deriv = sp.diff(beta_sym * xave_sym, beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dbeta_xave_depend_minuslog(k):
    """
    Analytical derivative of order k w.r.t. beta for -ln(<beta*x>)

    Returns sympy function with expression for derivative.
    """
    deriv = sp.diff(-sp.log(beta_sym * xave_sym), beta_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@lru_cache(maxsize=100)
def dvol_xave(k):
    """
    Analytical derivative of order k w.r.t. L for average x

    Returns sympy function with expression for derivative.
    """
    deriv = sp.diff(xave_sym, vol_sym, k)
    return sp.lambdify([beta_sym, vol_sym], deriv, "numpy")


@docfiller_shared
def x_beta_extrap(order, beta0, beta, vol=1.0):
    """
    Analytical extrapolation and coefficients from beta0 to beta (at L=vol) using derivatives up to order

    Returns extrapolation as first output and unnormalized coefficients as second.

    Parameters
    ----------
    {order}
    {beta0}
    {beta}
    {vol}
    """
    dbeta = beta - beta0

    out = []
    tot = 0.0
    for k in range(order + 1):
        val = dbeta_xave(k)(beta0, vol)
        out.append(val)
        tot += val / np.math.factorial(k) * (dbeta**k)
    return tot, np.array(out)


@docfiller_shared
def x_beta_extrap_minuslog(order, beta0, beta, vol=1.0):
    """
    Same as x_beta_extrap but with -ln<x>.

    Parameters
    ----------
    {order}
    {beta0}
    {beta}
    {vol}
    """
    dbeta = beta - beta0

    out = np.zeros(order + 1)
    tot = 0.0
    for o in range(order + 1):
        out[o] = dbeta_xave_minuslog(o)(beta0, vol)
        tot += out[o] * ((dbeta) ** o) / np.math.factorial(o)

    return tot, out


@docfiller_shared
def x_beta_extrap_depend(order, beta0, beta, vol=1.0):
    """
    Same as x_beta_extrap but for <beta*x>.

    Parameters
    ----------
    {order}
    {beta0}
    {beta}
    {vol}
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


@docfiller_shared
def x_beta_extrap_depend_minuslog(order, beta0, beta, vol=1.0):
    """
    Same as x_beta_extrap but with -ln<beta*x>.

    Parameters
    ----------
    {order}
    {beta0}
    {beta}
    {vol}
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


@docfiller_shared
def x_vol_extrap(order, vol0, vol, beta=1.0):
    """
    Analytical extrapolation coefficients from vol0 to vol (at beta) using derivatives up to order

    Returns extrapolation as first output and unnormalized coefficients as second.

    Parameters
    ----------
    {order}
    {vol0}
    {vol}
    {beta}
    """
    dvol = vol - vol0

    out = []
    tot = 0.0

    for k in range(order + 1):
        val = dvol_xave(k)(beta, vol0)
        out.append(val)
        tot += val / np.math.factorial(k) * (dvol**k)
    return tot, np.array(out)


@docfiller_shared
def generate_data(shape, beta, vol=1.0, r=None):
    """
    Generates data points in specified shape, where the first index is the number of samples and the second is the number of independent IG particles
    Sample will be at beta with L=vol
    Returns tuple of the particle positions in each configuration and the potential energy of each sampled configuration.

    r may be specified as an array of random numbers instead of shape

    Parameters
    ----------
    {shape}
    {beta}
    {vol}
    {r}
    """
    positions = x_sample(shape, beta, vol, r=r)
    x = positions.mean(axis=-1)
    u = positions.sum(axis=-1)
    return (x, u)

"""Holds analytic ideal gas model.
Useful for testing and examining fundamental behavior.
"""

import numpy as np
import sympy as sym
from scipy.stats import norm


class IGmodel:
    """Defines a 1D ideal gas in an external field. The position, x, may vary from 0 to L,
    with the field acting linearly on x, U(x) = a*x, where for simplicity we let a=1.
    This is a useful class to use for testing.
    """

    # Define some symbols and functions used across the class
    # All such classes will have identical symbols and functions, which is desirable here
    # Because volume is of secondary interest, set default parameter for this so that
    # it does not need to be specified (keeps older code compatible, too)
    b, l = sym.symbols("b l")
    avgXsym = (1 / b) - l / (sym.exp(b * l) - 1)
    avgXlambdify = sym.lambdify([b, l], avgXsym, "numpy")

    @classmethod
    def avgX(cls, B, L=1.0):
        """Average position x at the inverse temperature B"""
        return cls.avgXlambdify(B, L)

    @classmethod
    def varX(cls, B, L=1.0):
        """Variance in position, x at the inverse temperature B"""
        term1 = 1.0 / (B**2)
        term2 = (L**2) * np.exp(B * L) / ((np.exp(B * L) - 1) ** 2)
        return term1 - term2

    @classmethod
    def PofX(
        cls, x, B, L=1.0
    ):  # This will also give P(U) exactly for single particle if a = 1
        """Canonical probability of position x for single particle at inverse temperature B"""
        numer = B * np.exp(-B * x)
        denom = 1.0 - np.exp(-B * L)
        return numer / denom

    @classmethod
    def cdfX(cls, x, B, L=1.0):  # Cumulative distribution function for X
        """Cumulative probability density for position x for single particle at inverse temperature B"""
        numer = 1.0 - np.exp(-B * x)
        denom = 1.0 - np.exp(-B * L)
        return numer / denom

    def __init__(self, nParticles=1000):
        self.nP = nParticles  # Number of particles

    def sampleX(self, B, s, L=1.0):
        """Samples s samples of x from the probability density at inverse temperature B
        Does sampling based on inversion of cumulative distribution function
        """
        randvec = np.random.rand(s)
        randx = -(1.0 / B) * np.log(1.0 - randvec * (1.0 - np.exp(-B * L)))
        return randx

    def sampleU(self, B, s=1000, L=1.0):
        """Samples s (=1000 by default) potential energy values from a system self.nP particles.
        Particle positions are randomly sampled with sampleX at the inverse temperature B.
        """
        # Really just resampling the sum of x values many times to get distribution of U for large N
        randu = np.zeros(s)
        for i in range(s):
            randu[i] = np.sum(self.sampleX(B, self.nP, L=L))
        return randu

    def PofU(self, U, B, L=1.0):
        """In the large-N limit, the probability of the potential energy is Normal, so provides that"""
        # Provides P(U) in the limit of a large number of particles (becomes Gaussian)
        avgU = self.nP * self.avgX(B, L=L)
        stdU = np.sqrt(self.nP * self.varX(B, L=L))
        return norm.pdf(U, avgU, stdU)

    def pertAnalytic(self, B, B0, L=1.0):
        """Analytical perturbation of the system from B0 to B.
        Nice check to see if get same thing as avgX
        """

        # Really just the same as average of x, but it's a nice check all the math
        def pertNumer(B, B0):
            prefac = B0 / (1.0 - np.exp(-B0 * L))
            term1 = (1.0 - np.exp(-B * L)) / (B**2)
            term2 = L * np.exp(-B * L) / B
            return prefac * (term1 - term2)

        def pertDenom(B, B0):
            prefac = B0 / B
            numer = 1.0 - np.exp(-B * L)
            denom = 1.0 - np.exp(-B0 * L)
            return prefac * numer / denom

        return pertNumer(B, B0) / pertDenom(B, B0)

    def extrapAnalytic(self, B, B0, order, L=1.0):
        """Analytical extrapolation from B0 to B at specified order.
        Same as if used infinite number of symbols, so only includes truncation error.
        """
        dBeta = B - B0
        outvec = np.zeros(order + 1)
        outval = 0.0
        for k in range(order + 1):
            thisdiff = sym.diff(self.avgXsym, self.b, k)
            outvec[k] = thisdiff.subs({self.b: B0, self.l: L})
            outval += outvec[k] * (dBeta**k) / np.math.factorial(k)
        return (outval, outvec)

    def extrapAnalyticVolume(self, L, L0, order, B=1.0):
        """Analytical extrapolation from a reference system L0 to new L at specified order.
        Must also specify inverse temperature B if want it to be other than 1.0
        """
        dL = L - L0
        outvec = np.zeros(order + 1)
        outval = 0.0
        for k in range(order + 1):
            thisdiff = sym.diff(self.avgXsym, self.l, k)
            outvec[k] = thisdiff.subs({self.b: B, self.l: L0})
            outval += outvec[k] * (dL**k) / np.math.factorial(k)
        return (outval, outvec)

    # Want to be able to create sample data set we can work with at a reference beta
    def genData(self, B, nConfigs=100000, L=1.0):
        """Generates nConfigs data points of the model at inverse temperature beta.
        Returns are the average x values and the potential energy values of each data point.
        """
        allX = self.sampleX(B, nConfigs * self.nP, L=L)
        allConfigs = np.reshape(allX, (nConfigs, self.nP))
        obsX = np.average(allConfigs, axis=1)
        obsU = np.sum(allConfigs, axis=1)
        return (obsX, obsU)

# Should house old scripts for extrapolation/interpolation that do not use classes
# Really just for legacy and potentially testing

"""Legacy (deprecated) code kept only for testing and reference purposes.
SHOULD NOT BE USED.
"""

import numpy as np
from scipy.special import factorial

try:
    from pymbar import mbar
except ImportError:
    print(
        "Could not find pymbar - will not import and functions involving this will not work."
    )

from .utilities import buildAvgFuncs, symDerivAvgX


def extrapWithSamples(B, B0, x, U, order):
    """Uses symbolic logic to perform extrapolation to arbitrarily high order.
    Makes use of the buildAvgDict and symDerivAvgX functions defined above.
    B is the inverse temperature to extrapolate to (can be an array of values),
    B0 is the reference, x is the reference observable values, and U is the
    reference potential energy values. Order is the highest order expansion
    coefficient (derivative) to compute. The function returns both the extrapolated
    value and the derivatives. Vector-valued observables are allowed for x, but
    the first dimension should run over independent observations and match the size
    of U.
    """
    if x.shape[0] != U.shape[0]:
        print(
            "First observable dimension (%i) and size of potential energy"
            " array (%i) don't match!" % (x.shape[0], U.shape[0])
        )
        raise ValueError("x and U must have same shape in first dimension")

    # Next need to make sure x has at least two dimensions
    if len(x.shape) == 1:
        x = np.reshape(
            x, (x.shape[0], 1)
        )  # Rows are independent observations, columns x vectors

    # And make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
        B = [B]
    B = np.array(B)
    dBeta = B - B0

    outvec = np.zeros((order + 1, x.shape[1]))  # kth order derivative on kth row
    outval = np.zeros(
        (B.shape[0], x.shape[1])
    )  # each row is extrapolation to different beta

    # Get functions defining averages of observable, potential energy, etc.
    avgUfunc, avgXUfunc = buildAvgFuncs(x, U, order)

    # Loop over orders
    for o in range(order + 1):
        # Get derivative function at this order
        oDeriv = symDerivAvgX(o)
        # Evaluate it using the appropriate mappings to averages
        # MUST provide average U then XU because of how symDerivAvgX returns
        outvec[o] = oDeriv(avgUfunc, avgXUfunc)
        # Perform extrapolation using same deriatives and averages, just have many dBeta
        # Taking the tensor product of two (really should be) 1D arrays to get the right shape
        outval += np.tensordot((dBeta**o), outvec[o], axes=0) / np.math.factorial(o)

    return (outval, outvec)


def extrapWeighted(B, refB1, refB2, x1, x2, u1, u2, order1, order2, m=20):
    """Performs extrapolation from two points to an interior point and weights with a
    Minkowski-like function proposed by Mahynski, Errington, and Shen (2017).
    """

    def weightsMinkowski(d1, d2, m=20):
        w1 = 1.0 - (d1**m) / ((d1**m) + (d2**m))
        w2 = 1.0 - (d2**m) / ((d1**m) + (d2**m))
        return [w1, w2]

    ext1, derivs1 = extrapWithSamples(B, refB1, x1, u1, order1)
    ext2, derivs2 = extrapWithSamples(B, refB2, x2, u2, order2)

    # Make sure B is an array to handle case if it is
    # Also ensures w1 and w2 can be multiplied by the extrapolations correctly
    if isinstance(B, (int, float)):
        B = [B]
    B = np.array(B)

    w1, w2 = weightsMinkowski(abs(refB1 - B), abs(refB2 - B), m=m)

    # Transpose to get right multiplication (each row of exti is different beta)
    w1T = np.array([w1]).T
    w2T = np.array([w2]).T
    outval = (ext1 * w1T + ext2 * w2T) / (w1T + w2T)

    return (outval, derivs1, derivs2)


def interpPolyMultiPoint(B, refB, x, U, order):
    """refB is an array of beta values of at least length 2.
    x and U should be arrays for data and potential energy at each beta value
    (so their first dimension should be the same as refB).
    B are the beta values at which to interpolate using a polynomial.
    order is the maximum order derivative used at each point where data is provided.
    Returns polynomial values at specified betas and polynomial coefficients.
    """
    refB = np.array(refB)

    if x.shape[0] != U.shape[0]:
        print(
            "First observable dimension (%i) and size of potential energy"
            " array (%i) don't match!" % (x.shape[0], U.shape[0])
        )
        raise ValueError("x and U must have same shape in first dimension")

    if (x.shape[0] != refB.shape[0]) or (U.shape[0] != refB.shape[0]):
        print("First dimension of data must match number of provided beta values.")
        raise ValueError(
            "For interpolation, first dimension of xData, uData, and refB must match."
        )

    # Want to be able to handle vector-value observables
    # So make sure x has 3 dimensions, even if technically observable is scalar
    if len(x.shape) == 2:
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # While we're at it, also make B into an array if it isn't, just for convenience
    if isinstance(B, (int, float)):
        B = [B]
    B = np.array(B)

    # Define the order of the polynomial wer're going to compute
    pOrder = (
        refB.shape[0] * (order + 1) - 1
    )  # Also the number of coefficients we solve for minus 1

    # Need to put together systems of equations to solve
    # Will have to solve one system for each component of a vector-valued observable
    # Fortunately, matrix to invert will be same for each value of beta regardless of observable
    # Just the values we want the polynomial to match with (derivVals) will be different
    derivVals = np.zeros((pOrder + 1, x.shape[2]))
    mat = np.zeros((pOrder + 1, pOrder + 1))

    # Loop to get all values and derivatives at each point up to desired order
    for i, beta in enumerate(refB):
        # Just need derivatives, which is essentially same cost as computing extrapolation
        # But don't care about what point we extrapolate to or the value we get
        thisext, thisderivs = extrapWithSamples(
            np.average(refB), beta, x[i], U[i], order
        )

        # Loop over observable elements, with unique derivatives for each
        for j in range(x.shape[2]):
            derivVals[(order + 1) * i : (order + 1) * (i + 1), j] = thisderivs[:, j]

        # Loop over orders, filling out matrix for solving systems of equations
        for j in range(order + 1):
            # Suppress warnings about divide by zero since we actually want this to return infinity
            with np.errstate(divide="ignore"):
                mat[((order + 1) * i) + j, :] = (
                    ((np.ones(pOrder + 1) * beta) ** (np.arange(pOrder + 1) - j))
                    * factorial(np.arange(pOrder + 1))
                    / factorial(np.arange(pOrder + 1) - j)
                )

    # The above formula works everywhere except where the matrix should have zeros
    # Instead of zeros, it inserts infinity, so fix this
    # (apparently scipy's factorial returns 0 for negatives)
    mat[np.isinf(mat)] = 0.0

    # And solve a system of equations for the polynomial coefficients of each observable element
    matInv = np.linalg.inv(mat)
    coeffs = np.zeros((pOrder + 1, x.shape[2]))
    for j in range(x.shape[2]):
        coeffs[:, j] = np.dot(matInv, derivVals[:, j])

    # Calculate the polynomial interpolation values at each desired beta
    outvals = np.zeros((len(B), x.shape[2]))  # Each row is a different beta value
    for i, beta in enumerate(B):
        betaPower = beta ** (np.arange(pOrder + 1))
        betaPower = np.array([betaPower]).T
        outvals[i] = np.sum(coeffs * betaPower, axis=0)

    return (outvals, coeffs)


def perturbWithSamples(B, refB, x, U, useMBAR=False):
    """Computes observable x (can be a vector) at a set of perturbed temperatures
    of B (array) from the original refB using potential energies at each config
    and standard reweighting. Uses MBAR code instead of mine if desired.
    """
    if x.shape[0] != U.shape[0]:
        print(
            "First observable dimension (%i) and size of potential energy"
            " array (%i) don't match!" % (x.shape[0], U.shape[0])
        )
        raise ValueError("x and U must have same shape in first dimension")

    # Check shape of observables and add dimension if needed
    # Note that for observables with more than 1 dimension, things won't work
    if len(x.shape) == 1:
        x = np.array([x]).T

    # While we're at it, also make B into an array if it isn't, just for convenience
    if isinstance(B, (int, float)):
        B = [B]
    B = np.array(B)

    if useMBAR:
        mbarObj = mbar.MBAR(np.array([refB * U]), [U.shape[0]])
        outval = np.zeros((len(B), x.shape[1]))
        for i in range(len(B)):
            outval[i, :] = mbarObj.computeMultipleExpectations(x.T, B[i] * U)[0]

    else:
        # Compute what goes in the exponent and subtract out the maximum
        # Don't need to bother storing for later because compute ratio
        dBeta = B - refB
        dBetaU = (-1.0) * np.tensordot(dBeta, U, axes=0)
        dBetaUdiff = dBetaU - np.array([np.max(dBetaU, axis=1)]).T
        expVals = np.exp(dBetaUdiff)

        # And perform averaging
        numer = np.dot(expVals, x) / float(x.shape[0])
        denom = np.average(expVals, axis=1)
        outval = numer / np.array([denom]).T

    return outval

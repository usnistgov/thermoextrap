"""Implements all utilities needed for extrapolation/interpolation.
Mainly this is the code to automatically compute derivatives with sympy.
Also includes other useful functions.
"""


import numpy as np
import sympy as sym
from scipy.special import binom


def buildAvgFuncs(xvals, uvals, order):
    """Defines sympy functions mapping specific sympy function representations to values.
    We will let u(i) be the average of u**i and xu(i) be the average of x*(u**i). In
    other words, providing an integer to the function u or xu will produce the desired
    average quantity. Once the symbolic derivative is defined as a lambdified function
    of two sympy functions, can just input the custom sympy functions defined here to
    substitute in all the right values. Note that if the observable is vector-valued
    the functions will return vectors for averages.
    """
    # To allow for vector-valued observables, must make sure uvals can be transposed
    uvalsT = np.array([uvals]).T

    # First define dictionaries we will use to define the functions
    # Keys will be integers representing different orders
    dictu = {}
    dictxu = {}

    for o in range(order + 1):
        dictu[o] = np.average(uvals**o)
        dictxu[o] = np.average(xvals * (uvalsT**o), axis=0)

    def ufunc(order):
        return dictu[order]

    def xufunc(order):
        return dictxu[order]

    return ufunc, xufunc

    # class ufunc(sym.Function):
    #     """Modified sympy Function class to output powers of potential energies."""

    #     avgdict = copy.deepcopy(dictu)

    #     @classmethod
    #     def eval(cls, x):
    #         return cls.avgdict[x]

    # class xufunc(sym.Function):
    #     """Modified sympy Function class to output powers of potential energies."""
    #     avgdict = copy.deepcopy(dictxu)

    #     @classmethod
    #     def eval(cls, x):
    #         return cls.avgdict[x]

    # return (ufunc, xufunc)


def symDerivAvgX(order):
    """A function to compute the derivative to arbitrary order using symbolic logic.
    Returns a substituted string that can be substituted again to get an actual value.
    """
    # First define some consistent symbols
    b = sym.symbols("b")  # Beta or inverse temperature

    f = sym.Function("f")(
        b
    )  # Functions representing the numerator and denominator of an average
    z = sym.Function("z")(b)

    u = sym.Function("u")  # Functions that will represent various averages
    xu = sym.Function("xu")

    avgFunc = f / z
    thisderiv = avgFunc.diff(b, order)
    # Pick out what we want to substitute by object type
    tosub = thisderiv.atoms(sym.Function, sym.Derivative)

    # When we sub in, must do in order of highest to lowest derivatives, then functions
    # Otherwise substitution doesn't work because derivatives computed recursively by sympy
    for o in range(order + 1)[::-1]:
        subvals = {}

        if o == 0:
            for d in tosub:
                if isinstance(d, sym.Function):
                    if str(d) == "f(b)":
                        subvals[d] = xu(0) * z

        else:
            for d in tosub:
                if isinstance(d, sym.Derivative) and d.derivative_count == o:
                    if str(d.expr) == "f(b)":
                        subvals[d] = (
                            ((-1) ** d.derivative_count) * xu(d.derivative_count) * z
                        )
                    elif str(d.expr) == "z(b)":
                        subvals[d] = (
                            ((-1) ** d.derivative_count) * u(d.derivative_count) * z
                        )

        # Substitute derivatives for functions u and xu at this order
        thisderiv = thisderiv.subs(subvals)

        # To allow for vector-valued function inputs and to gain speed, lambdify
        thisderiv = sym.expand(sym.simplify(thisderiv))

    returnfunc = sym.lambdify((u, xu), thisderiv, "numpy")
    return returnfunc


def buildAvgFuncsDependent(xvals, uvals, order):
    """Same as buildAvgFuncs, but for an observable that explicitly depends on
     the variable we're extrapolating over. In this case, xvals should be 3D.
     The first dimension is time, just like uvals, the LAST dimension should
     be elements of the observable vector, and the second (or middle) dimension
     should match order and be DERIVATIVES of the observable vector elements to
     the order of the index of that dimension. So the first index of zero on
     this dimension is the zeroth-order derivative, which is just the observable
     itself. As an example, if you performed Widom insertions, have that the
     observable and its derivatives are:
            x = exp(-B*dU)
        dx/dB = -dU*exp(-B*dU)
    d^2x/dB^2 = (dU^2)*exp(-B*dU)
           etc.
    """
    # Make sure have provided derivatives in observable up to desired order
    if xvals.shape[1] < order + 1:
        print(
            "Maximum provided order of derivatives of observable (%i) "
            "is less than desired order (%i)." % (xvals.shape[1] - 1, order)
        )
        print("Setting order to match.")
        order = xvals.shape[1] - 1
    elif xvals.shape[1] >= order + 1:
        xvals = xvals[:, : order + 1, :]

    # To allow for vector-valued observables, must make sure uvals can be transposed
    uvalsT = np.array([uvals]).T

    # First define dictionaries we will use to define the functions
    # Keys will be integers representing different orders
    dictu = {}
    dictxu = {}

    for o in range(order + 1):
        dictu[o] = np.average(uvals**o)
        for j in range(order + 1):
            dictxu[(j, o)] = np.average(xvals[:, j, :] * (uvalsT**o), axis=0)

    # class ufunc(sym.Function):
    #     """Modified sympy Function class to output powers of potential energies."""

    #     avgdict = copy.deepcopy(dictu)

    #     @classmethod
    #     def eval(cls, x):
    #         return cls.avgdict[x]

    # class xufunc(sym.Function):
    #     """Modified sympy Function class to output product of observable and potential energies."""

    #     avgdict = copy.deepcopy(dictxu)

    #     @classmethod
    #     def eval(cls, x, y):
    #         return cls.avgdict[(x, y)]

    def ufunc(order):
        return dictu[order]

    def xufunc(x, y):
        return dictxu[x, y]

    return (ufunc, xufunc)


def symDerivAvgXdependent(order):
    """Same as symDerivAvgX except for one line when substituting for f(b) and its
    derivatives. Instead of substituting xu(i), it substitutes xu(i,j) so that
    derivatives are possible with the observable depending explicitly on the
    extrapolation variable. This is meant to be used with buildAvgFuncsDependent.
    """
    # First define some consistent symbols
    b = sym.symbols("b")  # Beta or inverse temperature
    k = sym.symbols("k")

    f = sym.Function("f")(
        b
    )  # Functions representing the numerator and denominator of an average
    z = sym.Function("z")(b)

    u = sym.Function("u")  # Functions that will represent various averages
    xu = sym.Function("xu")

    avgFunc = f / z
    thisderiv = avgFunc.diff(b, order)
    # Pick out what we want to substitute by object type
    tosub = thisderiv.atoms(sym.Function, sym.Derivative)

    # When we sub in, must do in order of highest to lowest derivatives, then functions
    # Otherwise substitution doesn't work because derivatives computed recursively by sympy
    for o in range(order + 1)[::-1]:
        subvals = {}

        if o == 0:
            for d in tosub:
                if isinstance(d, sym.Function):
                    if str(d) == "f(b)":
                        subvals[d] = xu(0, 0) * z

        else:
            for d in tosub:
                if isinstance(d, sym.Derivative) and d.derivative_count == o:
                    if str(d.expr) == "f(b)":
                        # Instead of substituting f(k)(b) = <x*(-u^k)> = xu(k), we want to do...
                        # (4th order as an example)
                        # f(4)(b) = xu(4,0) - 4*xu(3,1) + 6*xu(2,2) - 4*xu(1,3) + xu(0,4)
                        #        = <x(4)> - 4*<x(3)*u> + 6*<x(2)*u^2> - 4*<x(1)*u^3> + <x*u^4>
                        # In the above, f(4) or x(4) represents the 4th derivative of f or x with
                        # respect to the extrapolation variable b.
                        subvals[d] = (
                            sym.Sum(
                                ((-1) ** k)
                                * sym.binomial(d.derivative_count, k)
                                * xu(d.derivative_count - k, k),
                                (k, 0, d.derivative_count),
                            ).doit()
                            * z
                        )
                    elif str(d.expr) == "z(b)":
                        subvals[d] = (
                            ((-1) ** d.derivative_count) * u(d.derivative_count) * z
                        )

        # Substitute derivatives for functions u and xu at this order
        thisderiv = thisderiv.subs(subvals)

        # To allow for vector-valued function inputs and to gain speed, lambdify
        thisderiv = sym.expand(sym.simplify(thisderiv))

    returnfunc = sym.lambdify((u, xu), thisderiv, "numpy")
    return returnfunc


def extrapToPoly(B0, derivs):
    """Converts an extrapolation around a reference point to a polynomial over all real
    numbers by collecting terms. Input is the reference state point and the derivatives
    at that state point (starting with the zeroth derivative, which is just the
    observable value). Only works for SINGLE observable element if observable is a
    vector (so derivs must be a 1D array).
    """
    coeffs = np.zeros(len(derivs))
    for k, d in enumerate(derivs):
        for l in range(k + 1):
            coeffs[l] += ((-B0) ** (k - l)) * d * binom(k, l) / np.math.factorial(k)
    return coeffs


def bootstrapPolyCoeffs(extModel, n=100, order=3):
    """Determines uncertainty in polynomial coefficients determined from derivatives
    via extrapToPoly function. This will only reliably work if provided an
    ExtrapModel object for which extModel.train returns the derivatives and
    extModel.refB returns the reference point and the data can be resampled
    from extModel.resampleData. Might make more sense to include this in the
    class definition, but don't want to be inherited by other classes.
    """
    bShape = (n,) + extModel.params[: order + 1, :].shape
    bootStraps = np.zeros(bShape)
    for i in range(n):
        thisx, thisU = extModel.resampleData()
        thisParams = extModel.train(extModel.refB, thisx, thisU, saveParams=False)
        thisCoeffs = np.zeros(thisParams[: order + 1, :].shape)
        for j in range(thisParams.shape[1]):
            thisCoeffs[:, j] = extrapToPoly(extModel.refB, thisParams[: order + 1, j])
        bootStraps[i] = thisCoeffs
    bootStd = np.std(bootStraps, ddof=1, axis=0)
    return bootStd

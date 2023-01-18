"""Interpolation classes
"""

import numpy as np
from scipy.special import factorial

from .extrap import ExtrapModel


class ExtrapWeightedModel(ExtrapModel):
    """Model for extrapolation using two data sets at two different state points and
    weighting extrapolations from each with a Minkowski-like distance.
    """

    # Only need to redefine the train, predict, and resampleData functions
    def train(self, refB, xData, uData, saveParams=True):
        """This is the function used to set the parameters of the model. For extrapolation
        these are just the derivatives at the reference state. You can change this function
        for more complex parameters like the polynomial coefficients for interpolation.
        If saveParams is False, it will simply return the parameters without setting
        their values in self. This is useful for bootstrapping.
        """
        # Next need to make sure x has at least three dimensions for extrapolation
        # First should be 2, next should be number of samples in each dataset,
        # and the last should be the length of the observable vector.
        # Note that currently ragged data is not allowed, but I don't check for this!
        # (data sets at each state point must have the same number of samples)
        if (xData.shape[0] != 2) or (uData.shape[0] != 2):
            print(
                "Must provide observable and potential energy data from 2 state points!"
            )
            print(
                "First dimensions of provided data are not 2"
                " but %i and %i" % (xData.shape[0], uData.shape[0])
            )
            raise ValueError("xData and uData must have first dimension of 2")

        if isinstance(refB, (int, float)):
            print(
                "Must provide 2 reference beta values as a list or array,"
                "but got only a number."
            )
            raise TypeError(
                "refB must be a list or array of length 2, not float or int"
            )

        refB = np.array(refB)
        if refB.shape[0] != 2:
            print("Need exactly 2 reference beta values, but got %i" % refB.shape[0])
            raise ValueError("refB must be a list or array of exactly length 2")

        if len(xData.shape) == 2:
            xData = np.reshape(xData, (xData.shape[0], xData.shape[1], 1))
            # Rows are independent observations, columns elements of observable x

        params1 = self.calcDerivVals(refB[0], xData[0], uData[0])
        params2 = self.calcDerivVals(refB[1], xData[1], uData[1])
        params = np.array([params1, params2])

        if saveParams:
            self.refB = refB
            self.x = xData
            self.U = uData
            self.params = params

        return params

    # A function to calculate model prediction at other state points
    def predict(self, B, order=None, params=None, refB=None):
        """Like trainParams, this is a function that can be modified to create new
        extrapolation procedures. Just uses information collected from training
        to make predictions at specific state points, B, up to specified order.
        Can also specify parameters to use. If params is None, will just use
        the parameters found during training. If refB is None, will just use
        self.refB.
        """

        def weightsMinkowski(d1, d2, m=20):
            w1 = 1.0 - (d1**m) / ((d1**m) + (d2**m))
            w2 = 1.0 - (d2**m) / ((d1**m) + (d2**m))
            return [w1, w2]

        # Use parameters for estimate - for extrapolation, the parameters are the derivatives
        if params is None:
            if self.params is None:
                raise TypeError(
                    "self.params is None - need to train model before predicting"
                )
            params = self.params

        if refB is None:
            if self.refB is None:
                raise TypeError("self.refB is None - need to specify reference beta")
            refB = self.refB

        if order is None:
            order = self.maxOrder

        # Make sure B is an array, even if just has one element
        if isinstance(B, (int, float)):
            B = [B]
        B = np.array(B)
        dBeta = np.zeros((2, B.shape[0]))
        dBeta[0] = B - refB[0]
        dBeta[1] = B - refB[1]

        # Perform extrapolation from both reference points
        predictVals = np.zeros((2, B.shape[0], self.x.shape[-1]))
        for o in range(order + 1):
            predictVals[0] += np.tensordot(
                (dBeta[0] ** o), params[0, o], axes=0
            ) / np.math.factorial(o)
            predictVals[1] += np.tensordot(
                (dBeta[1] ** o), params[1, o], axes=0
            ) / np.math.factorial(o)

        w1, w2 = weightsMinkowski(abs(dBeta[0]), abs(dBeta[1]))

        # Transpose to get right multiplication (each row of exti is different beta)
        w1T = np.array([w1]).T
        w2T = np.array([w2]).T
        outVals = (predictVals[0] * w1T + predictVals[1] * w2T) / (w1T + w2T)

        return outVals

    def resampleData(self):
        """Function to resample the data, mainly for use in providing bootstrapped estimates.
        Should be adjusted to match the data structure.
        """
        if self.x is None:
            raise TypeError(
                "self.x is None - need to define data in model (i.e. train)"
            )

        sampX = np.zeros(self.x.shape)
        sampU = np.zeros(self.U.shape)

        for i in range(self.x.shape[0]):
            sampSize = self.x[i].shape[0]
            randInds = np.random.choice(sampSize, size=sampSize, replace=True)
            sampX[i] = self.x[i, randInds, :]
            sampU[i] = self.U[i, randInds]

        return (sampX, sampU)


class InterpModel(ExtrapModel):
    """Model for interpolation that inherits most functions and variables from extrapolation
    model. Just need to change train and predict functions!
    """

    # Only need to redefine the train, predict, and resampleData functions
    def train(self, refB, xData, uData, saveParams=True):
        """This is the function used to set the parameters of the model. For extrapolation
        these are just the derivatives at the reference state. You can change this function
        for more complex parameters like the polynomial coefficients for interpolation.
        If saveParams is False, it will simply return the parameters without setting
        their values in self. This is useful for bootstrapping.
        """
        refB = np.array(refB)

        if xData.shape[0] != uData.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy"
                " array (%i) don't match!" % (xData.shape[0], uData.shape[0])
            )
            raise ValueError("x and U must have same shape in first dimension")

        if (xData.shape[0] != refB.shape[0]) or (uData.shape[0] != refB.shape[0]):
            print("First dimension of data must match number of provided beta values.")
            raise ValueError(
                "For interpolation, first dimension of xData, uData, and refB must match."
            )

        # Want to be able to handle vector-value observables
        # So make sure x has 3 dimensions, even if technically observable is scalar
        # Note that currently ragged data is not allowed, but I don't check for this!
        # (data sets at each state point must have the same number of samples)
        if len(xData.shape) == 2:
            xData = np.reshape(xData, (xData.shape[0], xData.shape[1], 1))

        # Define the order of the polynomial we're going to compute
        order = self.maxOrder
        pOrder = (
            refB.shape[0] * (order + 1) - 1
        )  # Also the number of coefficients we solve for minus 1

        # Need to put together systems of equations to solve
        # Will have to solve one system for each component of a vector-valued observable
        # Fortunately, matrix to invert same for each value of beta regardless of observable
        # Just the values we want the polynomial to match with (derivVals) will be different
        derivVals = np.zeros((pOrder + 1, xData.shape[2]))
        mat = np.zeros((pOrder + 1, pOrder + 1))

        # Loop to get all values and derivatives at each point up to desired order
        for i, beta in enumerate(refB):
            thisderivs = self.calcDerivVals(beta, xData[i], uData[i])

            # Loop over observable elements, with unique derivatives for each
            for j in range(xData.shape[2]):
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

        # And solve system of equations for the polynomial coefficients of each observable element
        matInv = np.linalg.inv(mat)
        coeffs = np.zeros((pOrder + 1, xData.shape[2]))
        for j in range(xData.shape[2]):
            coeffs[:, j] = np.dot(matInv, derivVals[:, j])

        if saveParams:
            self.refB = refB
            self.x = xData
            self.U = uData
            self.params = coeffs

        return coeffs

    # A function to calculate model prediction at other state points
    def predict(self, B, order=None, params=None, refB=None):
        """Like trainParams, this is a function that can be modified to create new
        extrapolation procedures. Just uses information collected from training
        to make predictions at specific state points, B, up to self.maxOrder.
        Training requires selecting the order, so can't pick when predicting.
        So just use self.maxOrder as the highest order derivative information
        to use throughout the entire model.
        Can also specify parameters to use. If params is None, will just use
        the parameters found during training. refB will be ignored as it is
        not needed once the polynomial is known.
        """
        # Use parameters for estimate
        if params is None:
            if self.params is None:
                raise TypeError(
                    "self.params is None - need to train model before predicting"
                )
            params = self.params

        if order is None:
            order = self.maxOrder

        # Make sure B is an array, even if just has one element
        if isinstance(B, (int, float)):
            B = [B]
        B = np.array(B)

        # Infer polynomial order from parameters (which are polynomial coefficients)
        pOrder = len(params) - 1

        # Calculate the polynomial interpolation values at each desired beta
        outvals = np.zeros(
            (len(B), self.x.shape[-1])
        )  # Each row is a different beta value
        for i, beta in enumerate(B):
            betaPower = beta ** (np.arange(pOrder + 1))
            betaPower = np.array([betaPower]).T
            outvals[i] = np.sum(params * betaPower, axis=0)

        return outvals

    def resampleData(self):
        """Function to resample the data, mainly for use in providing bootstrapped estimates.
        Should be adjusted to match the data structure.
        """
        if self.x is None:
            raise TypeError(
                "self.x is None - need to define data in model (i.e. train)"
            )

        sampX = np.zeros(self.x.shape)
        sampU = np.zeros(self.U.shape)

        for i in range(self.x.shape[0]):
            sampSize = self.x[i].shape[0]
            randInds = np.random.choice(sampSize, size=sampSize, replace=True)
            sampX[i] = self.x[i, randInds, :]
            sampU[i] = self.U[i, randInds]

        return (sampX, sampU)


class VolumeExtrapWeightedModel(ExtrapWeightedModel):
    """Class to hold information about a VOLUME extrapolation. This can be trained by providing
    data at the reference state and can then be evaluated to obtain estimates at
    arbitrary other states. Note that refB is now the reference volume and self.U will
    actually represent the virial, not the potential energy.
    """

    # Can't go to higher order in practice, so don't return any symbolic derivatives
    # Instead, just use this to check and make sure not asking for order above 1
    def calcDerivFuncs(self):
        """Calculates symbolic derivative functions and returns lambdified functions."""
        if self.maxOrder > 1:
            print(
                "Volume extrapolation cannot go above 1st order without derivatives of forces."
            )
            print("Setting order to 1st order.")
            self.maxOrder = 1
        return None

    # And given data, calculate numerical values of derivatives up to maximum order
    # Will be very helpful when generalize to different extrapolation techniques
    # (and interpolation)
    def calcDerivVals(self, refV, x, W):
        """Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives. Only go to first order for volume extrapolation. And
        here W represents the virial instead of the potential energy.
        """

        if x.shape[0] != W.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy"
                " array (%i) don't match!" % (x.shape[0], W.shape[0])
            )
            raise ValueError("x and U must have same shape in first dimension")

        wT = np.array([W]).T
        avgX = np.average(x, axis=0)
        avgW = np.average(W)
        avgXW = np.average(x * wT, axis=0)
        derivVals = np.zeros((2, x.shape[1]))
        derivVals[0] = avgX
        derivVals[1] = (avgXW - avgX * avgW) / (
            3.0 * refV
        )  # Should check last term and add if needed

        return derivVals


class VolumeInterpModel(InterpModel):
    """Class to hold information about a VOLUME interpolation. This can be trained by providing
    data at the reference state and can then be evaluated to obtain estimates at
    arbitrary other states. Note that refB is now the reference volume and self.U will
    actually represent the virial, not the potential energy.
    """

    # Can't go to higher order in practice, so don't return any symbolic derivatives
    # Instead, just use this to check and make sure not asking for order above 1
    def calcDerivFuncs(self):
        """Calculates symbolic derivative functions and returns lambdified functions."""
        if self.maxOrder > 1:
            print(
                "Volume extrapolation cannot go above 1st order without derivatives of forces."
            )
            print("Setting order to 1st order.")
            self.maxOrder = 1
        return None

    # And given data, calculate numerical values of derivatives up to maximum order
    # Will be very helpful when generalize to different extrapolation techniques
    # (and interpolation)
    def calcDerivVals(self, refV, x, W):
        """Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives. Only go to first order for volume extrapolation. And
        here W represents the virial instead of the potential energy.
        """

        if x.shape[0] != W.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy"
                " array (%i) don't match!" % (x.shape[0], W.shape[0])
            )
            raise ValueError("x and U must have same shape in first dimension")

        wT = np.array([W]).T
        avgX = np.average(x, axis=0)
        avgW = np.average(W)
        avgXW = np.average(x * wT, axis=0)
        derivVals = np.zeros((2, x.shape[1]))
        derivVals[0] = avgX
        derivVals[1] = (avgXW - avgX * avgW) / (
            3.0 * refV
        )  # Should check last term and add if needed

        return derivVals

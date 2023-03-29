"""Main extrapolation classes
"""

import numpy as np

from .utilities import buildAvgFuncs, symDerivAvgX


class ExtrapModel:
    """Class to hold information about an extrapolation. This can be trained by providing
    data at the reference state and can then be evaluated to obtain estimates at
    arbitrary other states.
    """

    # Otherwise, it just needs to define some variables
    def __init__(self, maxOrder=6, refB=None, xData=None, uData=None):
        self.maxOrder = maxOrder  # Maximum order to calculate derivatives
        self.derivF = (
            self.calcDerivFuncs()
        )  # Perform sympy differentiation up to max order
        self.refB = refB
        self.x = xData
        self.U = uData
        self.params = None
        if (refB is not None) and (xData is not None) and (uData is not None):
            self.params = self.train(refB, xData, uData, saveParams=True)

    # Calculates symbolic derivatives up to maximum order given data
    # Returns list of functions that can be used to evaluate derivatives for specific data
    def calcDerivFuncs(self):
        """Calculates symbolic derivative functions and returns lambdified functions."""
        derivs = []
        for o in range(self.maxOrder + 1):
            derivs.append(symDerivAvgX(o))
        return derivs

    # And given data, calculate numerical values of derivatives up to maximum order
    # Will be very helpful when generalize to different extrapolation techniques
    # (and interpolation)
    def calcDerivVals(self, refB, x, U):
        """Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives.
        """

        if x.shape[0] != U.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy"
                " array (%i) don't match!" % (x.shape[0], U.shape[0])
            )
            raise ValueError("x and U must have same shape in first dimension")

        avgUfunc, avgXUfunc = buildAvgFuncs(x, U, self.maxOrder)
        derivVals = np.zeros((self.maxOrder + 1, x.shape[1]))
        for o in range(self.maxOrder + 1):
            derivVals[o] = self.derivF[o](avgUfunc, avgXUfunc)

        return derivVals

    # The below can be modified to change the extrapolation technique or to use interpolation
    def train(self, refB, xData, uData, saveParams=True):
        """This is the function used to set the parameters of the model. For extrapolation
        these are just the derivatives at the reference state. You can change this function
        for more complex parameters like the polynomial coefficients for interpolation.
        If saveParams is False, it will simply return the parameters without setting
        their values in self. This is useful for bootstrapping.
        """
        # Next need to make sure x has at least two dimensions for extrapolation
        if len(xData.shape) == 1:
            xData = np.reshape(xData, (xData.shape[0], 1))
            # Rows are independent observations, columns elements of observable x

        params = self.calcDerivVals(refB, xData, uData)

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
        the parameters found during training. If refB is None, will use self.refB.
        """
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
        dBeta = B - refB

        predictVals = np.zeros((B.shape[0], self.x.shape[-1]))
        for o in range(order + 1):
            predictVals += np.tensordot(
                (dBeta**o), params[o], axes=0
            ) / np.math.factorial(o)

        return predictVals

    def resampleData(self):
        """Function to resample the data, mainly for use in providing bootstrapped estimates.
        Should be adjusted to match the data structure.
        """
        if self.x is None:
            raise TypeError(
                "self.x is None - need to define data in model (i.e. train)"
            )

        sampSize = self.x.shape[0]
        randInds = np.random.choice(sampSize, size=sampSize, replace=True)
        sampX = self.x[randInds]
        sampU = self.U[randInds]
        return (sampX, sampU)

    # A method to obtain uncertainty estimates via bootstrapping
    def bootstrap(self, B, order=None, n=100):
        """Obtain estimates of uncertainty in model predictions via bootstrapping.
        Should not need to change this function - instead modify resampleData
        to match with the data structure. If B is None or a length zero array,
        i.e. no new state points are provided, then the std in the PARAMETERS
        of the model are reported from bootstrapping. Note that to change the
        REFERENCE state point and data, MUST RETRAIN!
        """
        if order is None:
            order = self.maxOrder

        # First make sure B isn't just a value
        if isinstance(B, (int, float)):
            B = [B]

        # If B is not a value but an empty array, set it to None
        if B is not None and len(B) == 0:
            print(
                "No state points provided to bootstrap prediction at - bootstrapping parameters."
            )
            B = None

        # If provided/not None make sure B is an array, even if just has one element
        if B is not None:
            B = np.array(B)
            # Last dimension should be observable vector size
            bootStraps = np.zeros((n, B.shape[0], self.x.shape[-1]))
        else:
            bShape = (n,) + self.params.shape
            bootStraps = np.zeros(bShape)

        # Loop for as many resamples as we want
        for i in range(n):
            thisx, thisU = self.resampleData()
            # Train for this resampled data, but don't alter model
            thisParams = self.train(self.refB, thisx, thisU, saveParams=False)
            if B is not None:
                # Predict the new value with the specific parameters
                bootStraps[i, :, :] = self.predict(B, order=order, params=thisParams)
            else:
                bootStraps[i] = thisParams

        # Compute uncertainty
        bootStd = np.std(bootStraps, ddof=1, axis=0)
        return bootStd


class VolumeExtrapModel(ExtrapModel):
    """Class to hold information about a VOLUME extrapolation. This can be trained by providing
    data at the reference state and can then be evaluated to obtain estimates at
    arbitrary other states. Note that refB is now the reference volume and self.U will
    actually represent the virial, not the potential energy. Will only go up to first order
    with derivative information, as after that derivatives of forces are needed.
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

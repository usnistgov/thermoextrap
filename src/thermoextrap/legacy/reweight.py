"""Provides reweighting techniques in same format as extrapolation/interpolation.
This includes perturbation and a wrapper on MBAR.
"""

import numpy as np

try:
    from pymbar import mbar
except ImportError:
    print(
        "Could not find pymbar - will not import and functions involving this will not work."
    )

from .interp import InterpModel


class PerturbModel:
    """Class to hold information about a perturbation."""

    # Otherwise, it just needs to define some variables
    def __init__(self, refB=None, xData=None, uData=None):
        self.refB = refB
        self.x = xData
        self.U = uData
        self.params = None
        if (refB is not None) and (xData is not None) and (uData is not None):
            self.params = self.train(refB, xData, uData, saveParams=True)

    def train(self, refB, xData, uData, saveParams=True):
        """This is the function used to set the parameters of the model. For perturbation
        it's just the observable and potential energy data.
        """
        # Next need to make sure x has at least two dimensions for extrapolation
        if len(xData.shape) == 1:
            xData = np.reshape(xData, (xData.shape[0], 1))
            # Rows are independent observations, columns elements of observable x

        # Also check if observable data matches up with potential energy
        if xData.shape[0] != uData.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy"
                " array (%i) don't match!" % (xData.shape[0], uData.shape[0])
            )
            raise ValueError("x and U must have same shape in first dimension")

        params = [xData, uData]

        if saveParams:
            self.refB = refB
            self.x = xData
            self.U = uData
            self.params = params

        return params

    # A function to calculate model prediction at other state points
    def predict(self, B, params=None, refB=None, useMBAR=False):
        """Performs perturbation at state of interest."""
        # Check if have parameters
        if params is None:
            # Use trained parameters if you have them
            if self.params is None:
                raise TypeError(
                    "self.params is None - need to train model before predicting"
                )
            params = self.params

        if refB is None:
            if self.refB is None:
                raise TypeError("self.refB is None - need to specify reference beta")
            refB = self.refB

        # Specify "parameters" as desired data to use
        x = params[0]
        U = params[1]

        # Make sure B is an array, even if just has one element
        if isinstance(B, (int, float)):
            B = [B]
        B = np.array(B)

        if useMBAR:
            mbarObj = mbar.MBAR(np.array([refB * U]), [U.shape[0]])
            predictVals = np.zeros((len(B), x.shape[1]))
            for i in range(len(B)):
                predictVals[i, :] = mbarObj.computeMultipleExpectations(x.T, B[i] * U)[
                    0
                ]

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
            predictVals = numer / np.array([denom]).T

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
        sampX = self.x[randInds, :]
        sampU = self.U[randInds]
        return (sampX, sampU)

    # A method to obtain uncertainty estimates via bootstrapping
    def bootstrap(self, B, n=100, useMBAR=False):
        """Obtain estimates of uncertainty in model predictions via bootstrapping."""
        if self.params is None:
            raise TypeError(
                "self.params is None - need to train model before bootstrapping"
            )

        # Make sure B is an array, even if just has one element
        if isinstance(B, (int, float)):
            B = [B]
        B = np.array(B)

        # Last dimension should be observable vector size
        bootStraps = np.zeros((n, B.shape[0], self.x.shape[-1]))

        # Loop for as many resamples as we want
        for i in range(n):
            thisx, thisU = self.resampleData()
            # "Train", which here just packages the data, but don't change model params
            thisParams = self.train(self.refB, thisx, thisU, saveParams=False)
            # Predict the new value with the resampled data
            bootStraps[i, :, :] = self.predict(B, params=thisParams, useMBAR=useMBAR)

        # Compute uncertainty
        bootStd = np.std(bootStraps, ddof=1, axis=0)
        return bootStd


class MBARModel(InterpModel):
    """Very similar to interpolation model so inheriting this class.
    Must also have at least two reference states and will use as many as
    provided to make estimate. Resampling will be the same, just need to
    change the train and predict functions.
    """

    def train(self, refB, xData, uData, saveParams=True):
        """Trains and returns a pymbar MBAR object as the model "parameters." """
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

        # Remember, no ragged data, otherwise the below won't work right
        allN = np.ones(xData.shape[0]) * xData.shape[1]
        allU = uData.flatten()
        Ukn = np.tensordot(refB, allU, axes=0)
        mbarObj = mbar.MBAR(Ukn, allN)

        if saveParams:
            self.refB = refB
            self.x = xData
            self.U = uData
            self.params = mbarObj

        return mbarObj

    def predict(self, B, order=None, params=None, refB=None):
        """Leaving in the order variable, even though it will be ignored.
        params should be an pymbar MBAR object and this will just wrap the
        computeExpectations function. Note that refB is ignored because it
        is not needed once data at refB has been incorporated into the mbar
        object. To include more data or data at other state points, need
        to retrain.
        """
        # Check if have parameters
        if params is None:
            # Use trained parameters if you have them
            if self.params is None:
                raise TypeError(
                    "self.params is None - need to train model before predicting"
                )
            params = self.params

        # Make sure B is an array, even if just has one element
        if isinstance(B, (int, float)):
            B = [B]
        B = np.array(B)

        allU = self.U.flatten()
        predictVals = np.zeros((len(B), self.x.shape[2]))
        x = np.reshape(self.x, (self.x.shape[0] * self.x.shape[1], self.x.shape[2]))

        for i in range(len(B)):
            predictVals[i, :] = params.computeMultipleExpectations(x.T, B[i] * allU)[0]

        return predictVals

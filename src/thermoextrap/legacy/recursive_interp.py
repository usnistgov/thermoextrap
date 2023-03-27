"""Holds recursive interpolation class.
This includes the recursive training algorithm and consistency checks.
"""

import numpy as np
from scipy.stats import norm

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Could not find matplotlib - plotting will fail, so ensure that all"
        " doPlot options are set to False, which is the default."
    )

from .ig import IGmodel
from .interp import InterpModel


class RecursiveInterp:
    """Class to perform a recursive interpolation (maybe using weighted extrapolation)
    and then save the information necessary to predict arbitrary interior points.
    Training performs recursive algorithm to meet desired accuracy.
    Prediction uses the learned piecewise function.
    """

    def __init__(self, model, edgeB, maxOrder=1, errTol=0.01):
        self.model = (
            model  # The model object used for interpolation, like ExtrapWeightedModel
        )
        self.modelParams = []  # Model params for piecewise intervals
        self.modelParamErrs = []  # Bootstrapped uncertainties in model parameters
        self.xData = (
            []
        )  # Observable data generated at each edge point - CONSIDER WAYS TO SAVE MEMORY
        self.uData = []  # Potential energy data generated at each edge point
        self.edgeB = np.array(
            edgeB
        )  # Values of state points that we interpolate between
        # Start with outer edges, but will add points as needed
        self.maxOrder = maxOrder  # Maximum order of derivatives to use - default is 1
        self.tol = errTol  # Default bootstrap absolute relative error tolerance of 1%
        # i.e. sigma_bootstrap/|interpolated value| <= 0.01

    def getData(self, B):
        """Obtains data at the specified state point.
        Can modify to run MD or MC simulation, load trajectory or data files, etc.
        MUST return two things, the observable data and the potential energy data
        with the rows being separate configurations/time steps and the columns
        of the observable being the observable vector elements. The observable
        data can be 1 or 2 dimensional but the potential energy data should have
        only one dimension.
        This function just uses the toy ideal gas model that comes with lib_extrap.
        """
        datModel = IGmodel(nParticles=1000)
        xdata, udata = datModel.genData(B, nConfigs=10000)
        return xdata, udata

    def recursiveTrain(
        self,
        B1,
        B2,
        xData1=None,
        xData2=None,
        uData1=None,
        uData2=None,
        recurseDepth=0,
        recurseMax=10,
        Bavail=None,
        verbose=False,
        doPlot=False,
        plotCompareFunc=None,
    ):
        """Recursively trains interpolating models on successively smaller intervals
        until error tolerance is reached. The next state point to subdivide an
        interval is chosen as the point where the bootstrapped error is the largest.
        If Bavail is not None, the closest state point value in this list will be
        used instead. This is useful when data has already been generated at
        specific state points and you do not wish to generate more.
        """
        if recurseDepth > recurseMax:
            raise RecursionError("Maximum recursion depth reached.")

        if verbose:
            print(f"\nInterpolating from points {B1:f} and {B2:f}")
            print("Recursion depth on this branch: %i" % recurseDepth)

        # Generate data somehow if not provided
        if xData1 is None:
            xData1, uData1 = self.getData(B1)
        if xData2 is None:
            xData2, uData2 = self.getData(B2)

        # And format it for training interpolation models
        xData = np.array([xData1, xData2])
        uData = np.array([uData1, uData2])

        # Train the model and get parameters we want to use for THIS interpolation
        # Have to save parameters because want to use SAME data when bootstrapping
        # So part of saving parameters is updating the data that's used in the model
        self.model.train([B1, B2], xData, uData, saveParams=True)

        # Decide if need more data to extrapolate from
        # Check convergence at grid of values between edges, using worst case to check
        Bvals = np.linspace(B1, B2, num=50)
        predictVals = self.model.predict(Bvals, order=self.maxOrder)
        bootErr = self.model.bootstrap(Bvals, order=self.maxOrder)
        # Be careful to catch /0.0
        relErr = np.zeros(bootErr.shape)
        for i in range(bootErr.shape[0]):
            for j in range(bootErr.shape[1]):
                if abs(predictVals[i, j]) == 0.0:
                    # If value is exactly zero, either really unlucky
                    # Or inherently no error because it IS zero - assume this
                    relErr[i, j] = 0.0
                else:
                    relErr[i, j] = bootErr[i, j] / abs(predictVals[i, j])

        # Checking maximum over both tested interior state points AND observable values
        # (if observable is a vector, use element with maximum error
        checkInd = np.unravel_index(np.argmax(relErr), relErr.shape)
        checkVal = relErr[checkInd]

        if verbose:
            print("Maximum bootstrapped error within interval: %f" % checkVal)

        # Check if bootstrapped uncertainty in estimate is small enough
        # If so, we're done
        if checkVal <= self.tol:
            newB = None
        # If not, we want to return the state point with the maximum error
        else:
            # Select closest value of state points in list if provided
            if Bavail is not None:
                Bavail = np.array(Bavail)
                newBInd = np.argmin(abs(Bavail - Bvals[checkInd[0]]))
                newB = Bavail[newBInd]
            else:
                newB = Bvals[
                    checkInd[0]
                ]  # First dimension of prediction is along beta values

        if verbose:
            if newB is not None:
                print("Selected new extrapolation point: %f" % newB)
            else:
                print("No additional extrapolation points necessary on this interval.")

        # Do some plotting just as a visual for how things are going, if desired
        if doPlot:
            interpVals = np.linspace(B1, B2, 20)
            interp = self.model.predict(interpVals, order=self.maxOrder)[:, 0]
            plt.clf()
            plt.plot(interpVals, interp)
            if newB is not None:
                plt.plot([newB, newB], [np.min(interp), np.max(interp)], "k:")
            if plotCompareFunc is not None:
                plt.plot(interpVals, plotCompareFunc(interpVals), "k--")
            plt.xlabel(r"$\beta$")
            plt.ylabel(r"Observable, $X$")
            plt.gcf().tight_layout()
            plt.show(block=False)
            plt.pause(5)
            plt.close()

        if newB is not None:
            # Add the new point to the list of edge points and recurse
            insertInd = np.where(self.edgeB > newB)[0][0]
            self.edgeB = np.insert(self.edgeB, insertInd, newB)
            recurseDepth += 1
            self.recursiveTrain(
                B1,
                newB,
                xData1=xData1,
                uData1=uData1,
                xData2=None,
                uData2=None,
                recurseDepth=recurseDepth,
                recurseMax=recurseMax,
                Bavail=Bavail,
                verbose=verbose,
                doPlot=doPlot,
                plotCompareFunc=plotCompareFunc,
            )
            self.recursiveTrain(
                newB,
                B2,
                xData1=None,
                uData1=None,
                xData2=xData2,
                uData2=uData2,
                recurseDepth=recurseDepth,
                recurseMax=recurseMax,
                Bavail=Bavail,
                verbose=verbose,
                doPlot=doPlot,
                plotCompareFunc=plotCompareFunc,
            )
        else:
            # If we don't need to add extrapolation points, add this region to piecewise function
            # Do this by adding in parameters for this region
            # Appending should work because code will always go with lower interval first
            self.modelParams.append(self.model.params)
            # And also append uncertainties by bootstrapping
            self.modelParamErrs.append(self.model.bootstrap(None))
            # Also add this data to what we save - hopefully have enough memory
            self.xData.append(xData1)
            self.uData.append(uData1)
            if B2 == self.edgeB[-1]:
                self.xData.append(xData2)
                self.uData.append(uData2)
            return

    def sequentialTrain(self, Btrain, verbose=False):
        """Trains sequentially without recursion. List of state point values is provided and
        training happens just on those without adding points.
        """

        # Check for overlap in self.edgeB and Btrain and merge as needed
        for Bval in Btrain:
            if Bval not in self.edgeB:
                self.edgeB = np.hstack((self.edgeB, [Bval]))
                self.xData.append(None)
                self.uData.append(None)
        sortInds = np.argsort(self.edgeB)
        self.xData = [self.xData[i] for i in sortInds]
        self.uData = [self.uData[i] for i in sortInds]
        self.edgeB = np.sort(self.edgeB)

        # Set self.modelParams and self.modelParamErrs to empty lists
        # Will recompute all in case have new intervals
        self.modelParams = []
        self.modelParamErrs = []

        # Loop over pairs of edge points
        for i in range(len(self.edgeB) - 1):
            B1 = self.edgeB[i]
            B2 = self.edgeB[i + 1]

            if verbose:
                print(f"\nInterpolating from points {B1:f} and {B2:f}")

            # Generate data somehow if not provided
            try:
                xData1 = self.xData[i]
                uData1 = self.uData[i]
                if xData1 is None:
                    xData1, uData1 = self.getData(B1)
                    self.xData[i] = xData1
                    self.uData[i] = uData1
            except IndexError:
                xData1, uData1 = self.getData(B1)
                self.xData.append(xData1)
                self.uData.append(uData1)
            try:
                xData2 = self.xData[i + 1]
                uData2 = self.uData[i + 1]
                if xData2 is None:
                    xData2, uData2 = self.getData(B2)
                    self.xData[i + 1] = xData2
                    self.uData[i + 1] = uData2
            except IndexError:
                xData2, uData2 = self.getData(B2)
                self.xData.append(xData2)
                self.uData.append(uData2)

            # And format data for training interpolation models
            xData = np.array([xData1, xData2])
            uData = np.array([uData1, uData2])

            # Train the model and get parameters we want to use for THIS interpolation
            # Have to save parameters because want to use SAME data when bootstrapping
            # So part of saving parameters is updating the data that's used in the model
            self.model.train([B1, B2], xData, uData, saveParams=True)

            if verbose:
                # Check if need more data to extrapolate from (just report info on this)
                Bvals = np.linspace(B1, B2, num=50)
                predictVals = self.model.predict(Bvals, order=self.maxOrder)
                bootErr = self.model.bootstrap(Bvals, order=self.maxOrder)
                # Be careful to catch /0.0
                relErr = np.zeros(bootErr.shape)
                for i in range(bootErr.shape[0]):
                    for j in range(bootErr.shape[1]):
                        if abs(predictVals[i, j]) == 0.0:
                            # If value is exactly zero, either really unlucky
                            # Or inherently no error because it IS zero - assume this
                            relErr[i, j] = 0.0
                        else:
                            relErr[i, j] = bootErr[i, j] / abs(predictVals[i, j])

                # Checking maximum over both tested interior state points AND observable values
                # (if observable is a vector, use element with maximum error
                checkInd = np.unravel_index(np.argmax(relErr), relErr.shape)
                checkVal = relErr[checkInd]
                print("Maximum bootstrapped error within interval: %f" % checkVal)
                print("At point: %f" % Bvals[checkInd[0]])

            # Add in parameters for this region
            # Appending should work because code will always go with lower interval first
            self.modelParams.append(self.model.params)
            # And also append uncertainties by bootstrapping
            self.modelParamErrs.append(self.model.bootstrap(None))
            # Also add this data to what we save - hopefully have enough memory

        return

    def predict(self, B):
        """Makes a prediction using the trained piecewise model.
        Note that the function will not produce output if asked to extrapolate outside
        the range it was trained on.
        """
        # Make sure we've done some training
        if len(self.modelParams) == 0:
            print("First must train the piecewise model!")
            raise ValueError("Must train before predicting")

        # For each state point in B, select a piecewise model to use
        predictVals = np.zeros((len(B), self.model.x.shape[2]))

        for i, beta in enumerate(B):
            # Check if out of lower bound
            if beta < self.edgeB[0]:
                print(
                    "Have provided point {:f} below interpolation function"
                    " interval edges ({}).".format(beta, str(self.edgeB))
                )
                raise IndexError("Interpolation point below range")

            # Check if out of upper bound
            if beta > self.edgeB[-1]:
                print(
                    "Have provided point {:f} above interpolation function"
                    " interval edges ({}).".format(beta, str(self.edgeB))
                )
                raise IndexError("Interpolation point above range")

            # And get correct index for interpolating polynomial
            paramInd = np.where(self.edgeB <= beta)[0][-1]

            # Don't want to train model (already done!) but need to manually specify
            # both the parameters AND the reference state points
            # For the latter, must set manually
            if paramInd == len(self.edgeB) - 1:
                self.model.refB = np.array(
                    [self.edgeB[paramInd - 1], self.edgeB[paramInd]]
                )
                predictVals[i] = self.model.predict(
                    beta, params=self.modelParams[paramInd - 1], order=self.maxOrder
                )[0, :]
            else:
                self.model.refB = np.array(
                    [self.edgeB[paramInd], self.edgeB[paramInd + 1]]
                )
                predictVals[i] = self.model.predict(
                    beta, params=self.modelParams[paramInd], order=self.maxOrder
                )[0, :]

        return predictVals

    def checkPolynomialConsistency(self, doPlot=False):
        """If the interpolation model is a polynomial, checks to see if the polynomials
        are locally consistent. In other words, we want the coefficients between
        neighboring regions to match closely to each other, and to the larger region
        composed of the two neighboring sub-regions. Essentially, this checks to make
        sure the local curvature is staying constant as you zoom in. If it is, your
        function in the region is well-described by the given order of polynomial
        and you can have higher confidence in the resulting model output. Will also
        generate plots as a visual check if desired.
        """
        if not isinstance(self.model, InterpModel):
            print(
                "Can only check polynomial consistency with a polynomial interpolation model class."
            )
            raise TypeError("Incorrect class provided")

        if len(self.modelParams) == 0:
            print(
                "No model parameters found. Must train model before checking consistency."
            )
            raise ValueError("self.modelParams is length 0 - must train model first")

        if len(self.modelParams) == 1:
            print("Single interpolation region. No point in checking consistency.")
            raise ValueError("self.modelParams is length 1 - nothing to check")

        # Need to subdivide the full interval into pairs of neighboring intervals
        # Easiest way is to take state point edge values in sliding sets of three
        allInds = np.arange(self.edgeB.shape[0])
        nrows = allInds.size - 3 + 1
        n = allInds.strides[0]
        edgeSets = np.lib.stride_tricks.as_strided(
            allInds, shape=(nrows, 3), strides=(n, n)
        )

        # Will record and return p-values from hypothesis tests
        allPvals = []

        # Before loop, set up plot if wanted
        if doPlot:
            pColors = plt.cm.cividis(np.arange(len(edgeSets)) / float(len(edgeSets)))
            pFig, pAx = plt.subplots()
            plotYmin = 1e10
            plotYmax = -1e10

        # Loop over sets of three edges
        for i, aset in enumerate(edgeSets):
            # Start with regions we already have coefficients for
            reg1Coeffs = self.modelParams[aset[0]]
            reg1Err = self.modelParamErrs[aset[0]]
            reg2Coeffs = self.modelParams[aset[1]]
            reg2Err = self.modelParamErrs[aset[1]]
            z12 = (reg1Coeffs - reg2Coeffs) / np.sqrt(reg1Err**2 + reg2Err**2)
            # Assuming Gaussian distributions for coefficients
            # This is implicit in returning bootstrap standard deviation as estimate of uncertainty
            # If DON'T want to assume this, bootstrap function should return confidence intervals
            # And that will require a good bit of re-coding throughout this whole class
            # p12 = 2.0*norm.cdf(-abs(z12)) #Null hypothesis that coefficients same
            p12 = norm.cdf(abs(z12)) - norm.cdf(
                -abs(z12)
            )  # Null hypothesis coefficients different

            # To check full interval, must retrain model with data
            fullCoeffs = self.model.train(
                self.edgeB[aset[[0, 2]]],
                np.array([self.xData[aset[0]], self.xData[aset[2]]]),
                np.array([self.uData[aset[0]], self.uData[aset[2]]]),
                saveParams=True,
            )
            fullErr = self.model.bootstrap(None)
            z1full = (reg1Coeffs - fullCoeffs) / np.sqrt(reg1Err**2 + fullErr**2)
            # p1full = 2.0*norm.cdf(-abs(z1full))
            p1full = norm.cdf(abs(z1full)) - norm.cdf(-abs(z1full))
            z2full = (reg2Coeffs - fullCoeffs) / np.sqrt(reg2Err**2 + fullErr**2)
            # p2full = 2.0*norm.cdf(-abs(z2full))
            p2full = norm.cdf(abs(z2full)) - norm.cdf(-abs(z2full))

            allPvals.append(np.vstack((p12, p1full, p2full)))
            print(
                "Interval with edges %s (indices %s):"
                % (str(self.edgeB[aset]), str(aset))
            )
            print("\tP-values between regions:")
            print(p12)
            print("\tP-values for full and 1 :")
            print(p1full)
            print("\tP-values for full and 2 :")
            print(p2full)

            if doPlot:
                plotPoints = np.linspace(self.edgeB[aset[0]], self.edgeB[aset[2]], 50)
                plotFull = np.polynomial.polynomial.polyval(
                    plotPoints, fullCoeffs[:, 0]
                )
                plotReg1 = np.polynomial.polynomial.polyval(
                    plotPoints, reg1Coeffs[:, 0]
                )
                plotReg2 = np.polynomial.polynomial.polyval(
                    plotPoints, reg2Coeffs[:, 0]
                )
                pAx.plot(plotPoints, plotFull, color=pColors[i], linestyle="-")
                pAx.plot(plotPoints, plotReg1, color=pColors[i], linestyle=":")
                pAx.plot(plotPoints, plotReg2, color=pColors[i], linestyle="--")
                allPlotY = np.hstack((plotFull, plotReg1, plotReg2))
                if np.min(allPlotY) < plotYmin:
                    plotYmin = np.min(allPlotY)
                if np.max(allPlotY) > plotYmax:
                    plotYmax = np.max(allPlotY)

        if doPlot:
            for edge in self.edgeB:
                pAx.plot([edge] * 2, [plotYmin, plotYmax], "k-")
            pAx.set_xlabel(r"$\beta$")
            pAx.set_ylabel(r"$\langle x \rangle$")
            pFig.tight_layout()
            plt.show()

        return allPvals

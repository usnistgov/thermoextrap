"""
Recursive interpolation (:mod:`~thermoextrap.recursive_interp`)
===============================================================

Holds recursive interpolation class.
This includes the recursive training algorithm and consistency checks.

See :ref:`notebooks/temperature_interp:recursive interpolation` for example usage.
"""


import numpy as np

# import xarray as xr
from scipy import stats

# TODO: Change this to point to the "new" ideagas.py
# TODO: rework this code to be cleaner
# from ..legacy.ig import IGmodel
from .core import idealgas
from .core._deprecate import deprecate, deprecate_kwarg
from .core.data import factory_data_values
from .core.models import ExtrapModel, InterpModel

try:
    import matplotlib.pyplot as plt

    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False
    # print(
    #     "Could not find matplotlib - plotting will fail, so ensure that all"
    #     " doPlot options are set to False, which is the default."
    # )


def _has_plt():
    if _HAS_PLT:
        pass
    else:
        raise ImportError("install matplotlib for this functionality")


class RecursiveInterp:
    """
    Class to perform a recursive interpolation (maybe using weighted extrapolation)
    and then save the information necessary to predict arbitrary interior points.
    Training performs recursive algorithm to meet desired accuracy.
    Prediction uses the learned piecewise function.

    Parameters
    ----------
    model_cls : type
        Class of each model.
    derivatives : :class:`thermoextrap.models.Derivatives`
    edge_beta : array-like
        values of `beta` at edges
    max_order : int, default=1
        Maximum order.
    tol : float, default=0.01
        Error tolerance.
    """

    @deprecate_kwarg("edgeB", "edge_beta")
    @deprecate_kwarg("maxOrder", "max_order")
    @deprecate_kwarg("errTol", "tol")
    def __init__(self, model_cls, derivatives, edge_beta, max_order=1, tol=0.01):
        self.model_cls = (
            model_cls  # The model CLASS used for interpolation, like InterpModel
        )
        self.derivatives = derivatives  # Derivatives object describing how derivatives will be calculated
        self.states = (
            []
        )  # List of ExtrapModel objects sharing same Derivatives but different Data
        self.edge_beta = np.array(
            edge_beta
        )  # Values of state points that we interpolate between
        # Start with outer edges, but will add points as needed
        self.max_order = max_order  # Maximum order of derivatives to use - default is 1
        self.tol = tol  # Default bootstrap absolute relative error tolerance of 1%
        # i.e. sigma_bootstrap/|interpolated value| <= 0.01

    @deprecate_kwarg("B", "beta")
    def get_data(self, beta):
        """
        Obtains data at the specified state point.
        Can modify to run MD or MC simulation, load trajectory or data files, etc.
        MUST return two things, the observable data and the potential energy data
        with the rows being separate configurations/time steps and the columns
        of the observable being the observable vector elements. The observable
        data can be 1 or 2 dimensional but the potential energy data should have
        only one dimension.
        This function just uses the toy ideal gas model that comes with lib_extrap.
        """

        npart, nconfig = 1000, 10000
        xdata, udata = idealgas.generate_data(shape=(nconfig, npart), beta=beta)

        # datModel = IGmodel(nParticles=1000)
        # xdata, udata = datModel.genData(B, nConfigs=10000)
        # Need to also change data object kwargs based on data when change getData
        data = factory_data_values(uv=udata, xv=xdata, order=self.max_order)
        return data

    getData = deprecate("getData", get_data, "0.2.0")

    @deprecate_kwarg("B1", "beta1")
    @deprecate_kwarg("B2", "beta2")
    @deprecate_kwarg("recurseDepth", "recurse_depth")
    @deprecate_kwarg("recurseMax", "recurse_max")
    @deprecate_kwarg("Bavail", "beta_avail")
    @deprecate_kwarg("doPlot", "do_plot")
    @deprecate_kwarg("plotCompareFunc", "plot_func")
    def recursive_train(
        self,
        beta1,
        beta2,
        data1=None,
        data2=None,
        recurse_depth=0,
        recurse_max=10,
        beta_avail=None,
        verbose=False,
        do_plot=False,
        plot_func=None,
    ):
        """
        Recursively trains interpolating models on successively smaller intervals
        until error tolerance is reached. The next state point to subdivide an
        interval is chosen as the point where the bootstrapped error is the largest.
        If beta_avail is not None, the closest state point value in this list will be
        used instead. This is useful when data has already been generated at
        specific state points and you do not wish to generate more.
        """
        if recurse_depth > recurse_max:
            raise RecursionError("Maximum recursion depth reached.")

        if verbose:
            print(f"\nInterpolating from points {beta1:f} and {beta2:f}")
            print("Recursion depth on this branch: %i" % recurse_depth)

        # Generate data somehow if not provided
        if data1 is None:
            data1 = self.get_data(beta1)
        if data2 is None:
            data2 = self.get_data(beta2)

        # For each set of data, create an ExtrapModel object
        extrap1 = ExtrapModel(
            alpha0=beta1, data=data1, derivatives=self.derivatives, order=self.max_order
        )
        extrap2 = ExtrapModel(
            alpha0=beta2, data=data2, derivatives=self.derivatives, order=self.max_order
        )

        # Now create interpolating model based on state collection of the two
        this_model = self.model_cls((extrap1, extrap2))

        # Decide if need more data to extrapolate from
        # Check convergence at grid of values between edges, using worst case to check
        Bvals = np.linspace(beta1, beta2, num=50)
        predictVals = this_model.predict(Bvals, order=self.max_order)
        bootErr = (
            this_model.resample(nrep=100)
            .predict(Bvals, order=self.max_order)
            .std("rep")
        )

        relErr = bootErr / abs(predictVals)
        # Be careful to catch divide by zero
        relErr = relErr.fillna(0.0)  # Catches 0.0/0.0, so replaces NaN with 0
        relErr = relErr.where(relErr != np.inf).fillna(
            0.0
        )  # Replaces Inf with NaN, then NaN to 0
        # If value is exactly zero, either really unlucky
        # Or inherently no error because it IS zero - assume the latter

        # Checking maximum over both tested interior state points AND observable values
        # (if observable is a vector, use element with maximum error
        checkInd = np.unravel_index(relErr.argmax(), relErr.shape)
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
            if beta_avail is not None:
                beta_avail = np.array(beta_avail)
                newBInd = np.argmin(abs(beta_avail - Bvals[checkInd[0]]))
                newB = beta_avail[newBInd]
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
        if do_plot:
            _has_plt()
            if "val" in predictVals.dims:
                toplot = predictVals.isel(val=0)
            else:
                toplot = predictVals
            plt.clf()
            plt.plot(Bvals, toplot)
            if newB is not None:
                plt.plot([newB, newB], [np.min(toplot), np.max(toplot)], "k:")
            if plot_func is not None:
                plt.plot(Bvals, plot_func(Bvals), "k--")
            plt.xlabel(r"$\beta$")
            plt.ylabel(r"Observable, $X$")
            plt.gcf().tight_layout()
            plt.show(block=False)
            plt.pause(5)
            plt.close()

        if newB is not None:
            # Add the new point to the list of edge points and recurse
            insertInd = np.where(self.edge_beta > newB)[0][0]
            self.edge_beta = np.insert(self.edge_beta, insertInd, newB)
            recurse_depth += 1
            self.recursive_train(
                beta1,
                newB,
                data1=data1,
                data2=None,
                recurse_depth=recurse_depth,
                recurse_max=recurse_max,
                beta_avail=beta_avail,
                verbose=verbose,
                do_plot=do_plot,
                plot_func=plot_func,
            )
            self.recursive_train(
                newB,
                beta2,
                data1=None,
                data2=data2,
                recurse_depth=recurse_depth,
                recurse_max=recurse_max,
                beta_avail=beta_avail,
                verbose=verbose,
                do_plot=do_plot,
                plot_func=plot_func,
            )
        else:
            # If we don't need to add extrapolation points, add this region to piecewise function
            # Do this by adding ExtrapModel object in this region, which also saves the data
            # Appending should work because code will always go with lower interval first
            self.states.append(extrap1)
            if beta2 == self.edge_beta[-1]:
                self.states.append(extrap2)
            return

    recursiveTrain = deprecate("recursiveTrain", recursive_train, "0.2.0")

    @deprecate_kwarg("Btrain", "beta_train")
    def sequential_train(self, beta_train, verbose=False):
        """
        Trains sequentially without recursion. List of state point values is provided and
        training happens just on those without adding points.
        """

        # Check for overlap in self.edge_beta and beta_train and merge as needed
        # Fill in None in self.states where we have not yet trained
        for Bval in beta_train:
            if Bval not in self.edge_beta:
                self.edge_beta = np.hstack((self.edge_beta, [Bval]))
                self.states = self.states + [
                    None,
                ]
        sort_inds = np.argsort(self.edge_beta)
        self.states = [self.states[i] for i in sort_inds]
        self.edge_beta = np.sort(self.edge_beta)

        # Loop over pairs of edge points
        for i in range(len(self.edge_beta) - 1):
            B1 = self.edge_beta[i]
            B2 = self.edge_beta[i + 1]

            if verbose:
                print(f"\nInterpolating from points {B1:f} and {B2:f}")

            # Check if already have ExtrapModel with data for B1
            if self.states[i] is None:
                data1 = self.getData(B1)
                extrap1 = ExtrapModel(
                    alpha0=B1,
                    data=data1,
                    derivatives=self.derivatives,
                    order=self.max_order,
                )
                self.states[i] = extrap1
            else:
                extrap1 = self.states[i]

            # And for B2
            if self.states[i + 1] is None:
                data2 = self.getData(B2)
                extrap2 = ExtrapModel(
                    alpha0=B2,
                    data=data2,
                    derivatives=self.derivatives,
                    order=self.max_order,
                )
                self.states[i + 1] = extrap2
            else:
                extrap2 = self.states[i + 1]

            # Train the model and get interpolation
            this_model = self.model_cls((extrap1, extrap2))

            if verbose:
                # Check if need more data to extrapolate from (just report info on this)
                Bvals = np.linspace(B1, B2, num=50)
                predictVals = this_model.predict(Bvals, order=self.max_order)
                bootErr = (
                    this_model.resample(nrep=100)
                    .predict(Bvals, order=self.max_order)
                    .std("rep")
                )

                relErr = bootErr / abs(predictVals)
                # Be careful to catch divide by zero
                relErr = relErr.fillna(0.0)  # Catches 0.0/0.0, so replaces NaN with 0
                relErr = relErr.where(relErr != np.inf).fillna(
                    0.0
                )  # Replaces Inf with NaN, then NaN to 0
                # If value is exactly zero, either really unlucky
                # Or inherently no error because it IS zero - assume the latter

                # Checking maximum over both tested interior state points AND observable values
                # (if observable is a vector, use element with maximum error
                checkInd = np.unravel_index(relErr.argmax(), relErr.shape)
                checkVal = relErr[checkInd]
                print("Maximum bootstrapped error within interval: %f" % checkVal)
                print("At point: %f" % Bvals[checkInd[0]])

        return

    sequentialTrain = deprecate("sequentialTrain", sequential_train, "0.2.0")

    @deprecate_kwarg("B", "beta")
    def predict(self, beta):
        """
        Makes a prediction using the trained piecewise model.
        Note that the function will not produce output if asked to extrapolate outside
        the range it was trained on.
        """
        # Make sure we've done some training
        if len(self.states) == 0:
            print("First must train the piecewise model!")
            raise ValueError("Must train before predicting")

        # For each state point in beta, select a piecewise model to use
        if "val" in self.states[0].data.xv.dims:
            predictVals = np.zeros((len(beta), self.states[0].data.xv["val"].size))
        else:
            predictVals = np.zeros(len(beta))

        for i, beta in enumerate(beta):
            # Check if out of lower bound
            if beta < self.edge_beta[0]:
                print(
                    "Have provided point {:f} below interpolation function"
                    " interval edges ({}).".format(beta, str(self.edge_beta))
                )
                raise IndexError("Interpolation point below range")

            # Check if out of upper bound
            if beta > self.edge_beta[-1]:
                print(
                    "Have provided point {:f} above interpolation function"
                    " interval edges ({}).".format(beta, str(self.edge_beta))
                )
                raise IndexError("Interpolation point above range")

            # Get indices for bracketing state points
            lowInd = np.where(self.edge_beta <= beta)[0][-1]
            try:
                hiInd = np.where(self.edge_beta > beta)[0][0]
            except IndexError:
                # With above logic, must have beta = self.edge_beta[-1]
                # Which would make lowInd = len(self.edge_beta)-1
                # Shift interval down
                lowInd -= 1
                hiInd = len(self.edge_beta) - 1

            # Create interpolation object and predict
            this_model = self.model_cls((self.states[lowInd], self.states[hiInd]))
            predictVals[i] = this_model.predict(beta, order=self.max_order)

        return predictVals

    @deprecate_kwarg("doPlot", "do_plot")
    def check_poly_consistency(self, do_plot=False):
        """
        If the interpolation model is a polynomial, checks to see if the polynomials
        are locally consistent. In other words, we want the coefficients between
        neighboring regions to match closely to each other, and to the larger region
        composed of the two neighboring sub-regions. Essentially, this checks to make
        sure the local curvature is staying constant as you zoom in. If it is, your
        function in the region is well-described by the given order of polynomial
        and you can have higher confidence in the resulting model output. Will also
        generate plots as a visual check if desired.
        """
        if self.model_cls != InterpModel:
            print(
                "Can only check polynomial consistency with a polynomial interpolation model class."
            )
            raise TypeError("Incorrect class provided")

        if len(self.states) == 0:
            print(
                "No model parameters found. Must train model before checking consistency."
            )
            raise ValueError("self.states is length 0 - must train model first")

        if len(self.states) == 2:
            print("Single interpolation region. No point in checking consistency.")
            raise ValueError("self.states is length 2 - nothing to check")

        # Need to subdivide the full interval into pairs of neighboring intervals
        # Easiest way is to take state point edge values in sliding sets of three
        allInds = np.arange(self.edge_beta.shape[0])
        nrows = allInds.size - 3 + 1
        n = allInds.strides[0]
        edgeSets = np.lib.stride_tricks.as_strided(
            allInds, shape=(nrows, 3), strides=(n, n)
        )

        # Will record and return p-values from hypothesis tests
        allPvals = []

        # Before loop, set up plot if wanted
        if do_plot:
            _has_plt()
            pColors = plt.cm.cividis(np.linspace(0.0, 1.0, len(edgeSets)))
            pFig, pAx = plt.subplots()
            plotYmin = 1e10
            plotYmax = -1e10

        # Loop over sets of three edges
        for i, aset in enumerate(edgeSets):
            reg1Model = self.model_cls((self.states[aset[0]], self.states[aset[1]]))
            reg1Coeffs = reg1Model.coefs(order=self.max_order)
            reg1Err = (
                reg1Model.resample(nrep=100).coefs(order=self.max_order).std("rep")
            )
            reg2Model = self.model_cls((self.states[aset[1]], self.states[aset[2]]))
            reg2Coeffs = reg2Model.coefs(order=self.max_order)
            reg2Err = (
                reg2Model.resample(nrep=100).coefs(order=self.max_order).std("rep")
            )
            z12 = (reg1Coeffs - reg2Coeffs) / np.sqrt(reg1Err**2 + reg2Err**2)
            # Assuming Gaussian distributions for coefficients
            # This is implicit in returning bootstrap standard deviation as estimate of uncertainty
            # If DON'T want to assume this, bootstrap function should return confidence intervals
            # And that will require a good bit of re-coding throughout this whole class
            # p12 = 2.0*stats.norm.cdf(-abs(z12)) #Null hypothesis that coefficients same
            p12 = stats.norm.cdf(abs(z12)) - stats.norm.cdf(
                -abs(z12)
            )  # Null hypothesis coefficients different

            # To check full interval, must retrain model with data
            fullModel = self.model_cls((self.states[aset[0]], self.states[aset[2]]))
            fullCoeffs = fullModel.coefs(order=self.max_order)
            fullErr = (
                fullModel.resample(nrep=100).coefs(order=self.max_order).std("rep")
            )
            z1full = (reg1Coeffs - fullCoeffs) / np.sqrt(reg1Err**2 + fullErr**2)
            # p1full = 2.0*stats.norm.cdf(-abs(z1full))
            p1full = stats.norm.cdf(abs(z1full)) - stats.norm.cdf(-abs(z1full))
            z2full = (reg2Coeffs - fullCoeffs) / np.sqrt(reg2Err**2 + fullErr**2)
            # p2full = 2.0*stats.norm.cdf(-abs(z2full))
            p2full = stats.norm.cdf(abs(z2full)) - stats.norm.cdf(-abs(z2full))

            allPvals.append(np.vstack((p12, p1full, p2full)))
            print(
                "Interval with edges %s (indices %s):"
                % (str(self.edge_beta[aset]), str(aset))
            )
            print("\tP-values between regions:")
            print(p12)
            print("\tP-values for full and 1 :")
            print(p1full)
            print("\tP-values for full and 2 :")
            print(p2full)

            if do_plot:
                plotPoints = np.linspace(
                    self.edge_beta[aset[0]], self.edge_beta[aset[2]], 50
                )
                plotFull = np.polynomial.polynomial.polyval(plotPoints, fullCoeffs)
                plotReg1 = np.polynomial.polynomial.polyval(plotPoints, reg1Coeffs)
                plotReg2 = np.polynomial.polynomial.polyval(plotPoints, reg2Coeffs)
                pAx.plot(plotPoints, plotFull, color=pColors[i], linestyle="-")
                pAx.plot(plotPoints, plotReg1, color=pColors[i], linestyle=":")
                pAx.plot(plotPoints, plotReg2, color=pColors[i], linestyle="--")
                allPlotY = np.hstack((plotFull, plotReg1, plotReg2))
                if np.min(allPlotY) < plotYmin:
                    plotYmin = np.min(allPlotY)
                if np.max(allPlotY) > plotYmax:
                    plotYmax = np.max(allPlotY)

        if do_plot:
            _has_plt()
            for edge in self.edge_beta:
                pAx.plot([edge] * 2, [plotYmin, plotYmax], "k-")
            pAx.set_xlabel(r"$\beta$")
            pAx.set_ylabel(r"$\langle x \rangle$")
            pFig.tight_layout()
            plt.show()

        return allPvals

    checkPolynomialConsistency = deprecate(
        "checkPolynomialConsistency", check_poly_consistency, "0.2.0"
    )

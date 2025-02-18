"""
Recursive interpolation (:mod:`~thermoextrap.recursive_interp`)
===============================================================

Holds recursive interpolation class.
This includes the recursive training algorithm and consistency checks.

See :ref:`examples/usage/basic/temperature_interp:recursive interpolation` for example usage.
"""

# TODO(wpk): rework this code to be cleaner
import logging

import numpy as np
from cmomy.random import validate_rng

from . import idealgas
from .core._deprecate import deprecate, deprecate_kwarg
from .data import factory_data_values
from .models import ExtrapModel, InterpModel

logger = logging.getLogger(__name__)


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        msg = "must install matplotlib to use plotting"
        raise ImportError(msg) from err
    return plt


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
    rng : :class:`numpy.random.Generator`, optional
    """

    @deprecate_kwarg("edgeB", "edge_beta")
    @deprecate_kwarg("maxOrder", "max_order")
    @deprecate_kwarg("errTol", "tol")
    def __init__(
        self,
        model_cls,
        derivatives,
        edge_beta,
        max_order=1,
        tol=0.01,
        rng=None,
    ) -> None:
        self.model_cls = (
            model_cls  # The model CLASS used for interpolation, like InterpModel
        )
        self.derivatives = derivatives  # Derivatives object describing how derivatives will be calculated
        self.states = []  # List of ExtrapModel objects sharing same Derivatives but different Data
        self.edge_beta = np.array(
            edge_beta
        )  # Values of state points that we interpolate between
        # Start with outer edges, but will add points as needed
        self.max_order = max_order  # Maximum order of derivatives to use - default is 1
        self.tol = tol  # Default bootstrap absolute relative error tolerance of 1%
        # i.e. sigma_bootstrap/|interpolated value| <= 0.01

        self.rng = validate_rng(rng)

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
        xdata, udata = idealgas.generate_data(
            shape=(nconfig, npart), beta=beta, rng=self.rng
        )

        # datModel = IGmodel(nParticles=1000)
        # xdata, udata = datModel.genData(B, nConfigs=10000)
        # Need to also change data object kwargs based on data when change getData
        return factory_data_values(uv=udata, xv=xdata, order=self.max_order)

    getData = deprecate("getData", get_data, "0.2.0")  # noqa: N815

    @deprecate_kwarg("beta1", "beta1")
    @deprecate_kwarg("beta2", "beta2")
    @deprecate_kwarg("recurseDepth", "recurse_depth")
    @deprecate_kwarg("recurseMax", "recurse_max")
    @deprecate_kwarg("Bavail", "beta_avail")
    @deprecate_kwarg("doPlot", "do_plot")
    @deprecate_kwarg("plotCompareFunc", "plot_func")
    def recursive_train(  # noqa: C901, PLR0912, PLR0914, PLR0915
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
    ) -> None:
        """
        Recursively trains interpolating models on successively smaller intervals
        until error tolerance is reached. The next state point to subdivide an
        interval is chosen as the point where the bootstrapped error is the largest.
        If beta_avail is not None, the closest state point value in this list will be
        used instead. This is useful when data has already been generated at
        specific state points and you do not wish to generate more.
        """
        if do_plot:
            plt = _get_plt()

        if recurse_depth > recurse_max:
            msg = "Maximum recursion depth reached."
            raise RecursionError(msg)

        # TODO(wpk): Better logger
        if verbose:
            logger.setLevel(logging.INFO)

        logger.info("Interpolating from points %f and %f", beta1, beta2)
        logger.info("Recursion depth on this branch: %s", recurse_depth)

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
        beta_vals = np.linspace(beta1, beta2, num=50)
        predict_vals = this_model.predict(beta_vals, order=self.max_order)
        boot_err = (
            this_model.resample(sampler={"nrep": 100})
            .predict(beta_vals, order=self.max_order)
            .std("rep")
        )

        rel_err = boot_err / abs(predict_vals)
        # Be careful to catch divide by zero
        rel_err = rel_err.fillna(0.0)  # Catches 0.0/0.0, so replaces NaN with 0
        rel_err = rel_err.where(rel_err != np.inf).fillna(
            0.0
        )  # Replaces Inf with NaN, then NaN to 0
        # If value is exactly zero, either really unlucky
        # Or inherently no error because it IS zero - assume the latter

        # Checking maximum over both tested interior state points AND observable values
        # (if observable is a vector, use element with maximum error
        check_ind = np.unravel_index(rel_err.argmax(), rel_err.shape)
        check_val = rel_err[check_ind]

        logger.info("Maximum bootstrapped error within interval: %s", check_val)

        # Check if bootstrapped uncertainty in estimate is small enough
        # If so, we're done
        if check_val <= self.tol:
            new_beta = None
        # If not, we want to return the state point with the maximum error
        elif beta_avail is not None:
            beta_avail = np.array(beta_avail)
            new_beta_ind = np.argmin(abs(beta_avail - beta_vals[check_ind[0]]))
            new_beta = beta_avail[new_beta_ind]
        else:
            new_beta = beta_vals[
                check_ind[0]
            ]  # First dimension of prediction is along beta values

        if new_beta is not None:
            logger.info("Selected new extrapolation point: %f", new_beta)
        else:
            logger.info(
                "No additional extrapolation points necessary on this interval."
            )

        # Do some plotting just as a visual for how things are going, if desired
        if do_plot:
            if "val" in predict_vals.dims:
                toplot = predict_vals.isel(val=0)
            else:
                toplot = predict_vals
            plt.clf()
            plt.plot(beta_vals, toplot)
            if new_beta is not None:
                plt.plot([new_beta, new_beta], [np.min(toplot), np.max(toplot)], "k:")
            if plot_func is not None:
                plt.plot(beta_vals, plot_func(beta_vals), "k--")
            plt.xlabel(r"$\beta$")
            plt.ylabel(r"Observable, $X$")
            plt.gcf().tight_layout()
            plt.show(block=False)
            plt.pause(5)
            plt.close()

        if new_beta is not None:
            # Add the new point to the list of edge points and recurse
            insert_ind = np.where(self.edge_beta > new_beta)[0][0]
            self.edge_beta = np.insert(self.edge_beta, insert_ind, new_beta)
            recurse_depth += 1
            self.recursive_train(
                beta1,
                new_beta,
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
                new_beta,
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

    recursiveTrain = deprecate("recursiveTrain", recursive_train, "0.2.0")  # noqa: N815

    @deprecate_kwarg("Btrain", "beta_train")
    def sequential_train(self, beta_train, verbose=False) -> None:
        """
        Trains sequentially without recursion. List of state point values is provided and
        training happens just on those without adding points.
        """
        # Check for overlap in self.edge_beta and beta_train and merge as needed
        # Fill in None in self.states where we have not yet trained
        for beta_val in beta_train:
            if beta_val not in self.edge_beta:
                self.edge_beta = np.hstack((self.edge_beta, [beta_val]))
                self.states = [*self.states, None]
        sort_inds = np.argsort(self.edge_beta)
        self.states = [self.states[i] for i in sort_inds]
        self.edge_beta = np.sort(self.edge_beta)

        # Loop over pairs of edge points
        for i in range(len(self.edge_beta) - 1):
            beta1 = self.edge_beta[i]
            beta2 = self.edge_beta[i + 1]

            if verbose:
                logger.setLevel(logging.INFO)
            logger.info("Interpolating from points %f and %f", beta1, beta2)

            # Check if already have ExtrapModel with data for beta1
            if self.states[i] is None:
                data1 = self.get_data(beta1)
                extrap1 = ExtrapModel(
                    alpha0=beta1,
                    data=data1,
                    derivatives=self.derivatives,
                    order=self.max_order,
                )
                self.states[i] = extrap1
            else:
                extrap1 = self.states[i]

            # And for beta2
            if self.states[i + 1] is None:
                data2 = self.get_data(beta2)
                extrap2 = ExtrapModel(
                    alpha0=beta2,
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
                beta_vals = np.linspace(beta1, beta2, num=50)
                predict_vals = this_model.predict(beta_vals, order=self.max_order)
                boot_err = (
                    this_model.resample(sampler={"nrep": 100})
                    .predict(beta_vals, order=self.max_order)
                    .std("rep")
                )

                rel_err = boot_err / abs(predict_vals)
                # Be careful to catch divide by zero
                rel_err = rel_err.fillna(0.0)  # Catches 0.0/0.0, so replaces NaN with 0
                rel_err = rel_err.where(rel_err != np.inf).fillna(
                    0.0
                )  # Replaces Inf with NaN, then NaN to 0
                # If value is exactly zero, either really unlucky
                # Or inherently no error because it IS zero - assume the latter

                # Checking maximum over both tested interior state points AND observable values
                # (if observable is a vector, use element with maximum error
                check_ind = np.unravel_index(rel_err.argmax(), rel_err.shape)
                check_val = rel_err[check_ind]
                logger.info("Maximum bootstrapped error within interval: %f", check_val)
                logger.info("At point: %f", beta_vals[check_ind[0]])

    sequentialTrain = deprecate("sequentialTrain", sequential_train, "0.2.0")  # noqa: N815

    @deprecate_kwarg("B", "beta")
    def predict(self, beta):
        """
        Makes a prediction using the trained piecewise model.
        Note that the function will not produce output if asked to extrapolate outside
        the range it was trained on.
        """
        # Make sure we've done some training
        if len(self.states) == 0:
            msg = "Must train before predicting"
            raise ValueError(msg)

        # For each state point in beta, select a piecewise model to use
        if "val" in self.states[0].data.xv.dims:
            predict_vals = np.zeros((len(beta), self.states[0].data.xv["val"].size))
        else:
            predict_vals = np.zeros(len(beta))

        for i, beta_val in enumerate(beta):
            # Check if out of lower bound
            if beta_val < self.edge_beta[0]:
                msg = f"""\
                Have provided point {beta_val:f} below interpolation function
                interval edges ({self.edge_beta!s}).
                """
                raise IndexError(msg)

            # Check if out of upper bound
            if beta_val > self.edge_beta[-1]:
                msg = f"""\
                Have provided point {beta_val:f} above interpolation function
                interval edges ({self.edge_beta!s}).
                """
                raise IndexError(msg)

            # Get indices for bracketing state points
            low_ind = np.where(self.edge_beta <= beta_val)[0][-1]
            try:
                hi_ind = np.where(self.edge_beta > beta_val)[0][0]
            except IndexError:
                # With above logic, must have beta_val = self.edge_beta[-1]
                # Which would make low_ind = len(self.edge_beta)-1
                # Shift interval down
                low_ind -= 1
                hi_ind = len(self.edge_beta) - 1

            # Create interpolation object and predict
            this_model = self.model_cls((self.states[low_ind], self.states[hi_ind]))
            predict_vals[i] = this_model.predict(beta_val, order=self.max_order)

        return predict_vals

    @deprecate_kwarg("doPlot", "do_plot")
    def check_poly_consistency(self, do_plot=False):  # noqa: PLR0914, PLR0915
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
        from scipy import stats

        if do_plot:
            plt = _get_plt()

        if self.model_cls != InterpModel:
            msg = "Incorrect class provided. Can only check polynomial consistency with a polynomial interpolation model class."
            raise TypeError(msg)

        if len(self.states) == 0:
            msg = "No model parameters found. Must train model before checking consistency."
            raise ValueError(msg)

        if len(self.states) == 2:
            msg = "Single interpolation region. No point in checking consistency."
            raise ValueError(msg)

        # Need to subdivide the full interval into pairs of neighboring intervals
        # Easiest way is to take state point edge values in sliding sets of three
        all_inds = np.arange(self.edge_beta.shape[0])
        nrows = all_inds.size - 3 + 1
        n = all_inds.strides[0]
        edge_sets = np.lib.stride_tricks.as_strided(
            all_inds, shape=(nrows, 3), strides=(n, n)
        )

        # Will record and return p-values from hypothesis tests
        all_pvals = []

        # Before loop, set up plot if wanted
        if do_plot:
            pcolors = plt.cm.cividis(np.linspace(0.0, 1.0, len(edge_sets)))
            pfig, pax = plt.subplots()
            plotymin = 1e10
            plotymax = -1e10

        # Loop over sets of three edges
        for i, aset in enumerate(edge_sets):
            reg1model = self.model_cls((self.states[aset[0]], self.states[aset[1]]))
            reg1coeffs = reg1model.coefs(order=self.max_order)
            reg1err = (
                reg1model.resample(sampler={"nrep": 100})
                .coefs(order=self.max_order)
                .std("rep")
            )
            reg2model = self.model_cls((self.states[aset[1]], self.states[aset[2]]))
            reg2coeffs = reg2model.coefs(order=self.max_order)
            reg2err = (
                reg2model.resample(sampler={"nrep": 100})
                .coefs(order=self.max_order)
                .std("rep")
            )
            z12 = (reg1coeffs - reg2coeffs) / np.sqrt(reg1err**2 + reg2err**2)
            # Assuming Gaussian distributions for coefficients
            # This is implicit in returning bootstrap standard deviation as estimate of uncertainty
            # If DON'T want to assume this, bootstrap function should return confidence intervals
            # And that will require a good bit of re-coding throughout this whole class
            # p12 = 2.0*stats.norm.cdf(-abs(z12)) #Null hypothesis that coefficients same
            p12 = stats.norm.cdf(abs(z12)) - stats.norm.cdf(
                -abs(z12)
            )  # Null hypothesis coefficients different

            # To check full interval, must retrain model with data
            fullmodel = self.model_cls((self.states[aset[0]], self.states[aset[2]]))
            fullcoeffs = fullmodel.coefs(order=self.max_order)
            fullerr = (
                fullmodel.resample(sampler={"nrep": 100})
                .coefs(order=self.max_order)
                .std("rep")
            )
            z1full = (reg1coeffs - fullcoeffs) / np.sqrt(reg1err**2 + fullerr**2)
            # p1full = 2.0*stats.norm.cdf(-abs(z1full))
            p1full = stats.norm.cdf(abs(z1full)) - stats.norm.cdf(-abs(z1full))
            z2full = (reg2coeffs - fullcoeffs) / np.sqrt(reg2err**2 + fullerr**2)
            # p2full = 2.0*stats.norm.cdf(-abs(z2full))
            p2full = stats.norm.cdf(abs(z2full)) - stats.norm.cdf(-abs(z2full))

            all_pvals.append(np.vstack((p12, p1full, p2full)))
            logger.info(
                # f"Interval with edges {self.edge_beta[aset]!s} (indices {aset!s}):"
                "Interval with edges %s (indices %s):",
                self.edge_beta[aset],
                aset,
            )
            logger.info("P-values between regions: %s", p12)
            logger.info("P-values for full and 1 : %s", p1full)
            logger.info("P-values for full and 2 : %s", p2full)

            if do_plot:
                plotpoints = np.linspace(
                    self.edge_beta[aset[0]], self.edge_beta[aset[2]], 50
                )
                plotfull = np.polynomial.polynomial.polyval(plotpoints, fullcoeffs)
                plotreg1 = np.polynomial.polynomial.polyval(plotpoints, reg1coeffs)
                plotreg2 = np.polynomial.polynomial.polyval(plotpoints, reg2coeffs)
                pax.plot(plotpoints, plotfull, color=pcolors[i], linestyle="-")
                pax.plot(plotpoints, plotreg1, color=pcolors[i], linestyle=":")
                pax.plot(plotpoints, plotreg2, color=pcolors[i], linestyle="--")
                allploty = np.hstack((plotfull, plotreg1, plotreg2))
                plotymin = min(np.min(allploty), plotymin)
                plotymax = max(np.max(allploty), plotymax)

        if do_plot:
            for edge in self.edge_beta:
                pax.plot([edge] * 2, [plotymin, plotymax], "k-")
            pax.set_xlabel(r"$\beta$")
            pax.set_ylabel(r"$\langle x \rangle$")
            pfig.tight_layout()
            plt.show()

        return all_pvals

    checkPolynomialConsistency = deprecate(  # noqa: N815
        "checkPolynomialConsistency", check_poly_consistency, "0.2.0"
    )

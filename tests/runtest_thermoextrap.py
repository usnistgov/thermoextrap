# Need to set this code up with very specific tests
# So set random number seed, run things through and make sure matches every time
# Think about what exactly to test here
# Should include exactly testing IG model as well as other functions using this model
# Importantly, need to determine what goes here versus in the tutorial.
# There will likely be overlap between the two, but this should really be the test.

import numpy as np

from thermoextrap import *

# Set random number seed
# Will actually need to re-do for each test, otherwise if skip test will change results
np.random.seed(42)

# Set some output options
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

# Define some conditions to test at
betaVals = np.array([1.0, 10.0])
betaCheck = np.array([2.0, 5.0, 8.0])
lVals = np.array([1.0, 10.0])
lCheck = np.array([2.0, 5.0, 8.0])
checkOrderExt = 2
checkOrderInt = 1

################## Testing Ideal Gas Model ########################
print("\n\nTesting Ideal Gas Model")
m = IGmodel(nParticles=1000)

# Generate data for a number of scenarios and check first two moments against analytical
for B in betaVals:
    for L in lVals:
        xdat, udat = m.genData(B, nConfigs=1000, L=L)
        xConfig = m.sampleX(B, m.nP * 1000, L=L)
        print("\tB=%f, L=%f" % (B, L))
        print(
            "\t\tSample vs Analytic <x>:    %f vs %f"
            % (np.average(xdat), m.avgX(B, L=L))
        )
        print(
            "\t\tSample vs Analytic var(x): %f vs %f"
            % (np.var(xConfig, ddof=1), m.varX(B, L=L))
        )
        print(
            "\t\tSample vs Analytic <U>:    %f vs %f"
            % (np.average(udat), m.nP * m.avgX(B, L=L))
        )
        print(
            "\t\tSample vs Analytic var(U): %f vs %f"
            % (np.var(udat, ddof=1), m.nP * m.varX(B, L=L))
        )

# And generate some data to use with other tests
xdat1, udat1 = m.genData(betaVals[0], nConfigs=10000, L=lVals[0])
wdat1 = -betaVals[0] * m.nP * xdat1  # For ideal gas, virial is -N*<x>
xdat2, udat2 = m.genData(betaVals[1], nConfigs=10000, L=lVals[0])
wdat2 = -betaVals[1] * m.nP * xdat2
xdat3, udat3 = m.genData(betaVals[0], nConfigs=10000, L=lVals[1])
wdat3 = -betaVals[0] * m.nP * xdat3
xdat4, udat4 = m.genData(betaVals[1], nConfigs=10000, L=lVals[1])
wdat4 = -betaVals[1] * m.nP * xdat4


################## Testing Extrapolation Routines ######################
np.random.seed(42)

print("\n\nTesting Extrapolation Routines")
analyticTvals, analyticTderivs = m.extrapAnalytic(betaCheck, betaVals[0], checkOrderExt)
analyticVvals, analyticVderivs = m.extrapAnalyticVolume(lCheck, lVals[0], checkOrderExt)

print("\n\tIn temperature for IG:")
print("\t\tAnalytical values:       ", analyticTvals)
print("\t\tAnalytical derivatives:  ", analyticTderivs)
# Test with 1D data
extTmodel = ExtrapModel()
extTparams = extTmodel.train(refB=betaVals[0], xData=xdat1, uData=udat1)
extTpredict = extTmodel.predict(betaCheck, order=checkOrderExt)
extTparamErr = extTmodel.bootstrap(None, order=checkOrderExt)
extTpredictErr = extTmodel.bootstrap(betaCheck, order=checkOrderExt)
print("\t\t1D data:")
print("\t\tEstimated values:        \n", extTpredict)
print("\t\tEstimated derivatives:   \n", extTparams)
print("\t\tBootstrapped std values: \n", extTpredictErr)
print("\t\tBootstrapped std derivs: \n", extTparamErr)
# Test with 3D data by replicating
# Simultaneously test other features by initiating model in different way
np.random.seed(42)

extTmodel = ExtrapModel(
    maxOrder=checkOrderExt,
    refB=betaVals[0],
    xData=np.tile(np.reshape(xdat1, (xdat1.shape[0], 1)), (1, 3)),
    uData=udat1,
)
extTpredict = extTmodel.predict(betaCheck)
extTparamErr = extTmodel.bootstrap(None)
extTpredictErr = extTmodel.bootstrap(betaCheck)
print("\t\t3D data:")
print("\t\tEstimated values:        \n", extTpredict)
print("\t\tEstimated derivatives:   \n", extTmodel.params)
print("\t\tBootstrapped std values: \n", extTpredictErr)
print("\t\tBootstrapped std derivs: \n", extTparamErr)
# Check perturbation, too
# Do not look at parameters or error, though
# For this class, parameters are the data points
np.random.seed(42)

pertTmodel = PerturbModel(refB=betaVals[0], xData=xdat1, uData=udat1)
pertTpredict = pertTmodel.predict(betaCheck)
pertTpredictErr = pertTmodel.bootstrap(betaCheck)
print("\t\tPerturbation:")
print("\t\tEstimated values:        \n", pertTpredict)
print("\t\tBootstrapped std values: \n", pertTpredictErr)

# Probably don't need to check this anymore, right?
# If can check against the analytical results of the ideal gas, why do we need extra code?
# ext1, extderivs1 = extrapWithSamples(bcheck, betavals[0], xdat1, udat1, checkorder)
# ext2, extderivs2 = extrapWithSamples(bcheck, betavals[1], xdat2, udat2, checkorder)
# pertval = perturbWithSamples(bcheck, betavals[0], xdat1, udat1, useMBAR=False)

np.random.seed(42)

print("\n\tIn volume for IG:")
print("\t\tAnalytical values:       \n", analyticVvals)
print("\t\tAnalytical derivatives:  \n", analyticVderivs)
extVmodel = VolumeExtrapModel(
    maxOrder=checkOrderExt, refB=lVals[0], xData=xdat1, uData=wdat1
)
print(extVmodel.params)
# Need to correct for dimensionality (1D instead of 3D)
extVmodel.params[1, 0] *= 3.0
# And add correction term, which for ideal gas is just <x> / L
extVmodel.params[1, 0] += extVmodel.params[0, 0] / lVals[0]
extVpredict = extVmodel.predict(lCheck)
extVparamErr = extVmodel.bootstrap(None)
extVpredictErr = extVmodel.bootstrap(lCheck)
print("\t\tEstimated values:        \n", extVpredict)
print("\t\tEstimated derivatives:   \n", extVmodel.params)
print("\t\tBootstrapped std values: \n", extVpredictErr)
print("\t\tBootstrapped std derivs: \n", extVparamErr)


################## Testing Interpolation Routines ######################
np.random.seed(42)

print("\n\nTesting Interpolation Routines")
analyticTvals2, analyticTderivs2 = m.extrapAnalytic(
    betaCheck, betaVals[1], checkOrderExt
)
analyticVvals2, analyticVderivs2 = m.extrapAnalyticVolume(
    lCheck, lVals[1], checkOrderExt
)

print("\n\tIn temperature for IG:")
print("\t\tAnalytical values (1):       ", analyticTvals)
print("\t\tAnalytical derivatives (1):  ", analyticTderivs)
print("\t\tAnalytical values (2):       ", analyticTvals2)
print("\t\tAnalytical derivatives (2):  ", analyticTderivs2)
# Weighted extrapolation
intTmodel = ExtrapWeightedModel(
    maxOrder=checkOrderInt,
    refB=betaVals,
    xData=np.array([xdat1, xdat2]),
    uData=np.array([udat1, udat2]),
)
intTpredict = intTmodel.predict(betaCheck)
intTparamErr = intTmodel.bootstrap(None)
intTpredictErr = intTmodel.bootstrap(betaCheck)
print("\t\tWeighted extrapolation:")
print("\t\tEstimated values:        \n", intTpredict)
print("\t\tEstimated derivatives:   \n", intTmodel.params)
print("\t\tBootstrapped std values: \n", intTpredictErr)
print("\t\tBootstrapped std derivs: \n", intTparamErr)
# Polynomial interpolation
np.random.seed(42)

intTmodel = InterpModel(
    maxOrder=checkOrderInt,
    refB=betaVals,
    xData=np.array([xdat1, xdat2]),
    uData=np.array([udat1, udat2]),
)
intTpredict = intTmodel.predict(betaCheck)
intTparamErr = intTmodel.bootstrap(None)
intTpredictErr = intTmodel.bootstrap(betaCheck)
print("\t\tPolynomial interpolation:")
print("\t\tEstimated values:        \n", intTpredict)
print("\t\tEstimated coefficients:  \n", intTmodel.params)
print("\t\tBootstrapped std values: \n", intTpredictErr)
print("\t\tBootstrapped std coeffs: \n", intTparamErr)
# MBAR
np.random.seed(42)

try:
    # Similar to perturbation, don't check parameters or their bootstrapped error
    with np.errstate(invalid="ignore"):
        intTmodel = MBARModel(
            refB=betaVals,
            xData=np.array([xdat1, xdat2]),
            uData=np.array([udat1, udat2]),
        )
        intTpredict = intTmodel.predict(betaCheck)
        intTpredictErr = intTmodel.bootstrap(betaCheck)
        print("\t\tMBAR:")
        print("\t\tEstimated values:        \n", intTpredict)
        print("\t\tBootstrapped std values: \n", intTpredictErr)
except NameError:
    print("pymbar not found, skipping MBAR testing")


# Again, shouldn't need these anymore?
# extW, extWderivs1, extWderivs2 = extrapWeighted(bcheck, betavals[0], betavals[1], xdat1, xdat2, udat1, udat2, checkorder, checkorder)
# intval, intcoeffs = interpPolyMultiPoint(bcheck, betavals, np.array([xdat1, xdat2]),
#                                         np.array([udat1, udat2]), checkorder)

np.random.seed(42)

print("\n\tIn volume for IG:")
print("\t\tAnalytical values (1):       ", analyticVvals)
print("\t\tAnalytical derivatives (1):  ", analyticVderivs)
print("\t\tAnalytical values (2):       ", analyticVvals2)
print("\t\tAnalytical derivatives (2):  ", analyticVderivs2)
# Weighted extrapolation
intVmodel = VolumeExtrapWeightedModel(
    maxOrder=checkOrderInt,
    refB=lVals,
    xData=np.array([xdat1, xdat3]),
    uData=np.array([wdat1, wdat3]),
)
# Need to correct for dimensionality
intVmodel.params[:, 1, 0] *= 3.0
# And add correction term, which for ideal gas is just <x> / L
intVmodel.params[:, 1, 0] += intVmodel.params[:, 0, 0] / lVals
intVpredict = intVmodel.predict(lCheck)
intVparamErr = intVmodel.bootstrap(None)
intVpredictErr = intVmodel.bootstrap(lCheck)
print("\t\tWeighted extrapolation:")
print("\t\tEstimated values:        \n", intVpredict)
print("\t\tEstimated derivatives:   \n", intVmodel.params)
print("\t\tBootstrapped std values: \n", intVpredictErr)
print("\t\tBootstrapped std derivs: \n", intVparamErr)
# Polynomial interpolation
# np.random.seed(42)
#
# Can't actually check for ideal gas unless modify dimensionality in calcDerivs
# intVmodel = VolumeInterpModel(maxOrder=checkOrderInt, refB=lVals,
#                              xData=np.array([xdat1, xdat3]),
#                              uData=np.array([wdat1, wdat3]))
# intVpredict = intVmodel.predict(lCheck)
# intVparamErr = intVmodel.bootstrap(None)
# intVpredictErr = intVmodel.bootstrap(lCheck)
# print('\t\tPolynomial interpolation:')
# print('\t\tEstimated values:        ', intVpredict)
# print('\t\tEstimated coefficients:  ', intVmodel.params)
# print('\t\tBootstrapped std values: ', intVpredictErr)
# print('\t\tBootstrapped std coeffs: ', intVparamErr)


################## Testing Recursive Interpolation ######################
np.random.seed(42)

print("\n\nTesting Recursive Interpolation")
intTmodel = InterpModel(maxOrder=checkOrderInt)
compareFunc = m.avgX
piecewiseInterp = RecursiveInterp(
    intTmodel, betaVals, maxOrder=checkOrderInt, errTol=0.005
)
piecewiseInterp.recursiveTrain(
    betaVals[0], betaVals[1], verbose=True, doPlot=False, plotCompareFunc=compareFunc
)
pVals = piecewiseInterp.checkPolynomialConsistency(doPlot=False)

# pinterp = InterpModel(maxOrder=2)
# mbarinterp = MBARModel()
# compareFunc = m.avgX
# piecewise = RecursiveInterp(pinterp, betavals, maxOrder=1, errTol=0.01)
# piecewise.recursiveTrain(betavals[0], betavals[-1],
#                         Bavail=np.arange(betavals[0], betavals[-1]+1.0, 1.0),
#                         verbose=True, doPlot=True, plotCompareFunc=compareFunc)
# ppredict = piecewise.predict(bcheck)
# print(ppredict[:,0])
# print(compareFunc(np.array(bcheck)))
#
# print(piecewise.predict([betavals[0], betavals[-1]]))
# print(piecewise.predict([betavals[0]-0.5, betavals[0], betavals[-1]]))
# print(piecewise.predict([betavals[0], betavals[-1], betavals[-1]+1]))
#
#
# pinterppvals = piecewise.checkPolynomialConsistency(doPlot=True)

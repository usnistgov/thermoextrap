
#Need to set this code up with very specific tests
#So set random number seed, run things through and make sure matches every time
#Think about what exactly to test here
#Should include exactly testing IG model as well as other functions using this model
#Importantly, need to determine what goes here versus in the tutorial.
#There will likely be overlap between the two, but this should really be the test.

import numpy as np
from lib_extrap import *

m = IGmodel(nParticles=1000)
betavals = [1.0, 10.0]
bcheck = [2.0, 5.0, 8.0]
checkorder = 2
xdat1, udat1 = m.genData(betavals[0], nConfigs=100)
xdat2, udat2 = m.genData(betavals[1], nConfigs=100)

ext1, extderivs1 = extrapWithSamples(bcheck, betavals[0], xdat1, udat1, checkorder)
ext2, extderivs2 = extrapWithSamples(bcheck, betavals[1], xdat2, udat2, checkorder)

extmodel = ExtrapModel()
extparams1 = extmodel.train(betavals[0], xdat1, udat1, saveParams=True)
print(extparams1)
print(extderivs1)
extpredict1 = extmodel.predict(bcheck, order=checkorder)
print(extpredict1)
print(ext1)
exterr1 = extmodel.bootstrap(bcheck, order=checkorder)
print(exterr1)
extparamerr = extmodel.bootstrap(None, order=checkorder)
print(extparamerr)
ext1poly = extrapToPoly(extmodel.refB, extmodel.params[:4,0])
print(ext1poly)
ext1polyerr = bootstrapPolyCoeffs(extmodel, order=3)
print(ext1polyerr)

extmodel2 = ExtrapModel(refB=betavals[1], xData=xdat2, uData=udat2)
print(extmodel2.params)
print(extderivs2)
extpredict2 = extmodel2.predict(bcheck, order=checkorder)
print(extpredict2)
print(ext2)
ext2poly = extrapToPoly(extmodel2.refB, extmodel2.params[:4,0])
print(ext2poly)
ext2polyerr = bootstrapPolyCoeffs(extmodel2, order=3)
print(ext2polyerr)
print('\n')


extW, extWderivs1, extWderivs2 = extrapWeighted(bcheck, betavals[0], betavals[1], xdat1, xdat2, udat1, udat2, checkorder, checkorder)

extWmodel = ExtrapWeightedModel(refB=betavals, xData=np.array([xdat1, xdat2]),
                                uData=np.array([udat1, udat2]))
print(extWmodel.params)
print(extWderivs1)
print(extWderivs2)
extWpredict = extWmodel.predict(bcheck, order=checkorder)
print(extWpredict)
print(extW)
extWerr = extWmodel.bootstrap(bcheck, order=checkorder)
print(extWerr)
extWparamerr = extWmodel.bootstrap([], order=checkorder)
print(extWparamerr)
print('\n')


intval, intcoeffs = interpPolyMultiPoint(bcheck, betavals, np.array([xdat1, xdat2]),
                                         np.array([udat1, udat2]), checkorder)

intmodel = InterpModel(refB=betavals, xData=np.array([xdat1, xdat2]),
                       uData=np.array([udat1, udat2]), maxOrder=1)
print(intmodel.params)
print(intcoeffs)
intpredict = intmodel.predict(bcheck)
print(intpredict)
print(intval)
interr = intmodel.bootstrap(bcheck)
print(interr)
intparamerr = intmodel.bootstrap(None)
print(intparamerr)
print('\n')


pertval = perturbWithSamples(bcheck, betavals[0], xdat1, udat1, useMBAR=False)

pertmodel = PerturbModel(refB=betavals[0], xData=xdat1, uData=udat1)
print(pertmodel.params)
pertpredict = pertmodel.predict(bcheck)
print(pertpredict)
print(pertval)
perterr = pertmodel.bootstrap(bcheck)
print(perterr)
print('\n')


#Highly recommended to supress runtime warnings when using MBAR
with np.errstate(invalid='ignore'):
  mbarmodel = MBARModel(refB=betavals, xData=np.array([xdat1, xdat2]), 
                        uData=np.array([udat1, udat2]))
  print(mbarmodel.params)
  mbarpredict = mbarmodel.predict(bcheck)
  print(mbarpredict)
  mbarerr = mbarmodel.bootstrap(bcheck)
  print(mbarerr)
  #Can't bootstrap the MBAR object the same way we bootstrap the model parameters
  #Will throw a messy error
print('\n')


pinterp = InterpModel(maxOrder=2)
#mbarinterp = MBARModel()
compareFunc = m.avgX
piecewise = RecursiveInterp(pinterp, betavals, maxOrder=1, errTol=0.01)
piecewise.recursiveTrain(betavals[0], betavals[-1],
                         Bavail=np.arange(betavals[0], betavals[-1]+1.0, 1.0),
                         verbose=True, doPlot=True, plotCompareFunc=compareFunc)
ppredict = piecewise.predict(bcheck)
print(ppredict[:,0])
print(compareFunc(np.array(bcheck)))

print(piecewise.predict([betavals[0], betavals[-1]]))
print(piecewise.predict([betavals[0]-0.5, betavals[0], betavals[-1]]))
print(piecewise.predict([betavals[0], betavals[-1], betavals[-1]+1]))


pinterppvals = piecewise.checkPolynomialConsistency(doPlot=True)



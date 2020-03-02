#Library of functions useful for extrapolation (or interpolation)
#Makes heavy use of sympy to go to arbitrary order derivatives


import time
import copy
import numpy as np
from sympy import *
from pymbar import mbar
from scipy.stats import norm
from scipy.special import binom
from scipy.special import factorial
import matplotlib.pyplot as plt


def buildAvgFuncs(xvals, uvals, order):
  """Defines sympy functions mapping specific sympy function representations to values.
     We will let u(i) be the average of u**i and xu(i) be the average of x*(u**i). In
     other words, providing an integer to the function u or xu will produce the desired
     average quantity. Once the symbolic derivative is defined as a lambdified function
     of two sympy functions, can just input the custom sympy functions defined here to
     substitute in all the right values. Note that if the observable is vector-valued 
     the functions will return vectors for averages.
  """
  #To allow for vector-valued observables, must make sure uvals can be transposed
  uvalsT = np.array([uvals]).T

  #First define dictionaries we will use to define the functions
  #Keys will be integers representing different orders
  dictu = {}
  dictxu = {}

  for o in range(order+1):
    dictu[o] = np.average(uvals**o)
    dictxu[o] = np.average(xvals*(uvalsT**o), axis=0)

  class ufunc(Function):
    avgdict = copy.deepcopy(dictu)
    @classmethod
    def eval(cls, x):
      return cls.avgdict[x]

  class xufunc(Function):
    avgdict = copy.deepcopy(dictxu)
    @classmethod
    def eval(cls, x):
      return cls.avgdict[x]

  return (ufunc, xufunc)


def symDerivAvgX(order):
  """A function to compute the derivative to arbitrary order using symbolic logic.
     Returns a substituted string that can be substituted again to get an actual value.
  """
  #First define some consistent symbols
  b = symbols('b') #Beta or inverse temperature

  f = Function('f')(b) #Functions representing the numerator and denominator of an average
  z = Function('z')(b)

  u = Function('u') #Functions that will represent various averages
  xu = Function('xu')

  avgFunc = f / z
  thisderiv = avgFunc.diff(b, order)
  #Pick out what we want to substitute by object type
  tosub = thisderiv.atoms(Function, Derivative) 

  #When we sub in, must do in order of highest to lowest derivatives, then functions
  #Otherwise substitution doesn't work because derivatives computed recursively by sympy
  for o in range(order+1)[::-1]:
    subvals = {}

    if o == 0:
      for d in tosub:
        if isinstance(d, Function):
          if str(d) == 'f(b)':
            subvals[d] = xu(0)*z

    else:
      for d in tosub:
        if isinstance(d, Derivative) and d.derivative_count == o:
          if str(d.expr) == 'f(b)':
            subvals[d] = ((-1)**d.derivative_count)*xu(d.derivative_count)*z
          elif str(d.expr) == 'z(b)':
            subvals[d] = ((-1)**d.derivative_count)*u(d.derivative_count)*z

    #Substitute derivatives for functions u and xu at this order
    thisderiv = thisderiv.subs(subvals)

    #To allow for vector-valued function inputs and to gain speed, lambdify
    thisderiv = expand(simplify(thisderiv))

  returnfunc = lambdify((u, xu), thisderiv, "numpy")
  return returnfunc


def buildAvgFuncsDependent(xvals, uvals, order):
  """Same as buildAvgFuncs, but for an observable that explicitly depends on
     the variable we're extrapolating over. In this case, xvals should be 3D.
     The first dimension is time, just like uvals, the second dimension should
     be elements of the observable vector, and the last dimension should match
     order and be DERIVATIVES of the observable vector elements to the order of
     the index of that dimension. So the first index of zero on this dimension
     is the zeroth-order derivative, which is just the observable itself.
     As an example, if you performed Widom insertions, have that the observable
     and its derivatives are:
            x = exp(-B*dU)
        dx/dB = -dU*exp(-B*dU)
    d^2x/dB^2 = (dU^2)*exp(-B*dU)
           etc.
  """
  #Make sure have provided derivatives in observable up to desired order
  if xvals.shape[2] < order+1:
    print('Maximum provided order of derivatives of observable (%i) is less than desired order (%i).'%(xvals.shape[2]-1, order))
    print('Setting order to match.')
    order = xvals.shape[2] - 1
  elif xvals.shape[2] >= order+1:
    xvals = xvals[:,:,:order+1]

  #To allow for vector-valued observables, must make sure uvals can be transposed
  uvalsT = np.array([uvals]).T

  #First define dictionaries we will use to define the functions
  #Keys will be integers representing different orders
  dictu = {}
  dictxu = {}

  for o in range(order+1):
    dictu[o] = np.average(uvals**o)
    for j in range(order+1):
      dictxu[(j,o)] = np.average(xvals[:,:,j]*(uvalsT**o), axis=0)

  class ufunc(Function):
    avgdict = copy.deepcopy(dictu)
    @classmethod
    def eval(cls, x):
      return cls.avgdict[x]

  class xufunc(Function):
    avgdict = copy.deepcopy(dictxu)
    @classmethod
    def eval(cls, x, y):
      return cls.avgdict[(x,y)]

  return (ufunc, xufunc)


def symDerivAvgXdependent(order):
  """Same as symDerivAvgX except for one line when substituting for f(b) and its
     derivatives. Instead of substituting xu(i), it substitutes xu(i,j) so that 
     derivatives are possible with the observable depending explicitly on the
     extrapolation variable. This is meant to be used with buildAvgFuncsDependent.
  """
  #First define some consistent symbols
  b = symbols('b') #Beta or inverse temperature

  f = Function('f')(b) #Functions representing the numerator and denominator of an average
  z = Function('z')(b)

  u = Function('u') #Functions that will represent various averages
  xu = Function('xu')

  avgFunc = f / z
  thisderiv = avgFunc.diff(b, order)
  #Pick out what we want to substitute by object type
  tosub = thisderiv.atoms(Function, Derivative) 

  #When we sub in, must do in order of highest to lowest derivatives, then functions
  #Otherwise substitution doesn't work because derivatives computed recursively by sympy
  for o in range(order+1)[::-1]:
    subvals = {}

    if o == 0:
      for d in tosub:
        if isinstance(d, Function):
          if str(d) == 'f(b)':
            subvals[d] = xu(0,0)*z

    else:
      for d in tosub:
        if isinstance(d, Derivative) and d.derivative_count == o:
          if str(d.expr) == 'f(b)':
            #Instead of substituting f(k)(b) = <x*(-u^k)> = xu(k), we want to do...
            #(4th order as an example)
            #f(4)(b) = xu(4,0) - 4*xu(3,1) + 6*xu(2,2) - 4*xu(1,3) + xu(0,4)
            #        = <x(4)> - 4*<x(3)*u> + 6*<x(2)*u^2> - 4*<x(1)*u^3> + <x*u^4>
            #In the above, f(4) or x(4) represents the 4th derivtive of f or x with
            #respect to the extrapolation variable b.
            subvals[d] = Sum(((-1)**k)*binomial(d.derivative_count,k)*xu(d.derivative_count-k,k), (k,0,d.derivative_count)).doit()*z
          elif str(d.expr) == 'z(b)':
            subvals[d] = ((-1)**d.derivative_count)*u(d.derivative_count)*z

    #Substitute derivatives for functions u and xu at this order
    thisderiv = thisderiv.subs(subvals)

    #To allow for vector-valued function inputs and to gain speed, lambdify
    thisderiv = expand(simplify(thisderiv))

  returnfunc = lambdify((u, xu), thisderiv, "numpy")
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
    for l in range(k+1):
      coeffs[l] += ((-B0)**(k-l))*d*binom(k,l)/np.math.factorial(k)
  return coeffs


def bootstrapPolyCoeffs(extModel, n=100, order=3):
  """Determines uncertainty in polynomial coefficients determined from derivatives
     via extrapToPoly function. This will only reliably work if provided an
     ExtrapModel object for which extModel.train returns the derivatives and
     extModel.refB returns the reference point and the data can be resampled
     from extModel.resampleData. Might make more sense to include this in the 
     class definition, but don't want to be inherited by other classes.
  """
  bShape = (n,) + extModel.params[:order+1,:].shape
  bootStraps = np.zeros(bShape)
  for i in range(n):
    thisx, thisU = extModel.resampleData()
    thisParams = extModel.train(extModel.refB, thisx, thisU, saveParams=False)
    thisCoeffs = np.zeros(thisParams[:order+1,:].shape)
    for j in range(thisParams.shape[1]):
      thisCoeffs[:,j] = extrapToPoly(extModel.refB, thisParams[:order+1,j])
    bootStraps[i] = thisCoeffs
  bootStd = np.std(bootStraps, ddof=1, axis=0)
  return bootStd


class ExtrapModel:
  """Class to hold information about an extrapolation. This can be trained by providing
     data at the reference state and can then be evaluated to obtain estimates at
     arbitrary other states.
  """

  #Otherwise, it just needs to define some variables
  def __init__(self, maxOrder=6, refB=None, xData=None, uData=None):
    self.maxOrder = maxOrder #Maximum order to calculate derivatives
    self.derivF = self.calcDerivFuncs() #Perform sympy differentiation up to max order
    self.refB = refB
    self.x = xData
    self.U = uData
    self.params = None
    if (refB is not None) and (xData is not None) and (uData is not None):
      self.params = self.train(refB, xData, uData, saveParams=True)

  #Calculates symbolic derivatives up to maximum order given data
  #Returns list of functions that can be used to evaluate derivatives for specific data
  def calcDerivFuncs(self):
    derivs = []
    for o in range(self.maxOrder+1):
      derivs.append(symDerivAvgX(o))
    return derivs

  #And given data, calculate numerical values of derivatives up to maximum order
  #Will be very helpful when generalize to different extrapolation techniques
  #(and interpolation)
  def calcDerivVals(self, refB, x, U):
    """Calculates specific derivative values at B with data x and U up to max order.
       Returns these derivatives.
    """

    if x.shape[0] != U.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], U.shape[0]))
      return

    avgUfunc, avgXUfunc = buildAvgFuncs(x, U, self.maxOrder)
    derivVals = np.zeros((self.maxOrder+1, x.shape[1]))
    for o in range(self.maxOrder+1):
      derivVals[o] = self.derivF[o](avgUfunc, avgXUfunc)

    return derivVals

  #The below can be modified to change the extrapolation technique or to use interpolation
  def train(self, refB, xData, uData, saveParams=True):
    """This is the function used to set the parameters of the model. For extrapolation
       these are just the derivatives at the reference state. You can change this function
       for more complex parameters like the polynomial coefficients for interpolation.
       If saveParams is False, it will simply return the parameters without setting
       their values in self. This is useful for bootstrapping.
    """
    #Next need to make sure x has at least two dimensions for extrapolation
    if len(xData.shape) == 1:
      xData = np.reshape(xData, (xData.shape[0], 1))
      #Rows are independent observations, columns elements of observable x

    params = self.calcDerivVals(refB, xData, uData)

    if saveParams:
      self.refB = refB
      self.x = xData
      self.U = uData
      self.params = params

    return params

  #A function to calculate model prediction at other state points
  def predict(self, B, order=None, params=None, refB=None):
    """Like trainParams, this is a function that can be modified to create new
       extrapolation procedures. Just uses information collected from training
       to make predictions at specific state points, B, up to specified order.
       Can also specify parameters to use. If params is None, will just use
       the parameters found during training. If refB is None, will use self.refB.
    """
    #Use parameters for estimate - for extrapolation, the parameters are the derivatives
    if params is None:
      if self.params is None:
        return
      else:
        params = self.params

    if refB is None:
      if self.refB is None:
        return
      else:
        refB = self.refB

    if order is None:
      order = self.maxOrder

    #Make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
      B = [B]
    B = np.array(B)
    dBeta = (B - refB)

    predictVals = np.zeros((B.shape[0], self.x.shape[-1]))
    for o in range(order+1):
      predictVals += np.tensordot((dBeta**o), params[o], axes=0)/np.math.factorial(o)

    return predictVals

  def resampleData(self):
    """Function to resample the data, mainly for use in providing bootstrapped estimates.
       Should be adjusted to match the data structure.
    """
    if self.x is None:
      return

    sampSize = self.x.shape[0]
    randInds = np.random.choice(sampSize, size=sampSize, replace=True)
    sampX = self.x[randInds, :]
    sampU = self.U[randInds]
    return (sampX, sampU)

  #A method to obtain uncertainty estimates via bootstrapping
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

    #First make sure B isn't just a value
    if isinstance(B, (int, float)):
      B = [B]

    #If B is not a value but an empty array, set it to None
    if B is not None and len(B) == 0:
      print('No state points provided to boostrap prediction at - bootstrapping parameters.')
      B = None

    #If provided/not None make sure B is an array, even if just has one element
    if B is not None:
      B = np.array(B)
      #Last dimension should be observable vector size
      bootStraps = np.zeros((n, B.shape[0], self.x.shape[-1]))
    else:
      bShape = (n,) + self.params.shape
      bootStraps = np.zeros((bShape))

    #Loop for as many resamples as we want
    for i in range(n):
      thisx, thisU = self.resampleData()
      #Train for this resampled data, but don't alter model
      thisParams = self.train(self.refB, thisx, thisU, saveParams=False)
      if B is not None:
        #Predict the new value with the specific parameters
        bootStraps[i,:,:] = self.predict(B, order=order, params=thisParams)
      else:
        bootStraps[i] = thisParams

    #Compute uncertainty
    bootStd = np.std(bootStraps, ddof=1, axis=0)
    return bootStd


class ExtrapWeightedModel(ExtrapModel):
  """Model for extrapolation using two data sets at two different state points and
     weighting extrapolations from each with a Minkowski-like distance.
  """

  #Only need to redefine the train, predict, and resampleData functions
  def train(self, refB, xData, uData, saveParams=True):
    """This is the function used to set the parameters of the model. For extrapolation
       these are just the derivatives at the reference state. You can change this function
       for more complex parameters like the polynomial coefficients for interpolation.
       If saveParams is False, it will simply return the parameters without setting
       their values in self. This is useful for bootstrapping.
    """
    #Next need to make sure x has at least three dimensions for extrapolation
    #First should be 2, next should be number of samples in each dataset,
    #and the last should be the length of the observable vector.
    #Note that currently ragged data is not allowed, but I don't check for this!
    #(data sets at each state point must have the same number of samples)
    if (xData.shape[0] != 2) or (uData.shape[0] != 2):
      print('Must provide observable and potential energy data from 2 state points!')
      print('First dimensions of provided data are not 2 but %i and %i'%(xData.shape[0], uData.shape[0]))
      return

    if isinstance(refB, (int, float)):
      print('Must provide 2 reference beta values as a list or array, but got only a number.')
    refB = np.array(refB)
    if refB.shape[0] != 2:
      print('Need exactly 2 reference beta values, but got %i'%refB.shape[0])

    if len(xData.shape) == 2:
      xData = np.reshape(xData, (xData.shape[0], xData.shape[1], 1))
      #Rows are independent observations, columns elements of observable x

    params1 = self.calcDerivVals(refB[0], xData[0], uData[0])
    params2 = self.calcDerivVals(refB[1], xData[1], uData[1])
    params = np.array([params1, params2])

    if saveParams:
      self.refB = refB
      self.x = xData
      self.U = uData
      self.params = params

    return params

  #A function to calculate model prediction at other state points
  def predict(self, B, order=None, params=None, refB=None):
    """Like trainParams, this is a function that can be modified to create new
       extrapolation procedures. Just uses information collected from training
       to make predictions at specific state points, B, up to specified order.
       Can also specify parameters to use. If params is None, will just use
       the parameters found during training. If refB is None, will just use
       self.refB.
    """
    def weightsMinkowski(d1, d2, m=20):
      w1 = 1.0 - (d1**m) / ((d1**m)+ (d2**m))
      w2 = 1.0 - (d2**m) / ((d1**m)+ (d2**m))
      return [w1, w2]

    #Use parameters for estimate - for extrapolation, the parameters are the derivatives
    if params is None:
      if self.params is None:
        return
      else:
        params = self.params

    if refB is None:
      if self.refB is None:
        return
      else:
        refB = self.refB

    if order is None:
      order = self.maxOrder

    #Make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
      B = [B]
    B = np.array(B)
    dBeta = np.zeros((2, B.shape[0]))
    dBeta[0] = (B - refB[0])
    dBeta[1] = (B - refB[1])

    #Perform extrapolation from both reference points
    predictVals = np.zeros((2, B.shape[0], self.x.shape[-1]))
    for o in range(order+1):
      predictVals[0] += np.tensordot((dBeta[0]**o), params[0,o], axes=0)/np.math.factorial(o)
      predictVals[1] += np.tensordot((dBeta[1]**o), params[1,o], axes=0)/np.math.factorial(o)

    w1, w2 = weightsMinkowski(abs(dBeta[0]), abs(dBeta[1]))

    #Transpose to get right multiplication (each row of exti is different beta)
    w1T = np.array([w1]).T
    w2T = np.array([w2]).T
    outVals = (predictVals[0]*w1T + predictVals[1]*w2T) / (w1T+w2T)

    return outVals

  def resampleData(self):
    """Function to resample the data, mainly for use in providing bootstrapped estimates.
       Should be adjusted to match the data structure.
    """
    if self.x is None:
      return

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

  #Only need to redefine the train, predict, and resampleData functions
  def train(self, refB, xData, uData, saveParams=True):
    """This is the function used to set the parameters of the model. For extrapolation
       these are just the derivatives at the reference state. You can change this function
       for more complex parameters like the polynomial coefficients for interpolation.
       If saveParams is False, it will simply return the parameters without setting
       their values in self. This is useful for bootstrapping.
    """
    refB = np.array(refB)

    if xData.shape[0] != uData.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(xData.shape[0], uData.shape[0]))
      return

    if (xData.shape[0] != refB.shape[0]) or (uData.shape[0] != refB.shape[0]):
      print('First dimension of data must match number of provided beta values.')
      return

    #Want to be able to handle vector-value observables
    #So make sure x has 3 dimensions, even if technically observable is scalar
    #Note that currently ragged data is not allowed, but I don't check for this!
    #(data sets at each state point must have the same number of samples)
    if len(xData.shape) == 2:
      xData = np.reshape(xData, (xData.shape[0], xData.shape[1], 1))

    #Define the order of the polynomial wer're going to compute
    order = self.maxOrder
    pOrder = refB.shape[0]*(order+1) - 1 #Also the number of coefficients we solve for minus 1

    #Need to put together systems of equations to solve
    #Will have to solve one system for each component of a vector-valued observable 
    #Fortunately, matrix to invert same for each value of beta regardless of observable
    #Just the values we want the polynomial to match with (derivVals) will be different
    derivVals = np.zeros((pOrder+1, xData.shape[2]))
    mat = np.zeros((pOrder+1, pOrder+1))

    #Loop to get all values and derivatives at each point up to desired order
    for i, beta in enumerate(refB):
      thisderivs = self.calcDerivVals(beta, xData[i], uData[i])

      #Loop over observable elements, with unique derivatives for each
      for j in range(xData.shape[2]):
        derivVals[(order+1)*i:(order+1)*(i+1), j] = thisderivs[:,j]

      #Loop over orders, filling out matrix for solving systems of equations
      for j in range(order+1):
        #Suppress warnings about divide by zero since we actually want this to return infinity
        with np.errstate(divide='ignore'):
          mat[((order+1)*i)+j, :] = ( ((np.ones(pOrder+1)*beta)**(np.arange(pOrder+1) - j))
                                    * factorial(np.arange(pOrder+1))/factorial(np.arange(pOrder+1)-j) )

    #The above formula works everywhere except where the matrix should have zeros
    #Instead of zeros, it inserts infinity, so fix this
    #(apparently scipy's factorial returns 0 for negatives)
    mat[np.isinf(mat)] = 0.0

    #And solve system of equations for the polynomial coefficients of each observable element
    matInv = np.linalg.inv(mat)
    coeffs = np.zeros((pOrder+1, xData.shape[2]))
    for j in range(xData.shape[2]):
      coeffs[:,j] = np.dot(matInv, derivVals[:,j])

    if saveParams:
      self.refB = refB
      self.x = xData
      self.U = uData
      self.params = coeffs

    return coeffs

  #A function to calculate model prediction at other state points
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
    #Use parameters for estimate
    if params is None:
      if self.params is None:
        return
      else:
        params = self.params

    if order is None:
      order = self.maxOrder

    #Make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
      B = [B]
    B = np.array(B)

    #Infer polynomial order from parameters (which are polynomial coefficients)
    pOrder = len(params) - 1

    #Calculate the polynomial interpolation values at each desired beta
    outvals = np.zeros((len(B), self.x.shape[-1])) #Each row is a different beta value
    for i, beta in enumerate(B):
      betaPower = beta**(np.arange(pOrder+1))
      betaPower = np.array([betaPower]).T
      outvals[i] = np.sum(params*betaPower, axis=0)

    return outvals

  def resampleData(self):
    """Function to resample the data, mainly for use in providing bootstrapped estimates.
       Should be adjusted to match the data structure.
    """
    if self.x is None:
      return

    sampX = np.zeros(self.x.shape)
    sampU = np.zeros(self.U.shape)

    for i in range(self.x.shape[0]):
      sampSize = self.x[i].shape[0]
      randInds = np.random.choice(sampSize, size=sampSize, replace=True)
      sampX[i] = self.x[i, randInds, :]
      sampU[i] = self.U[i, randInds]

    return (sampX, sampU)


class MBARModel(InterpModel):
  """Very similar to interpolation model so inheriting this class.
     Must also have at least two reference states and will use as many as 
     provided to make estimate. Resampling will be the same, just need to
     change the train and predict functions.
  """

  def train(self, refB, xData, uData, saveParams=True):
    """Trains and returns a pymbar MBAR object as the model "parameters."
    """
    refB = np.array(refB)

    if xData.shape[0] != uData.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(xData.shape[0], uData.shape[0]))
      return

    if (xData.shape[0] != refB.shape[0]) or (uData.shape[0] != refB.shape[0]):
      print('First dimension of data must match number of provided beta values.')
      return

    #Want to be able to handle vector-value observables
    #So make sure x has 3 dimensions, even if technically observable is scalar
    #Note that currently ragged data is not allowed, but I don't check for this!
    #(data sets at each state point must have the same number of samples)
    if len(xData.shape) == 2:
      xData = np.reshape(xData, (xData.shape[0], xData.shape[1], 1))

    #Remember, no ragged data, otherwise the below won't work right
    allN = np.ones(xData.shape[0])*xData.shape[1]
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
    #Check if have parameters
    if params is None:
      #Use trained parameters if you have them
      if self.params is None:
        return
      else:
        params = self.params

    #Make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
      B = [B]
    B = np.array(B)

    allU = self.U.flatten()
    predictVals = np.zeros((len(B), self.x.shape[2]))
    x = np.reshape(self.x, (self.x.shape[0]*self.x.shape[1], self.x.shape[2]))

    for i in range(len(B)):
      predictVals[i,:] = params.computeMultipleExpectations(x.T, B[i]*allU)[0]

    return predictVals


class PerturbModel:
  """Class to hold information about a perturbation. 
  """

  #Otherwise, it just needs to define some variables
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
    #Next need to make sure x has at least two dimensions for extrapolation
    if len(xData.shape) == 1:
      xData = np.reshape(xData, (xData.shape[0], 1))
      #Rows are independent observations, columns elements of observable x

    #Also check if observable data matches up with potential energy
    if xData.shape[0] != uData.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(xData.shape[0], uData.shape[0]))
      return

    params = [xData, uData]

    if saveParams:
      self.refB = refB
      self.x = xData
      self.U = uData
      self.params = params

    return params

  #A function to calculate model prediction at other state points
  def predict(self, B, params=None, refB=None, useMBAR=False):
    """Performs perturbation at state of interest.
    """
    #Check if have parameters
    if params is None:
      #Use trained parameters if you have them
      if self.params is None:
        return
      else:
        params = self.params

    if refB is None:
      if self.refB is None:
        return
      else:
        refB = self.refB

    #Specify "parameters" as desired data to use
    x = params[0]
    U = params[1]

    #Make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
      B = [B]
    B = np.array(B)

    if useMBAR:
      mbarObj = mbar.MBAR(np.array([refB*U]), [U.shape[0]])
      predictVals = np.zeros((len(B), x.shape[1]))
      for i in range(len(B)):
        predictVals[i,:] = mbarObj.computeMultipleExpectations(x.T, B[i]*U)[0]

    else:
      #Compute what goes in the exponent and subtract out the maximum
      #Don't need to bother storing for later because compute ratio
      dBeta = B - refB
      dBetaU = (-1.0)*np.tensordot(dBeta, U, axes=0)
      dBetaUdiff = dBetaU - np.array([np.max(dBetaU, axis=1)]).T
      expVals = np.exp(dBetaUdiff)

      #And perform averaging
      numer = np.dot(expVals, x) / float(x.shape[0])
      denom = np.average(expVals, axis=1)
      predictVals = numer / np.array([denom]).T

    return predictVals

  def resampleData(self):
    """Function to resample the data, mainly for use in providing bootstrapped estimates.
       Should be adjusted to match the data structure.
    """
    if self.x is None:
      return

    sampSize = self.x.shape[0]
    randInds = np.random.choice(sampSize, size=sampSize, replace=True)
    sampX = self.x[randInds, :]
    sampU = self.U[randInds]
    return (sampX, sampU)

  #A method to obtain uncertainty estimates via bootstrapping
  def bootstrap(self, B, n=100, useMBAR=False):
    """Obtain estimates of uncertainty in model predictions via bootstrapping.
    """
    if self.params is None:
      return

    #Make sure B is an array, even if just has one element
    if isinstance(B, (int, float)):
      B = [B]
    B = np.array(B)

    #Last dimension should be observable vector size
    bootStraps = np.zeros((n, B.shape[0], self.x.shape[-1])) 

    #Loop for as many resamples as we want
    for i in range(n):
      thisx, thisU = self.resampleData()
      #"Train", which here just packages the data, but don't change model params
      thisParams = self.train(self.refB, thisx, thisU, saveParams=False)
      #Predict the new value with the resampled data
      bootStraps[i,:,:] = self.predict(B, params=thisParams, useMBAR=useMBAR)

    #Compute uncertainty
    bootStd = np.std(bootStraps, ddof=1, axis=0)
    return bootStd


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
    print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], U.shape[0]))
    return

  #Next need to make sure x has at least two dimensions
  if len(x.shape) == 1:
    x = np.reshape(x, (x.shape[0], 1)) #Rows are independent observations, columns x vectors

  #And make sure B is an array, even if just has one element
  if isinstance(B, (int, float)):
    B = [B]
  B = np.array(B)
  dBeta = (B - B0)

  outvec = np.zeros((order+1, x.shape[1])) #kth order derivative on kth row
  outval = np.zeros((B.shape[0], x.shape[1])) #each row is extrapolation to different beta

  #Get functions defining averages of observable, potential energy, etc.
  avgUfunc, avgXUfunc = buildAvgFuncs(x, U, order)

  #Loop over orders
  for o in range(order+1):
    #Get derivative function at this order
    oDeriv = symDerivAvgX(o)
    #Evaluate it using the appropriate mappings to averages
    #MUST provide average U then XU because of how symDerivAvgX returns
    outvec[o] = oDeriv(avgUfunc, avgXUfunc)
    #Perform extrapolation using same deriatives and averages, just have many dBeta
    #Taking the tensor product of two (really should be) 1D arrays to get the right shape
    outval += np.tensordot((dBeta**o), outvec[o], axes=0) / np.math.factorial(o)

  return (outval, outvec)


def extrapWeighted(B, refB1, refB2, x1, x2, u1, u2, order1, order2, m=20):
  """Performs extrapolation from two points to an interior point and weights with a 
     Minkowski-like function proposed by Mahynski, Errington, and Shen (2017).
  """
  def weightsMinkowski(d1, d2, m=20):
    w1 = 1.0 - (d1**m) / ((d1**m)+ (d2**m))
    w2 = 1.0 - (d2**m) / ((d1**m)+ (d2**m))
    return [w1, w2]

  ext1, derivs1 = extrapWithSamples(B, refB1, x1, u1, order1)
  ext2, derivs2 = extrapWithSamples(B, refB2, x2, u2, order2)

  #Make sure B is an array to handle case if it is
  #Also ensures w1 and w2 can be multiplied by the extrapolations correctly
  if isinstance(B, (int, float)):
    B = [B]
  B = np.array(B)

  w1, w2 = weightsMinkowski(abs(refB1-B), abs(refB2-B), m=m)

  #Transpose to get right multiplication (each row of exti is different beta)
  w1T = np.array([w1]).T
  w2T = np.array([w2]).T
  outval = (ext1*w1T + ext2*w2T) / (w1T+w2T)

  return (outval, derivs1, derivs2)


def interpPolyMultiPoint(B, refB, x, U, order):
  """refB is an array of beta values of at least length 2.
     x and U should be arrays for data and potential energy at each beta value
     (so their first dimension should be the same as refB).
     B are the beta values at which to interpolate using a polynomial.
     order is the maximum order derivative used at each point where data is provided.
     Returns polynomial values at specified betas and polynomial coefficients.
  """
  from scipy.special import factorial

  refB = np.array(refB)

  if x.shape[0] != U.shape[0]:
    print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], U.shape[0]))
    return

  if (x.shape[0] != refB.shape[0]) or (U.shape[0] != refB.shape[0]):
    print('First dimension of data must match number of provided beta values.')
    return

  #Want to be able to handle vector-value observables
  #So make sure x has 3 dimensions, even if technically observable is scalar
  if len(x.shape) == 2:
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

  #While we're at it, also make B into an array if it isn't, just for convenience
  if isinstance(B, (int, float)):
    B = [B]
  B = np.array(B)

  #Define the order of the polynomial wer're going to compute
  pOrder = refB.shape[0]*(order+1) - 1 #Also the number of coefficients we solve for minus 1

  #Need to put together systems of equations to solve
  #Will have to solve one system for each component of a vector-valued observable 
  #Fortunately, matrix to invert will be same for each value of beta regardless of observable
  #Just the values we want the polynomial to match with (derivVals) will be different
  derivVals = np.zeros((pOrder+1, x.shape[2]))
  mat = np.zeros((pOrder+1, pOrder+1))

  #Loop to get all values and derivatives at each point up to desired order
  for i, beta in enumerate(refB):
    #Just need derivatives, which is essentially same cost as computing extrapolation
    #But don't care about what point we extrapolate to or the value we get
    thisext, thisderivs = extrapWithSamples(np.average(refB), beta, x[i], U[i], order)

    #Loop over observable elements, with unique derivatives for each
    for j in range(x.shape[2]):
      derivVals[(order+1)*i:(order+1)*(i+1), j] = thisderivs[:,j]

    #Loop over orders, filling out matrix for solving systems of equations
    for j in range(order+1):
      #Suppress warnings about divide by zero since we actually want this to return infinity
      with np.errstate(divide='ignore'):
        mat[((order+1)*i)+j, :] = ( ((np.ones(pOrder+1)*beta)**(np.arange(pOrder+1) - j))
                                    * factorial(np.arange(pOrder+1))/factorial(np.arange(pOrder+1)-j) )

  #The above formula works everywhere except where the matrix should have zeros
  #Instead of zeros, it inserts infinity, so fix this
  #(apparently scipy's factorial returns 0 for negatives)
  mat[np.isinf(mat)] = 0.0

  #And solve a system of equations for the polynomial coefficients of each observable element
  matInv = np.linalg.inv(mat)
  coeffs = np.zeros((pOrder+1, x.shape[2]))
  for j in range(x.shape[2]):
    coeffs[:,j] = np.dot(matInv, derivVals[:,j])

  #Calculate the polynomial interpolation values at each desired beta
  outvals = np.zeros((len(B), x.shape[2])) #Each row is a different beta value
  for i, beta in enumerate(B):
    betaPower = beta**(np.arange(pOrder+1))
    betaPower = np.array([betaPower]).T
    outvals[i] = np.sum(coeffs*betaPower, axis=0)

  return (outvals, coeffs)


def perturbWithSamples(B, refB, x, U, useMBAR=False):
  """Computes observable x (can be a vector) at a set of perturbed temperatures
     of B (array) from the original refB using potential energies at each config
     and standard reweighting. Uses MBAR code instead of mine if desired.
  """
  if x.shape[0] != U.shape[0]:
    print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], U.shape[0]))
    return

  #Check shape of observables and add dimension if needed
  #Note that for observables with more than 1 dimension, things won't work
  if len(x.shape) == 1:
    x = np.array([x]).T

  #While we're at it, also make B into an array if it isn't, just for convenience
  if isinstance(B, (int, float)):
    B = [B]
  B = np.array(B)

  if useMBAR:
    from pymbar import mbar
    mbarObj = mbar.MBAR(np.array([refB*U]), [U.shape[0]])
    outval = np.zeros((len(B), x.shape[1]))
    for i in range(len(B)):
      outval[i,:] = mbarObj.computeMultipleExpectations(x.T, B[i]*U)[0]

  else:
    #Compute what goes in the exponent and subtract out the maximum
    #Don't need to bother storing for later because compute ratio
    dBeta = B - refB
    dBetaU = (-1.0)*np.tensordot(dBeta, U, axes=0)
    dBetaUdiff = dBetaU - np.array([np.max(dBetaU, axis=1)]).T
    expVals = np.exp(dBetaUdiff)

    #And perform averaging
    numer = np.dot(expVals, x) / float(x.shape[0])
    denom = np.average(expVals, axis=1)
    outval = numer / np.array([denom]).T

  return outval


class VolumeExtrapModel(ExtrapModel):
  """Class to hold information about a VOLUME extrapolation. This can be trained by providing
     data at the reference state and can then be evaluated to obtain estimates at
     arbitrary other states. Note that refB is now the reference volume and self.U will
     actually represent the virial, not the potential energy. Will only go up to first order
     with derivative information, as after that derivatives of forces are needed.
  """

  #Can't go to higher order in practice, so don't return any symbolic derivatives
  #Instead, just use this to check and make sure not asking for order above 1
  def calcDerivFuncs(self):
    if self.maxOrder > 1:
      print('Volume extrapolation cannot go above 1st order without derivatives of forces.')
      print('Setting order to 1st order.')
      self.maxOrder = 1
    return None

  #And given data, calculate numerical values of derivatives up to maximum order
  #Will be very helpful when generalize to different extrapolation techniques
  #(and interpolation)
  def calcDerivVals(self, refV, x, W):
    """Calculates specific derivative values at B with data x and U up to max order.
       Returns these derivatives. Only go to first order for volume extrapolation. And
       here W represents the virial instead of the potential energy.
    """

    if x.shape[0] != W.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], W.shape[0]))
      return

    wT = np.array([W]).T
    avgX = np.average(x, axis=0)
    avgW = np.average(W)
    avgXW = np.average(x*wT, axis=0)
    derivVals = np.zeros((2, x.shape[1]))
    derivVals[0] = avgX
    derivVals[1] = (avgXW - avgX*avgW) / (3.0*refV)

    return derivVals


class VolumeExtrapWeightedModel(ExtrapWeightedModel):
  """Class to hold information about a VOLUME extrapolation. This can be trained by providing
     data at the reference state and can then be evaluated to obtain estimates at
     arbitrary other states. Note that refB is now the reference volume and self.U will
     actually represent the virial, not the potential energy.
  """

  #Can't go to higher order in practice, so don't return any symbolic derivatives
  #Instead, just use this to check and make sure not asking for order above 1
  def calcDerivFuncs(self):
    if self.maxOrder > 1:
      print('Volume extrapolation cannot go above 1st order without derivatives of forces.')
      print('Setting order to 1st order.')
      self.maxOrder = 1
    return None

  #And given data, calculate numerical values of derivatives up to maximum order
  #Will be very helpful when generalize to different extrapolation techniques
  #(and interpolation)
  def calcDerivVals(self, refV, x, W):
    """Calculates specific derivative values at B with data x and U up to max order.
       Returns these derivatives. Only go to first order for volume extrapolation. And
       here W represents the virial instead of the potential energy.
    """

    if x.shape[0] != W.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], W.shape[0]))
      return

    wT = np.array([W]).T
    avgX = np.average(x, axis=0)
    avgW = np.average(W)
    avgXW = np.average(x*wT, axis=0)
    derivVals = np.zeros((2, x.shape[1]))
    derivVals[0] = avgX
    derivVals[1] = (avgXW - avgX*avgW) / (3.0*refV)

    return derivVals


class VolumeInterpModel(InterpModel):
  """Class to hold information about a VOLUME interpolation. This can be trained by providing
     data at the reference state and can then be evaluated to obtain estimates at
     arbitrary other states. Note that refB is now the reference volume and self.U will
     actually represent the virial, not the potential energy.
  """

  #Can't go to higher order in practice, so don't return any symbolic derivatives
  #Instead, just use this to check and make sure not asking for order above 1
  def calcDerivFuncs(self):
    if self.maxOrder > 1:
      print('Volume extrapolation cannot go above 1st order without derivatives of forces.')
      print('Setting order to 1st order.')
      self.maxOrder = 1
    return None

  #And given data, calculate numerical values of derivatives up to maximum order
  #Will be very helpful when generalize to different extrapolation techniques
  #(and interpolation)
  def calcDerivVals(self, refV, x, W):
    """Calculates specific derivative values at B with data x and U up to max order.
       Returns these derivatives. Only go to first order for volume extrapolation. And
       here W represents the virial instead of the potential energy.
    """

    if x.shape[0] != W.shape[0]:
      print('First observable dimension (%i) and size of potential energy array (%i) don\'t match!'%(x.shape[0], W.shape[0]))
      return

    wT = np.array([W]).T
    avgX = np.average(x, axis=0)
    avgW = np.average(W)
    avgXW = np.average(x*wT, axis=0)
    derivVals = np.zeros((2, x.shape[1]))
    derivVals[0] = avgX
    derivVals[1] = (avgXW - avgX*avgW) / (3.0*refV)

    return derivVals


class IGmodel:
  """Defines a 1D ideal gas in an external field. The position, x, may vary from 0 to L,
     with the field acting linearly on x, U(x) = a*x, where for simplicity we let a=1.
     This is a useful class to use for testing.
  """

  #Define some symbols and functions used across the class
  #All such classes will have identical symbols and functions, which is desirable here
  #Because volume is of secondary interest, set default parameter for this so that
  #it does not need to be specified (keeps older code compatible, too)
  b, l = symbols('b l')
  avgXsym = (1/b) - l/(exp(b*l) - 1)
  avgXlambdify = lambdify([b, l], avgXsym, "numpy")

  @classmethod
  def avgX(cls, B, L=1.0):
    """Average position x at the inverse temperature B
    """
    return cls.avgXlambdify(B, L)

  @classmethod
  def varX(cls, B, L=1.0):
    """Variance in position, x at the inverse temperature B
    """
    term1 = 1.0/(B**2)
    term2 = (L**2)*np.exp(B*L)/((np.exp(B*L) - 1)**2)
    return (term1 - term2)

  @classmethod
  def PofX(cls, x, B, L=1.0): #This will also give P(U) exactly for single particle if a = 1
    """Canonical probability of position x for single particle at inverse temperature B
    """
    numer = B*np.exp(-B*x)
    denom = 1.0 - np.exp(-B*L)
    return (numer / denom)

  @classmethod
  def cdfX(cls, x, B, L=1.0): #Cumulative distribution function for X
    """Cumulative probability density for position x for single particle at inverse temperature B
    """
    numer = 1.0 - np.exp(-B*x)
    denom = 1.0 - np.exp(-B*L)
    return (numer / denom)

  def __init__(self, nParticles=1000):
    self.nP = nParticles #Number of particles

  def sampleX(self, B, s, L=1.0):
    """Samples s samples of x from the probability density at inverse temperature B
       Does sampling based on inversion of cumulative distribution function
    """
    randvec = np.random.random(size=s)
    randx = -(1.0/B)*np.log(1.0 - randvec*(1.0 - np.exp(-B*L)))
    return randx

  def sampleU(self, B, s=1000, L=1.0): #Really just resampling the sum of x values many times to get distribution of U for large N
    """Samples s (=1000 by default) potential energy values from a system self.nP particles.
       Particle positions are randomly sampled with sampleX at the inverse temperature B.
    """
    randu = np.zeros(s)
    for i in range(s):
        randu[i] = np.sum(self.sampleX(B, self.nP, L=L))
    return randu

  def PofU(self, U, B, L=1.0): #Provides P(U) in the limit of a large number of particles (becomes Gaussian)
    """In the large-N limit, the probability of the potential energy is Normal, so provides that
    """
    avgU = self.nP*self.avgX(B, L=L)
    stdU = np.sqrt(self.nP*self.varX(B, L=L))
    return norm.pdf(U, avgU, stdU)

  def pertAnalytic(self, B, B0, L=1.0): #Really just the same as average of x, but it's a nice check of all the math
    """Analytical perturbation of the system from B0 to B.
       Nice check to see if get same thing as avgX
    """
    def pertNumer(B, B0):
      prefac = B0 / (1.0 - np.exp(-B0*L))
      term1 = (1.0 - np.exp(-B*L)) / (B**2)
      term2 = L*np.exp(-B*L) / B
      return (prefac*(term1 - term2))

    def pertDenom(B, B0):
      prefac = B0 / B
      numer = 1.0 - np.exp(-B*L)
      denom = 1.0 - np.exp(-B0*L)
      return (prefac*numer/denom)

    return (pertNumer(B, B0) / pertDenom(B, B0))

  def extrapAnalytic(self, B, B0, order, L=1.0):
    """Analytical extrapolation from B0 to B at specified order.
       Same as if used infinite number of symbols, so only includes truncation error.
    """
    dBeta = B-B0
    outvec = np.zeros(order+1)
    outval = 0.0
    for k in range(order+1):
        thisdiff = diff(self.avgXsym, self.b, k)
        outvec[k] = thisdiff.subs({self.b:B0, self.l:L})
        outval += outvec[k]*(dBeta**k)/np.math.factorial(k)
    return (outval, outvec)

  def extrapAnalyticVolume(self, L, L0, order, B=1.0):
    """Analytical extrapolation from a reference system L0 to new L at specified order.
       Must also specify inverse temperature B if want it to be other than 1.0
    """
    dL = L - L0
    outvec = np.zeros(order+1)
    outval = 0.0
    for k in range(order+1):
        thisdiff = diff(self.avgXsym, self.l, k)
        outvec[k] = thisdiff.subs({self.b:B, self.l:L0})
        outval += outvec[k]*(dL**k)/np.math.factorial(k)
    return (outval, outvec)

  #Want to be able to create sample data set we can work with at a reference beta
  def genData(self, B, nConfigs=100000, L=1.0):
    """Generates nConfigs data points of the model at inverse temperature beta.
       Returns are the average x values and the potential energy values of each data point.
    """
    allX = self.sampleX(B, nConfigs*self.nP, L=L)
    allConfigs = np.reshape(allX, (nConfigs, self.nP))
    obsX = np.average(allConfigs, axis=1)
    obsU = np.sum(allConfigs, axis=1)
    return (obsX, obsU)


class RecursiveInterp:
  """Class to perform a recursive interpolation (maybe using weighted extrapolation)
     and then save the information necessary to predict arbitrary interior points.
     Training performs recursive algorithm to meet desired accuracy.
     Prediction uses the learned piecewise function.
  """

  def __init__(self, model, edgeB, maxOrder=1, errTol=0.01):
    self.model = model #The model object used for interpolation, like ExtrapWeightedModel
    self.modelParams = [] #Model params for piecewise intervals
    self.modelParamErrs = [] #Bootstrapped uncertainties in model parameters
    self.xData = [] #Observable data generated at each edge point - CONSIDER WAYS TO SAVE MEMORY
    self.uData = [] #Potential energy data generated at each edge point
    self.edgeB = np.array(edgeB) #Values of state points that we interpolate between
                                 #Start with outer edges, but will add points as needed
    self.maxOrder = maxOrder #Maximum order of derivatives to use - default is 1
    self.tol = errTol #Default bootstrap absolute relative error tolerance of 1%
                      #i.e. sigma_bootstrap/|interpolated value| <= 0.01

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

  def recursiveTrain(self, B1, B2, xData1=None, xData2=None, uData1=None, uData2=None, recurseDepth=0, recurseMax=10, Bavail=None, verbose=False, doPlot=False, plotCompareFunc=None):
    """Recursively trains interpolating models on successively smaller intervals
       until error tolerance is reached. The next state point to subdivide an
       interval is chosen as the point where the bootstrapped error is the largest.
       If Bavail is not None, the closest state point value in this list will be
       used instead. This is useful when data has already been generated at
       specific state points and you do not wish to generate more.
    """
    if recurseDepth > recurseMax:
      return

    if verbose:
      print('\nInterpolating from points %f and %f'%(B1, B2))
      print('Recursion depth on this branch: %i'%recurseDepth)

    #Generate data somehow if not provided
    if xData1 is None:
       xData1, uData1 = self.getData(B1)
    if xData2 is None:
       xData2, uData2 = self.getData(B2)

    #And format it for training interpolation models
    xData = np.array([xData1, xData2])
    uData = np.array([uData1, uData2])

    #Train the model and get parameters we want to use for THIS interpolation
    #Have to save parameters because want to use SAME data when bootstrapping
    #So part of saving parameters is updating the data that's used in the model
    thisParams = self.model.train([B1, B2], xData, uData, saveParams=True)

    #Decide if need more data to extrapolate from
    #Check convergence at grid of values between edges, using worst case to check
    Bvals = np.linspace(B1, B2, num=50)
    predictVals = self.model.predict(Bvals, order=self.maxOrder)
    bootErr = self.model.bootstrap(Bvals, order=self.maxOrder)
    #Be careful to catch /0.0
    relErr = np.zeros(bootErr.shape)
    for i in range(bootErr.shape[0]):
      for j in range(bootErr.shape[1]):
        if abs(predictVals[i,j]) == 0.0:
          #If value is exactly zero, either really unlucky
          #Or inherently no error because it IS zero - assume this
          relErr[i,j] = 0.0
        else:
          relErr[i,j] = bootErr[i,j] / abs(predictVals[i,j])

    #Checking maximum over both tested interior state points AND observable values
    #(if observable is a vector, use element with maximum error
    checkInd = np.unravel_index(np.argmax(relErr), relErr.shape)
    checkVal = relErr[checkInd]

    if verbose:
      print('Maximum bootstrapped error within interval: %f'%checkVal)

    #Check if bootstrapped uncertainty in estimate is small enough
    #If so, we're done
    if checkVal <= self.tol:
      newB = None
    #If not, we want to return the state point with the maximum error
    else:
      #Select closest value of state points in list if provided
      if Bavail is not None:
        Bavail = np.array(Bavail)
        newBInd = np.argmin(abs(Bavail - Bvals[checkInd[0]]))
        newB = Bavail[newBInd]
      else:
        newB = Bvals[checkInd[0]] #First dimension of prediction is along beta values

    if verbose:
      if newB is not None:
        print('Selected new extrapolation point: %f'%newB)
      else:
        print('No additional extrapolation points necessary on this interval.')

    #Do some plotting just as a visual for how things are going, if desired
    if doPlot:
      interpVals = np.linspace(B1, B2, 20)
      interp = self.model.predict(interpVals, order=self.maxOrder)[:,0]
      plt.clf()
      plt.plot(interpVals, interp)
      if newB is not None:
        plt.plot([newB, newB], [np.min(interp), np.max(interp)], 'k:')
      if plotCompareFunc is not None:
        plt.plot(interpVals, plotCompareFunc(interpVals), 'k--')
      plt.xlabel(r'$\beta$')
      plt.ylabel(r'Observable, $X$')
      plt.gcf().tight_layout()
      plt.show(block=False)
      plt.pause(5)
      #time.sleep(5)
      plt.close()

    if newB is not None:
      #Add the new point to the list of edge points and recurse
      insertInd = np.where(self.edgeB > newB)[0][0]
      self.edgeB = np.insert(self.edgeB, insertInd, newB)
      recurseDepth += 1
      self.recursiveTrain(B1, newB,
                          xData1=xData1, uData1=uData1,
                          xData2=None, uData2=None,
                          recurseDepth=recurseDepth, recurseMax=recurseMax,
                          Bavail=Bavail, verbose=verbose,
                          doPlot=doPlot, plotCompareFunc=plotCompareFunc)
      self.recursiveTrain(newB, B2,
                          xData1=None, uData1=None,
                          xData2=xData2, uData2=uData2,
                          recurseDepth=recurseDepth, recurseMax=recurseMax,
                          Bavail=Bavail, verbose=verbose,
                          doPlot=doPlot, plotCompareFunc=plotCompareFunc)
    else:
      #If we don't need to add extrapolation points, add this region to piecewise function
      #Do this by adding in parameters for this region
      #Appending should work because code will always go with lower interval first
      self.modelParams.append(self.model.params)
      #And also append uncertainties by bootstrapping
      self.modelParamErrs.append(self.model.bootstrap(None))
      #Also add this data to what we save - hopefully have enough memory
      self.xData.append(xData1)
      self.uData.append(uData1)
      if B2 == self.edgeB[-1]:
        self.xData.append(xData2)
        self.uData.append(uData2)
      return

  def predict(self, B):
    """Makes a prediction using the trained piecewise model.
       Note that the function will not produce output if asked to extrapolate outside
       the range it was trained on.
    """
    #Make sure we've done some training
    if len(self.modelParams) == 0:
      print("First must train the piecewise model!")
      return

    #For each state point in B, select a piecewise model to use
    predictVals = np.zeros((len(B), self.model.x.shape[2]))

    for i, beta in enumerate(B):

      #Check if out of lower bound
      try:
        paramInd = np.where(self.edgeB <= beta)[0][-1]
      except IndexError:
        print("Have provided point %f below interpolation function interval edges (%s)."%(beta, str(self.edgeB)))
        return

      #Check if out of upper bound
      if beta > self.edgeB[-1]:
        print("Have provided point %f above interpolation function interval edges (%s)."%(beta, str(self.edgeB)))
        return

      #Don't want to train model (already done!) but need to manually specify
      #both the parameters AND the reference state points
      #For the latter, must set manually
      if paramInd == len(self.edgeB)-1:
        self.model.refB = np.array([self.edgeB[paramInd-1], self.edgeB[paramInd]])
        predictVals[i] = self.model.predict(beta,
                                            params=self.modelParams[paramInd-1],
                                            order=self.maxOrder)[0,:]
      else:
        self.model.refB = np.array([self.edgeB[paramInd], self.edgeB[paramInd+1]])
        predictVals[i] = self.model.predict(beta,
                                            params=self.modelParams[paramInd],
                                            order=self.maxOrder)[0,:]

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
      print('Can only check polynomial consistency with a polynomial interpolation model class.')
      return

    if len(self.modelParams) == 0:
      print('No model parameters found. Must train model before checking consistency.')
      return

    if len(self.modelParams) == 1:
      print('Single interpolation region. No point in checking consistency.')
      return

    #Need to subdivide the full interval into pairs of neighboring intervals
    #Easiest way is to take state point edge values in sliding sets of three
    allInds = np.arange(self.edgeB.shape[0])
    nrows = allInds.size - 3 + 1
    n = allInds.strides[0]
    edgeSets = np.lib.stride_tricks.as_strided(allInds, shape=(nrows,3), strides=(n,n))

    #Will record and return p-values from hypothesis tests
    allPvals = []

    #Before loop, set up plot if wanted
    if doPlot:
      pColors = plt.cm.cividis(np.arange(len(edgeSets))/float(len(edgeSets)))
      pFig, pAx = plt.subplots()
      plotYmin = 1E+10
      plotYmax = -1E+10

    #Loop over sets of three edges
    for i, aset in enumerate(edgeSets):
      #Start with regions we already have coefficients for
      reg1Coeffs = self.modelParams[aset[0]]
      reg1Err = self.modelParamErrs[aset[0]]
      reg2Coeffs = self.modelParams[aset[1]]
      reg2Err = self.modelParamErrs[aset[1]]
      z12 = (reg1Coeffs - reg2Coeffs) / np.sqrt(reg1Err**2 + reg2Err**2)
      #Assuming Gaussian distributions for coefficients
      #This is implicit in returning bootstrap standard deviation as estimate of uncertainty
      #If DON'T want to assume this, boostrap function should return confidence intervals
      #And that will require a good bit of re-coding throughout this whole class
      p12 = 2.0*norm.cdf(-abs(z12))

      #To check full interval, must retrain model with data
      fullCoeffs = self.model.train(self.edgeB[aset[[0,2]]],
                                    np.array([self.xData[aset[0]], self.xData[aset[2]]]),
                                    np.array([self.uData[aset[0]], self.uData[aset[2]]]),
                                    saveParams=True)
      fullErr = self.model.bootstrap(None)
      z1full = (reg1Coeffs - fullCoeffs) / np.sqrt(reg1Err**2 + fullErr**2)
      p1full = 2.0*norm.cdf(-abs(z1full))
      z2full = (reg2Coeffs - fullCoeffs) / np.sqrt(reg2Err**2 + fullErr**2)
      p2full = 2.0*norm.cdf(-abs(z2full))

      allPvals.append(np.vstack((p12, p1full, p2full)))
      print('Interval with edges %s (indices %s):'%(str(self.edgeB[aset]), str(aset)))
      print('\tP-values between regions:')
      print(p12)
      print('\tP-values for full and 1 :')
      print(p1full)
      print('\tP-values for full and 2 :')
      print(p2full)

      if doPlot:
        plotPoints = np.linspace(self.edgeB[aset[0]], self.edgeB[aset[2]], 50)
        plotFull = np.polynomial.polynomial.polyval(plotPoints, fullCoeffs[:,0])
        plotReg1 = np.polynomial.polynomial.polyval(plotPoints, reg1Coeffs[:,0])
        plotReg2 = np.polynomial.polynomial.polyval(plotPoints, reg2Coeffs[:,0])
        pAx.plot(plotPoints, plotFull, color = pColors[i], linestyle='-')
        pAx.plot(plotPoints, plotReg1, color = pColors[i], linestyle=':')
        pAx.plot(plotPoints, plotReg2, color = pColors[i], linestyle='--')
        allPlotY = np.hstack((plotFull, plotReg1, plotReg2))
        if np.min(allPlotY) < plotYmin:
          plotYmin = np.min(allPlotY)
        if np.max(allPlotY) > plotYmax:
          plotYmax = np.max(allPlotY)

    if doPlot:
      for edge in self.edgeB:
        pAx.plot([edge]*2, [plotYmin, plotYmax], 'k-')
      pAx.set_xlabel(r'$\beta$')
      pAx.set_ylabel(r'$\langle x \rangle$')
      pFig.tight_layout()
      plt.show()

    return allPvals



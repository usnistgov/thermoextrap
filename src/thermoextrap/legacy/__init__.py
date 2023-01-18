"""Library of functions useful for thermodynamic extrapolation and interpolation.
"""
#
# Thermodynamic Extrapolation/Interpolation Library
# This repository contains code used and described in the paper "Extrapolation and interpolation strategies for efficiently estimating structural observables as a function of temperature and density."
# If you find this code useful in producing published works, please provide an appropriate citation.
# Contributors: Jacob I. Monroe, Harold W. Hatch, Bill Krekelberg
#

from .extrap import *
from .ig import *
from .interp import *
from .recursive_interp import *
from .reweight import *

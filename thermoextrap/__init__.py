"""Library of functions useful for thermodynamic extrapolation and interpolation.
"""
#
# Thermodynamic Extrapolation/Interpolation Library
# This repository contains code used and described in the paper "Extrapolation and interpolation strategies for efficiently estimating structural observables as a function of temperature and density."
# If you find this code useful in producing published works, please provide an appropriate citation.
# Contributors: Jacob I. Monroe, Harold W. Hatch, Bill Krekelberg
#

import pkg_resources

from .extrap import *
from .ig import *
from .interp import *
from .recursive_interp import *
from .reweight import *

try:
    __version__ = pkg_resources.get_distribution("thermoextrap").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

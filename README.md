# Thermodynamic Extrapolation/Interpolation Library
This repository contains code used and described in the paper "Extrapolation and interpolation strategies for efficiently estimating structural observables as a function of temperature and density."
If you find this code useful in producing published works, please provide an appropriate citation.

Code included here can be used to perform thermodynamic extrapolation and interpolation of observables calculated from molecular simulations.
This allows for more efficient use of simulation data for calculating how observables change with simulation conditions, including temperature, density, pressure, chemical potential, or force field parameters.
Users are highly encourage to work through the Jupyter Notebook tutorial (Ideal_Gas_Example.ipynb) presenting examples for a variety of different observable functional forms.
We only guarantee that this code is functional for the test cases we present here or for which it has previously been applied
Additionally, the code may be in continuous development at any time.
Use at your own risk and always check to make sure the produced results make sense.
If bugs are found, please report them.
If specific features would be helpful just let us know and we will be happy to work with you to come up with a solution.

# Dependencies
- python 3 (python 2 may also work but is not tested or officially supported)
- numpy
- scipy
- sympy
- pymbar (optional --- for comparisons)

# Installation
Currently, installation is simple.
Clone this repository and place lib_extrap.py into a directory pointed to by your system and/or python path.
You should the be able to load in all classes and functions with `from lib_extrap import *`.
To test installation, run test_lib_extrap.py and diff the output against test_output.txt.
More sophisticated packaging and robust installation is in the works.

# Contact
Questions may be addressed to Jacob Monroe at jacob.monroe@nist.gov.



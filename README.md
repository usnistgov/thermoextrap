# Thermodynamic Extrapolation/Interpolation Library
This repository contains code used and described in:

Monroe, J. I.; Hatch, H. W.; Mahynski, N. A.; Shell, M. S.; Shen, V. K. Extrapolation and Interpolation Strategies for Efficiently Estimating Structural Observables as a Function of Temperature and Density. J. Chem. Phys. 2020, 153 (14), 144101. https://doi.org/10.1063/5.0014282.

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
- matplotlib (optional --- for visual consistency checks)
- pymbar (optional --- for comparisons)

# Installation
Move into the repository directory and run
```
pip install .
```
It is recommended that this is done in a python environment, such as one created with conda.
The above command will also install numpy, scipy, and sympy if these packages are not detected.
To disable automatic installation of these packages, use
```
pip install --no-deps .
```
Note, however, that if these packages are not installed, the code will not work.
The initial version of the code was developed and tested with numpy 1.17.2, scipy 1.3.1, and sympy 1.4.
Earlier versions of these packages may also be compatible, but have not been tested.
The code may also be used without pip installation by placing the libextrap directory into a directory pointed to by your system and/or python path.

The matplotlib and pymbar packages are optional.
The main body of the code will run without these packages, with the exception of reweighting.py which requires pymbar for all of its functionality as this involves performing perturbation theory or MBAR predictions.
Plotting is only used for visual consistency checks for polynomial interpolation in recursive_interp.py and is disabled by default.
To install matplotlib or pymbar, you can use you favorite package manager like pip or conda.
Directions for installing pymbar may include additional subtleties which may be found [here](https://pymbar.readthedocs.io/en/master/getting_started.html#installing-pymbar).

With succesful installation, you should be able to load in all classes and functions with `from thermoextrap import *`.
The exception is utilities.py, which contains low-level code that will not be necessary for most use cases.
If you want to import specific modules rather than everything, you of course can also do that.
To test installation, run `python test_thermoextrap.py` and diff the output against test_output.txt.
If pymbar is not installed, the output will differ by a single test to check MBAR.

# Contact
Questions may be addressed to Jacob Monroe at jacob.monroe@nist.gov.



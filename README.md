<!-- markdownlint-disable MD041 -->

[![Repo][repo-badge]][repo-link] [![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: black][black-badge]][black-link]

<!-- For more badges, see
https://shields.io/category/other
https://naereen.github.io/badges/
[pypi-badge]: https://badge.fury.io/py/thermoextrap
-->

[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
[pypi-badge]: https://img.shields.io/pypi/v/thermoextrap
[pypi-link]: https://pypi.org/project/thermoextrap
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/thermoextrap/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/usnistgov/thermoextrap
[conda-badge]: https://img.shields.io/conda/v/wpk-nist/thermoextrap
[conda-link]: https://anaconda.org/wpk-nist/thermoextrap
[license-badge]: https://img.shields.io/pypi/l/cmomy?color=informational
[license-link]: https://github.com/usnistgov/thermoextrap/blob/main/LICENSE

<!-- other links -->

[cmomy]: https://github.com/usnistgov/cmomy
[gpr-link]:
  https://github.com/usnistgov/thermoextrap/tree/main/examples/gpr_active_learning
[notebook-link]:
  https://github.com/usnistgov/thermoextrap/tree/main/examples/usage

# `thermoextrap`: Thermodynamic Extrapolation/Interpolation Library

This repository contains code used and described in references [^fn1] [^fn2].

[^fn1]:
    [Extrapolation and Interpolation Strategies for Efficiently Estimating Structural Observables as a Function of Temperature and Density](https://doi.org/10.1063/5.0014282)

[^fn2]:
    Leveraging Uncertainty Estimates and Derivative Information in Gaussian
    Process Regression for Expedited Data Collection in Molecular Simulations.
    In preparation.

## Overview

If you find this code useful in producing published works, please provide an
appropriate citation. Note that the second citation is focused on adding
features that make use of GPR models based on derivative information produced by
the core code base. For now, the GPR code, along with more information, may be
found under [here][gpr-link]. In a future release, we expect this to be fully
integrated into the code base rather than a standalone module.

Code included here can be used to perform thermodynamic extrapolation and
interpolation of observables calculated from molecular simulations. This allows
for more efficient use of simulation data for calculating how observables change
with simulation conditions, including temperature, density, pressure, chemical
potential, or force field parameters. Users are highly encourage to work through
the [Jupyter Notebooks][notebook-link] presenting examples for a variety of
different observable functional forms. We only guarantee that this code is
functional for the test cases we present here or for which it has previously
been applied Additionally, the code may be in continuous development at any
time. Use at your own risk and always check to make sure the produced results
make sense. If bugs are found, please report them. If specific features would be
helpful just let us know and we will be happy to work with you to come up with a
solution.

## Features

- Fast calculation of derivatives

## Status

This package is actively used by the author. Please feel free to create a pull
request for wanted features and suggestions!

## Quick start

`thermoextrap` may be installed with either (recommended)

```bash
conda install -c wpk-nist thermoextrap
```

or

```bash
pip install thermoextrap
```

If you use pip, then you can include additional dependencies using

```bash
pip install thermoextrap[all]
```

If you install `thermoextrap` with conda, there are additional optional
dependencies that take some care for installation. We recommend installing the
following via `pip`, as the versions on the conda/conda-forge channels are often
a bit old.

```bash
pip install tensorflow tensorflow-probability gpflow
```

To install from source do the following:

```bash
git clone git@github.com:usnistgov/thermoextrap.git
cd thermoextrap
pip install . [-e]
```

To (optionally) include the example data do the following:

```bash
git submodule update --init  --recursive
```

## Example usage

```python
import thermoextrap

```

<!-- end-docs -->

## Documentation

See the [documentation][docs-link] for a look at `thermoextrap` in action.

To have a look at using `thermoextrap` with Gaussian process regression, look in
the [gpr](examples/usage/gpr) and
[gpr_active_learning](examples/gpr_active_learning) directories.

## License

This is free software. See [LICENSE][license-link].

## Related work

This package extensively uses the [cmomy] package to handle central comoments.

## Contact

Questions may be addressed to Bill Krekelberg at william.krekelberg@nist.gov or
Jacob Monroe at jacob.monroe@uark.edu.

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).

# cmomy

[![image](https://img.shields.io/pypi/v/cmomy.svg)](https://pypi.python.org/pypi/cmomy)

[![image](https://img.shields.io/travis/wpk-nist-gov/cmomy.svg)](https://travis-ci.com/wpk-nist-gov/cmomy)

Central (co)moment calculation/manipulation

-   Free software: NIST license

## Overview

`cmomy` is an open source package to calculate central moments and
co-moments in a numerical stable and direct way. Behind the scenes,
`cmomy` makes use of [Numba](https://numba.pydata.org/) to rapidly
calculate moments. A good introduction to the type of formulas used can
be found
[here](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance).

## Features

-   Fast calculation of central moments and central co-moments with
    weights
-   Support for scalor or vector inputs
-   numpy and xarray api\'s

## Installation

Use one of the following

``` bash
pip install cmomy
```

or

``` bash
conda install -c wpk-nist cmomy
```

## Note on caching

This code makes extensive use of the numba python package. This uses a
jit compiler to speed up vital code sections. This means that the first
time a funciton called, it has to compile the underlying code. However,
caching has been implemented. Therefore, the very first time you run a
function, it may be slow. But all subsequent uses (including other
sessions) will be already compiled.

## Testing

Tests are packaged with the distribution intentionally. To test code
run:

``` bash
pytest --pyargs cmomy
```

By running the tests once, you create a cache of the numba code for most
cases. The first time you run the tests, it will take a while (about 1.5
min on my machine). However, subsequent runs will be much faster (about
3 seconds on my machine).

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[wpk-nist-gov/cookiecutter-pypackage](https://github.com/wpk-nist-gov/cookiecutter-pypackage)
Project template forked from
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).

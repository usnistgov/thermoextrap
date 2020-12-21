# Package for analyzing central moments

For example usage, look at [example usage](examples/example_usage.ipynb)

# Installation

## Install needed packages

If using conda, then

``` {.bash org-language="sh"}
conda install numpy xarray numba
```

## From source

``` {.bash org-language="sh"}
cd directory/path
git clone https://github.com/wpk-nist-gov/cmomy.git

# full install (note that versioning is iffy.  -I forces reinstall)
pip install -I -U .

# for editable
pip install -e .
```

To run tests:

``` {.bash org-language="sh"}
pytest -v
```

Note that tests are a bit slow (about 1.5 min). This is due to numba jit
compiler having to compile all underlying functions for all cases.

## pip install

``` {.bash org-language="sh"}
pip install -I -U git+https://github.com/wpk-nist-gov/cmomy.git@develop
```

# Note on caching

This code makes extensive use of the numba python package. This uses a
jit compiler to speed up vital code sections. This means that the first
time a funciton called, it has to compile the underlying code. However,
caching has been implemented. Therefore, the very first time you run a
function, it may be slow. But all subsequent uses (including other
sessions) will be already compiled.

# Testing

Tests are packaged with the distribution intentionally. To test code
run:

``` {.bash org-language="sh"}
pytest -x -v --pyargs cmomy
```

By running the tests once, you create a cache of the numba code for most
cases. The first time you run the tests, it will take a while (about 1.5
min on my machine). However, subsequent runs will be much faster (about
3 seconds on my machine).

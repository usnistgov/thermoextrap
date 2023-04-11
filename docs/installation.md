```{highlight} shell
```

# Installation

## Stable release

To install thermodynamic-extrapolation, run this command in your terminal:

```
$ pip install thermoextrap
```

or, if you use conda, run:

```
$ conda install c wpk-nist thermoextrap
```

## Additional dependencies

To utilize the full potential of `thermoextrap`, additional dependencies are needed.  This can be done via pip by using:

```
$ pip install thermoextrap[all]
```

If using conda, then you'll have to manually install some dependencies.  For example, you can run:

```
$ conda install bottleneck dask pymbar<4.0
```

At this time, it is recommended to install the Gaussian Process Regression (GPR) dependencies via pip, as the conda-forge recipes are slightly out of date:

```
$ pip install tensorflow tensorflow-probability gpflow
```

## From sources

The sources for thermodynamic-extrapolation can be downloaded from the [Github repo].

You can either clone the public repository:

```console
$ git clone git://github.com/usnistgov/thermoextrap.git
```

Once you have a copy of the source, you can install it with:

```console
$ pip install .
```

[github repo]: https://github.com/usnistgov/thermoextrap

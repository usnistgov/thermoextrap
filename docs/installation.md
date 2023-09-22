# Installation

## Stable release

To install thermodynamic-extrapolation, run this command in your terminal:

```bash
pip install thermoextrap
```

or, if you use conda, run:

```bash
conda install -c conda-forge thermoextrap
```

## Additional dependencies

To utilize the full potential of `thermoextrap`, additional dependencies are
needed. This can be done via pip by using:

```bash
pip install thermoextrap[all]
```

If using conda, then you'll have to manually install some dependencies. For
example, you can run:

```bash
conda install bottleneck dask pymbar>=4.0
```

At this time, it is recommended to install the Gaussian Process Regression (GPR)
dependencies via pip, as the conda-forge recipes are slightly out of date:

```bash
pip install tensorflow tensorflow-probability "gpflow>=2.6.0"
```

## From sources

See [](./contributing) for details on installing from source.

# GP Models Utilizing Derivative Information and Active Learning

All of the scripts and notebooks contained here make use of the `gpr_active` module of `thermo-extrap`.
The code contained here and in `gpr_active` was developed as part of the a publication that will soon be submitted.
Commit 91d905b934e290a4b775e098a9230371decc5ae3 (release NUMBER) is the stable version of the code used in that paper.
The code from that commit should be seen as a preliminary version, with updates currently in progress.
That being said, this page describes the basic organization and functionality of the `gpr_active` module.
We will do our best to keep this and related tutorial notebooks updated as we develop the code further, but stay tuned as major changes are possible.

## Requirements and Dependencies

For the core tools in `gpr_active`, the key dependencies are sympy and GPflow, which are installed by default with thermo-extrap.
Due to signicant changes in how GPflow handles likelihoods and custom models, we currently require the GPflow version to be less then 2.6.0.
Plans to ensure compatibility with newer versions of GPflow are underway.

For everything in the upcoming publication, though, the molecular simulation packages OpenMM, FEASST, and CASSANDRA, are also required.
That means these packages also need to be installed to run the scripts and notebooks found here.
An environment file that can be used to install a similar environment to that used in the paper is available as `environment_active.yml`.
To install this with conda, use `conda env create --file environment_active.yml`, then, after activating the environment, following the directions for installing thermo-extrap and FEASST with pip.

## Gaussian Process Models

The components of all Gaussian Process (GP) models are housed in `gpr_active.gp_models`.
A custom kernel function `DerivativeKernel` forms the basis of using derivative information in the GP models.
Behind the scenes, this function uses sympy to compute necessary derivatives of a provided sympy expression representing the kernel.
Unique combinations of derivative orders are identified, the derivative function determined, and the results stored and stitched back together at the end to produce the covariance matrix.
This is possible because a derivative is a linear operator on the covariance kernel, meaning that derivatives of the kernel provide various covariances between observations at different derivative orders.

Other key functions are the custom likelihood `HetGaussianDeriv` and the GP model itself `HeteroscedasticGPR`.
The former builds a likelihood model that takes covariances between derivatives into account and also allows for heteroscedasticity (different uncertainties for different data points, including different derivative orders).
The latter, `HeteroscedasticGPR` is a heteroscedastic GP model making use of the likelihood just described and the `DerivativeKernel` and behaves much like other GP models in GPflow.
Note, though, that `predict_y` is not implemented as that would require estimates of the uncertainty at new points.
For heteroscedastic uncertainties, that would require a model of how the uncertainty varied with input location, which has not yet been implemented.

## Active Learning

Tools that assist in performing active learning protocols are found in `gpr_active.active_utils`.
Though this includes functions for building and training GPR models, the focus is on classes and methods that enable active learning.
Pre-eminent among these are classes for housing data and keeping track of simulations performed during active learning.
Since every simulation environment and project will be unique, users are encouraged to use these as guidelines and templates for creating their own classes for managing data collection.
More generally useful are classes describing active learning update strategies, metrics, and stopping criteria.
These can easily be inherited to construct new active learning protocols without fundamentally changing the primary active learning function.
While this function is fairly general in its structure, it does make specific use of some of the other classes described, which, as highlighted, vary in their generalizability and transferability to new situations.

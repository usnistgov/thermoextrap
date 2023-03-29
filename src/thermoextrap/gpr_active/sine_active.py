# Written by Jacob I. Monroe, NIST employee

"""
Generates sine data with controlled uncertainty and other features to test GP
models and active learning strategies.
"""

import numpy as np


def noise_func(x, s, n):
    """
    Function to produce heteroscedastic noise based on given x values
    Must also provide a slope and noise base scaling since model is
        noise = scale*(slope*(x - x_min) + cos(x)^2)
    i.e., sum of linear model and periodic function
    Note that produces variance for each x location, not sample of noise.

    Inputs:
        x - input points
        s - slope
        n - noise magnitude
    """
    linear_term = s * (x - np.min(x))
    cos_term = np.cos(x) ** 2
    return n * (linear_term + cos_term)


def make_data(
    x_vals,
    fac=1.0,
    phase_shift=0.0,
    noise=0.1,
    slope=0.1,
    order_scale=1.0,
    max_order=4,
    rng=np.random.default_rng(42),
):
    """
    Creates data with heteroscedastic noise around sin(x).

    Inputs:
        x_vals - values at which y data should be generated
        fac - (1.0) scaling factor in front of sine, fac*sin(x)
        phase_shift - (0.0) phase shift of sine function, sin(x + shift)
        noise - (0.1) magnitude of noise produced by noise_func
        slope - (0.1) slope of linear portion of noise in noise_func
        order_scale - (1.0) scale for variance with derivative order so
                            that derivative uncertainty is multiplied by
                            exp(order_scale*order), or log scales linearly
        max_order - (4) maximum order of derivative info to generate

    Outputs:
        X - x values tiled and with added column to indicate derivative orders
            should be ready for input to a GP model
        Y - y values at each x
        Y_err - variance in each y value - assumes all values and derivatives independent
                (diagonal covariance matrix)
    """
    if isinstance(x_vals, (float, int)):
        x_vals = [x_vals]
    x_vals = np.array(x_vals)

    y_vals = fac * np.sin(x_vals + phase_shift)
    y_err = (fac**2) * noise_func(x_vals, slope, noise)
    for i in range(1, max_order + 1):
        if i % 2 == 0:
            deriv_vals = fac * np.sin(x_vals + phase_shift)
        else:
            deriv_vals = fac * np.cos(x_vals + phase_shift)
        this_noise = (
            (fac**2) * noise_func(x_vals, slope, noise) * np.exp(order_scale * i)
        )
        if i % 4 >= 2:
            deriv_vals *= -1
        y_vals = np.hstack([y_vals, deriv_vals])
        y_err = np.hstack([y_err, this_noise])

    X = np.vstack(
        [
            np.tile(x_vals, (max_order + 1)),
            np.hstack([np.ones(x_vals.shape[0]) * k for k in range(max_order + 1)]),
        ]
    ).T

    # Sample outputs Y from Gaussian
    Y = rng.normal(y_vals, np.sqrt(y_err))[:, None]
    # Also adding noise to estimate of noise
    Y_err = (y_err * np.exp(0.5 * (rng.random(len(y_err)) - 0.5)))[:, None]

    return X, Y, Y_err

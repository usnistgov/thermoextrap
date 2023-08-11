from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # some weird stuff happens with sympy and pyright
    # this should stop those errors:
    import importlib

    import cmomy
    import numpy as np
    import pandas as pd
    import pymbar
    import scipy
    import xarray as xr

    sp = importlib.import_module("sympy")


else:
    import lazy_loader as lazy

    np = lazy.load("numpy")
    xr = lazy.load("xarray")
    sp = lazy.load("sympy")
    pd = lazy.load("pandas")
    scipy = lazy.load("scipy")
    pymbar = lazy.load("pymbar")
    cmomy = lazy.load("cmomy")


def _has_pymbar():
    out = False
    try:
        _ = pymbar.__version__
        out = True
    except ImportError:
        out = False
    return out


__all__ = ["np", "xr", "sp", "pd", "scipy", "pymbar", "cmomy", "_has_pymbar"]

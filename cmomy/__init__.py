import pkg_resources

from .central import CentralMoments, central_moments
from .resample import (
    bootstrap_confidence_interval,
    randsamp_freq,
    xbootstrap_confidence_interval,
)
from .xcentral import xcentral_moments, xCentralMoments

try:
    __version__ = pkg_resources.get_distribution("cmomy").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"

__all__ = [
    "CentralMoments",
    "central_moments",
    "xCentralMoments",
    "xcentral_moments",
    "bootstrap_confidence_interval",
    "randsamp_freq",
    "xbootstrap_confidence_interval",
    "__version__",
]

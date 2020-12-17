from .central import CentralMoments, central_moments
from .resample import (
    bootstrap_confidence_interval,
    randsamp_freq,
    xbootstrap_confidence_interval,
)
from .xcentral import xcentral_moments, xCentralMoments

__all__ = (
    CentralMoments,
    central_moments,
    xCentralMoments,
    xcentral_moments,
    bootstrap_confidence_interval,
    randsamp_freq,
    xbootstrap_confidence_interval,
)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

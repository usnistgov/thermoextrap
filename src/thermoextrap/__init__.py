"""Classes/routines to deal with thermodynamic extrapolation."""

# TODO: move data, idealgas, models to top level.
from . import beta, data, idealgas, lnpi, models, volume, volume_idealgas
from .core.xrutils import xrwrap_alpha, xrwrap_uv, xrwrap_xv
from .data import (
    DataCentralMoments,
    DataCentralMomentsVals,
    DataValues,
    DataValuesCentral,
    factory_data_values,
    resample_indices,
)

# expose some data/models
from .models import (
    Derivatives,
    ExtrapModel,
    ExtrapWeightedModel,
    InterpModel,
    InterpModelPiecewise,
    MBARModel,
    PerturbModel,
    StateCollection,
)

# updated versioning scheme
try:
    from ._version import __version__
except Exception:
    __version__ = "999"


__all__ = [
    "ExtrapModel",
    "ExtrapWeightedModel",
    "InterpModel",
    "InterpModelPiecewise",
    "MBARModel",
    "PerturbModel",
    "StateCollection",
    "Derivatives",
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "DataValues",
    "DataValuesCentral",
    "factory_data_values",
    "resample_indices",
    "xrwrap_xv",
    "xrwrap_uv",
    "xrwrap_alpha",
    "idealgas",
    "data",
    "models",
    "beta",
    "lnpi",
    "volume",
    "volume_idealgas",
    "__version__",
]

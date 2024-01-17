"""Classes/routines to deal with thermodynamic extrapolation."""

# TODO(wpk): move data, idealgas, models to top level.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (  # noqa: TCH004
        beta,
        data,
        idealgas,
        lnpi,
        models,
        volume,
        volume_idealgas,
    )
    from .core.xrutils import xrwrap_alpha, xrwrap_uv, xrwrap_xv  # noqa: TCH004
    from .data import (
        DataCentralMoments,  # noqa: TCH004
        DataCentralMomentsVals,  # noqa: TCH004
        DataValues,  # noqa: TCH004
        DataValuesCentral,  # noqa: TCH004
        factory_data_values,  # noqa: TCH004
        resample_indices,  # noqa: TCH004
    )

    # expose some data/models
    from .models import (
        Derivatives,  # noqa: TCH004
        ExtrapModel,  # noqa: TCH004
        ExtrapWeightedModel,  # noqa: TCH004
        InterpModel,  # noqa: TCH004
        InterpModelPiecewise,  # noqa: TCH004
        MBARModel,  # noqa: TCH004
        PerturbModel,  # noqa: TCH004
        StateCollection,  # noqa: TCH004
    )
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=[
            "beta",
            "data",
            "idealgas",
            "lnpi",
            "models",
            "volume",
            "volume_idealgas",
        ],
        submod_attrs={
            "data": [
                "DataCentralMoments",
                "DataCentralMomentsVals",
                "DataValues",
                "DataValuesCentral",
                "factory_data_values",
                "resample_indices",
            ],
            "models": [
                "Derivatives",
                "ExtrapModel",
                "ExtrapWeightedModel",
                "InterpModel",
                "InterpModelPiecewise",
                "MBARModel",
                "PerturbModel",
                "StateCollection",
            ],
            "core.xrutils": ["xrwrap_alpha", "xrwrap_uv", "xrwrap_xv"],
        },
    )


# updated versioning scheme
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("thermoextrap")
except PackageNotFoundError:  # pragma: no cover
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

"""Classes/routines to deal with thermodynamic extrapolation."""

# TODO(wpk): move data, idealgas, models to top level.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (  # noqa: TC004
        beta,
        data,
        idealgas,
        lnpi,
        models,
        random,
        volume,
        volume_idealgas,
    )
    from .core.xrutils import xrwrap_alpha, xrwrap_uv, xrwrap_xv  # noqa: TC004
    from .data import (
        DataCentralMoments,  # noqa: TC004
        DataCentralMomentsVals,  # noqa: TC004
        DataValues,  # noqa: TC004
        DataValuesCentral,  # noqa: TC004
        factory_data_values,  # noqa: TC004
    )

    # expose some data/models
    from .models import (
        Derivatives,  # noqa: TC004
        ExtrapModel,  # noqa: TC004
        ExtrapWeightedModel,  # noqa: TC004
        InterpModel,  # noqa: TC004
        InterpModelPiecewise,  # noqa: TC004
        MBARModel,  # noqa: TC004
        PerturbModel,  # noqa: TC004
        StateCollection,  # noqa: TC004
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
            "random",
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
    "DataCentralMoments",
    "DataCentralMomentsVals",
    "DataValues",
    "DataValuesCentral",
    "Derivatives",
    "ExtrapModel",
    "ExtrapWeightedModel",
    "InterpModel",
    "InterpModelPiecewise",
    "MBARModel",
    "PerturbModel",
    "StateCollection",
    "__version__",
    "beta",
    "data",
    "factory_data_values",
    "idealgas",
    "lnpi",
    "models",
    "random",
    "volume",
    "volume_idealgas",
    "xrwrap_alpha",
    "xrwrap_uv",
    "xrwrap_xv",
]

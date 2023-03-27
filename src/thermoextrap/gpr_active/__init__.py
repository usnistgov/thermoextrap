"""Modules to apply Gaussian process regression to thermodynamic extrapolation."""

from . import active_utils, gp_models, ig_active

__all__ = [
    "gp_models",
    "active_utils",
    "ig_active",
]

# from .gp_models import DerivativeKernel
# from .active_utils import DataWrapper
# from .ig_active import IG_DataWrapper

# from . import active_utils


# __all__ = [
#     "DerivativeKernel",
#     "DataWrapper",
#     "IG_DataWrapper",
#     "active_utils",
# ]


# Old Stuff in to have in docstring.
# .. autosummary::
#     :toctree: generated/
#     :template: autodocsumm/module.rst

#     thermoextrap.gpr_active.gp_models
#     thermoextrap.gpr_active.active_utils
#     thermoextrap.gpr_active.ig_active

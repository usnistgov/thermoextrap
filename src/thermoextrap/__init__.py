"""Classes/routines to deal with thermodynamic extrapolation."""
# pylint: disable=duplicate-code
# ruff:file-ignore[non-empty-init-module]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# To change top level imports edit __init__.pyi
import lazy_loader as _lazy  # pyright: ignore[reportMissingTypeStubs]

<<<<<<< before updating
__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)  # pyright: ignore[reportUnknownVariableType]

try:
=======
try:  # ruff:ignore[non-empty-init-module]
>>>>>>> after updating
    __version__ = _version("thermoextrap")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

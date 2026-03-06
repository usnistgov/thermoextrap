"""Class to override numba cache location/tagging"""

# ruff: noqa: D102
# pyright: reportMissingTypeStubs=false, reportImplicitOverride=false
from __future__ import annotations

import hashlib
import os
import platform
import sys
from functools import cached_property
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numba
from numba.core.caching import UserProvidedCacheLocator

if TYPE_CHECKING:
    from typing import Any


class _Config:
    @cached_property
    def magic(self) -> str:
        magic = os.getenv("NUMBA_CACHE_TOOLS_MAGIC", "0")
        if (path := Path(magic)).is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        return magic


config = _Config()


class HashCacheLocator(UserProvidedCacheLocator):
    """User defined caching function"""

    def get_source_stamp(self) -> Any:
        mtime = config.magic
        size: str = hashlib.sha256(
            Path(cast("str", self._py_file)).read_bytes()
        ).hexdigest()
        return mtime, size


class SharedHashCacheLocator(HashCacheLocator):
    """With standard location"""

    @classmethod
    def get_suitable_cache_subpath(cls, py_file: str) -> str:
        """Given the Python file path, compute a suitable path inside the
        cache directory.

        This will reduce a file path that is too long, which can be a problem
        on some operating system (i.e. Windows 7).
        """
        path = Path(py_file).resolve()
        rel_dir = path.resolve().parent

        for _ in range(100):
            name = rel_dir.name
            if name in {"src", "site-packages"}:
                break
            rel_dir = rel_dir.parent
        else:
            msg = f"Can't determine source path for file {py_file}"
            raise ValueError(msg)

        subpath = path.parent.relative_to(rel_dir)
        package_version = import_module(subpath.parts[0]).__version__

        return "_".join(
            [
                *subpath.parts,
                config.magic,
                package_version,
                sys.platform,
                platform.python_version(),
                numba.__version__,
            ]
        )

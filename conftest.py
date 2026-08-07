"""Settings for doctests and nbval tests."""
# ruff:file-ignore[missing-type-function-argument,undocumented-public-function]

import pytest


def pytest_collectstart(collector) -> None:
    if collector.fspath and collector.fspath.ext == ".ipynb":
        collector.skip_compare += (
            "text/html",
            "application/javascript",
            "stderr",
        )


def pytest_ignore_collect(collection_path) -> None:  # ruff:ignore[unused-function-argument]
    return False


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

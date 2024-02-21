"""Settings for doctests and nbval tests."""


def pytest_collectstart(collector) -> None:
    if collector.fspath and collector.fspath.ext == ".ipynb":
        collector.skip_compare += (
            "text/html",
            "application/javascript",
            "stderr",
        )


def pytest_ignore_collect(collection_path, path, config) -> None:  # noqa: ARG001
    import sys

    if sys.version_info < (3, 9):
        if "thermoextrap/tests" in str(collection_path):
            return False
        return True

    return False

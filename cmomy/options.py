"""
This sets up optional values
"""

NMAX = "nmax"
FASTMATH = "fastmath"
PARALLEL = "parallel"
CACHE = "cache"


OPTIONS = {NMAX: 20, PARALLEL: True, CACHE: True, FASTMATH: True}


def _isbool(x):
    isinstance(x, bool)


def _isint(x):
    isinstance(x, int)


def _isstr(x):
    isinstance(x, str)


_VALIDATORS = {
    NMAX: _isint,
    PARALLEL: _isbool,
    CACHE: _isbool,
    FASTMATH: _isbool,
}

_SETTERS = {}


class set_options(object):
    """Set options for xarray in a controlled context.
    Currently supported options:
    - `NMAX` : max moment size
    You can use ``set_options`` either as a context manager:
    >>> with xr.set_options(use_tqdm=True, tqdm_min_len_calc=50):
    ...     c.xge.betaOmega()
    ...
    Or to set global options:
    >>> xr.set_options(tqdm_min_len_calc=50)
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)

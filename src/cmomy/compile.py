"""Set of routines to call most cases to pre-compile numba functions."""

from __future__ import absolute_import

import numpy as np

from .central import CentralMoments


def compile_numba_funcs():
    """Attempt at function to compile all funcs."""
    dtype = float
    for val_shape in [(), (1,), (1, 1)]:
        for mom in [(1,), (1, 1)]:
            s = CentralMoments.zeros(val_shape=val_shape, mom=mom)

            val = np.empty(val_shape, dtype=dtype)
            vals = np.empty((1,) + val_shape, dtype=dtype)

            if len(mom) == 2:
                val = (val, val)
                vals = (vals, vals)

            data = np.empty(s.shape, dtype=dtype)
            datas = np.empty((1,) + s.shape, dtype=dtype)

            s.push_val(val)
            s.push_vals(vals)

            s.push_data(data)
            s.push_datas(datas)

            if len(val_shape) > 1:
                for parallel in [False, True]:
                    s.resample_and_reduce(nrep=2, parallel=parallel)

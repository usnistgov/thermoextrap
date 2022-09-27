"""Common docstrings."""

from textwrap import dedent

import pandas.util._decorators as pd_dec

from .options import DOC_SUB


def docfiller(*args, **kwargs):
    """To fill common docs.

    Taken from pandas.utils._decorators

    """

    if DOC_SUB:
        return pd_dec.doc(*args, **kwargs)
    else:

        def decorated(func):
            return func

        return decorated


# fmt: off
_shared_docs = {
    "copy":
    """
    copy : bool, optional
        If True, copy the data.  If False, attempt to use view.
    """,
    "copy_kws":
    """
    copy_kws : mapping, optional
        extra arguments to copy
    """,
    "verify":
    """
    verify : bool, optional
        If True, make sure data is c-contiguous
    """,
    "check_shape":
    """
    check_shape : bool, optional
        If True, check that shape of resulting object is correct.
    """,
    "mom":
    """
    mom : int or (int, int)
        Order or moments.  If integer or length one tuple, then moments are for
        a single variable.  If length 2 tuple, then comoments of two variables
    """,
    "mom_ndim":
    """
    mom_ndim : {1, 2}
        Value indicates if moments (``mom_ndim = 1``) or comoments (``mom_ndim=2``).
    """,
    "val_shape":
    """
    val_shape: tuple, optional
        Shape of `values` part of data.  That is, the non-moment dimensions.
    """,
    "shape":
    """
    shape : tuple, optional
        Total shape.  ``shape = val_shape + tuple(m+1 for m in mom)``
    """,
    "dtype":
    """
    dtype : dtype, optional
        Optional ``dtype`` for output data.
    """,
    "zeros_kws":
    """
    zeros_kws : mapping, optional
        Optional parameters to :func:`numpy.zeros`
    """,
    "axis":
    """
    axis : int
        Axis to reduce along.
    """,
    "broadcast":
    """
    broadcast : bool, optional
        If True, and ``x=(x0, x1)``, then perform 'smart' broadcasting.
        In this case, if ``x1.ndim = 1`` and ``len(x1) == x0.shape[axis]``, then
        broadcast `x1` to ``x0.shape``.
    """,
    "freq":
    """
    freq : array of int, optional
        Array of shape ``(nrep, size)`` where `nrep` is the number of replicates and
        ``size = self.shape[axis]``.  `freq` is the weight that each sample contributes
        to resamples values.  See :func:`~cmomy.resample.randsamp_freq`
    """,
    "indices":
    """
    indices : array of int, optional
        Array of shape ``(nrep, size)``.  If passed, create `freq` from indices.
        See :func:`~cmomy.resample.randsamp_freq`.
    """,
    "nrep":
    """
    nrep : int, optional
        Number of replicates.  Create `freq` with this many replicates.
        See :func:`~cmomy.resample.randsamp_freq`
    """,
    "pushed":
    """
    pushed : same as object
        Same as object, with new data pushed onto `self.data`
    """,
    "resample_kws":
    """
    resample_kws : mapping
        Extra arguments to :func:`~cmomy.resample.resample_vals`
    """,
    "full_output":
    """
    full_output : bool, optional
        If True, also return `freq` array
    """,
    "convert_kws":
    """
    convert_kws : mapping
        Extra arguments to :func:`~cmomy.convert.to_central_moments` or :func:`~cmomy.convert.to_central_comoments`
    """,
    "dims":
    """
    dims : hashable or sequence of hashable, optional
        Dimension of resulting :class:`xarray.DataArray`.

        * If ``len(dims) == self.ndim``, then dims specifies all dimensions.
        * If ``len(dims) == self.val_ndim``, ``dims = dims + mom_dims``

        Default to ``('dim_0', 'dim_1', ...)``

    """,
    "mom_dims":
    """
    mom_dims : hashable or tuple of hashable
        Name of moment dimensions.  Defaults to ``(mom_0, ...)``
    """,
    "attrs":
    """
    attrs : mapping
        Attributes of output
    """,
    "coords":
    """
    coords : mapping
        Coordinates of output
    """,
    "name":
    """
    name : hashable
        Name of output
    """,
    "indexes":
    """
    indexes : Any
        indexes attribute.  This is ignored.
    """,
    "template":
    """
    template : DataArray
        If present, output will have attributes of `template`.
        Overrides other options.
    """,
    "dim":
    """
    dim : hashable, optional
        Dimension to reduce along.
    """,
    "rep_dim":
    """
    rep_dim : hashable, optional
        Name of new 'replicated' dimension:
    """,
    "klass": "Same as calling class"
    ,
    "T_Array": "ndarray",
}
# fmt: on


def _prepare_shared_docs(shared_docs):
    return {k: dedent(v).strip() for k, v in _shared_docs.items()}


_shared_docs = _prepare_shared_docs(_shared_docs)


# add in xr_params
_shared_docs["xr_params"] = "\n".join(
    [
        _shared_docs[k]
        for k in ["dims", "mom_dims", "attrs", "coords", "name", "indexes", "template"]
    ]
)


docfiller_shared = docfiller(**_shared_docs)

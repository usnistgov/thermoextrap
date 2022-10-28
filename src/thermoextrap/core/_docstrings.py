"""Common docstrings."""

from __future__ import annotations

from ._docfiller import AttributeDict, docfiller, prepare_shared_docs

# fmt: off
SHARED_DOCS_CMOMY = prepare_shared_docs(**{
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
        Name of moment dimensions.  Defaults to ``('xmom', 'umom')``
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
    "rec_dim":
    """
    rec_dim: hashable, optional
        Name of dimension for 'records', i.e., multiple observations.
    """,
})


SHARED_DOCS_CMOMY['xr_params'] = "\n".join(
    [
        SHARED_DOCS_CMOMY[k]
        for k in ["dims", "attrs", "coords", "name", "indexes", "template"]
    ]
)


SHARED_DOCS_XTRAP = prepare_shared_docs(**{
    "uv" :
    """
    uv : xarray.DataArray
        raw values of u (energy)
    """,
    "xv" :
    """
    xv : xarray.DataArray
        raw values of x (observable)
    """,
    "w":
    """
    w : array-like, optional
        optional weigth array.  Note that this array/xarray must be conformable to uv, xv
    """,
    "order" :
    """
    order : int
        maximum order of moments/expansion to calculate
    """,
    "umom_dim" :
    """
    umom_dim : str, default='umom'
        Name of dimension for moment of energy `u`.
    """,
    "xmom_dim" :
    """
    xmom_dim : str, default='xmom'
        Name of dimension for moments of observable `x`.
    """,
    "deriv_dim" :
    """
    deriv_dim : str, default=None
        if deriv_dim is a string, then this is the name of the derivative dimension
        and xarray objects will have a derivative
    """,
    "skipna" :
    """
    skipna : bool, default=False
        if True, skip nan values
    """,
    "chunk" :
    """
    chunk : bool, optional
        chunking of xarray objects
    """,
    "compute" :
    """
    compute : bool, optional
        whether to perform compute step on xarray outputs
    """,
    "meta" :
    """
    meta : dict, optional
        extra meta data/parameters to be caried along with object and child objects.
        if 'checker' in meta, then perform a callback of the form meta['checker](self, meta)
        this can also be used to hotwire things like derivs_args.
        Values passed through method `resample_meta`
    """,
    "meta_kws" :
    """
    meta_kws : mapping, optional
        Optional parameters for meta.
    """,
    "x_is_u" :
    """
    x_is_u : bool, default=False
        if True, treat `xv = uv` and do adjust u/du accordingly
    """,
    "uv_xv_array":
    """
    uv : array-like
        raw values of u (energy)
        if not DataArray, wrap with `xrwrap_uv`
    xv : xarray.DataArray
        raw values of x (observable)
        if not DataArray, wrap with `xrwrap_xv`
    """,
    "val_dims":
    """
    val_dims : str or list-like
        Names of extra dimensions
    """,
    "xalpha":
    """
    xalpha : bool, default=False
        Flag whether `u` depends on variable `alpha`.
    """,
    "central":
    """
    central : bool
        If True, Use central moments.  Otherwise, use raw moments.
    """,
    "dxduave":
    """
    dxduave : xCentralMoments
        Central moments object.
    """,
    "expand":
    """
    expand : bool
        If True, apply :func:`sympy.expand`
    """,
    "post_func":
    """
    post_func : str or callable
        Transformation of base function.
        For example, `post_fuc = -sympy.log` is equivalent to passing `minus_log=True`
        If a string, then apply the following standard functions

        * minus_log : post_func = -sympy.log
        * pow_i : post_func = lambda f: pow(f, i).  E.g., `pow_2` => pow(f, 2)
    """
})


SHARED_DOCS_BETA = prepare_shared_docs(**{
    "n_order":
    """
    n : int
        Order of moment.
    """,
    "d_order":
    """
    d : int
        Order of derivative of ``x``.
    """,
    "beta":
    """
    beta : float
        reference value of inverse temperature
    """,
    "data":
    """
    data : Data object
        Instance of :class:`thermoextrap.AbstractData`.
    """,
    "alpha_name":
    """
    alpha_name, str, default='beta'
        name of expansion parameter
    """,
})


SHARED_DOCS_VOLUME = prepare_shared_docs(**{
    "volume" :
    """
    volume : float
        Reference value of system volume.
    """,
    "ndim":
    """
    ndim : int, default = 3
        Number of dimensions of the system.
    """,
    "dxdqv":
    """
    dxdqv : array-like
        values of `sum dx/dq_i q_i` where `q_i` is the ith coordinate.
    """,
})
# fmt: on

SHARED_DOCS = AttributeDict(
    {
        "cmomy": SHARED_DOCS_CMOMY,
        "xtrap": SHARED_DOCS_XTRAP,
        "beta": SHARED_DOCS_BETA,
        "volume": SHARED_DOCS_VOLUME,
    }
)


# @lru_cache
def _factory_get_mapping(
    names=None,
    *,
    dotted=False,
    shared_docs=None,
    **kws,
):
    if names is None:
        names = ()
    elif isinstance(names, str):
        names = (names,)
    names = tuple(name.lower() for name in names)

    names_parsed = []
    for name in names:
        if name == "all":
            add_in = list(SHARED_DOCS.keys())
        elif name == "default":
            add_in = ["cmomy", "xtrap"]
        else:
            assert name in SHARED_DOCS.keys()
            add_in = [name]

        for n in add_in:
            if n not in names_parsed:
                names_parsed.append(n)

    if shared_docs is None:
        shared_docs = SHARED_DOCS
    else:
        shared_docs = AttributeDict.from_dict(shared_docs)

    for name in names_parsed:
        assert name in shared_docs.keys()

    if dotted:
        mapping = AttributeDict({k: shared_docs[k] for k in names_parsed})
    else:
        mapping = AttributeDict()
        for name in names_parsed:
            mapping.entries.update(**shared_docs[name])

    mapping.entries.update(**prepare_shared_docs(**kws))
    return mapping


def factory_docfiller_shared(names=None, *, dotted=False, shared_docs=None, **kws):
    """
    Create a decorator for filling in documentation.

    Based on pandas private method. This is not ideal, but going with it for now.

    Parameters
    ----------
    names : hashable or sequence of hashable, optional
        names of keys in ``shared_docs`` to include in expansion.
    dotted : bool, default=False
    If True,

    """

    mapping = _factory_get_mapping(
        names=names, dotted=dotted, shared_docs=shared_docs, **kws
    )
    return docfiller(**mapping)

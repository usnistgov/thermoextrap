"""Common docstrings."""

from __future__ import annotations

from .external.docfiller import DocFiller

_docstring_cmomy = """\
Parameters
----------
copy : bool, optional
    If True, copy the data.  If False, attempt to use view.
copy_kws : mapping, optional
    extra arguments to copy
verify : bool, optional
    If True, make sure data is c-contiguous
check_shape : bool, optional
    If True, check that shape of resulting object is correct.
mom : int or tuple of int
    Order or moments.  If integer or length one tuple, then moments are for
    a single variable.  If length 2 tuple, then comoments of two variables
mom_ndim : {1, 2}
    Value indicates if moments (``mom_ndim = 1``) or comoments (``mom_ndim=2``).
val_shape : tuple, optional
    Shape of `values` part of data.  That is, the non-moment dimensions.
shape : tuple, optional
    Total shape.  ``shape = val_shape + tuple(m+1 for m in mom)``
dtype : dtype, optional
    Optional ``dtype`` for output data.
zeros_kws : mapping, optional
    Optional parameters to :func:`numpy.zeros`
axis : int
    Axis to reduce along.
broadcast : bool, optional
    If True, and ``x=(x0, x1)``, then perform 'smart' broadcasting.
    In this case, if ``x1.ndim = 1`` and ``len(x1) == x0.shape[axis]``, then
    broadcast `x1` to ``x0.shape``.
freq : array of int, optional
    Array of shape ``(nrep, size)`` where `nrep` is the number of replicates and
    ``size = self.shape[axis]``.  `freq` is the weight that each sample contributes
    to resamples values.  See :func:`~cmomy.resample.randsamp_freq`
indices : array of int, optional
    Array of shape ``(nrep, size)``.  If passed, create `freq` from indices.
    See :func:`~cmomy.resample.randsamp_freq`.
nrep : int, optional
    Number of replicates.  Create `freq` with this many replicates.
    See :func:`~cmomy.resample.randsamp_freq`
pushed : same as object
    Same as object, with new data pushed onto `self.data`
resample_kws : mapping
    Extra arguments to :func:`~cmomy.resample.resample_vals`
full_output : bool, optional
    If True, also return `freq` array
convert_kws : mapping
    Extra arguments to :func:`~cmomy.convert.to_central_moments` or :func:`~cmomy.convert.to_central_comoments`
dims : hashable or sequence of hashable, optional
    Dimension of resulting :class:`xarray.DataArray`.

    * If ``len(dims) == self.ndim``, then dims specifies all dimensions.
    * If ``len(dims) == self.val_ndim``, ``dims = dims + mom_dims``

    Default to ``('dim_0', 'dim_1', ...)``
mom_dims : hashable or tuple of hashable
    Name of moment dimensions.  Defaults to ``('xmom', 'umom')``
attrs : mapping
    Attributes of output
coords : mapping
    Coordinates of output
name : hashable
    Name of output
indexes : Any
    indexes attribute.  This is ignored.
template : DataArray
    If present, output will have attributes of `template`.
    Overrides other options.
dim : hashable, optional
    Dimension to reduce along.
rep_dim : hashable, optional
    Name of new 'replicated' dimension:
rec_dim : hashable, optional
    Name of dimension for 'records', i.e., multiple observations.
"""

DOCFILLER_CMOMY = DocFiller.from_docstring(
    _docstring_cmomy, combine_keys="parameters"
).assign_combined_key(
    "xr_params", ["dims", "attrs", "coords", "name", "indexes", "template"]
)


# add uv_xv_array
_docstring_xtrap = """\
Parameters
----------
uv : xarray.DataArray
    raw values of u (energy)
xv : xarray.DataArray
    raw values of x (observable)
w : array-like, optional
    optional weight array.  Note that this array/xarray must be conformable to uv, xv
order : int
    maximum order of moments/expansion to calculate
umom_dim : str, default='umom'
    Name of dimension for moment of energy `u`.
xmom_dim : str, default='xmom'
    Name of dimension for moments of observable `x`.
deriv_dim : str, default=None
    if deriv_dim is a string, then this is the name of the derivative dimension
    and xarray objects will have a derivative
skipna : bool, default=False
    if True, skip nan values
chunk : bool, optional
    chunking of xarray objects
compute : bool, optional
    whether to perform compute step on xarray outputs
meta : dict, optional
    extra meta data/parameters to be carried along with object and child objects.
    if 'checker' in meta, then perform a callback of the form meta['checker](self, meta)
    this can also be used to override things like derivs_args.
    Values passed through method `resample_meta`
meta_kws : mapping, optional
    Optional parameters for meta.
x_is_u : bool, default=False
    if True, treat `xv = uv` and do adjust u/du accordingly
uv_array | uv : array-like
    raw values of u (energy)
    if not DataArray, wrap with `xrwrap_uv`
xv_array | xv : xarray.DataArray
    raw values of x (observable)
    if not DataArray, wrap with `xrwrap_xv`
val_dims : str or sequence of str
    Names of extra dimensions
xalpha : bool, default=False
    Flag whether `u` depends on variable `alpha`.
central : bool
    If True, Use central moments.  Otherwise, use raw moments.
dxduave : xCentralMoments
    Central moments object.
expand : bool
    If True, apply :meth:`~sympy.core.expr.Expr.expand`
post_func : str or callable
    Transformation of base function.
    For example, `post_fuc = -sympy.log` is equivalent to passing `minus_log=True`
    If a string, then apply the following standard functions

    * minus_log : post_func = -sympy.log
    * pow_i : post_func = lambda f: pow(f, i).  E.g., `pow_2` => pow(f, 2)

"""

DOCFILLER_XTRAP = DocFiller.from_docstring(
    _docstring_xtrap, combine_keys="parameters"
).assign_combined_key("uv_xv_array", ["uv_array", "xv_array"])


_docstring_beta = """\
Parameters
----------
n_order | n : int
    Order of moment.
d_order | d : int
    Order of derivative of ``x``.
beta : float
    reference value of inverse temperature
data : object
    Instance of data object, e.g. :class:`thermoextrap.data.DataCentralMoments`
alpha_name : str, default='beta'
    name of expansion parameter
"""

DOCFILLER_BETA = DocFiller.from_docstring(_docstring_beta, combine_keys="parameters")


_docstring_volume = """\
Parameters
----------
volume : float
    Reference value of system volume.
ndim : int, default=3
    Number of dimensions of the system.
dxdqv : array-like
    values of `sum dx/dq_i q_i` where `q_i` is the ith coordinate.
"""

DOCFILLER_VOLUME = DocFiller.from_docstring(
    _docstring_volume, combine_keys="parameters"
)

SHARED_DOCS = {
    "cmomy": DOCFILLER_CMOMY.data,
    "xtrap": DOCFILLER_XTRAP.data,
    "beta": DOCFILLER_BETA.data,
    "volume": DOCFILLER_VOLUME.data,
}


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

    for name in names_parsed:
        assert name in shared_docs.keys()

    if dotted:
        mapping = {k: shared_docs[k] for k in names_parsed}
    else:
        mapping = {}
        for name in names_parsed:
            mapping.update(**shared_docs[name])

    mapping.update(**kws)
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
    return DocFiller(mapping)()

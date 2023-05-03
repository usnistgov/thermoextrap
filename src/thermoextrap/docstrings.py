"""Common docstrings."""

from __future__ import annotations

from cmomy.docstrings import docfiller as DOCFILLER_CMOMY

# from .external.docfiller import DocFiller
from module_utilities.docfiller import DocFiller

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


DOCFILLER_SHARED = DocFiller.concat(
    cmomy=DOCFILLER_CMOMY,
    xtrap=DOCFILLER_XTRAP,
    beta=DOCFILLER_BETA,
    volume=DOCFILLER_VOLUME,
)

"""Central moments/comoments routines."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
)

import numpy as np
import xarray as xr
from numpy.core.numeric import normalize_axis_index  # type: ignore
from numpy.typing import ArrayLike, DTypeLike

from . import convert
from ._docstrings import docfiller_shared
from ._typing import ArrayOrder, Moments, T_CentralMoments
from .resample import randsamp_freq, resample_vals
from .utils import _axis_expand_broadcast  # _cached_ones,; _my_broadcast,
from .utils import _shape_insert_axis, _shape_reduce

if TYPE_CHECKING:
    from .xcentral import xCentralMoments


###############################################################################
# central mom/comoments routines
###############################################################################
def _central_moments(
    vals: ArrayLike,
    mom: Moments,
    w: np.ndarray | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate central mom along axis."""

    if isinstance(mom, tuple):
        mom = mom[0]

    x = np.asarray(vals, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    if w is None:
        w = np.ones_like(x)
    else:
        w = _axis_expand_broadcast(
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = (mom + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try rolling
            out = np.moveaxis(out, -1, 0)
        assert out.shape == shape

    wsum = w.sum(axis=0)
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", w, x) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, mom + 1).reshape(*shape)

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", w, dx) * wsum_inv

    if last:
        out = np.moveaxis(out, 0, -1)
    return out


def _central_comoments(
    vals: Tuple[np.ndarray, np.ndarray],
    mom: Moments,
    w: Optional[np.ndarray] = None,
    axis: int = 0,
    last: bool = True,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate central co-mom (covariance, etc) along axis."""

    if isinstance(mom, int):
        mom = (mom,) * 2

    mom = tuple(mom)
    assert len(mom) == 2

    # change x to tuple of inputs
    assert isinstance(vals, tuple) and len(vals) == 2
    x, y = vals

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    y = _axis_expand_broadcast(
        y,
        x.shape,
        axis,
        roll=False,
        broadcast=broadcast,
        expand=broadcast,
        dtype=dtype,
        order=order,
    )

    if w is None:
        w = np.ones_like(x)
    else:
        w = _axis_expand_broadcast(
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    assert w.shape == x.shape
    assert y.shape == x.shape

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        y = np.moveaxis(y, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = tuple(x + 1 for x in mom) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try moving axis
            out = np.moveaxis(out, [-2, -1], [0, 1])
        assert out.shape == shape

    wsum = w.sum(axis=0)
    wsum_inv = 1.0 / wsum

    xave = np.einsum("r...,r...->...", w, x) * wsum_inv
    yave = np.einsum("r...,r...->...", w, y) * wsum_inv

    shape = (-1,) + (1,) * (x.ndim)
    p0 = np.arange(0, mom[0] + 1).reshape(*shape)
    p1 = np.arange(0, mom[1] + 1).reshape(*shape)

    dx = (x[None, ...] - xave) ** p0
    dy = (y[None, ...] - yave) ** p1

    out[...] = np.einsum("r...,ir...,jr...->ij...", w, dx, dy) * wsum_inv

    out[0, 0, ...] = wsum
    out[1, 0, ...] = xave
    out[0, 1, ...] = yave

    if last:
        out = np.moveaxis(out, [0, 1], [-2, -1])
    return out


@docfiller_shared
def central_moments(
    x: np.ndarray | Tuple[np.ndarray, np.ndarray],
    mom: Moments,
    w: np.ndarray | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
    broadcast: bool = False,
) -> np.ndarray:
    """Calculate central moments or comoments along axis.

    Parameters
    ----------
    vals : array-like or tuple of array-like
        if calculating moments, then this is the input array.
        if calculating comoments, then pass in tuple of values of form (x, y)
    {mom}
    w : array-like, optional
        Weights. If passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    {axis}
    last : bool, default=True
        if True, put mom as last dimension.
        Otherwise, mom will be in first dimension
    {dtype}
    {broadcast}
    out : array
        if present, use this for output data
        Needs to have shape of either (mom,) + shape or shape + (mom,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape=shape + mom_shape or mom_shape + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:], and `mom_shape` is the shape of
        the moment part, either (mom+1,) or (mom0+1, mom1+1).  Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment.

    See Also
    --------
    CentralMoments


    Examples
    --------
    create data:

    >>> np.random.seed(0)
    >>> x = np.random.rand(10)

    Generate first 2 central moments:

    >>> moments = central_moments(x=x, mom=2)
    >>> print(moments)
    [10.          0.61576628  0.03403099]

    Generate moments with weights

    >>> w = np.random.rand(10)
    >>> central_moments(x=x, w=w, mom=2)
    array([5.47343366, 0.65419879, 0.0389794 ])


    Generate co-momoments

    >>> y = np.random.rand(10)
    >>> central_moments(x=(x, y), w=w, mom=(2, 2))
    array([[ 5.47343366e+00,  6.94721195e-01,  5.13060985e-02],
           [ 6.54198786e-01,  1.16001132e-02, -2.63166301e-03],
           [ 3.89794046e-02, -3.36141683e-03,  2.30236918e-03]])

    """

    if isinstance(mom, int):
        mom = (mom,)

    kws = dict(
        vals=x, mom=mom, w=w, axis=axis, last=last, dtype=dtype, order=order, out=out
    )
    if len(mom) == 1:
        return _central_moments(**kws)  # type: ignore
    else:
        kws["broadcast"] = broadcast
        return _central_comoments(**kws)  # type: ignore


from .abstract_central import CentralMomentsABC


###############################################################################
# Classes
###############################################################################
class CentralMoments(CentralMomentsABC[np.ndarray]):
    """
    Parameters
    ----------
    data : ndarray
        Moments collection array.
    """

    def __new__(cls, data: np.ndarray, mom_ndim: Literal[1, 2] = 1):  # noqa: D102
        return super().__new__(cls, data=data, mom_ndim=mom_ndim)

    def __init__(self, data: np.ndarray, mom_ndim: Literal[1, 2] = 1) -> None:
        if mom_ndim not in (1, 2):
            raise ValueError(
                "mom_ndim must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )

        if not isinstance(data, np.ndarray):
            raise ValueError(f"data must be an np.ndarray.  Passed type {type(data)}")

        self._mom_ndim = mom_ndim

        if data.ndim < self.mom_ndim:
            raise ValueError("not enough dimensions in data")

        self._data = data
        self._data_flat = self._data.reshape(self.shape_flat)

        if any(m <= 0 for m in self.mom):
            raise ValueError("moments must be positive")

    @property
    def values(self) -> np.ndarray:
        """Accesses for self.data"""
        return self._data

    ###########################################################################
    # SECTION: top level creation/copy/new
    ###########################################################################
    def new_like(
        self: T_CentralMoments,
        data: np.ndarray | None = None,
        copy: bool = False,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        strict: bool = False,
        **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        output : CentralMoments


        Examples
        --------
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10), mom=3, axis=0)
        >>> da
        <CentralMoments(val_shape=(), mom=(3,))>
        array([1.00000000e+01, 6.15766283e-01, 3.40309867e-02, 3.81976735e-03])

        >>> da2 = da.new_like().zero()
        >>> da2
        <CentralMoments(val_shape=(), mom=(3,))>
        array([0., 0., 0., 0.])

        >>> da
        <CentralMoments(val_shape=(), mom=(3,))>
        array([1.00000000e+01, 6.15766283e-01, 3.40309867e-02, 3.81976735e-03])

        """

        if data is None:
            data = np.zeros_like(self._data, order="C")
            copy = verify = check_shape = False

        kws.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kws = dict(
                dict(
                    mom=self.mom,
                    val_shape=self.val_shape,
                    dtype=self.dtype,
                ),
                **kws,
            )

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            check_shape=check_shape,
            **kws,
        )

    ###########################################################################
    # SECTION: To/from xarray
    ###########################################################################
    @docfiller_shared
    def to_xarray(
        self,
        dims: Hashable | Sequence[Hashable] | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: Hashable | Tuple[Hashable, Hashable] | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xr.DataArray:
        """
        Create a :class:`DataArray` representation of underlying data.

        Parameters
        ----------
        {xr_params}
        {copy}

        Returns
        -------
        output : DataArray


        Examples
        --------
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 1, 2), axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])

        Default constructor

        >>> da.to_xarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_xarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])

        """

        if template is not None:
            out = template.copy(data=self.data)
        else:
            if dims is None:
                dims = tuple(f"dim_{i}" for i in range(self.val_ndim))
            elif isinstance(dims, str):
                dims = (dims,)
            dims = tuple(dims)  # type: ignore

            if len(dims) == self.ndim:
                dims_output = dims  # type: ignore

            elif len(dims) == self.val_ndim:
                if mom_dims is None:
                    mom_dims = tuple(f"mom_{i}" for i in range(self.mom_ndim))
                elif isinstance(mom_dims, Hashable):
                    mom_dims = (mom_dims,)
                mom_dims = tuple(mom_dims)

                assert len(mom_dims) == self.mom_ndim

                dims_output = dims + mom_dims

            else:
                raise ValueError(
                    f"Problem with {dims}, {mom_dims}.  Total length should be {self.ndim}"
                )
            out = xr.DataArray(
                self.data, dims=dims_output, coords=coords, attrs=attrs, name=name
            )

        if copy:
            out = out.copy()

        return out

    @docfiller_shared
    def to_xcentralmoments(
        self,
        dims: Hashable | Sequence[Hashable] | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: Hashable | Tuple[Hashable, Hashable] | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xCentralMoments:
        """
        Create an DataArray representation of underlying data.

        Parameters
        ----------
        {xr_params}
        {copy}

        Returns
        --------
        output : xCentralMoments

        See Also
        --------
        to_xarray

        Examples
        --------
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 1, 2), axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])

        Default constructor

        >>> da.to_xcentralmoments()
        <xCentralMoments(val_shape=(1, 2), mom=(2,))>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_xcentralmoments()
        <xCentralMoments(val_shape=(1, 2), mom=(2,))>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.        ,  0.52056625,  0.08147257],
                [10.        ,  0.6425434 ,  0.06334664]]])

        """  # noqa: D409
        from .xcentral import xCentralMoments

        data = self.to_xarray(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            template=template,
            copy=copy,
        )
        return xCentralMoments(data=data, mom_ndim=self.mom_ndim)

    ###########################################################################
    # SECTION: pushing routines
    ###########################################################################
    #  -> np.ndarray | float | Tuple[float|np.ndarray, None|float|np.ndarray] :
    def _verify_value(
        self,
        x: np.ndarray | float,
        target: np.ndarray | Tuple[int, ...] | str | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,  # included here for consistency
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Tuple[int, ...] | None = None,
        other: np.ndarray | None = None,
        **kwargs,
    ):  # type: ignore
        """Verify input values.

        Parameters
        ----------
        x : array
        target : tuple or array
            If tuple, this is the target shape to be used to Make target.
            If array, this is the target array
        Optinal target that has already been rolled.  If this is passed, and
        x will be broadcast/expanded, can expand to this shape without the need
        to reorder,
        """

        x = np.asarray(x, dtype=self.dtype)

        if isinstance(target, str):
            if target == "val":
                target_shape = self.val_shape
            elif target == "vals":
                target_shape = _shape_insert_axis(self.val_shape, axis, x.shape[axis])  # type: ignore
            elif target == "data":
                target_shape = self.shape
            elif target == "datas":
                # make sure axis in limits
                axis = normalize_axis_index(axis, self.val_ndim + 1)
                # if axis < 0:
                #     axis += self.ndim - self.mom_ndim
                target_shape = _shape_insert_axis(self.shape, axis, x.shape[axis])  # type: ignore
            elif target == "var":
                target_shape = self.shape_var
            elif target == "vars":
                assert other is not None
                target_shape = _shape_insert_axis(self.shape_var, axis, other.shape[axis])  # type: ignore
            else:
                raise ValueError(f"unknown string target name {target}")

            target_output = x

        elif isinstance(target, tuple):
            target_shape = target
            target_output = x

        elif isinstance(target, np.ndarray):
            target_shape = target.shape
            target_output = None

        else:
            raise ValueError("unknown target type")

        x = _axis_expand_broadcast(
            x,
            target_shape,
            axis,
            verify=False,
            expand=expand,
            broadcast=broadcast,
            dtype=self.dtype,
            roll=False,
        )

        # check shape:
        assert (
            x.shape == target_shape
        ), f"x.shape = {x.shape} not equal target_shape={target_shape}"

        # move axis
        if axis is not None:
            if axis != 0:
                x = np.moveaxis(x, axis, 0)
            nrec = (x.shape[0],)  # type: ignore
        else:
            nrec = ()  # type: ignore

        if shape_flat is not None:
            x = x.reshape(nrec + shape_flat)

        if x.ndim == 0:
            x = x[()]

        if target_output is None:
            return x
        else:
            return x, target_output

    def push_data(self: T_CentralMoments, data: np.ndarray) -> T_CentralMoments:
        """
        Returns
        -------
        pushed : CentralMoments
            Same object as caller, with updated data

        Examples
        --------
        >>> np.random.seed(0)
        >>> xs = np.random.rand(2, 10)
        >>> datas = [central_moments(x=x, mom=2) for x in xs]
        >>> da = CentralMoments.from_data(datas[0], mom_ndim=1)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([10.        ,  0.61576628,  0.03403099])


        >>> da.push_data(datas[1])
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.        ,  0.58155482,  0.07612921])


        Which is equivalaent to
        >>> CentralMoments.from_vals(xs.reshape(-1), mom=2)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.        ,  0.58155482,  0.07612921])

        """
        data = self._check_data(data)
        self._push.data(self._data_flat, data)
        return self

    def push_datas(
        self: T_CentralMoments,
        datas: np.ndarray,
        axis: int = 0,
        **kwargs,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        pushed : CentralMoments
            Same object as caller, with updated data

        Examples
        --------
        >>> np.random.seed(0)
        >>> xs = np.random.rand(2, 10)
        >>> datas = np.array([central_moments(x=x, mom=2) for x in xs])
        >>> da = CentralMoments.zeros(mom=2)
        >>> da.push_datas(datas, axis=0)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.        ,  0.58155482,  0.07612921])


        Which is equivalaent to
        >>> CentralMoments.from_vals(xs.reshape(-1), mom=2)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.        ,  0.58155482,  0.07612921])
        """
        datas = self._check_datas(datas, axis=axis, **kwargs)
        self._push.datas(self._data_flat, datas)
        return self

    def push_val(
        self: T_CentralMoments,
        x: float | np.ndarray | Tuple[float, float] | Tuple[np.ndarray, np.ndarray],
        w: np.ndarray | float | None = None,
        broadcast: bool = False,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        pushed : CentralMoments
            Same object as caller, with updated data

        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> y = np.random.rand(10)
        >>> w = np.random.rand(10)

        >>> da = CentralMoments.zeros(val_shape=(2,), mom=(2, 2))
        >>> for xx, yy, ww in zip(x, y, w):
        ...     _ = da.push_val(x=(xx, yy), w=ww, broadcast=True)

        >>> da
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 4.50477700e-01, -1.54375666e-02,  7.57296738e-04],
                [ 7.97586261e-02,  2.07156860e-04,  3.27773648e-03]],
        <BLANKLINE>
               [[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 6.74378211e-01, -4.03657689e-02, -3.81631321e-04],
                [ 6.41716062e-02,  9.64873805e-03,  6.37017023e-03]]])


        Which is the same as

        >>> CentralMoments.from_vals(x=(x, y), w=w, mom=(2, 2), broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 4.50477700e-01, -1.54375666e-02,  7.57296738e-04],
                [ 7.97586261e-02,  2.07156860e-04,  3.27773648e-03]],
        <BLANKLINE>
               [[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 6.74378211e-01, -4.03657689e-02, -3.81631321e-04],
                [ 6.41716062e-02,  9.64873805e-03,  6.37017023e-03]]])

        """

        if self.mom_ndim == 1:
            ys = ()
        else:
            assert isinstance(x, tuple) and len(x) == self.mom_ndim
            x, *ys = x  # type: ignore

        xr, target = self._check_val(x, "val")  # type: ignore
        yr = tuple(self._check_val(y, target=target, broadcast=broadcast) for y in ys)  # type: ignore
        wr = self._check_weight(w, target)  # type: ignore
        self._push.val(self._data_flat, *((wr, xr) + yr))
        return self

    def push_vals(
        self: T_CentralMoments,
        x: np.ndarray | Tuple[np.ndarray, np.ndarray],
        w: np.ndarray | None = None,
        axis: int = 0,
        broadcast: bool = False,
        **kwargs,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        pushed : CentralMoments
            Same object as caller, with updated data

        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> y = np.random.rand(10)
        >>> w = np.random.rand(10)

        >>> da = CentralMoments.zeros(val_shape=(2,), mom=(2, 2))
        >>> da.push_vals(x=(x, y), w=w, broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 4.50477700e-01, -1.54375666e-02,  7.57296738e-04],
                [ 7.97586261e-02,  2.07156860e-04,  3.27773648e-03]],
        <BLANKLINE>
               [[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 6.74378211e-01, -4.03657689e-02, -3.81631321e-04],
                [ 6.41716062e-02,  9.64873805e-03,  6.37017023e-03]]])


        Which is the same as

        >>> CentralMoments.from_vals(x=(x, y), w=w, mom=(2, 2), broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 4.50477700e-01, -1.54375666e-02,  7.57296738e-04],
                [ 7.97586261e-02,  2.07156860e-04,  3.27773648e-03]],
        <BLANKLINE>
               [[ 5.55439698e+00,  6.07634023e-01,  5.96006381e-02],
                [ 6.74378211e-01, -4.03657689e-02, -3.81631321e-04],
                [ 6.41716062e-02,  9.64873805e-03,  6.37017023e-03]]])

        """
        if self.mom_ndim == 1:
            ys = ()
        else:
            assert len(x) == self.mom_ndim
            x, *ys = x  # type: ignore

        xr, target = self._check_vals(x, axis=axis, target="vals", **kwargs)  # type: ignore
        yr = tuple(  # type: ignore
            self._check_vals(y, target=target, axis=axis, broadcast=broadcast, **kwargs)  # type: ignore
            for y in ys  # type: ignore
        )  # type: ignore
        wr = self._check_weights(w, target=target, axis=axis, **kwargs)
        self._push.vals(self._data_flat, *((wr, xr) + yr))
        return self

    ###########################################################################
    # SECTION: Manipulation
    ###########################################################################
    @docfiller_shared
    def reshape(
        self: T_CentralMoments,
        shape: Tuple[int, ...],
        copy: bool = True,
        copy_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Create a new object with reshaped data.

        Parameters
        ----------
        shape : tuple
            shape of values part of data.
        {copy}
        {copy_kws}
        **kws
            Parameters to :meth:`from_data`

        Returns
        -------
        output : CentralMoments
            output object with reshaped data

        See Also
        --------
        numpy.reshape
        from_data

        Examples
        --------
        >>> x = np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 2, 3), mom=2)
        >>> da
        <CentralMoments(val_shape=(2, 3), mom=(2,))>
        array([[[10.        ,  0.45494641,  0.04395725],
                [10.        ,  0.60189056,  0.08491604],
                [10.        ,  0.6049404 ,  0.09107171]],
        <BLANKLINE>
               [[10.        ,  0.53720667,  0.05909394],
                [10.        ,  0.42622908,  0.08434857],
                [10.        ,  0.47326641,  0.05907737]]])

               [[10.        ,  0.53720667,  0.05909394],
                [10.        ,  0.42622908,  0.08434857],
                [10.        ,  0.47326641,  0.05907737]]])

        >>> da.reshape(shape=(-1,))
        <CentralMoments(val_shape=(6,), mom=(2,))>
        array([[10.        ,  0.45494641,  0.04395725],
               [10.        ,  0.60189056,  0.08491604],
               [10.        ,  0.6049404 ,  0.09107171],
               [10.        ,  0.53720667,  0.05909394],
               [10.        ,  0.42622908,  0.08434857],
               [10.        ,  0.47326641,  0.05907737]])



        """
        self._raise_if_scalar()
        new_shape = shape + self.mom_shape
        data = self._data.reshape(new_shape)

        return type(self).from_data(
            data=data,
            mom_ndim=self.mom_ndim,
            mom=self.mom,
            val_shape=None,
            copy=copy,
            copy_kws=copy_kws,
            verify=True,
            check_shape=True,
            dtype=self.dtype,
            **kws,
        )

    @docfiller_shared
    def moveaxis(
        self: T_CentralMoments,
        source: int | Tuple[int, ...],
        destination: int | Tuple[int, ...],
        copy: bool = True,
        copy_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Move axis from source to destination.

        Parameters
        ----------
        source : int or sequence of int
            Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
            Destination positions for each of the original axes. These must also be
            unique.
        {copy}
        {copy_kws}

        Returns
        -------
        result : CentralMoments
            CentralMoments object with with moved axes. This array is a view of the input array.


        Examples
        --------
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 1, 2, 3), axis=0)
        >>> da.moveaxis((2, 1), (0, 2))
        <CentralMoments(val_shape=(3, 1, 2), mom=(2,))>
        array([[[[10.        ,  0.45494641,  0.04395725],
                 [10.        ,  0.53720667,  0.05909394]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[10.        ,  0.60189056,  0.08491604],
                 [10.        ,  0.42622908,  0.08434857]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[10.        ,  0.6049404 ,  0.09107171],
                 [10.        ,  0.47326641,  0.05907737]]]])

        """
        self._raise_if_scalar()

        def _internal_check_val(v) -> Tuple[int, ...]:
            if isinstance(v, int):
                v = (v,)
            else:
                v = tuple(v)
            return tuple(self._wrap_axis(x) for x in v)

        source = _internal_check_val(source)
        destination = _internal_check_val(destination)
        data = np.moveaxis(self.data, source, destination)

        # use from data for extra checks
        # return self.new_like(data=data, copy=copy, *args, **kwargs)
        return type(self).from_data(
            data,
            mom=self.mom,
            mom_ndim=self.mom_ndim,
            val_shape=data.shape[: -self.mom_ndim],
            copy=copy,
            copy_kws=copy_kws,
            **kws,
        )

    ###########################################################################
    # SECTION: Constructors
    ###########################################################################
    @classmethod
    def zeros(
        cls: Type[T_CentralMoments],
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        mom_ndim: int | None = None,
        shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        output : CentralMoments
        """

        if shape is None:
            assert mom is not None
            if isinstance(mom, int):
                mom = (mom,)
            if mom_ndim is None:
                mom_ndim = len(mom)
            assert len(mom) == mom_ndim

            if val_shape is None:
                val_shape = ()
            elif isinstance(val_shape, int):
                val_shape = (val_shape,)
            shape = val_shape + tuple(x + 1 for x in mom)

        else:
            assert mom_ndim is not None

        if dtype is None:
            dtype = float

        if zeros_kws is None:
            zeros_kws = {}
        data = np.zeros(shape=shape, dtype=dtype, **zeros_kws)

        kws = dict(kws, verify=False, copy=False, check_shape=False)
        return cls.from_data(data=data, mom_ndim=mom_ndim, **kws)

    @classmethod
    def from_data(
        cls: Type[T_CentralMoments],
        data: np.ndarray,
        mom_ndim: int | None = None,
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
        # **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        output : CentralMoments

        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(20)
        >>> data = central_moments(x=x, mom=2)

        >>> da = CentralMoments.from_data(data=data, mom_ndim=1)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.        ,  0.58155482,  0.07612921])

        """
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if verify:
            data_verified = np.asarray(data, dtype=dtype, order="C")
        else:
            data_verified = data

        if check_shape:
            if val_shape is None:
                val_shape = data_verified.shape[:-mom_ndim]
            mom = cls._check_mom(mom, mom_ndim, data_verified.shape)

            if data_verified.shape != val_shape + tuple(x + 1 for x in mom):  # type: ignore
                raise ValueError(
                    f"{data.shape} does not conform to {val_shape} and {mom}"
                )

        if copy and data_verified is data:
            if copy_kws is None:
                copy_kws = {}
            data_verified = data_verified.copy(**copy_kws)

        return cls(data=data_verified, mom_ndim=mom_ndim)

    @classmethod
    def from_datas(
        cls: Type[T_CentralMoments],
        datas: np.ndarray,
        mom_ndim: int | None = None,
        axis: int = 0,
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = True,
        check_shape: bool = True,
        **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        output : CentralMoments

        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> datas = central_moments(x=x, mom=2, axis=0)
        >>> datas
        array([[10.        ,  0.52056625,  0.08147257],
               [10.        ,  0.6425434 ,  0.06334664]])

        Reduce along a dimension
        >>> da = CentralMoments.from_datas(datas, axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.        ,  0.58155482,  0.07612921])

        """

        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if verify:
            datas = np.asarray(datas, dtype=dtype)
        datas, axis = cls._datas_axis_to_first(datas, axis=axis, mom_ndim=mom_ndim)
        if check_shape:
            if val_shape is None:
                val_shape = datas.shape[1:-mom_ndim]

            mom = cls._check_mom(mom, mom_ndim, datas.shape)
            assert datas.shape[1:] == val_shape + tuple(x + 1 for x in mom)  # type: ignore

        if dtype is None:
            dtype = datas.dtype

        return cls.zeros(
            shape=datas.shape[1:], mom_ndim=mom_ndim, dtype=dtype, **kws
        ).push_datas(datas=datas, axis=0)

    @classmethod
    def from_vals(
        cls: Type[T_CentralMoments],
        x: np.ndarray | Tuple[np.ndarray, np.ndarray],
        w: np.ndarray | None = None,
        axis: int = 0,
        dim: Hashable | None = None,
        mom: Moments = 2,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        out : CentralMoments

        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(100, 3)
        >>> da = CentralMoments.from_vals(x, axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[1.00000000e+02, 5.01679603e-01, 8.78720022e-02],
               [1.00000000e+02, 4.86570021e-01, 8.52865420e-02],
               [1.00000000e+02, 5.22257956e-01, 7.84813449e-02]])
        """

        mom_ndim = cls._mom_ndim_from_mom(mom)
        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast(np.ndarray, x0)
        if val_shape is None:
            val_shape = _shape_reduce(x0.shape, axis)
        if dtype is None:
            dtype = x0.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_vals(
            x=x, axis=axis, w=w, broadcast=broadcast
        )

    @classmethod
    def from_resample_vals(
        cls: Type[T_CentralMoments],
        x: np.ndarray | Tuple[np.ndarray, np.ndarray],
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: int | None = None,
        w: np.ndarray | None = None,
        axis: int = 0,
        mom: Moments = 2,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        full_output: bool = False,
        **kws,
    ) -> T_CentralMoments | Tuple[T_CentralMoments, np.ndarray]:
        """
        Returns
        -------
        out : CentralMoments
        freq : array, optional


        Examples
        --------
        >>> np.random.seed(0)
        >>> ndat, nrep = 10, 3
        >>> x = np.random.rand(ndat)
        >>> from cmomy.resample import numba_random_seed
        >>> numba_random_seed(0)
        >>> da, freq = CentralMoments.from_resample_vals(
        ...     x, nrep=nrep, axis=0, full_output=True
        ... )
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[10.        ,  0.5776884 ,  0.01741024],
               [10.        ,  0.78718604,  0.03810531],
               [10.        ,  0.56333822,  0.02592309]])

        Note that this is equivalent to (though in general faster than)

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
        >>> x_resamp = np.take(x, indices, axis=0)
        >>> da = CentralMoments.from_vals(x_resamp, axis=1, mom=2)
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[10.        ,  0.5776884 ,  0.01741024],
               [10.        ,  0.78718604,  0.03810531],
               [10.        ,  0.56333822,  0.02592309]])

        """

        mom_ndim = cls._mom_ndim_from_mom(mom)

        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast(np.ndarray, x0)
        freq = randsamp_freq(
            nrep=nrep,
            freq=freq,
            indices=indices,
            size=x0.shape[axis],
            check=True,
        )

        if resample_kws is None:
            resample_kws = {}

        data = resample_vals(
            x,
            freq=freq,
            mom=mom,
            axis=axis,
            w=w,
            mom_ndim=mom_ndim,
            parallel=parallel,
            **resample_kws,
            broadcast=broadcast,
        )
        out = cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            verify=True,
            check_shape=True,
            copy=False,
            **kws,
        )

        if full_output:
            return out, freq
        else:
            return out

        return out

    @classmethod
    def from_raw(
        cls: Type[T_CentralMoments],
        raw: np.ndarray,
        mom_ndim: int | None = None,
        mom: Moments | None = None,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        out : CentralMoments


        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10)
        >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

        >>> dx_raw = CentralMoments.from_raw(raw_x, mom_ndim=1)
        >>> dx_raw.mean()
        0.6157662833145425
        >>> dx_raw.cmom()
        array([1.        , 0.        , 0.03403099, 0.00381977, 0.00258793])

        Which is equivalent to creating raw moments from values
        >>> dx_cen = CentralMoments.from_vals(x, axis=0, mom=4)
        >>> dx_cen.mean()
        0.6157662833145425
        >>> dx_cen.cmom()
        array([1.        , 0.        , 0.03403099, 0.00381977, 0.00258793])


        But note that calculating using from_raw can lead to
        numerical issues.  For example

        >>> y = x + 10000
        >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
        >>> dy_raw = CentralMoments.from_raw(raw_y, mom_ndim=1)
        >>> dy_raw.mean() - 10000
        0.6157662833156792
        >>> dy_raw.cmom()  # note that these don't match dx_raw, which they should
        array([ 1.00000000e+00,  0.00000000e+00,  3.40309463e-02,  4.77443448e-03,
               -1.83500492e+01])

        >>> dy_cen = CentralMoments.from_vals(y, axis=0, mom=4)
        >>> dy_cen.mean() - 10000
        0.6157662833156792
        >>> dy_cen.cmom()  # this matches above
        array([1.        , 0.        , 0.03403099, 0.00381977, 0.00258793])

        """

        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if convert_kws is None:
            convert_kws = {}

        if mom_ndim == 1:
            data = convert.to_central_moments(raw, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            data = convert.to_central_comoments(raw, dtype=dtype, **convert_kws)
        else:
            raise ValueError(f"unknown mom_ndim {mom_ndim}")

        kws = dict(dict(verify=True, check_shape=True), **kws)

        return cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            copy=False,
            **kws,
        )

    @classmethod
    def from_raws(
        cls: Type[T_CentralMoments],
        raws: np.ndarray,
        mom_ndim: int | None = None,
        mom: Moments | None = None,
        axis: int = 0,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        **kws,
    ) -> T_CentralMoments:
        """
        Returns
        -------
        output : CentralMoments

        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> raws = (x[..., None] ** np.arange(4)).mean(axis=0)
        >>> raws
        array([[1.        , 0.52056625, 0.35246178, 0.25901614],
               [1.        , 0.6425434 , 0.47620866, 0.37399554]])
        >>> dx = CentralMoments.from_raws(raws, axis=0, mom_ndim=1)
        >>> dx.mean()
        0.5815548245225974
        >>> dx.cmom()
        array([ 1.        ,  0.        ,  0.07612921, -0.01299943])

        This is equivalent to

        >>> da = CentralMoments.from_vals(x.reshape(-1), axis=0, mom=3)
        >>> da.mean()
        0.5815548245225974
        >>> da.cmom()
        array([ 1.        ,  0.        ,  0.07612921, -0.01299943])

        """
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if convert_kws is None:
            convert_kws = {}
        if mom_ndim == 1:
            datas = convert.to_central_moments(raws, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            datas = convert.to_central_comoments(raws, dtype=dtype, **convert_kws)
        else:
            raise ValueError(f"unknown mom_ndim {mom_ndim}")

        return cls.from_datas(
            datas=datas,
            axis=axis,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    # --------------------------------------------------
    # mom_ndim == 1 specific
    # --------------------------------------------------

    @staticmethod
    def _raise_if_not_1d(mom_ndim: int) -> None:
        if mom_ndim != 1:
            raise NotImplementedError("only available for mom_ndim == 1")

    # special, 1d only methods
    def push_stat(
        self: T_CentralMoments,
        a: np.ndarray | float,
        v: np.ndarray | float = 0.0,
        w: np.ndarray | float | None = None,
        broadcast: bool = True,
    ) -> T_CentralMoments:
        """Push statisics onto self."""
        self._raise_if_not_1d(self.mom_ndim)

        ar, target = self._check_val(a, target="val")
        vr = self._check_var(v, broadcast=broadcast)
        wr = self._check_weight(w, target=target)
        self._push.stat(self._data_flat, wr, ar, vr)
        return self

    def push_stats(
        self: T_CentralMoments,
        a: np.ndarray,
        v: np.ndarray | float = 0.0,
        w: np.ndarray | float | None = None,
        axis: int = 0,
        broadcast: bool = True,
    ) -> T_CentralMoments:
        """Push multiple statistics onto self."""
        self._raise_if_not_1d(self.mom_ndim)

        ar, target = self._check_vals(a, target="vals", axis=axis)
        vr = self._check_vars(v, target=target, axis=axis, broadcast=broadcast)
        wr = self._check_weights(w, target=target, axis=axis)
        self._push.stats(self._data_flat, wr, ar, vr)
        return self

    @classmethod
    def from_stat(
        cls: Type[T_CentralMoments],
        a: ArrayLike | float,
        v: np.ndarray | float = 0.0,
        w: np.ndarray | float | None = None,
        mom: Moments = 2,
        val_shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        order: ArrayOrder | None = None,
        **kws,
    ) -> T_CentralMoments:
        """Create object from single weight, average, variance/covariance."""
        mom_ndim = cls._mom_ndim_from_mom(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        if val_shape is None and isinstance(a, np.ndarray):
            val_shape = a.shape
        if dtype is None:
            dtype = a.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_stat(
            w=w, a=a, v=v
        )

    @classmethod
    def from_stats(
        cls: Type[T_CentralMoments],
        a: np.ndarray,
        v: np.ndarray,
        w: np.ndarray | float | None = None,
        axis: int = 0,
        dim=None,
        mom: Moments = 2,
        val_shape: Tuple[int, ...] = None,
        dtype: DTypeLike | None = None,
        order: ArrayOrder | None = None,
        **kws,
    ) -> T_CentralMoments:
        """Create object from several statistics.

        Weights, averages, variances/covarainces along
        axis.
        """

        mom_ndim = cls._mom_ndim_from_mom(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        # get val_shape
        if val_shape is None:
            val_shape = _shape_reduce(a.shape, axis)
        return cls.zeros(val_shape=val_shape, dtype=dtype, mom=mom, **kws).push_stats(
            a=a,
            v=v,
            w=w,
            axis=axis,
        )

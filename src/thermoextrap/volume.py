"""
Volume extrapolation (:mod:`~thermoextrap.volume`)
==================================================

Note: This only handles volume expansion to first order.
Also, Only DataCentralMomentsVals like objects (with ``resample_values = True``) are supported.
"""
# pylint: disable=duplicate-code

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Generic, SupportsIndex

import attrs
from attrs import field
from attrs import validators as attv
from module_utilities import cached

from thermoextrap.core.validate import validator_xarray_typevar
from thermoextrap.data import DataCentralMomentsVals
from thermoextrap.models import Derivatives, ExtrapModel

from .core.docstrings import DOCFILLER_SHARED
from .core.typing import DataDerivArgs, DataT, OptionalKwsAny, SupportsData
from .core.typing_compat import override
from .core.xrutils import xrwrap_xv
from .data import DataCallbackABC

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from cmomy import IndexSampler
    from xarray.core.dataarray import DataArray

    from .core.typing_compat import Self

docfiller_shared = DOCFILLER_SHARED.levels_to_top(
    "cmomy", "xtrap", "beta", "volume"
).decorate

# Need funcs to pass to Coefs class
# Just needs to be indexable based on order, so...
# d^n X / d V^n = funcs[n](*args)
# Could create list of simple functions for each derivative order
# But here I'm trying something similar to what's in xtrapy already
# For any general observable, might need to modify Data class to also pass full x values
# (or other data, perhaps)
# This is because the last, custom term may need more information than W and x*W moments
# Though ALL of the observables in the paper end up with a unique term that is just
# some constant multiplied by an average of x (same with ideal gas, too).


class VolumeDerivFuncs:
    """
    Calculates specific derivative values at reference volume V with data x and W.
    Only go to first order for volume extrapolation.
    Here W represents the virial instead of the potential energy.
    """

    def __getitem__(self, order: SupportsIndex) -> Callable[..., Any]:
        # Check to make sure not going past first order
        if (order := int(order)) > 1:
            raise IndexError(
                "Volume derivatives cannot go past 1st order"
                + " and received %i" % order
                + "\n(because would need derivatives of forces)"
            )
        return self.create_deriv_func(order)

    # TODO(wpk): move this to just a functions
    @staticmethod
    def create_deriv_func(order: int) -> Callable[..., Any]:
        """Derivative function for Volume derivatives"""

        # Works only because of local scope
        # Even if order is defined somewhere outside of this class, won't affect returned func
        def func(
            beta_virial: Any,
            x_beta_virial: Any,
            dxdq: Any,
            volume: float,
            ndim: int = 1,
        ) -> Any:
            r"""
            Calculate function.  dxdq is <sum_{i=1}^N dy/dx_i x_i>.

            :math:`beta_virial=\beta \dot W`, :math:`x_beta_virial = x \dot \beta \dot W`, where `W`
            is the virial.
            """
            # Zeroth order derivative
            return (
                x_beta_virial[0]
                if order == 0
                else (-x_beta_virial[0] * beta_virial[1] + x_beta_virial[1] + dxdq)
                / (volume * ndim)
            )

        return func


@lru_cache(5)
def factory_derivatives() -> Derivatives:
    """Factory function to provide coefficients of expansion."""
    deriv_funcs = VolumeDerivFuncs()
    return Derivatives(deriv_funcs)


@attrs.frozen
@docfiller_shared
class VolumeDataCallback(DataCallbackABC, Generic[DataT]):
    """
    Object to handle callbacks of metadata.

    Parameters
    ----------
    volume : float
        Reference value of system volume.
    {dxdqv}
    {ndim}

    See Also
    --------
    thermoextrap.data.DataCallbackABC
    """

    volume: float = field(validator=attv.instance_of(float))
    dxdqv: DataT = field(validator=validator_xarray_typevar)
    ndim: int = field(default=3, validator=attv.instance_of(int))

    _cache: dict[str, Any] = field(init=False, repr=False, factory=dict[str, "Any"])

    @override
    def check(self, data: SupportsData[Any]) -> None:
        pass

    @cached.meth
    def dxdq(self, rec_dim: str) -> DataT:
        return self.dxdqv.mean(rec_dim)

    @override
    def resample(
        self,
        data: SupportsData[Any],
        *,
        meta_kws: OptionalKwsAny,
        sampler: IndexSampler[Any],
        **kws: Any,
    ) -> Self:
        if not isinstance(data, DataCentralMomentsVals):
            msg = "resampling only possible with DataCentralMomentsVals object."
            raise NotImplementedError(msg)

        return self.new_like(dxdqv=self.dxdqv[sampler.indices])

    @override
    def deriv_args(
        self, data: SupportsData[Any], *, deriv_args: DataDerivArgs
    ) -> DataDerivArgs:
        if not isinstance(data, DataCentralMomentsVals):
            msg = "resampling only possible with DataCentralMomentsVals object."
            raise NotImplementedError(msg)

        return (
            *tuple(deriv_args),
            self.dxdq(data.rec_dim),
            self.volume,
            self.ndim,
        )


@docfiller_shared
def factory_extrapmodel(
    volume: float,
    uv: DataArray,
    xv: DataT,
    dxdqv: DataArray,
    ndim: int = 3,
    order: int = 1,
    alpha_name: str = "volume",
    rec_dim: str = "rec",
    val_dims: str = "val",
    rep_dim: str = "rep",
    **kws: Any,
) -> ExtrapModel[DataT]:
    """
    Factory function to create Extrapolation model for volume expansion.

    Parameters
    ----------
    {volume}
    volume : float
        reference value of volume
    uv, xv : array-like
        values for u and x
        Note that here, uv should be the temperature scaled virial `beta * virial`
    dxdqv : array-like
        values of `sum dx/dq_i q_i` where `q_i` is the ith coordinate
        This array is wrapped with `cmomy.data.xrwrap_xv`
    {ndim}
    order : int, default=1
        maximum order.  Only `order=1` is currently supported
    alpha_name, str, default='volume'
        name of expansion parameter
    {rec_dim}
    {val_dims}
    {rep_dim}
    **kws :
        Extra arguments to :class:`~.DataCentralMomentsVals`

    Returns
    -------
    extrapmodel : :class:`thermoextrap.models.ExtrapModel`

    """
    if order != 1:
        msg = "only order=1 is supported"
        raise ValueError(msg)

    dxdqv = xrwrap_xv(
        dxdqv, rec_dim=rec_dim, rep_dim=rep_dim, deriv_dim=None, val_dims=val_dims
    )

    meta = VolumeDataCallback(volume=volume, dxdqv=dxdqv, ndim=ndim)

    data = DataCentralMomentsVals(
        uv=uv,
        xv=xv,
        order=order,
        meta=meta,
        rec_dim=rec_dim,
        deriv_dim=None,
        resample_values=True,
        **kws,
    )

    derivatives = factory_derivatives()
    return ExtrapModel(
        alpha0=volume,
        data=data,
        derivatives=derivatives,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )

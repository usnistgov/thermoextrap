"""
Volume expansion for ideal gas (:mod:`~thermoextrap.volume_idealgas`)
=====================================================================
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, SupportsIndex

from .core.docstrings import DOCFILLER_SHARED
from .models import Derivatives, ExtrapModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from xarray.core.dataarray import DataArray

    from .core.typing import DataT, SupportsData

docfiller_shared = DOCFILLER_SHARED.levels_to_top(
    "cmomy", "xtrap", "beta", "volume"
).decorate


class VolumeDerivFuncsIG:
    """
    Calculates specific derivative values at reference_volume with data x and W.
    Only go to first order for volume extrapolation.
    Here W represents the virial instead of the potential energy.
    """

    def __init__(self, reference_volume: float = 1.0) -> None:
        # If do not set reference_volume, assumes virial data is already divided by the reference volume
        # If this is not the case, need to set reference_volume
        # Or if need reference_volume to also compute custom term, need to specify
        self.reference_volume = reference_volume

    def __getitem__(self, order: SupportsIndex) -> Callable[..., Any]:
        # Check to make sure not going past first order
        if (order := int(order)) > 1:
            raise IndexError(
                "Volume derivatives cannot go past 1st order"
                + " and received %i" % order
                + "\n(because would need derivatives of forces)"
            )
        return self.create_deriv_func(order)

    def create_deriv_func(self, order: int) -> Callable[..., Any]:
        # Works only because of local scope
        # Even if order is defined somewhere outside of this class, won't affect returned func

        def func(beta_virial: Any, x_beta_virial: Any) -> Any:
            if order == 0:
                # Zeroth order derivative
                deriv_val = x_beta_virial[0]

            else:
                # First order derivative
                deriv_val = (x_beta_virial[1] - x_beta_virial[0] * beta_virial[1]) / (
                    self.reference_volume
                )  # No 3 b/c our IG is 1D
                # Term unique to Ideal Gas... <x>/L
                # Replace with whatever is appropriate to observable of interest
                deriv_val += x_beta_virial[0] / self.reference_volume
            return deriv_val

        return func


@lru_cache(5)
def factory_derivatives(reference_volume: float = 1.0) -> Derivatives:
    """
    Factory function to provide coefficients of expansion.

    Parameters
    ----------
    reference_volume : float
        reference volume (default 1 - if already divided by volume no need to set)

    Returns
    -------
    derivatives : :class:`thermoextrap.models.Derivatives` object
        Object used to calculate moments
    """
    deriv_funcs = VolumeDerivFuncsIG(reference_volume=reference_volume)
    return Derivatives(deriv_funcs)


def factory_extrapmodel(
    volume: float,
    uv: DataArray,
    xv: DataT,
    order: int = 1,
    alpha_name: str = "volume",
    **kws: Any,
) -> ExtrapModel[DataT]:
    """
    Factory function to create Extrapolation model for volume expansion.

    Parameters
    ----------
    order : int
        maximum order
    volume : float
        reference value of volume
    uv, xv : DataArray or Dataset
        values for u and x
    alpha_name, str, default='volume'
        name of expansion parameter
    kws : dict
        extra arguments to `factory_datavalues`

    Returns
    -------
    extrapmodel : ExtrapModel
    """
    if order != 1:
        msg = "only first order supported"
        raise ValueError(msg)

    from .data import factory_data_values

    data = factory_data_values(
        uv=uv, xv=xv, order=order, central=False, xalpha=False, **kws
    )
    derivatives = factory_derivatives(reference_volume=volume)
    return ExtrapModel(
        alpha0=volume,
        data=data,
        derivatives=derivatives,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )


def factory_extrapmodel_data(
    volume: float,
    data: SupportsData[DataT],
    order: int | None = 1,
    alpha_name: str = "volume",
) -> ExtrapModel[DataT]:
    """
    Factory function to create Extrapolation model for volume expansion.

    Parameters
    ----------
    volume : float
        reference value of volume
    data : object
        Note that this data object should have central=False, deriv_dim=None
    alpha_name : str, default='volume'
        name of expansion parameter

    Returns
    -------
    extrapmodel : ExtrapModel
    """
    if order is None:
        order = data.order

    if order != 1:
        msg = "Only first order supported"
        raise ValueError(msg)
    if order > data.order:
        msg = f"{order=} > {data.order=}"
        raise ValueError(msg)
    if data.central:
        msg = "Only works with raw moments."
        raise ValueError(msg)
    if data.deriv_dim is not None:
        msg = "Missing deriv_dim"
        raise ValueError(msg)

    derivatives = factory_derivatives(reference_volume=volume)
    return ExtrapModel(
        alpha0=volume,
        data=data,
        derivatives=derivatives,
        order=order,
        minus_log=False,
        alpha_name=alpha_name,
    )

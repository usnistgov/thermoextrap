from warnings import warn

import numpy as np
import xarray as xr

import thermoextrap as xtrap
from thermoextrap.legacy.extrap import ExtrapModel

# from thermoextrap import xpan_vol as volxtrap
# from thermoextrap import xpan_vol_ig as volxtrap_ig

# from thermoextrap import xpan_vol, xpan_vol_ig


# With original extrapolation package
class VolumeExtrapModelIG(ExtrapModel):
    """Class to hold information about an extrapolation in size for a 1D system (e.g. our ideal gas model)"""

    # Can't go to higher order in practice, so don't return any symbolic derivatives
    # Instead, just use this to check and make sure not asking for order above 1
    def calcDerivFuncs(self) -> None:  # noqa: N802
        if self.maxOrder > 1:
            warn(
                "Volume extrapolation cannot go above 1st order without derivatives of forces. "
                "Setting order to 1st order.",
                stacklevel=1,
            )
            self.maxOrder = 1

    # And given data, calculate numerical values of derivatives up to maximum order
    # Will be very helpful when generalize to different extrapolation techniques
    # (and interpolation)
    @staticmethod
    def calcDerivVals(refL, x, W):  # noqa: N802, N803
        """
        Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives. Only go to first order for volume extrapolation. And
        here W represents the virial instead of the potential energy.
        """
        if x.shape[0] != W.shape[0]:
            msg = f"First observable dimension {x.shape[0]} and size of potential energy array {W.shape[0]} don't match!"
            raise ValueError(msg)
        w_transpose = np.array([W]).T
        x_ave = np.average(x, axis=0)
        w_ave = np.average(W)
        xw_ave = np.average(x * w_transpose, axis=0)
        deriv_vals = np.zeros((2, x.shape[1]))
        deriv_vals[0] = x_ave
        deriv_vals[1] = (xw_ave - x_ave * w_ave) / refL
        # Add the unique correction for the observable <x> in the ideal gas system
        # It turns out this is just <x> itself divided by L
        deriv_vals[1] += x_ave / refL
        return deriv_vals


def test_extrapmodel_vol(fixture) -> None:
    volume = 1.0
    volumes = [0.1, 0.5, 1.5, 2.0]

    em = VolumeExtrapModelIG(1, volume, fixture.x, fixture.u)

    xem_ig = xtrap.volume_idealgas.factory_extrapmodel(
        order=1,
        volume=volume,
        uv=fixture.u,
        xv=fixture.x,
    )

    xem = xtrap.volume.factory_extrapmodel(
        uv=fixture.u, xv=fixture.x, order=1, dxdqv=fixture.x, volume=volume, ndim=1
    )

    np.testing.assert_allclose(em.predict(volumes), xem_ig.predict(volumes))
    xr.testing.assert_allclose(xem_ig.predict(volumes), xem.predict(volumes))

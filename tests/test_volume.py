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
    def calcDerivFuncs(self):
        if self.maxOrder > 1:
            print(
                "Volume extrapolation cannot go above 1st order without derivatives of forces."
            )
            print("Setting order to 1st order.")
            self.maxOrder = 1
        return None

    # And given data, calculate numerical values of derivatives up to maximum order
    # Will be very helpful when generalize to different extrapolation techniques
    # (and interpolation)
    def calcDerivVals(self, refL, x, W):
        """Calculates specific derivative values at B with data x and U up to max order.
        Returns these derivatives. Only go to first order for volume extrapolation. And
        here W represents the virial instead of the potential energy.
        """
        if x.shape[0] != W.shape[0]:
            print(
                "First observable dimension (%i) and size of potential energy array (%i) don't match!"
                % (x.shape[0], W.shape[0])
            )
            return
        wT = np.array([W]).T
        avgX = np.average(x, axis=0)
        avgW = np.average(W)
        avgXW = np.average(x * wT, axis=0)
        derivVals = np.zeros((2, x.shape[1]))
        derivVals[0] = avgX
        derivVals[1] = (avgXW - avgX * avgW) / refL
        # Add the unique correction for the observable <x> in the ideal gas system
        # It turns out this is just <x> itself divided by L
        derivVals[1] += avgX / refL
        return derivVals


def test_extrapmodel_vol(fixture):
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

import numpy as np

from thermoextrap import idealgas
from thermoextrap.legacy.ig import IGmodel


def test_idealgas() -> None:
    model = IGmodel(1000)

    b = np.linspace(0.1, 10, 5)
    x = np.random.rand()
    np.testing.assert_allclose(idealgas.x_ave(b), model.avgX(b))
    np.testing.assert_allclose(idealgas.x_var(b), model.varX(b))
    np.testing.assert_allclose(idealgas.x_prob(x, b), model.PofX(x, b))
    np.testing.assert_allclose(
        idealgas.u_prob(x * 300, 1000, b), model.PofU(x * 300, b)
    )

    np.testing.assert_allclose(idealgas.x_cdf(x, b), model.cdfX(x, b))

    np.random.seed(0)
    _a = idealgas.x_sample(5, b[:, None])

    np.random.seed(0)
    _b = model.sampleX(b[:, None], 5)
    np.testing.assert_allclose(_a, _b)

    tota, coefa = model.extrapAnalytic(B=2.0, B0=1.0, order=5)
    totb, coefb = idealgas.x_beta_extrap(5, beta0=1.0, beta=2.0)
    np.testing.assert_allclose(tota, totb)
    np.testing.assert_allclose(coefa, coefb)

    beta = np.random.rand()
    np.random.seed(0)
    _xa, _ua = model.genData(beta, 100)

    np.random.seed(0)
    _xb, _ub = idealgas.generate_data((100, 1000), beta, 1.0)

    np.testing.assert_allclose(_xa, _xb)
    np.testing.assert_allclose(_ua, _ub)

    tota, coefa = idealgas.x_vol_extrap(order=5, vol0=0.3, beta=beta, vol=1.0)
    totb, coefb = model.extrapAnalyticVolume(1.0, 0.3, 5, beta)
    np.testing.assert_allclose(tota, totb)
    np.testing.assert_allclose(coefa, coefb)

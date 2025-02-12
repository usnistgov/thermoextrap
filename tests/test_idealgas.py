import numpy as np

from thermoextrap import idealgas
from thermoextrap.legacy.ig import IGmodel


def test_idealgas() -> None:
    model = IGmodel(1000)

    b = np.linspace(0.1, 10, 5)

    rng = np.random.default_rng(seed=1234)

    x = rng.random()
    np.testing.assert_allclose(idealgas.x_ave(b), model.avgX(b))
    np.testing.assert_allclose(idealgas.x_var(b), model.varX(b))
    np.testing.assert_allclose(idealgas.x_prob(x, b), model.PofX(x, b))
    np.testing.assert_allclose(
        idealgas.u_prob(x * 300, 1000, b), model.PofU(x * 300, b)
    )

    np.testing.assert_allclose(idealgas.x_cdf(x, b), model.cdfX(x, b))

    seed = 456
    a = idealgas.x_sample(5, b[:, None], rng=np.random.default_rng(seed))
    b_ = model.sampleX(b[:, None], 5, rng=np.random.default_rng(seed))
    np.testing.assert_allclose(a, b_)

    tota, coefa = model.extrapAnalytic(B=2.0, B0=1.0, order=5)
    totb, coefb = idealgas.x_beta_extrap(5, beta0=1.0, beta=2.0)
    np.testing.assert_allclose(tota, totb)
    np.testing.assert_allclose(coefa, coefb)

    beta = rng.random()
    xa, ua = model.genData(beta, 100, rng=np.random.default_rng(seed))
    xb, ub = idealgas.generate_data(
        (100, 1000), beta, 1.0, rng=np.random.default_rng(seed)
    )

    np.testing.assert_allclose(xa, xb)
    np.testing.assert_allclose(ua, ub)

    tota, coefa = idealgas.x_vol_extrap(order=5, vol0=0.3, beta=beta, vol=1.0)
    totb, coefb = model.extrapAnalyticVolume(1.0, 0.3, 5, beta)
    np.testing.assert_allclose(tota, totb, rtol=1e-5)
    np.testing.assert_allclose(coefa, coefb, rtol=1e-5)

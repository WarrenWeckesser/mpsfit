
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import gamma
from mpsfit import mpsfit


def test_two_sample_exponential():
    # Example 1 from https://en.wikipedia.org/wiki/Maximum_spacing_estimation
    # Exponential distribution with just two samples: x = [2, 4]
    # MPS estimate should be lambda = -ln(0.6)/2 (i.e. scale = -2/ln(0.6)).

    def cdf(x, theta):
        return gamma.cdf(x, 1, loc=0, scale=theta)

    def sf(x, theta):
        return gamma.sf(x, 1, loc=0, scale=theta)

    def pdf(x, theta):
        return gamma.pdf(x, 1, loc=0, scale=theta)

    def median(theta):
        return gamma.median(1, loc=0, scale=theta).item(0)

    x = np.array([2.0, 4.0])
    theta0 = 1.0
    params = mpsfit(x, theta0=theta0, cdf=cdf, sf=sf, pdf=pdf, median=median)
    lam = 1/params[0]
    assert_allclose(lam, -0.5*np.log(0.6), rtol=1e-7)

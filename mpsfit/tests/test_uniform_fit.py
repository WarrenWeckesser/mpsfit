
# import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import uniform
from mpsfit import mpsfit


def test_uniform_fit():
    # Example 2 from https://en.wikipedia.org/wiki/Maximum_spacing_estimation
    # a = loc
    # b = loc + scale
    #  <=>
    # loc = a
    # scale = b - a

    def cdf(x, theta):
        return uniform.cdf(x, loc=theta[0], scale=theta[1] - theta[0])

    def sf(x, theta):
        return uniform.sf(x, loc=theta[0], scale=theta[1] - theta[0])

    def pdf(x, theta):
        return uniform.pdf(x, loc=theta[0], scale=theta[1] - theta[0])

    def median(theta):
        return uniform.median(loc=theta[0], scale=theta[1] - theta[0]).item(0)

    loc = 1
    scale = 4
    rng = np.random.default_rng(6876652734511)
    x = rng.uniform(loc, loc + scale, size=20000)
    xmin = x.min()
    xmax = x.max()

    # Experimenting with the initial guess theta0...
    rng = x.ptp()
    pad = 0.025
    a0 = xmin - pad*rng
    b0 = xmax + pad*rng

    theta0 = np.array([a0, b0])

    # mean=True is needed here (i.e. compute the mean of the logs of the
    # spacings rather than just the sum).  To do: figure out why.  Scaling?
    # Some other numerical issue?
    params = mpsfit(x, theta0=theta0, cdf=cdf, sf=sf, pdf=pdf, median=median,
                    mean=True)
    n = len(x)
    ahat = (n*xmin - xmax)/(n - 1)
    bhat = (n*xmax - xmin)/(n - 1)
    assert_allclose(params, [ahat, bhat], rtol=1e-7)

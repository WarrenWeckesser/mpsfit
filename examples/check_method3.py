
import numpy as np
from scipy.stats import norm
from mpsfit._mpsfit import _mps_q_with_reps3, _mps_q_with_reps3_alt


# The PDF, CDF, survival function and median function for the normal
# distribution.


def pdf(x, theta):
    return norm.pdf(x, loc=theta[0], scale=theta[1])


def cdf(x, theta):
    return norm.cdf(x, loc=theta[0], scale=theta[1])


def sf(x, theta):
    return norm.sf(x, loc=theta[0], scale=theta[1])


def ppf(x, theta):
    return norm.ppf(x, loc=theta[0], scale=theta[1])


def median(theta):
    return norm.median(loc=theta[0], scale=theta[1])


# Input
x = np.array([-1.25, 0.5, 1.0, 4.0, 5.0])
counts = np.array([1, 2, 2, 1, 3])
delta = 0.005

theta = [-2, 4.1]
M = _mps_q_with_reps3_alt(theta, x, counts, cdf, sf, ppf, pdf, median, delta)
print(f"{M  = }")

# Compare to the result of _mps_q_with_reps3()
M3 = _mps_q_with_reps3(theta, x, counts, cdf, sf, pdf, median, delta)
print(f"{M3 = }")

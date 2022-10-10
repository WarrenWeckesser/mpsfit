
# Reproduce results from the paper:
#
#   M. Rahman, L. M. Pearson, U. R. Martinovic,
#   Method of product spacings in the two-parameter gamma distribution,
#   Journal of Statistical Research
#   2007, Vol. 41, No. 1, pp. 51–58.
#   ISSN 0256-422X

import numpy as np
from scipy.stats import gamma
from mpsfit import mpsfit
from gammafuncs import gamma_fit_mom


# The PDF, CDF, survival function and median function for the gamma
# distribution.

def pdf(x, theta):
    return gamma.pdf(x, theta[0], loc=0, scale=theta[1])


def cdf(x, theta):
    return gamma.cdf(x, theta[0], loc=0, scale=theta[1])


def sf(x, theta):
    return gamma.sf(x, theta[0], loc=0, scale=theta[1])


def median(theta):
    return gamma.median(theta[0], loc=0, scale=theta[1])


# This data set has ties.
failure_times = [620, 1285, 818, 871,  32, 253, 164, 560, 470, 260, 218,
                 393, 947, 399, 193, 531, 343, 376, 6, 860, 16, 1267, 151,
                 24, 89, 388, 106, 158, 1274, 32, 317, 85, 1512, 1792, 89,
                 1055, 352, 160, 689, 1119, 242, 103, 100, 152, 477, 403,
                 12, 134, 660, 1410, 250, 41, 47, 95, 76, 537, 101, 385,
                 195, 1279, 356, 1733, 2194, 763, 39, 460, 284, 103, 69,
                 158, 548, 381, 203, 1101, 32, 421, 515, 72, 1585, 176, 11,
                 565, 751, 500, 803, 555, 14, 45, 776, 1]

# Method of moments.
mom_shape, mom_scale = gamma_fit_mom(failure_times)

# MLE using gamma.fit().  Use the MOM data as the initial guess.
mle_shape, _, mle_scale = gamma.fit(failure_times, mom_shape, 0, mom_scale,
                                    floc=0)
theta0 = np.array([mle_shape, mle_scale])

# Maximum product spacing, using method 2 for handling ties.
# Method 2 agrees with the values quoted in the paper:
# alpha=0.80, beta=604.13.
mps2_shape, mps2_scale = mpsfit(failure_times, theta0=theta0,
                                cdf=cdf, sf=sf, pdf=pdf, median=median,
                                method=2)

print("Fit a gamma distribution to the failure times data set")
print(f"(sample size: {len(failure_times):d})")
print("                          shape   scale")
print(f"Method of moments:       {mom_shape:7.4f}  {mom_scale:8.4f}")
print(f"Maximum likelihood:      {mle_shape:7.4f}  {mle_scale:8.4f}")
print(f"Maximum product spacing: {mps2_shape:7.4f}  {mps2_scale:8.4f}")
print("MPS, from reference:      0.80    604.13")
print()

#
# Demonstration of Example 4.2 in "An Introduction to Statistical Modeling
# of Exteme Values" by Stuart Coles.  With c=1, the distribution of the peaks
# over threshold of the Frechet distribution (`invweibull` in `scipy.stats`)
# is the generalized Pareto distribution with shape parameter 1 and scale
# equal to the threshold.
#

import sys
import numpy as np
from scipy.stats import genpareto, invweibull
from scipy.special import chdtrc
from mpsfit import mpsfit


def pdf(x, theta):
    return genpareto.pdf(x, theta[0], loc=0, scale=theta[1])


def cdf(x, theta):
    return genpareto.cdf(x, theta[0], loc=0, scale=theta[1])


def sf(x, theta):
    return genpareto.sf(x, theta[0], loc=0, scale=theta[1])


def median(theta):
    return genpareto.median(theta[0], loc=0, scale=theta[1])


if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    print("seed =", seed)
else:
    seed = None

np.random.seed(seed)
sample_size = 1000000
c = 1.0
# invweibull is the distribution that is also commonly called the
# Frechet distribution.
sample = invweibull.rvs(c=c, size=sample_size)

threshold = 1000
peaks = sample[sample >= threshold] - threshold
print("sample size:", sample_size)
print("threshold:", threshold)
n = len(peaks)
print("num over threshold:", n)

theta0 = [c, threshold]
params, T = mpsfit(peaks, theta0=theta0,
                   cdf=cdf, sf=sf, pdf=pdf, median=median,
                   method=3, delta=0.001, returnT=True)

print('-'*72)
print(f"{params[0]:17.15f}  {params[1]:17.10f}  "
      f"(sample generated with c={c} threshold={threshold})")
print(f"{T = :f}")
p = chdtrc(n, T)
print(f"{p = :f}")

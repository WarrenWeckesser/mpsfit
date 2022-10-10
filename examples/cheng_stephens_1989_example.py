
import numpy as np
from scipy.stats import norm
from mpsfit import mpsfit

breaking_stress = np.array([
    27.55, 29.89, 30.07, 30.65, 31.23, 31.53, 31.53,
    31.82, 32.23, 32.28, 32.69, 32.98, 33.28, 33.28,
    33.74, 33.74, 33.86, 33.86, 33.86, 34.15, 34.15,
    34.15, 34.44, 34.62, 34.74, 34.74, 35.05, 35.03,
    35.32, 35.44, 35.61, 35.61, 35.73, 35.90, 36.20,
    36.78, 37.07, 37.36, 37.36, 37.36, 40.28,
])

# The PDF, CDF, survival function and median function for the normal
# distribution.


def pdf(x, theta):
    return norm.pdf(x, loc=theta[0], scale=theta[1])


def cdf(x, theta):
    return norm.cdf(x, loc=theta[0], scale=theta[1])


def sf(x, theta):
    return norm.sf(x, loc=theta[0], scale=theta[1])


def median(theta):
    return norm.median(loc=theta[0], scale=theta[1])


theta0 = [np.mean(breaking_stress), np.std(breaking_stress)]

# Note that Cheng & Stephens deal with repeated values using
# a delta of 0.005.

# Maximum product spacing, using method 2 for handling ties.
(mps2_loc, mps2_scale), T2 = mpsfit(breaking_stress, theta0=theta0,
                                    cdf=cdf, sf=sf, pdf=pdf, median=median,
                                    method=2, delta=0.005, returnT=True)

# Maximum product spacing, using method 3 for handling ties.
(mps3_loc, mps3_scale), T3 = mpsfit(breaking_stress, theta0=theta0,
                                    cdf=cdf, sf=sf, pdf=pdf, median=median,
                                    method=3, delta=0.005, returnT=True)


print()
print('-'*75)
print(f"MPS (2):    {mps2_loc:8.4f}  {mps2_scale:8.4f}  "
      f"(squared: {mps2_scale**2:8.4f}) T = {T2:.8g}")
print(f"MPS (3):    {mps3_loc:8.4f}  {mps3_scale:8.4f}  "
      f"(squared: {mps3_scale**2:8.4f}) T = {T2:.8g}")
print("MPS (ref):   34.072             (squared:   6.874)  T = 63.1")

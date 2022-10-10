# In the 2006 paper by Wong and Li [1], they apply MPS to four data sets from
# Castillo, et al [2].  Table 8 of Wong and Li (GPD is the generalized pareto
# distribution):
#
#     Estimated GPD parameters by MPS in four examples
#     ------------------------------------------------
#     Data  Threshold    gamma   sigma      M(theta)
#     ------------------------------------------------
#     Age    104.01       1.06    2.79       43.01
#     Wave    17.36       0.01    7.00       45.81
#     Flood   45.04      -0.03    9.62       60.53
#     Wind    36.82      -0.88    5.31       46.79
#
# According to Wong and Li, "The GPD was fitted to the excess over a threshold.
# The thresholds were taken from [Castillo, et al]."
# The only way I could get the result of this script to match Table 8 was to
# interpret that to mean the threshold is subtracted from each value in the
# data, and values less than or equal to 0 are discarded.
#
# [1] T. S. T. Wong and W. K. Li, A note on the estimation of extreme value
#     distributions using maximum product of spacings, IMS Lecture Notes-
#     Monograph Series, Time Series and Related Topics, Vol. 52 (2006) 272-283.
# [2] Castillo, E., Hadi, A. S., Balakrishnan, N. and Sarabia, J. M. (2005).
#     Extreme Value and Related Models with Applications in Engineering and
#     Science. John Wiley & Sons, Inc., New Jersey. MR2191401

import numpy as np
from mpsfit import mpsfit
from scipy.stats import genpareto

# Note that the shape parameter used by scipy.stats.genpareto, `c`, has the
# opposite sign of the shape parameter gamma used by Wong and Li.  I handle
# this in the function wrappers, so the values printed by the script follow the
# same convention as the paper.  Also note that in this example, the location
# parameter is fixed at 0.


def pdf(x, theta):
    return genpareto.pdf(x, -theta[0], loc=0, scale=theta[1])


def cdf(x, theta):
    return genpareto.cdf(x, -theta[0], loc=0, scale=theta[1])


def sf(x, theta):
    return genpareto.sf(x, -theta[0], loc=0, scale=theta[1])


def median(theta):
    return genpareto.median(-theta[0], loc=0, scale=theta[1])


def fit_exceedances(name, data, threshold, theta0, expected):
    exceedances = data[data > threshold] - threshold
    params = mpsfit(exceedances,
                    theta0=theta0, cdf=cdf, sf=sf, pdf=pdf, median=median,
                    method=3, delta=0.005)
    c, scale = params
    ref_c, ref_scale = expected
    print(f"{name:6} {c:8.4f} {scale:8.4f}    {ref_c:5.2f}   {ref_scale:5.2f}")


wind = np.loadtxt('castillo-et-al-data/Table1-1.txt')
flood = np.loadtxt('castillo-et-al-data/Table1-2.txt')
wave = np.loadtxt('castillo-et-al-data/Table1-3.txt')
age = np.loadtxt('castillo-et-al-data/Table1-5.txt')

# For most of these examples, the initial guess for the solution must
# be pretty close to the actual solution for the code to work.
# If it isn't, the solver usually generates the warning
#     RuntimeWarning: invalid value encountered in subtract
# and then fails to find a solution.
# It would be better if the code had some method for automatically
# generating a good initial guess--maybe the method of moments using
# L-moments.

print("             mpsfit         from the paper")
print("Data      gamma   sigma     gamma   sigma")

fit_exceedances('Age', data=age, threshold=104.01, theta0=[1.0, 2.5],
                expected=[1.06, 2.79])
fit_exceedances('Wave', data=wave, threshold=17.36, theta0=[0.1, 5],
                expected=[0.01, 7.00])
fit_exceedances('Flood', data=flood, threshold=45.04, theta0=[0, 10],
                expected=[-0.03, 9.62])
fit_exceedances('Wind', data=wind, threshold=36.82, theta0=[-1, 1],
                expected=[-0.88, 5.31])

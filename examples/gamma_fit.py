
import numpy as np
from scipy.stats import gamma
from mpsfit import mpsfit
from gammafuncs import gamma_fit_mom


def cdf(x, theta):
    return gamma.cdf(x, theta[0], loc=0, scale=theta[1])


def sf(x, theta):
    return gamma.sf(x, theta[0], loc=0, scale=theta[1])


def pdf(x, theta):
    return gamma.pdf(x, theta[0], loc=0, scale=theta[1])


def median(theta):
    return gamma.median(theta[0], loc=0, scale=theta[1])


rng = np.random.default_rng(12468279875651092)

# sample size
n = 200
shape = 5.5
scale = 0.25
x = gamma.rvs(shape, loc=0, scale=scale, size=n, random_state=rng)


print("Gamma distribution")
print("------------------")

print("shape =", shape, "  scale =", scale)
print("sample size:", n)
print()

print("method      shape      scale")
# Use the method of moments for the initial estimate of the parameters.
theta0 = gamma_fit_mom(x)
print("MOM     %10.6f %10.6f" % theta0)

mle = gamma.fit(x, floc=0)
print("MLE     %10.6f %10.6f" % (mle[0], mle[2]))

params = mpsfit(x, theta0=theta0, cdf=cdf, sf=sf, pdf=pdf, median=median)
print("MPS     %10.6f %10.6f" % tuple(params))
print()

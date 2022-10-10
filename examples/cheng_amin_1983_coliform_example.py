
import numpy as np
from scipy.stats import weibull_min
from mpsfit import mpsfit


conc = np.array([
     1364,  2154,  2236,  2518,  2527,
     2600,  3009,  3045,  4109,  5500,
     5800,  7200,  8400,  8400,  8900,
    11500, 12700, 15300, 18300, 20400,
])


def pdf(x, theta):
    return weibull_min.pdf(x, theta[1], loc=theta[0], scale=theta[2])


def cdf(x, theta):
    return weibull_min.cdf(x, theta[1], loc=theta[0], scale=theta[2])


def sf(x, theta):
    return weibull_min.sf(x,   theta[1], loc=theta[0], scale=theta[2])


def median(theta):
    return weibull_min.median(theta[1], loc=theta[0], scale=theta[2])


# theta = [alpha, beta, gamma]  (= [loc, shape, scale])
theta0 = np.array([1000, 1.0, 1000])

# Maximum product spacing, using method 2 for handling ties.
alpha, beta, gamma = mpsfit(conc, theta0=theta0,
                            cdf=cdf, sf=sf, pdf=pdf, median=median,
                            method=2)

print("                            alpha        beta     gamma")
print("                            (loc)        (shape)  (scale)")
print(f"Maximum product spacing:   {alpha:10.4f}  {beta:8.4f}  {gamma:10.4f}")
print("From the paper:             1085         0.95     6562")

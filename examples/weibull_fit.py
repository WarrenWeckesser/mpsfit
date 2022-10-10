
import numpy as np
from scipy.stats import weibull_max, genextreme
from mpsfit import mpsfit


# See:
# http://stackoverflow.com/questions/
#    38765996/fit-weibull-to-distribution-with-genextreme-and-weibull-min
# http://stats.stackexchange.com/questions/
#    132652/how-to-determine-which-distribution-fits-my-data-best


def gev_to_wmax(xi, mu, sigma):
    alpha = 1/xi
    a = sigma/xi
    b = mu + a
    return alpha, b, a


data = np.array([37.50,  46.79,  48.30,  46.04,  43.40,  39.25,  38.49,  49.51,
                 40.38,  36.98,  40.00,  38.49,  37.74,  47.92,  44.53,  44.91,
                 44.91,  40.00,  41.51,  47.92,  36.98,  43.40,  42.26,  41.89,
                 38.87,  43.02,  39.25,  40.38,  42.64,  36.98,  44.15,  44.91,
                 43.40,  49.81,  38.87,  40.00,  52.45,  53.13,  47.92,  52.45,
                 44.91,  29.54,  27.13,  35.60,  45.34,  43.37,  54.15,  42.77,
                 42.88,  44.26,  27.14,  39.31,  24.80,  16.62,  30.30,  36.39,
                 28.60,  28.53,  35.84,  31.10,  34.55,  52.65,  48.81,  43.42,
                 52.49,  38.00,  38.65,  34.54,  37.70,  38.11,  43.05,  29.95,
                 32.48,  24.63,  35.33,  41.34])


def cdf(x, theta):
    return weibull_max.cdf(x, theta[0], loc=theta[1], scale=theta[2])


def sf(x, theta):
    return weibull_max.sf(x, theta[0], loc=theta[1], scale=theta[2])


def pdf(x, theta):
    return weibull_max.pdf(x, theta[0], loc=theta[1], scale=theta[2])


def median(theta):
    return weibull_max.median(theta[0], loc=theta[1], scale=theta[2])


theta0 = weibull_max.fit(data)
wm_params = mpsfit(data, theta0=theta0, cdf=cdf, sf=sf, pdf=pdf, median=median)
print("Fit weibull_max")
print("weibull_max.fit: %12.8f %12.8f %12.8f" % theta0)
print("mpsfit:          %12.8f %12.8f %12.8f" % tuple(wm_params))


def cdf(x, theta):
    return genextreme.cdf(x, theta[0], loc=theta[1], scale=theta[2])


def sf(x, theta):
    return genextreme.sf(x, theta[0], loc=theta[1], scale=theta[2])


def pdf(x, theta):
    return genextreme.pdf(x, theta[0], loc=theta[1], scale=theta[2])


def median(theta):
    return genextreme.median(theta[0], loc=theta[1], scale=theta[2])


theta0 = genextreme.fit(data)
gev_params = mpsfit(data, theta0=theta0, cdf=cdf, sf=sf, pdf=pdf,
                    median=median)

print()
print("Fit genextreme, and express result in terms of weibull_max")
print("                   genextreme params.                      wmax params")
print("genextreme.fit:  %12.8f %12.8f %12.8f  %12.8f %12.8f %12.8f" %
      (tuple(theta0) + gev_to_wmax(*theta0)))
print("mpsfit:          %12.8f %12.8f %12.8f  %12.8f %12.8f %12.8f" %
      (tuple(gev_params) + tuple(gev_to_wmax(*gev_params))))

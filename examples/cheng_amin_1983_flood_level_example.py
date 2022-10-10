
import numpy as np
from scipy.stats import weibull_min
from mpsfit import mpsfit


level = np.array([
    0.654, 0.613, 0.315, 0.449, 0.297,
    0.402, 0.379, 0.423, 0.379, 0.3235,
    0.269, 0.740, 0.418, 0.412, 0.494,
    0.416, 0.338, 0.392, 0.484, 0.265,
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
theta0 = np.array([0.2, 1.0, 0.1])

mle = weibull_min.fit(level)

# Maximum product spacing, using method 2 for handling ties.
alpha, beta, gamma = mpsfit(level, theta0=theta0,
                            cdf=cdf, sf=sf, pdf=pdf, median=median, method=2)

print("                            alpha     beta        gamma")
print("                            (loc)     (shape)     (scale)")
print(f"MLE:                    {mle[1]:10.4f}  {mle[0]:8.4f}  {mle[2]:10.4f}")
print("MLE from the paper:         0.261     1.245       0.173")
print(f"Maximum product spacing:{alpha:10.4f}  {beta:8.4f}  {gamma:10.4f}")
print("MPS from the paper:         0.244     1.310       0.202")

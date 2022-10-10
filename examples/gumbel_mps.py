
import numpy as np
from mpsfit import mpsfit
from scipy.stats import gumbel_l, gumbel_r


data = np.array([ 436.69554799,  403.66261714,  700.05113046,  480.81943775,
                  540.59717961,  508.44020471,  441.58024658,  442.37747932,
                  462.29390664,  531.7675216 ,  454.13330111,  649.82483669,
                  362.30714799,  529.15843889,  510.43216302,  549.80251083,
                  419.92286214,  506.35441477,  667.942395  ,  613.4794213 ])


for dist in [gumbel_l, gumbel_r]:
    print(dist.name)

    def cdf(x, theta):
        return dist.cdf(x, loc=theta[0], scale=theta[1])

    def sf(x, theta):
        return dist.sf(x, loc=theta[0], scale=theta[1])

    def pdf(x, theta):
        return dist.pdf(x, loc=theta[0], scale=theta[1])

    def median(theta):
        return dist.median(loc=theta[0], scale=theta[1]).item(0)

    theta0 = dist.fit(data)
    print("MLE:", *theta0)
    params = mpsfit(data, theta0=theta0, cdf=cdf, sf=sf, pdf=pdf,
                    median=median)
    print("MPS:", *params)

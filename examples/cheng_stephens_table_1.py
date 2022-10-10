
import math
import numpy as np
from scipy.special import chdtri


print("Values of A/n  "
      "(Reproduce part of Table 1 from Cheng & Stephens (1989))")
print()
print("  n  α=0.9   α=0.95  α=0.99")
for n in [5, 10, 20]:
    print("%3d  " % n, end='')
    m = n + 1
    for alpha in [0.9, 0.95, 0.99]:
        gamma_m = m*(math.log(m) + np.euler_gamma) - 0.5 - 1/(12*m)
        sigma2_m = m*(np.pi**2/6 - 1) - 0.5 - 1/(6*m)
        # Cheng & Stephens (1989), Eq. (2.1):
        C1 = gamma_m - math.sqrt(0.5 * n * sigma2_m)
        C2 = math.sqrt(sigma2_m/(2*n))

        S = chdtri(n, 1 - alpha)
        A = C2*S + C1
        print("%5.3f   " % (A/n), end='')
    print()

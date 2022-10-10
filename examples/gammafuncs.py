
import numpy as np


def gamma_fit_mom(x):
    """
    Method of moments fit of a gamma distribution to `x`.

    Returns the estimated shape and scale.
    """
    var = np.var(x)
    mean = np.mean(x)
    mom_shape = mean**2 / var
    mom_scale = var / mean
    return mom_shape, mom_scale

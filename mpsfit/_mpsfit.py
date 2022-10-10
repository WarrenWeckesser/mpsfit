
import warnings
import math
import numpy as np
from scipy.special import entr
from scipy.optimize import fmin, fmin_bfgs


def _deltaF(x, theta, cdf, sf, median):
    # x must be a sorted 1-d numpy array.
    # The length of the return value is one more than x.size.

    med = median(theta)
    m = x.searchsorted(med)

    F = np.empty(m+2)
    F[0] = 0
    if m == len(x):
        F[1:-1] = cdf(x[:m], theta)
        F[-1] = 1
        deltaF = np.diff(F)
    else:
        F[1:] = cdf(x[:m+1], theta)
        deltaF = np.empty(len(x)+1)
        deltaF[:m+1] = np.diff(F)

        F = np.empty(len(x) - m + 1)
        F[-1] = 0
        F[:-1] = sf(x[m:], theta)
        deltaF[m+1:] = np.diff(F[::-1])[::-1]

    return deltaF


def _mps_q(theta, x, cdf, sf, median, mean=False):
    # x must be a sorted 1-d numpy array of unique values (no ties).
    # cdf, sf and median must be callables with signatures
    #     cdf(x, theta)
    #     sf(x, theta)
    #     median(theta)
    # respectively.
    #
    # The value returned by this function is the negative
    # sum of the log of delta-F, divided by len(x)+1. To use
    # the optimal value to compute the T-statistic, multiply
    # by len(x)+1 first.

    n = len(x)
    deltaF = _deltaF(x, theta, cdf, sf, median)
    if np.any(deltaF == 0):
        q = np.inf
    else:
        q = -np.log(deltaF).sum()
        if mean:
            q /= n + 1
    return q


"""
def _mps_q_with_reps1(theta, x, counts, cdf, sf, pdf, median):
    # x must be a sorted 1-d numpy array.
    # cdf must be a callable with signature cdf(x, theta).

    # First method from
    #
    # Umesh Singh, Sanjay Kumar Singh and Rajwant Kumar Singh,
    # Product Spacings as an Alternative to Likelihood for Bayesian Inferences,
    # Journal of Statistics Applications & Probability 3, No. 2, 179-188 (2014)
    #
    # for handling repeated values.  The justification for this method seems
    # hazy.  If x[i] is repeated, why should the value of F at the (possibly
    # distant) neighbor play a role in the repeated value's contribution to
    # the sum? So I think this method can be dropped.

    # XXX This function assumes that the lower and upper bounds of the support
    # do not occur in x!
    deltaF = _deltaF(x, theta, cdf, sf, median)
    # Adjust for ties.
    deltaF[:-1] /= counts
    deltaF[:-1] **= counts

    if np.any(deltaF == 0):
        q = np.inf
    else:
        ##q = -np.log(deltaF).sum() / len(x)
        ##q = -np.log(deltaF).sum() / np.sum(counts)
        q = -np.log(deltaF).sum()
    return q
"""


def _mps_q_with_reps2(theta, x, counts, cdf, sf, pdf, median, mean=False):
    # Second method from
    #
    # Umesh Singh, Sanjay Kumar Singh and Rajwant Kumar Singh,
    # Product Spacings as an Alternative to Likelihood for Bayesian Inferences,
    # Journal of Statistics Applications & Probability 3, No. 2, 179-188 (2014)
    #
    # for handling repeated values.  The value returned by this method is not
    # scaled properly for use in the T-statistic.

    # x must be a sorted 1-d numpy array.
    # cdf must be a callable with signature cdf(x, theta).
    # sf must be a callable with signature sf(x, theta).
    # pdf must be a callable with signature pdf(x, theta).
    # median must be a callable with signature median(theta).

    # XXX This function assumes that the lower and upper bounds of the support
    # do not occur in x!

    deltaF = _deltaF(x, theta, cdf, sf, median)

    # Adjust for ties.
    deltaF[:-1] *= pdf(x, theta)**(counts - 1)

    if np.any(deltaF == 0):
        q = np.inf
    else:
        # q = -np.log(deltaF).sum() / len(x)
        # q = -np.log(deltaF).sum() / np.sum(counts)
        q = -np.log(deltaF).sum()
        if mean:
            q /= np.sum(counts) + 1
    return q


def _mps_q_with_reps3(theta, x, counts, cdf, sf, pdf, median, delta,
                      mean=False):
    # Method outlined in section 4.2 of Cheng & Stephens (1989)
    # x must be a sorted 1-d numpy array.
    # cdf must be a callable with signature cdf(x, theta).
    # etc.

    # Suppose x is a value repeated r times.  In effect, this method
    # replaces the repeated values with r distinct points between
    # x - delta and x + delta, chosen so that their CDF values are evenly
    # spaced.

    # XXX Suppose x[i] is repeated.  What happens if x[i]-delta is less
    # than x[i-1]?  At a minimum, the code should check for this and raise
    # an error.  Perhaps better is to generate an array in which all the
    # repeated values are replaced by their "spread out" values, and then
    # that array is sorted again.  (But what to do if *that* array ends up
    # with repeated values?)

    # Currently delta must be a scalar.  But it might make sense
    # to also allow delta to be a vector with the same length as x.

    t = (counts > 1).astype(int) + 1
    # print("t:",t)
    xx = np.repeat(x, t)
    isupper = np.r_[0, (xx[1:] == xx[:-1]).astype(int)]
    islower = np.r_[isupper[1:], 0]
    # print("islower:", islower)
    # print("isupper:", isupper)
    xx += delta*(isupper - islower)
    deltaF = _deltaF(xx, theta, cdf, sf, median)
    # print("deltaF:", deltaF)
    if np.any(deltaF <= 0):
        return np.inf
    log_deltaF = np.log(deltaF)
    # Now replace any log_deltaF[i] that is associated with a
    # count r > 1 with (r - 1)*log_deltaF - (r - 1)*log(r - 1)
    rm1 = counts[counts > 1] - 1
    repmask = np.r_[isupper.astype(bool), False]
    # print("rm1:", rm1)
    log_deltaF[repmask] *= rm1
    log_deltaF[repmask] += entr(rm1)

    # XXX Divide by len(x) or sum(counts)? Or don't scale at all?
    # q = -log_deltaF.sum() / len(xx)
    # q = -log_deltaF.sum() / np.sum(counts)
    # q = -log_deltaF.sum() / (np.sum(counts)+1)
    q = -log_deltaF.sum()
    if mean:
        q /= (np.sum(counts) + 1)
    return q


def _mps_q_with_reps3_alt(theta, x, counts, cdf, sf, ppf, pdf, median, delta):
    # A different implementation of the method 3 objective function.
    # Requires ppf.
    xx = np.empty(counts.sum())
    k = 0
    for j in range(len(x)):
        if counts[j] == 1:
            xx[k] = x[j]
            k += 1
        else:
            F0 = cdf(x[j] - delta, theta)
            F1 = cdf(x[j] + delta, theta)
            xr = ppf(np.linspace(F0, F1, counts[j]), theta)
            xx[k:k+counts[j]] = xr
            k += counts[j]

    dF = _deltaF(xx, theta, cdf, sf, median)

    M = -np.sum(np.log(dF))
    return M


def mpsfit(x, *, theta0, cdf, sf, pdf, median, method=2, delta=None,
           returnT=False, mean=False):
    """
    Fit a distribution to data using the method of Maximal Product Spacing.

    The method is also known as Maximal Spacing Estimation.

    XXX API is temporary.

    Parameters
    ----------
    x : sequence
        The data set.  `x` is expected to be a one-dimensional sequence.
    theta0 : scalar or sequence
        The initial value of the parameters.
    cdf : callable
        The cumulative distribution function; must have the signature
            cdf(x, theta)
        and return the CDF value for each quantile in x.
    sf : callable
        The survival function of the distribution; must have the signature
            sf(x, theta)
    pdf : callable
        The probability density function of the distribution; must have the
        signature
            pdf(x, theta)
    median : callable
        The median of the distribution; must have the signature
            median(theta)
    method : [TO DO]
        Determines how repeated values in `x` are handled.
        This argument is ignored if there are no repeated values in `x`.
        TO DO: Explain the two methods.  Currently only the values 2 and 3
        are accepted.  This is a temporary API; the values should be something
        more descriptive.
    delta : float
        This argument is required when `method` is 3.
        TO DO: explain what this argument is.

    Return value
    ------------
    theta :
        The best fit parameters according to the MPS method.

    """
    # scipy.optimize.fmin is used.  If this is changed, an optimizer that
    # can handle an objective function that returns infinity must be used
    # (or the objective functions must be changed to not return infinity).
    #
    # TO DO:
    # o Add an option for fixing some parameter values.
    # o Add options for passing minimizer arguments (xtol, ftol, etc.) to fmin.
    # o Add goodness-of-fit statistics to the output. There is some information
    #   about goodness of fit (Moran's statistic, etc.) on wikipedia:
    #       https://en.wikipedia.org/wiki/Maximum_spacing_estimation
    #   See Cheng & Stephens, "A Goodness-Of-Fit Test Using Moran's Statistic
    #   with Estimated Parameters", Biometrika, Vol. 76, No. 2 (Jun., 1989),
    #   pp. 385-392

    if method not in [2, 3]:
        raise ValueError("method must be 2 or 3.")

    ux, counts = np.unique(x, return_counts=True)
    if ux.size != len(x):
        # There are repeated values in `x`.
        if method == 2:
            result = fmin(_mps_q_with_reps2, theta0,
                          args=(ux, counts, cdf, sf, pdf, median, mean),
                          xtol=1e-11, ftol=1e-12, maxiter=100000, disp=0,
                          full_output=True)
            params, fopt, niter, funccalls, warnflag = result
        else:  # method == 3
            if delta is None:
                raise ValueError('delta must be given when method is 3.')
            result = fmin(_mps_q_with_reps3, theta0,
                          args=(ux, counts, cdf, sf, pdf, median, delta, mean),
                          xtol=1e-11, ftol=1e-12, maxiter=100000, disp=0,
                          full_output=True)
            params, fopt, niter, funccalls, warnflag = result
    else:
        # `x` has no repeated values.  `method` is ignored in this case.
        # XXX With ftol=1e-12, I observed a case where `fmin` would always
        # hit the maximum number iteration (up tp 100000).
        result = fmin(_mps_q, theta0,
                      args=(ux, cdf, sf, median, mean),
                      xtol=1e-11, ftol=1e-10, maxiter=100000, disp=0,
                      full_output=True)
        params, fopt, niter, funccalls, warnflag = result
        if warnflag != 0:
            msgs = {1: "Maximum number of function evaluations made.",
                    2: "Maximum number of iterations reached."}
            warnings.warn("Minimization routine scipy.optimize.fmin() "
                          "returned warnflag=%d: %s" %
                          (warnflag, msgs[warnflag]))

    if returnT:
        # The notation here follows section 2 of Cheng & Stevens (1989).
        n = np.sum(counts)
        m = n + 1
        k = len(theta0)
        gamma_m = m*(math.log(m) + np.euler_gamma) - 0.5 - 1/(12*m)
        sigma2_m = m*(np.pi**2/6 - 1) - 0.5 - 1/(6*m)
        # Cheng & Stephens (1989), Eq. (2.1):
        C1 = gamma_m - math.sqrt(0.5 * n * sigma2_m)
        C2 = math.sqrt(sigma2_m/(2*n))
        if ux.size != len(x) and method != 3:
            # When there are repeated values, method 3 is used to
            # recompute fopt, using the parameters that were found
            # using method 2.  Method 2 will correctly find the
            # optimal parameters, but the actual value does not match
            # the formula required for computing the chi-squared
            # statistic.
            if delta is None:
                raise ValueError("When there are repeated values in x, "
                                 "delta must be given in order to compute "
                                 "the chi-squared statistic T.")
            # Recompute fopt using method 3 so it has the appropriate
            # scaling for the chi2 statistic.
            fopt = _mps_q_with_reps3(params, ux, counts, cdf, sf, pdf, median,
                                     delta, mean)
            print("recomputed fopt =", fopt)
        # Cheng & Stephens (1989), Eq. (2.3):
        T = (fopt + 0.5*k - C1) / C2

        # This is a hack.  Better would be a robust function to numerically
        # compute the Hessian. Even better would be an explicit function that
        # compute the analytical Hessian.
        print("Recomputing fopt to get the inverse Hessian")
        args = (ux, counts, cdf, sf, pdf, median, delta, mean)
        result = fmin_bfgs(_mps_q_with_reps3, x0=0.99*np.asarray(params),
                           args=args, gtol=1e-8, maxiter=10000,
                           full_output=True, disp=True)
        params2, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = result
        print("fopt:", fopt)
        print("warnflag:", warnflag)
        print(f"{params=}, {params2=}")
        print("inverse Hessian:")
        print(Bopt)
        print("SE(?):", np.sqrt(np.diag(Bopt)))

        return params, T
    else:
        return params

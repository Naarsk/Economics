import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import romb
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from scipy_distribution import dump_distribution, CustomDistribution
from itertools import product

##############################################################################
# 1) Define your characteristic function psi(t).
##############################################################################
def _psi_tld(k, loc, scale, alpha, lam, beta):
    term1 = -1j * loc * k
    term2 = scale ** alpha * ((lam ** alpha - (k ** 2 + lam ** 2) ** (alpha / 2)) / np.cos(np.pi * alpha / 2)) * np.cos(alpha * np.arctan(np.abs(k) / lam))
    term3 = (1 + 1j * np.sign(k) * beta * np.tan(alpha * np.arctan(np.abs(k) /lam)))
    return term1 - term2 * term3

##############################################################################
# 2) Define the PDF via inverse Fourier transform (real part).
##############################################################################

def _pdf_from_cf(x, loc, scale, alpha, lam, beta, k_max=50.0, ln_2_k_points=15, x_lim=250):

    if np.abs(x) > x_lim*scale:
        result = 0
    else:
        def integrand_real(k):
            return np.exp(-1j * k * x - _psi_tld(k, loc, scale, alpha, lam, beta)).real

        k_grid, dk = np.linspace(-k_max, k_max, 2**ln_2_k_points+1,retstep=True)

        # We take the real part of the integral:
        result = romb(integrand_real(k_grid), dx= float(dk))/ (2 * np.pi)

    return result

##############################################################################
# 3) Vectorize it
##############################################################################

pdf_from_cf = np.vectorize(_pdf_from_cf)

##############################################################################
# 4) Dump the values on file
##############################################################################

def dump_levy(loc, scale, alpha, lam, beta, filename='truncated_levy', x_min=-15, x_max =15, x_points=10000, k_max=50.0, ln_2_k_points=15):
    x_values = np.linspace(x_min,x_max,x_points)
    pdf_values = pdf_from_cf(x_values,loc, scale, alpha, lam, beta,k_max=k_max, ln_2_k_points=ln_2_k_points)
    filename = filename + '_' + str(np.round(loc,2)) + '_' +  str(np.round(scale,2)) + '_' +   str(np.round(alpha,2)) + '_' +   str(np.round(lam,2)) + '_' +   str(np.round(beta,2))
    dump_distribution(x_vals=x_values,pdf_vals=pdf_values,filename=filename)
    return filename

##############################################################################
# 5) Read the values from file
##############################################################################

def load_levy(filename):
    return CustomDistribution(data_file=filename, momtype=0, name="Truncated Skewed Lévy")


##############################################################################
# 6) Fit the PDF
##############################################################################

def histo_fit_levy(data, alpha_grid, lam_grid, beta_grid, n_bins =30, k_max=50.0, ln_2_k_points=15):

    x_values, bins, _ = plt.hist(data, bins = n_bins, density=True)
    plt.close()
    y_values = np.array(bins[1:]-bins[:-1])

    min_error=np.inf
    optimal_params = np.ones(3)

    for alpha, lam, beta in product(alpha_grid,lam_grid,beta_grid):

        error_values = pdf_from_cf(x_values,0,1,alpha,lam,beta,k_max=k_max,ln_2_k_points=ln_2_k_points)-y_values

        total_squared_error = np.sum(error_values**2)

        if total_squared_error < min_error:

            min_error = total_squared_error

            optimal_params = np.array([alpha, lam, beta])

    return optimal_params


##############################################################################

def log_likelihood(params, data, k_max=50.0, ln_2_k_points=15):
    """Compute negative log-likelihood for given parameters."""
    a, l, b = params  # Unpack parameters
    likelihoods = pdf_from_cf(data, 0, 1, a, l, b, k_max=k_max,ln_2_k_points=ln_2_k_points)

    # Ensure no log(0) issues: If any likelihood is 0, return -inf
    if np.any(likelihoods <= 0):
        return np.inf

    return -np.sum(np.log(likelihoods))  # Negative log-likelihood for minimization

def scipy_fit_levy(data, initial_guess, k_max=50.0, ln_2_k_points=15):
    """Fits the distribution parameters using MLE."""
    result = minimize(log_likelihood, args=(data, k_max, ln_2_k_points), x0=initial_guess,
                      method='Nelder-Mead')  # Optimization algorithm
    return result

def gs_fit_levy(data, alpha_grid, lam_grid, beta_grid, k_max=50.0, ln_2_k_points=15):

    max_loglikelihood = -np.inf

    optimal_params = np.ones(3)

    for alpha, lam, beta in product(alpha_grid, lam_grid, beta_grid):

        likelihoods = pdf_from_cf(data, 0, 1, alpha, lam, beta, k_max=k_max, ln_2_k_points=ln_2_k_points)

        if np.any(likelihoods <= 0):
            return 0

        total_log_likelihood = np.sum(np.log(likelihoods))

        if total_log_likelihood > max_loglikelihood:
            max_loglikelihood = log_likelihood

            optimal_params = np.array([alpha, lam, beta])

    return optimal_params


def symmetric_logspace(n, x_max, base=10):
    # Generate log-spaced values in positive domain
    positive = np.logspace(np.log10(x_max / base), np.log10(x_max), num=n, base=base)

    # Mirror to negative side and include zero
    return np.concatenate([-positive[::-1], [0], positive])


def kde_fit_levy(data, alpha_grid, lam_grid, beta_grid, n =100, x_max=15, k_max=50.0, ln_2_k_points=15):
    kde = gaussian_kde(data)

    x_values = symmetric_logspace(n, x_max)
    y_values = kde(x_values)

    min_error=np.inf

    optimal_params = np.ones(3)

    for alpha, lam, beta in product(alpha_grid,lam_grid,beta_grid):

        error_values = pdf_from_cf(x_values,0,1,alpha,lam,beta,k_max=k_max,ln_2_k_points=ln_2_k_points)-y_values

        total_squared_error = np.sum(error_values**2)

        if total_squared_error < min_error:

            min_error = total_squared_error

            optimal_params = np.array([alpha, lam, beta])

    return optimal_params

def kde_fit_levy_2(data, mu_grid, c_grid, alpha_grid, lam_grid, beta_grid, n =100, x_max=15, k_max=50.0, ln_2_k_points=15):
    kde = gaussian_kde(data)

    x_values = symmetric_logspace(n, x_max)
    y_values = kde(x_values)

    min_error=np.inf

    optimal_params = np.ones(5)

    for mu,c ,alpha, lam, beta in product(mu_grid, c_grid,alpha_grid,lam_grid,beta_grid):

        error_values = pdf_from_cf(x_values,mu,c,alpha,lam,beta,k_max=k_max,ln_2_k_points=ln_2_k_points)-y_values

        total_squared_error = np.sum(error_values**2)

        if total_squared_error < min_error:

            min_error = total_squared_error

            optimal_params = np.array([mu,c,alpha, lam, beta])

    return optimal_params
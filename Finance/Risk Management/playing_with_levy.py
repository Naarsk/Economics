import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import romb
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from itertools import product
import time




##############################################################################
# 3) Example usage: compute and plot the PDF over a range of x values.
##############################################################################
    # Choose some parameter values (you must set these appropriately):
mu = 0.0
c = 1.0
alpha = 1.212
lam = 0.02
beta = -0.09

plot = False
if plot:
    # Range of x for which we want the PDF:
    x_values = np.linspace(-10, 20, 1000)
    start_time = time.time()
    # Compute the PDF for each x:
    pdf_values_real = [pdf_from_cf(x, mu, c, alpha, lam, beta) for x in x_values]
    print("--- %s seconds for 1000 evaluations" % (time.time() - start_time))
    # It takes 12.5 seconds for 1000 evaluations
    pdf_gaussian = stats.norm.pdf(x_values, loc=mu, scale=c)
    # Plot the resulting PDF:
    plt.figure(figsize=(7, 5))
    plt.plot(x_values, pdf_values_real, label='Truncated Lévy')
    plt.plot(x_values, pdf_gaussian, label='Gaussian')

    plt.title("Truncated Lévy vs Gaussian PDF")
    plt.xlabel("x")
    plt.ylabel("pdf(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

save = False
if save:
    x_values = np.linspace(-15,15,10000)
    y = [pdf_from_cf(x, mu=0, c=1, alpha=.7, lam=0.02, beta=-0.9, k_max=50) for x in x_values]
    dump_distribution(x_values, y, 'skewed_truncated_levy')

moments = False
if moments:
    x_max = 150
    ln_2_x_points = 10
    x_grid, dx = np.linspace(-x_max, x_max, 2**ln_2_x_points+1,retstep=True)
    values = np.array([pdf_from_cf(x, mu, c, alpha, lam, beta, k_max=50) for x in x_grid])
    norm = values.sum()*dx
    mean = np.sum(values*x_grid)*dx
    second_moment = np.sum(values*x_grid**2)*dx
    print("Normalization: " , norm)
    print("Mean: ", mean)
    print("2nd Moment: ", second_moment)

make_dist = True
if make_dist:
    class TruncatedLevy(stats.rv_continuous):
        def __init__(self, alpha, lam, beta, x_max = 50,*args, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.lam = lam
            self.beta = beta
            self.x_max = x_max

            r_values = np.logspace(10**(-4), np.log(self.x_max), 50)
            l_values = -r_values

            self.pdf_values = pdf_from_cf(self.x_values, mu=0.0, c=1.0, alpha=self.alpha, lam=self.lam, beta=self.beta, k_max=50,x_lim=self.x_max)

            self.pdf_interpolator = interp1d(
                self.x_values,
                self.pdf_values,
                bounds_error=False,
                fill_value=0.0,  # PDF=0 outside the range
                kind='linear'
            )

            # Optionally, pre-compute CDF by cumulative integration
            # to implement _cdf() or _ppf(). For demonstration, we skip that.

        def _pdf(self, x):
            """
            Return the PDF at points x by interpolation.
            """
            return self.pdf_interpolator(x)
    start_time = time.time()
    # Now create an instance of the distribution.
    truncated_levy = TruncatedLevy(momtype=0, a=-250, b=250, name="Skewed Truncated Levy", alpha=1.2, lam=0.02, beta=-0.9)
    # Now stats.make_distribution will work because truncated_levy now has _shape_info set.
    x_values = np.linspace(-10, 20, 1000)
    # Compute the PDF for each x:
    pdf_values = truncated_levy.pdf(x_values)
    cdf_values = truncated_levy.cdf(x_values)
    print("--- %s seconds for 1000 evaluations" % (time.time() - start_time))
    # It takes 12.5 seconds for 1000 evaluations
    plt.figure(figsize=(7, 5))
    plt.plot(x_values, pdf_values, label='Truncated Lévy')
    plt.plot(x_values, cdf_values, label='Truncated Lévy')
    plt.show()

fitting = False
if fitting:

    def log_likelihood(params, data):
        """Compute negative log-likelihood for given parameters."""
        m, c, a, l, b = params  # Unpack parameters
        likelihoods = np.array([pdf_from_cf(x, m, c, a, l, b) for x in data])

        # Ensure no log(0) issues: If any likelihood is 0, return -inf
        if np.any(likelihoods == 0):
            return np.inf

        return -np.sum(np.log(likelihoods))  # Negative log-likelihood for minimization

    def fit_distribution(data, initial_guess):
        """Fits the distribution parameters using MLE."""
        result = minimize(log_likelihood, initial_guess, args=(data,),
                          method='Nelder-Mead')  # Optimization algorithm
        return result.x  # Returns best-fit parameters


    data = np.random.normal(0, 1, 1000)  # Example data

    # Initial guess for parameters (a, b, c)
    initial_guess = [0, 1, 1.6, 0.23, 0.27]

    # Fit the distribution
    best_params = fit_distribution(data, initial_guess)

    # Print the best fit parameters
    print("Best-fit parameters:", best_params)

    x_values = np.linspace(2*data.min(),2*data.max(),1000)

    y_values = pdf_from_cf(x_values, best_params[0],best_params[1],best_params[2],best_params[3],best_params[4])

    plt.hist(data, bins =30, density=True)
    plt.plot(x_values,y_values)
    plt.show()

histo_fitting = False
if histo_fitting:
    n_points = 7
    n_bins =30
    print("estimated time %s minutes" % (int(n_points**5*n_bins/1000*12.5/60)))
    data = np.random.normal(0, 1, 1000)
    x_values, bins, _ = plt.hist(data, bins = n_bins, density=True)
    y_values = np.array(bins[1:]-bins[:-1])

    mu_grid = np.linspace(-0.1,0.1,n_points)
    c_grid = np.linspace(0.9,1.1,n_points)
    alpha_grid = np.linspace(1.2,1.4,n_points)
    lam_grid = np.linspace(0.01,0.03,n_points)
    beta_grid = np.linspace(-0.1,-0.05,n_points)

    min_error=np.inf
    optimal_params = np.ones(5)
    for mu, c, alpha, lam, beta in product(mu_grid,c_grid,alpha_grid,lam_grid,beta_grid):
        error_values = pdf_from_cf(x_values,mu,c,alpha,lam,beta)-y_values
        total_squared_error = np.sum(error_values**2)
        if total_squared_error < min_error:
            min_error = total_squared_error
            optimal_params = np.array([mu, c, alpha, lam, beta])

    print(optimal_params)
    # Range of x for which we want the PDF:
    x_plot = np.linspace(x_values.min()*1.5, x_values.max()*1.5, 1000)

    # Compute the PDF for each x:
    pdf_values = [pdf_from_cf(x,optimal_params[0],optimal_params[1],optimal_params[2],optimal_params[3],optimal_params[4]) for x in x_values]
    plt.plot(x_plot,pdf_values)
    plt.show()
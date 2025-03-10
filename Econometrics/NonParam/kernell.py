import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set seed for replication
np.random.seed(12345678)

# Number of observations
n = 100

# Generate the data
X = np.random.normal(0, 1, n)
U = np.random.normal(0, 1, n)
Y = X**2 + U

# Generate the x-values at which pdf_X(x) and E(Y|X=x) are to be estimated
x_grid = np.arange(-5, 5, 0.01)

# Define Gaussian and uniform kernels
def gaussian_kernel(u):
    return norm.pdf(u)

def uniform_kernel(u):
    return 0.5 * np.where((u >= -1) & (u <= 1), 1, 0)

# Function for estimating pdf_X(x) and E(Y|X=x)
def kernel_smoother(y_data, x_data, x_values, n, h, kernel):
    d_matrix = np.outer(x_data, x_values) - np.outer(x_data, np.ones_like(x_values))
    k_matrix = kernel(d_matrix / h)
    den = np.sum(k_matrix, axis=0) / (n * h)
    num = np.sum(k_matrix * y_data[:, None], axis=0) / (n * h)
    est = {'f_hat': den, 'mu_hat': num / den}
    return est

# Function for estimating the LOO estimator of mu_hat
def mu_hat_loo(y_data, x_data, N, h, kernel):
    d_matrix = np.outer(x_data, x_data) - np.outer(x_data, np.ones_like(x_data))
    k_matrix = kernel(d_matrix / h)
    np.fill_diagonal(k_matrix, 0)
    num = np.sum(k_matrix * y_data[:, None], axis=0) / ((N - 1) * h)
    den = np.sum(k_matrix, axis=0) / ((N - 1) * h)
    return num / den

# Cross-validation function for Nadaraya-Watson estimator of E(Y|X)
def cv_mu_hat(y_data, x_data, N, h, kernel):
    mu_hat_loo_values = mu_hat_loo(y_data, x_data, N, h, kernel)
    ase_loo = np.mean((y_data - mu_hat_loo_values) ** 2)
    return ase_loo

# Grid of bandwidths
bw_grid = np.arange(0.1, 1, 0.01)

# Initialize arrays to store CV values
CV_values_gaussian = np.zeros_like(bw_grid)
CV_values_uniform = np.zeros_like(bw_grid)

# Loop over bandwidths and compute CV values
for i, h in enumerate(bw_grid):
    CV_values_gaussian[i] = cv_mu_hat(Y, X, n, h, gaussian_kernel)
    CV_values_uniform[i] = cv_mu_hat(Y, X, n, h, uniform_kernel)

# Find cross-validated bandwidths
cv_bw_gaussian = bw_grid[np.argmin(CV_values_gaussian)]
cv_bw_uniform = bw_grid[np.argmin(CV_values_uniform)]

# Obtain f_hat and mu_hat
f_hat_mu_hat_gaussian = kernel_smoother(Y, X, x_grid, n, cv_bw_gaussian, gaussian_kernel)
f_hat_gaussian = f_hat_mu_hat_gaussian['f_hat']
mu_hat_gaussian = f_hat_mu_hat_gaussian['mu_hat']

f_hat_mu_hat_uniform = kernel_smoother(Y, X, x_grid, n, cv_bw_uniform, uniform_kernel)
f_hat_uniform = f_hat_mu_hat_uniform['f_hat']
mu_hat_uniform = f_hat_mu_hat_uniform['mu_hat']

# Plot f_hat and mu_hat
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(x_grid, f_hat_gaussian, 'k--')
plt.title('Gaussian kernel')
plt.xlabel('x')
plt.ylabel(r'$\hat{f \, }(x)$')

plt.subplot(2, 2, 2)
plt.plot(x_grid, f_hat_uniform, 'k--')
plt.title('Uniform kernel')
plt.xlabel('x')
plt.ylabel(r'$\hat{f \, }(x)$')

plt.subplot(2, 2, 3)
plt.plot(x_grid, mu_hat_gaussian, 'k--')
plt.xlabel('x')
plt.ylabel(r'$\hat{\mu \, }(x)$')

plt.subplot(2, 2, 4)
plt.plot(x_grid, mu_hat_uniform, 'k--')
plt.xlabel('x')
plt.ylabel(r'$\hat{\mu \, }(x)$')

plt.tight_layout()
plt.savefig('den+reg-5+LOO-graphs.pdf')

# Plot cross-validation functions
plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.plot(bw_grid, CV_values_gaussian, 'k--')
plt.title('Gaussian kernel')
plt.xlabel('Bandwidth')
plt.ylabel('CV value')

plt.subplot(1, 2, 2)
plt.plot(bw_grid, CV_values_uniform, 'k--')
plt.title('Uniform kernel')
plt.xlabel('Bandwidth')
plt.ylabel('CV value')

plt.tight_layout()
plt.savefig('den+reg-5+LOO-cv.pdf')

plt.show()
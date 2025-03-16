import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, romb
from scipy.stats import norm

##############################################################################
# 1) Define your characteristic function psi(t).
##############################################################################
def psi_tld(k, mu, c, alpha, lam, beta):
    term1 = 1j * mu * k
    term2 = c**alpha * ((lam**alpha - (k**2 + lam**2)**(alpha/2)) / np.cos(np.pi * alpha / 2)) * np.cos(alpha * np.arctan(np.abs(k) / lam))
    term3 = (1 + 1j * np.sign(k) * beta * np.tan(alpha * np.arctan(np.abs(k) /lam)))
    return np.exp(-term1 + term2 * term3)


##############################################################################
# 2) Define the PDF via inverse Fourier transform (real part).
##############################################################################
def pdf_from_cf(x, mu, c, alpha, lam, beta, T=50.0):
    """
    Compute the PDF at x by numerically inverting psi(t).

    f(x) = 1/(2*pi) * integral_{-T}^{T} e^{-i t x} * psi(t) dt

    Parameters
    ----------
    x     : float
        The point at which we want the PDF.
    mu, alpha, c, lam : floats
        Parameters for the characteristic function.
    T     : float
        Truncation bound for integration. Must be chosen sufficiently large.

    Returns
    -------
    tuple
        Approximation to the PDF at x.
    """

    def integrand_real(k):
        return (np.exp(-1j * k * x) * psi_tld(k, mu, c, alpha, lam, beta)).real

    def integrand_imag(k):
        return (np.exp(-1j * k * x) * psi_tld(k, mu, c, alpha, lam, beta)).imag


    # We take the real part of the integral:
    result_real = quad(lambda t: integrand_real(t), -T, T)
    result_imag = quad(lambda t: integrand_imag(t), -T, T)

    return result_real[0],result_imag[0]


def pdf_from_cf_romberg(x, mu, c, alpha, lam, beta, k_max=50.0, ln_2_k_points=10):
    """
    Compute the PDF at x by numerically inverting psi(t).

    f(x) = 1/(2*pi) * integral_{-T}^{T} e^{-i t x} * psi(t) dt

    Parameters
    ----------
    x     : float
        The point at which we want the PDF.
    mu, alpha, c, lam : floats
        Parameters for the characteristic function.
    T     : float
        Truncation bound for integration. Must be chosen sufficiently large.

    Returns
    -------
    tuple
        Approximation to the PDF at x.
    """

    def integrand_real(k):
        return (np.exp(-1j * k * x) * psi_tld(k, mu, c, alpha, lam, beta)).real

    def integrand_imag(k):
        return (np.exp(-1j * k * x) * psi_tld(k, mu, c, alpha, lam, beta)).imag

    k_grid, dk = np.linspace(-k_max, k_max, 2**ln_2_k_points+1,retstep=True)

    # We take the real part of the integral:
    result_real = romb(integrand_real(k_grid), dx= float(dk))
    result_imag = romb(integrand_imag(k_grid), dx= float(dk))

    return result_real,result_imag

##############################################################################
# 3) Example usage: compute and plot the PDF over a range of x values.
##############################################################################
    # Choose some parameter values (you must set these appropriately):
mu = 1.0
c = 1.0
alpha = 0.7
lam = 0.02
beta = -0.9

# Range of x for which we want the PDF:
x_values = np.linspace(-5, 7.5, 500)

# Compute the PDF for each x:
# pdf_values = [pdf_from_cf(x, mu, c, alpha, lam, beta, T=30) for x in x_values]
pdf_values_real = [pdf_from_cf_romberg(x, mu, c, alpha, lam, beta, k_max=50)[0] / (2 * np.pi) for x in x_values]
pdf_values_imag = [pdf_from_cf_romberg(x, mu, c, alpha, lam, beta, k_max=50)[1] / (2 * np.pi) for x in x_values]
pdf_values_abs = np.sqrt(np.array(pdf_values_imag)**2 +np.array(pdf_values_real)**2)

pdf_gaussian = norm.pdf(x_values, loc=mu, scale=c)
# Plot the resulting PDF:
plt.figure(figsize=(7, 5))
plt.plot(x_values, pdf_values_real, label='Truncated Lévy')
#plt.plot(x_values, pdf_values_imag, label='Img Part')
plt.plot(x_values, pdf_gaussian, label='Gaussian')

plt.title("Truncated Lévy vs Gaussian PDF")
plt.xlabel("x")
plt.ylabel("pdf(x)")
plt.legend()
plt.grid(True)
plt.show()

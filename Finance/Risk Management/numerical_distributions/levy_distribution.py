import numpy as np
from scipy.integrate import romb
from scipy_distribution import dump_distribution, CustomDistribution

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

def dump_levy(loc, scale, alpha, lam, beta, filename='truncated_levy', x_min=-15, x_max =15, x_points=10000):
    x_values = np.linspace(x_min,x_max,x_points)
    pdf_values = pdf_from_cf(x_values,loc, scale, alpha, lam, beta)
    filename = filename + '_' + str(np.round(loc,2)) + '_' +  str(np.round(scale,2)) + '_' +   str(np.round(alpha,2)) + '_' +   str(np.round(lam,2)) + '_' +   str(np.round(beta,2))
    dump_distribution(x_vals=x_values,pdf_vals=pdf_values,filename=filename)
    return

##############################################################################
# 5) Read the values from file
##############################################################################

def load_levy(filename):
    return CustomDistribution(data_file=filename, momtype=0, name="Truncated Skewed Lévy", badvalue=0)
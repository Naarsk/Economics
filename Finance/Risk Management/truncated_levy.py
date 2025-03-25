import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from levy_distribution import pdf_from_cf_romberg

class TruncatedLevy(stats.rv_continuous):
    def __init__(self, loc, scale, alpha, lamb, beta, *args, **kwargs):
        self.loc = loc
        self.scale = scale
        self.alpha = alpha
        self.lamb = lamb
        self.beta = beta
        super().__init__(*args, **kwargs)

    def characteristic_function(self, k):
        """Define the characteristic function φ(t). Modify for different distributions."""
        term1 = 1j * self.loc * k
        term2 = self.scale ** self.alpha * ((self.lamb ** self.alpha - (k ** 2 + self.lamb ** 2) ** (self.alpha / 2)) / np.cos(np.pi * self.alpha / 2)) * np.cos(
            self.alpha * np.arctan(np.abs(k) / self.lamb))
        term3 = (1 + 1j * np.sign(k) * self.beta * np.tan(self.alpha * np.arctan(np.abs(k) / self.lamb)))
        return np.exp(-term1 + term2 * term3)

    def _pdf(self, x):
        """Compute the PDF using inverse Fourier transform of the CF."""
        return pdf_from_cf_romberg(x, self.loc,self.scale,self.alpha,self.lamb,self.beta)[0]


mu = 1.0
c = 1.0
alpha = 0.7
lam = 0.02
beta = -0.9

# Example usage
custom_dist = TruncatedLevy(loc=mu, scale=c, alpha=alpha, lamb=lam,beta=beta, name="custom_cf_dist")
samples = custom_dist.rvs(size=1000)



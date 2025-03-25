from typing import Union, Sequence
import numpy as np
from arch.typing import Float64Array, ArrayLike1D, ArrayLike
from arch.univariate import Distribution
from scipy_distribution import CustomDistribution

class CustomArchDistribution(Distribution):

    def __init__(self,data_file, seed=None):
        super().__init__(seed=seed)
        self.custom_dist = CustomDistribution(momtype=0, data_file=data_file)

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return np.empty(0), np.empty(0)

    def bounds(self, resids: Float64Array) -> list[tuple[float, float]]:
        return []

    def parameter_names(self):
        """Names of the parameters in the order used in other methods."""
        return []

    def ppf(self, pits: Union[float, Sequence[float], ArrayLike1D],
            parameters: Union[Sequence[float], ArrayLike1D, None] = None) -> Union[float, Float64Array]:
        return self.custom_dist.ppf(pits)

    def cdf(self, resids: Union[Sequence[float], ArrayLike1D],
            parameters: Union[Sequence[float], ArrayLike1D, None] = None) -> Float64Array:
        return self.custom_dist.cdf(resids)

    def moment(self, n: int, parameters: Union[Sequence[float], ArrayLike1D, None] = None) -> float:
        return self.custom_dist.moment(n)

    def partial_moment(self, n: int, z: float = 0.0,
                       parameters: Union[Sequence[float], ArrayLike1D, None] = None) -> float:
        return self.custom_dist.moment(n)

    def starting_values(self, resids):
        """
        Provides initial guesses for the parameter vector.
        For Laplace, a rough guess is the mean absolute residual.
        """
        return np.empty(0)

    def loglikelihood(
        self,
        parameters: Union[Sequence[float], ArrayLike1D],
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False,
    ) -> Union[float , Float64Array]:

        r"""Computes the log-likelihood of assuming residuals are normally
        distributed, conditional on the variance

        Parameters
        ----------
        parameters : ndarray
            The normal likelihood has no shape parameters. Empty since the
            standard normal has no shape parameters.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=-\frac{1}{2}\left(\ln2\pi+\ln\sigma^{2}
            +\frac{x^{2}}{\sigma^{2}}\right)

        """
        lls = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + resids ** 2.0 / sigma2)
        if individual:
            return lls
        else:
            return sum(lls)

    def _simulator(self, size: Union[int, tuple[int, ...]]) -> Float64Array:
        return self.custom_dist.rvs(size)

    def simulate(self, parameters):
        """
        Simulate random draws from Laplace(0, theta).
        Must return a 1D array of length nobs.
        """
        return self._simulator

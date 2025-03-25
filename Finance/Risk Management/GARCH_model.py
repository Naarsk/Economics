import numpy as np
import scipy

class GarchOneOne(object):

    def __init__(self, log_returns):
        self.results = None
        self.logReturns = log_returns
        self.sigma_2 = self.filter(self.optimization())
        self.coefficients = self.optimization()

    def filter(self, parameters):

        """Returns the variance expression of a GARCH(1,1) process."""

        # Slicing the parameters list
        omega = parameters[0]
        alpha = parameters[1]
        beta = parameters[2]

        # Length of logReturns
        length = len(self.logReturns)

        # Initializing an empty array
        sigma_2 = np.zeros(length)

        # Filling the array, if i == 0 then uses the long term variance.
        for i in range(length):
            if i == 0:
                sigma_2[i] = omega / (1 - alpha - beta)
            else:
                sigma_2[i] = omega + alpha * self.logReturns[i - 1] ** 2 + beta * sigma_2[i - 1]

        return sigma_2

    def log_likelihood(self, parameters):

        """Defines the log likelihood sum to be optimized given the parameters."""

        sigma_2 = self.filter(parameters)
        loglikelihood = - np.sum(-np.log(sigma_2) - self.logReturns ** 2 / sigma_2)

        return loglikelihood


    def optimization(self):

        """Optimizes the log likelihood function and returns estimated coefficients"""

        # Parameters initialization
        parameters = np.array([.000005, .1, .85])

        # Parameters optimization, scipy does not have a maximize function, so we minimize the opposite of the equation described earlier
        opt = scipy.optimize.minimize(self.log_likelihood, parameters,
                                      bounds=((.001, 1), (.001, 1), (.001, 1)))

        variance = opt.x[0] / (1 - opt.x[1] - opt.x[2])  # Times .01**2 because it concerns squared returns

        self.results = {'omega' : opt.x[0], 'alpha' : opt.x[1], 'beta' : opt.x[2], 'variance' : variance}

        return np.append(opt.x, variance)

class PowerGarchOneOne(object):

    def __init__(self, log_returns):
        self.results = None
        self.logReturns = log_returns *100
        self.sigma_2 = self.filter(self.optimization())
        self.coefficients = self.optimization()

    def filter(self, parameters):
        alpha_0 = parameters[0]
        alpha_1 = parameters[1]
        alpha_2 = parameters[2]
        alpha_3 = parameters[3]
        kappa = parameters[4]
        delta = parameters[5]

        # Length of logReturns
        length = len(self.logReturns)

        # Initializing an empty array
        phi = np.zeros(length)
        sigma_2 = np.zeros(length)
        epsilon = np.random.normal(0, 1, length)

        # Filling the array, if i == 0 then uses the long term variance.
        phi[0] = alpha_0 / (1 - alpha_1 - (2 / np.pi) ** (kappa / 2) * alpha_2 - (2 / np.pi) ** (
                kappa / 2) / 2 * alpha_3)

        for i in range(1,length):
            if epsilon[i] >= 0:
                phi[i] = alpha_0 + (alpha_1 + alpha_2 * epsilon[i] ** kappa)*phi[i-1]
            else:
                phi[i] = alpha_0 + (alpha_1 + (alpha_2 + alpha_3) * (-epsilon[i]) ** kappa)*phi[i-1]
            sigma_2[i] = np.abs(delta * phi[i] - delta + 1) ** (1 / (2 * delta))

        return sigma_2

    def log_likelihood(self, parameters):

        """Defines the log likelihood sum to be optimized given the parameters."""

        sigma_2 = self.filter(parameters)
        loglikelihood = - np.sum(-np.log(sigma_2) - self.logReturns ** 2 / sigma_2)

        return loglikelihood

    def optimization(self):

        """Optimizes the log likelihood function and returns estimated coefficients"""

        # Parameters initialization
        parameters = np.array([.000005, .827, .008, 0.072, 1.76, 0.88])

        # Parameters optimization, scipy does not have a maximize function, so we minimize the opposite of the equation described earlier
        opt = scipy.optimize.minimize(self.log_likelihood, parameters,
                                      bounds=((.00000001, 1), (.001, 1), (.001, 1), (.001, 1), (.001, 2), (.001, 1)))

        variance = np.abs(opt.x[5]*(opt.x[0] / (1 - opt.x[1] - (2 / np.pi) ** (opt.x[4] / 2) * opt.x[2] - (2 / np.pi) ** (
                opt.x[4] / 2) / 2 * opt.x[3]))-opt.x[5]+1)

        self.results = {'alpha_0': opt.x[0], 'alpha_1': opt.x[1], 'alpha_2': opt.x[2], 'alpha_3' : opt.x[3], 'kappa' : opt.x[4],  'delta' : opt.x[5],'variance': variance}

        return np.append(opt.x, variance)


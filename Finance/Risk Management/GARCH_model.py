import numpy as np
import scipy
import pandas as pd


class garchOneOne(object):

    def __init__(self, log_returns):
        self.logReturns = log_returns * 100
        self.sigma_2 = self.garch_filter(self.garch_optimization())
        self.coefficients = self.garch_optimization()

    def garch_filter(self, parameters):
        "Returns the variance expression of a GARCH(1,1) process."

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

    def garch_log_likelihood(self, parameters):

        "Defines the log likelihood sum to be optimized given the parameters."

        length = len(self.logReturns)

        sigma_2 = self.garch_filter(parameters)

        loglikelihood = - np.sum(-np.log(sigma_2) - self.logReturns ** 2 / sigma_2)

        return loglikelihood


    def garch_optimization(self):
        "Optimizes the log likelihood function and returns estimated coefficients"
        # Parameters initialization
        parameters = [.1, .05, .92]

        # Parameters optimization, scipy does not have a maximize function, so we minimize the opposite of the equation described earlier
        opt = scipy.optimize.minimize(self.garch_log_likelihood, parameters,
                                      bounds=((.001, 1), (.001, 1), (.001, 1)))

        variance = .01 ** 2 * opt.x[0] / (1 - opt.x[1] - opt.x[2])  # Times .01**2 because it concerns squared returns

        return np.append(opt.x, variance)
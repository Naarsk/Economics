import numdifftools as nd
import numpy as np


class UtilityFunction:
    def __init__(self, func, params):
        """
        Initialize a UtilityFunction object.

        Parameters
        ----------
        func : callable
            A function that takes two arguments (x and params) and returns
            the output of the utility function.
        params : list
            A list of parameters of the utility function.

        Returns
        -------
        None
        """
        self.func=func
        self.params=params

    def __call__(self, x):
        """
        Evaluate the utility function with the given variables and parameters.

        Parameters
        ----------
        x : list
            A list of the variables to evaluate the utility function with.

        Returns
        -------
        float
            The output of the utility function.
        """
        return self.func(x, self.params)

    def absolute_risk_aversion(self, x):
        return - nd.Derivative(self, order=2)(x)/nd.Derivative(self, order=1)(x)

class CRRA(UtilityFunction):
    def __init__(self, params):

        """
        Initialize a CRRA object.

        Parameters
        ----------
        params : list
            A list of parameters of the CRRA utility function.

        Returns
        -------
        None
        """

        def crra(x, gamma):
            """
            The constant relative risk aversion (CRRA) utility function.

            Parameters
            ----------
            x : float
                The consumption value to evaluate the CRRA utility function with.
            gamma : float
                The parameter of the CRRA utility function, which represents the
                individual's risk aversion.

            Returns
            -------
            float
                The output of the CRRA utility function.
            """
            return x**(1-gamma)/(1-gamma)

        super().__init__(crra, params)


    def __call__(self, x):
        """
        Evaluate the CRRA utility function with the given consumption value and parameters.

        Parameters
        ----------
        x : float
            The consumption value to evaluate the CRRA utility function with.

        Returns
        -------
        float
            The output of the CRRA utility function.

        """
        return super().__call__([x,])


class CARA(UtilityFunction):
    def __init__(self, params):


        """
        Initialize a CARA object.

        Parameters
        ----------
        params : list
            A list of parameters of the CARA utility function.

        Returns
        -------
        None
        """
        def cara(x, alpha):

            """
            The Constant Absolute Risk Aversion (CARA) utility function.

            Parameters
            ----------
            x : float
                The consumption value to evaluate the CARA utility function with.
            alpha : float
                The parameter of the CARA utility function, which represents the
                individual's risk aversion.

            Returns
            -------
            float
                The output of the CARA utility function.
            """
            return -np.exp(-alpha*x)

        super().__init__(cara, params)

    def __call__(self, x):

        """
        Evaluate the CARA utility function with the given consumption value and parameters.

        Parameters
        ----------
        x : float
            The consumption value to evaluate the CARA utility function with.

        Returns
        -------
        float
            The output of the CARA utility function.
        """
        return super().__call__([x, ])
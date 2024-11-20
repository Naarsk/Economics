from typing import Any

import numpy as np
from numpy import ndarray, dtype
from scipy.optimize import minimize


class ProductionFunction:
    def __init__(self, func, params : list[float]):
        """
        Initialize a ProductionFunction object.

        Parameters
        ----------
        func : callable
            A function that takes two arguments (capital and labor) and returns
            the output of the production function.
        params : list
            A list of parameters of the production function.

        Returns
        -------
        None
        """
        self.func=func
        self.params=params

    def __call__(self, inputs : list[float]):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        inputs : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return self.func(inputs, self.params)

    def cost_function(self, output_level, input_prices):
        """
        Calculate the minimum (long-run) cost to produce a given output level.

        Parameters
        ----------
        output_level : float
            The desired output level of the production function.
        input_prices : list
            A list of prices for the inputs to the production function.

        Returns
        -------
        float
            The minimum cost to produce the desired output level."""
        x0=np.ones(len(input_prices))
        result=minimize(lambda x: np.dot(x, input_prices), x0=x0, constraints={"type": "ineq", "fun": lambda x: self(x) - output_level})

        if result.success:
            return np.dot(result.x, input_prices)
        else:
            # raise ValueError("Optimization failed.")
            return 0

    def restricted_cost_function(self, output_level, input_prices, fixed_inputs : list [bool]):

        """
        Calculate the restricted (short-run) cost to produce a given output level.

        Parameters
        ----------
        output_level : float
            The desired output level of the production function.
        input_prices : list
            A list of prices for the inputs to the production function.
        fixed_inputs : list [bool]
            A list of booleans indicating whether the corresponding input is fixed.

        Returns
        -------
        float
            The restricted cost to produce the desired output level."""
        x0=np.ones(len(input_prices))
        arg_func = lambda x: np.dot(np.dot(x, np.diag(fixed_inputs)), input_prices)

        result=minimize(arg_func, x0=x0, constraints={"type": "ineq", "fun": lambda x: self(x) - output_level})
        if result.success:
            return result.x
        else:
            # raise ValueError("Optimization failed.")
            return 0


    def profit_function(self, output_level, input_prices):
        """
        Calculate the profit to produce a given output level.

        Parameters
        ----------
        output_level : float
            The desired output level of the production function.
        input_prices : list
            A list of prices for the inputs to the production function.

        Returns
        -------
        float
            The profit to produce the desired output level."""
        return 0

    def gradient(self, inputs : list[float]) -> ndarray[Any, dtype[Any]]:
        """
        Evaluate the gradient of the production function with the given variables and parameters.

        Parameters
        ----------
        inputs : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        ndarray
            The gradient of the production function.
        """
        # Convert inputs to a numpy array
        inputs = np.array(inputs)

        # Compute the gradient using finite differences
        eps = 1e-6
        gradient = np.zeros_like(inputs)
        for i in range(len(inputs)):
            inputs_plus_eps = inputs.copy()
            inputs_plus_eps[i] += eps
            inputs_minus_eps = inputs.copy()
            inputs_minus_eps[i] -= eps
            gradient[i] = (self.func(inputs_plus_eps, self.params) - self.func(inputs_minus_eps, self.params)) / (2 * eps)

        return gradient

class CobbDouglas(ProductionFunction):
    def __init__(self, alpha):

        """
        Initialize a CobbDouglas object.

        Parameters
        ----------
        alpha : float
            The exponent of capital in the Cobb-Douglas production function.

        Returns
        -------
        None
        """
        def cobb_douglas(x: list[2], params: list[float]):

            """
            Evaluate the Cobb-Douglas production function with the given variables and parameters.

            Parameters
            ----------
            x : list[2]
                A list of the variables to evaluate the production function with. The first element
                is capital, and the second element is labor.
            params : list[float]
                The parameter alpha in the Cobb-Douglas production function.
            Returns
            -------
            float
                The output of the Cobb-Douglas production function.
            """
            if params[0] < 0 or params[0] > 1:
                # raise ValueError("Alpha must be between 0 and 1.")
                return 0
            else:
                return x[0] ** params[0] * x[1] ** (1 - params[0])

        super().__init__(func=cobb_douglas, params=[alpha,])


class SolowCES(ProductionFunction):
    def __init__(self, sigma, gamma):
        """
        Initialize a CES object.

        Parameters
        ----------
        sigma : float
            The elasticity of substitution in the CES production function.
        gamma : float
            The distribution parameter in the CES production function.

        Returns
        -------
        None
        """
        def ces(x: list[5]):
            """
            The Constant Elasticity of Substitution (CES) production function.

            Parameters
            ----------
            x : list[5]
                A list with five elements, the first being the capital stock (k), the second being the labor (l), the third being the human capital (a_h), the fourth being the technological progress parameter for capital (a_k) and the fifth being the technological progress parameter for labor (a_l).

            Returns
            -------
            The output of the CES production function, which is given by a_h*(gamma*(a_k*k)**((sigma-1)/sigma) + a_l*(a_k*k)**((sigma-1)/sigma))**(sigma/(sigma-1)).
            """
            k=x[0]
            l=x[1]
            a_h=x[2]
            a_k=x[3]
            a_l=x[4]

            return a_h*(gamma*(a_k*k)**((sigma-1)/sigma) + (1-gamma)*(a_l*l)**((sigma-1)/sigma))**(sigma/(sigma-1))

        super().__init__(func=ces, params=[sigma,gamma])

    def __call__(self, inputs):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        inputs : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return super().__call__(inputs)


class Logit(ProductionFunction):
    def __init__(self, beta, func, params):
        """
        Initialize a Logit object.

        Parameters
        ----------
        beta : float
            The beta parameter in the Logit production function.

        Returns
        -------
        None
        """

        def logit(x: list[2]):
            """
            Evaluate the Logit production function with the given variables and parameters.

            Parameters
            ----------
            x : list[2]
                A list of the variables to evaluate the production function with. The first element
                is capital, and the second element is labor.
            Returns
            -------
            float
                The output of the Logit production function.
            """
            k = x[0]
            l = x[1]

            return beta * k + (1 - beta) * l

        super().__init__(logit, params)


class CES(ProductionFunction):
    def __init__(self, rho : float, weights : list):
        """
        Initialize a CES object.

        Parameters
        ----------
        rho : float
            The parameter in the CES production function.
        weights : list
            The distribution parameter in the CES production function.
        Returns
        -------
        None
        """
        def ces(x: list, params: list):
            """
            Evaluate the CES production function with the given variables and parameters.

            Parameters
            ----------
            x : list
                A list of the input variables to evaluate the production function with.
            params : list
                A list of 2 elements, the first being rho, the second the list of weights
            Returns
            -------
            float
                The output of the CES production function.
            """

            return sum(w * x[i]**params[0] for i, w in enumerate(params[1]))**(1/rho)

        super().__init__(func=ces, params=[rho,weights])

    def __call__(self, inputs):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        inputs : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return super().__call__(inputs)
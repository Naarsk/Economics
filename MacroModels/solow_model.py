import numpy as np
from scipy.optimize import approx_fprime
import pandas as pd

class SolowModel:
    """
    A class to represent the Solow model of economic growth.

    Attributes
    ----------
    saving_rate : float
        The fraction of income saved each period.
    depreciation_rate : float
        The rate at which capital depreciates each period.
    starting_capital : float
        The initial level of capital.
    production_function : callable
        A function representing the production technology. Takes two arguments: capital and labor.
    labor_growth_function : callable
        A function representing the rate of growth of the population. Takes one argument: the current population level.
    starting_pop_level : float
        The initial population level.
    time_horizon : int, optional
        The number of periods to simulate. Defaults to 1000.
    """
    def __init__(self, saving_rate, depreciation_rate, starting_capital, production_function, labor_growth_function, starting_pop_level,time_horizon = 1000):
        """
        Initialize a Solow model object.

        Parameters
        ----------
        saving_rate : float
            The fraction of income saved each period.
        depreciation_rate : float
            The rate at which capital depreciates each period.
        starting_capital : float
            The initial level of capital.
        production_function : callable
            A function representing the production technology. Takes two arguments: capital and labor.
        labor_growth_function : callable
            A function representing the rate of growth of the population. Takes one argument: the current population level.
        starting_pop_level : float
            The initial population level.
        time_horizon : int, optional
            The number of periods to simulate. Defaults to 1000.

        Returns
        -------
        None
        """
        self.saving_rate = saving_rate
        self.depreciation_rate = depreciation_rate
        self.starting_capital = starting_capital
        self.production_function = production_function
        self.labor_growth_function = labor_growth_function
        self.starting_pop_level = starting_pop_level
        self.capital = np.zeros(time_horizon)
        self.labor = np.zeros(time_horizon)
        self.shifter = np.zeros(time_horizon)
        self.rates = np.zeros(time_horizon)
        self.wages = np.zeros(time_horizon)
        self.income = np.zeros(time_horizon)
        self.income_per_cap = np.zeros(time_horizon)
        self.capital_per_cap = np.zeros(time_horizon)
        self.time = range(time_horizon)


    def evaluate(self):
        """
        Evaluate the Solow model.

        This function evaluates the Solow model for the specified time horizon and
        returns a pandas DataFrame with the time series of the capital stock, labor,
        income, income per capita, capital per capita, interest rate, and wage.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            A DataFrame with the time series of the capital stock, labor, income,
            income per capita, capital per capita, interest rate, and wage.
        """
        self.labor[0]=self.starting_pop_level
        self.capital[0]=self.starting_capital

        for t in self.time[0:-1]:
            self.income[t]=self.production_function([self.capital[t], self.labor[t]])
            self.income_per_cap[t] = self.income[t]/self.labor[t]
            self.capital_per_cap[t] = self.capital[t]/self.labor[t]

            self.labor[t+1] = self.labor_growth_function(self.labor[t])
            self.capital[t+1] = self.saving_rate*self.income[t]+(1-self.depreciation_rate)*self.capital[t]
            self.rates[t]=approx_fprime(np.array([self.capital[t],self.labor[t]]),self.production_function)[0]
            self.wages[t]=approx_fprime(np.array([self.capital[t],self.labor[t]]),self.production_function)[1]

        self.income[-1]=self.production_function([self.capital[-1], self.labor[-1]])
        self.income_per_cap[-1] = self.income[-1]/self.labor[-1]
        self.capital_per_cap[-1] = self.capital[-1]/self.labor[-1]
        self.rates[-1] = approx_fprime(np.array([self.capital[-1],self.labor[-1]]),self.production_function)[0]
        self.wages[-1] = approx_fprime(np.array([self.capital[-1],self.labor[-1]]),self.production_function)[1]

        return pd.DataFrame({'time':self.time,'capital':self.capital,'labor':self.labor,'income':self.income,'income_per_cap':self.income_per_cap,'capital_per_cap':self.capital_per_cap,'rates':self.rates,'wages':self.wages})


class ProductionFunction():
    def __init__(self, func, params):
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

    def __call__(self, x):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        x : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return self.func(x, self.params)

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
        def cobb_douglas(x: list[2], params: float):
            """
            The Cobb-Douglas production function.

            Parameters
            ----------
            x : list[2]
                A list with two elements, the first being the capital stock (k) and the second being the labor (l).

            Returns
            -------
            The output of the Cobb-Douglas production function, which is given by k ** alpha * l ** (1 - alpha).

            """
            k = x[0]
            l = x[1]
            alpha=params

            return k ** alpha * l ** (1 - alpha)

        super().__init__(func=cobb_douglas, params=alpha)

    def __call__(self, x):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        x : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return super().__call__(x)

class CES(ProductionFunction):
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
        def ces(x: list[5], params: list[2]):
            """
            The Constant Elasticity of Substitution (CES) production function.

            Parameters
            ----------
            x : list[5]
                A list with five elements, the first being the capital stock (k), the second being the labor (l), the third being the human capital (a_h), the fourth being the technological progress parameter for capital (a_k) and the fifth being the technological progress parameter for labor (a_l).

            Returns
            -------
            The output of the CES production function, which is given by a_h*(gamma*(a_k*k)**((sigma-1)/sigma) + a_l*(a_k*k)**((sigma-1)/sigma))**(sigma/(sigma-1)).
            :param params:

            """
            k=x[0]
            l=x[1]
            a_h=x[2]
            a_k=x[3]
            a_l=x[4]

            gamma=params[0]
            sigma=params[1]
            return a_h*(gamma*(a_k*k)**((sigma-1)/sigma) + a_l*(a_k*k)**((sigma-1)/sigma))**(sigma/(sigma-1))

        super().__init__(func=ces, params=[sigma,gamma])

        def _call__(self, x):
            """
            Evaluate the production function with the given variables and parameters.

            Parameters
            ----------
            x : list
                A list of the variables to evaluate the production function with.

            Returns
            -------
            float
                The output of the production function.
            """
            return super().__call__(x)


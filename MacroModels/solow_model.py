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

    def __init__(self, saving_rate, depreciation_rate, starting_capital, production_function, labor_growth_function,
                 starting_pop_level, time_horizon=1000):
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

    def __call__(self, *args, **kwargs):
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
        self.labor[0] = self.starting_pop_level
        self.capital[0] = self.starting_capital

        for t in self.time[0:-1]:
            self.income[t] = self.production_function([self.capital[t], self.labor[t]])
            self.income_per_cap[t] = self.income[t] / self.labor[t]
            self.capital_per_cap[t] = self.capital[t] / self.labor[t]

            self.labor[t + 1] = self.labor_growth_function(self.labor[t])
            self.capital[t + 1] = self.saving_rate * self.income[t] + (1 - self.depreciation_rate) * self.capital[t]
            self.rates[t] = approx_fprime(np.array([self.capital[t], self.labor[t]]), self.production_function)[0]
            self.wages[t] = approx_fprime(np.array([self.capital[t], self.labor[t]]), self.production_function)[1]

        self.income[-1] = self.production_function([self.capital[-1], self.labor[-1]])
        self.income_per_cap[-1] = self.income[-1] / self.labor[-1]
        self.capital_per_cap[-1] = self.capital[-1] / self.labor[-1]
        self.rates[-1] = approx_fprime(np.array([self.capital[-1], self.labor[-1]]), self.production_function)[0]
        self.wages[-1] = approx_fprime(np.array([self.capital[-1], self.labor[-1]]), self.production_function)[1]

        return pd.DataFrame({'time': self.time, 'capital': self.capital, 'labor': self.labor, 'income': self.income,
                             'income_per_cap': self.income_per_cap, 'capital_per_cap': self.capital_per_cap,
                             'rates': self.rates, 'wages': self.wages})


class BalancedGrowth(SolowModel):
    def __init__(self, saving_rate, depreciation_rate, starting_capital, labor_growth_function, starting_pop_level,
                 production_function, time_horizon=1000):
        super().__init__(saving_rate, depreciation_rate, starting_capital, production_function, labor_growth_function,
                         starting_pop_level, time_horizon)


class SustainedGrowth(SolowModel):
    def __init__(self, saving_rate, depreciation_rate, starting_capital, labor_growth_function, starting_pop_level,
                 production_function, time_horizon=1000):
        super().__init__(saving_rate, depreciation_rate, starting_capital, production_function, labor_growth_function,
                         starting_pop_level, time_horizon)


class NoGrowth(SolowModel):
    def __init__(self, saving_rate, depreciation_rate, starting_capital, labor_growth_function, starting_pop_level,
                 production_function, time_horizon=1000):
        super().__init__(saving_rate, depreciation_rate, starting_capital, production_function, labor_growth_function,
                         starting_pop_level, time_horizon)

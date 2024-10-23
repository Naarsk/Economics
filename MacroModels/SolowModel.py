import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from MacroModels.ExogenousVariables import ExogenousVariables, ExogenousFunctions
from MacroModels.GrowthFunctions import compounded_growth, identity
from MacroModels.ProductionFunctions import CobbDouglas

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
    """

    def __init__(self, saving_rate, depreciation_rate, starting_capital, production_function,
                 exogenous_variables: ExogenousVariables):
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

        Returns
        -------
        None
        """
        self.labor = exogenous_variables.labor
        self.hicks_progress = exogenous_variables.hicks_progress
        self.solow_progress = exogenous_variables.solow_progress
        self.harrods_progress = exogenous_variables.harrod_progress
        time_horizon=len(exogenous_variables.labor)

        self.saving_rate = saving_rate
        self.depreciation_rate = depreciation_rate
        self.starting_capital = starting_capital
        self.production_function = production_function

        self.capital = np.zeros(time_horizon)
        self.rental_rates = np.zeros(time_horizon)
        self.wages = np.zeros(time_horizon)
        self.income = np.zeros(time_horizon)
        self.income_per_cap = np.zeros(time_horizon)
        self.capital_per_cap = np.zeros(time_horizon)
        self.time = range(time_horizon)

    def __call__(self):
        """
        Evaluate the Solow model.

        This function evaluates the Solow model for the specified time horizon and
        returns a pandas DataFrame with the time series of the capital stock, labor,
        income, income per capita, capital per capita, interest rate, and wage.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the time series of the capital stock, labor, income,
            income per capita, capital per capita, interest rate, and wage.
        """

        self.capital[0] = self.starting_capital

        for t in self.time[:-1]:
            self.income[t] = self.hicks_progress[t] * self.production_function(
                [self.solow_progress[t] * self.capital[t], self.harrods_progress[t] * self.labor[t]])
            self.income_per_cap[t] = self.income[t] / self.labor[t]
            self.capital_per_cap[t] = self.capital[t] / self.labor[t]

            self.capital[t + 1] = self.saving_rate * self.income[t] + (1 - self.depreciation_rate) * self.capital[t]
            self.rental_rates[t] = approx_fprime(np.array([self.capital[t], self.labor[t]]), self.production_function)[0]
            self.wages[t] = approx_fprime(np.array([self.capital[t], self.labor[t]]), self.production_function)[1]

        self.income[-1] = self.production_function([self.capital[-1], self.labor[-1]])
        self.income_per_cap[-1] = self.income[-1] / self.labor[-1]
        self.capital_per_cap[-1] = self.capital[-1] / self.labor[-1]
        self.rental_rates[-1] = approx_fprime(np.array([self.capital[-1], self.labor[-1]]), self.production_function)[0]
        self.wages[-1] = approx_fprime(np.array([self.capital[-1], self.labor[-1]]), self.production_function)[1]

        return pd.DataFrame({'time': self.time, 'capital': self.capital, 'labor': self.labor, 'income': self.income,
                             'income_per_cap': self.income_per_cap, 'capital_per_cap': self.capital_per_cap,
                             'rates': self.rental_rates, 'wages': self.wages})


class BalancedGrowth(SolowModel):
    def __init__(self, saving_rate, depreciation_rate, starting_capital, starting_pop,
                 labor_growth_rate, elasticity, starting_hicks, hicks_growth_rate, time_horizon=1000):
        labor_growth_function = lambda x: compounded_growth(x, labor_growth_rate)
        hicks_growth_function = lambda x: compounded_growth(x, hicks_growth_rate)
        production_function = CobbDouglas(elasticity)
        exogenous_variables = ExogenousFunctions(starting_pop=starting_pop, starting_hicks=starting_hicks,
                                                 starting_solow=1, starting_harrods=1,
                                                 hicks_progress_function=hicks_growth_function,
                                                 solow_progress_function=identity, harrods_progress_function=identity,
                                                 time_horizon=time_horizon, labor_growth_function=labor_growth_function)
        super().__init__(saving_rate, depreciation_rate, starting_capital, production_function,
                         exogenous_variables=exogenous_variables)


class SustainedGrowth(SolowModel):
    def __init__(self, saving_rate, depreciation_rate, starting_capital, starting_pop, progress, labor_growth_rate, time_horizon=1000):
        labor_growth_function = lambda x: compounded_growth(x, labor_growth_rate)
        production_function = CobbDouglas(1)
        exogenous_variables = ExogenousFunctions(starting_pop=starting_pop, starting_hicks=progress, starting_solow=1,
                                                 starting_harrods=1, hicks_progress_function=identity,
                                                 solow_progress_function=identity, harrods_progress_function=identity,
                                                 time_horizon=time_horizon, labor_growth_function=labor_growth_function)
        super().__init__(saving_rate, depreciation_rate, starting_capital, production_function,
                         exogenous_variables=exogenous_variables)


class NoGrowth(SolowModel):
    def __init__(self, saving_rate, depreciation_rate, starting_capital, population, elasticity, time_horizon=1000):
        production_function = CobbDouglas(elasticity)
        exogenous_variables = ExogenousFunctions(starting_pop=population, starting_hicks=1,
                                                 starting_solow=1, starting_harrods=1,
                                                 hicks_progress_function=identity,
                                                 solow_progress_function=identity, harrods_progress_function=identity,
                                                 time_horizon=time_horizon, labor_growth_function=identity)
        super().__init__(saving_rate, depreciation_rate, starting_capital, production_function,
                         exogenous_variables=exogenous_variables)
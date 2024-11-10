import numpy as np

from Finance.UtilityFunctions import UtilityFunction
from MacroModels.ExogenousVariables import ExogenousVariables
from MacroModels.ProductionFunctions import ProductionFunction


class OverlappingGenerations:

    def __init__(self, depreciation_rate, discounting_rate, starting_capital, life_span, utility_function : UtilityFunction, production_function : ProductionFunction,
                 exogenous_variables: ExogenousVariables):
        self.labor = exogenous_variables.labor
        self.starting_capital = starting_capital
        self.production_function = production_function
        self.utility_function = utility_function
        self.depreciation_rate = depreciation_rate
        self.discounting_rate = discounting_rate
        self.discount_factor = 1 + self.discounting_rate
        self.life_span = life_span
        time_horizon=len(exogenous_variables.labor)
        self.capital = np.zeros(time_horizon)
        self.assets = np.zeros(time_horizon)
        self.rental_rates = np.zeros(time_horizon)
        self.rates_of_return = np.zeros(time_horizon)
        self.wages = np.zeros(time_horizon)
        self.income = np.zeros(time_horizon)
        self.consumption = np.zeros(time_horizon)
        self.time = range(time_horizon)
        self.elasticity = np.zeros(time_horizon)

    def target_function(self,t):

        fun=0
        for i in range(self.life_span):
            fun = fun + self.discount_factor**i * self.utility_function(self.consumption[t+i])

        return fun


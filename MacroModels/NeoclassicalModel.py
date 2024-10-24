import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime, fsolve

from Finance.UtilityFunctions import UtilityFunction
from MacroModels.ExogenousVariables import ExogenousVariables


class RamseyModel:

    def __init__(self, depreciation_rate, discounting_rate, starting_capital, utility_function : UtilityFunction, production_function,
                 exogenous_variables: ExogenousVariables):
        self.labor = exogenous_variables.labor
        self.starting_capital = starting_capital
        self.production_function = production_function
        self.utility_function = utility_function
        self.depreciation_rate = depreciation_rate
        self.discounting_rate = discounting_rate
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

    def __call__(self):
        self.capital[0] = self.starting_capital
        self.consumption[-1] = 0

        def euler_method(x,t):
            return x+(self.rates_of_return[t]-self.discounting_rate)/self.utility_function.absolute_risk_aversion(x)

        for t in self.time.__reversed__():
            self.consumption[t-1] = fsolve(euler_method, self.consumption[t], args=(t-1,))


        for t in self.time[:-1]:
            self.income[t] = self.production_function([ self.capital[t], self.labor[t]])
            self.wages[t] = approx_fprime(np.array([self.capital[t], self.labor[t]]), self.production_function)[1]

            self.rental_rates[t] = approx_fprime(np.array([self.capital[t], self.labor[t]]), self.production_function)[0]
            self.rates_of_return[t]=self.rental_rates[t]-self.depreciation_rate

            self.elasticity[t] = self.utility_function.absolute_risk_aversion(float(self.consumption[t])) * self.consumption[t]
            self.consumption[t+1] = self.consumption[t]*(1+(self.rates_of_return-self.discounting_rate)/self.elasticity[t])
            self.assets[t+1]=self.rates_of_return*self.assets[t]+self.wages[t]*self.labor[t]-self.consumption[t]*self.labor[t]


        return pd.DataFrame({'time': self.time, 'capital': self.capital, 'labor': self.labor, 'income': self.income,
                             'rates': self.rates_of_return, 'wages': self.wages})

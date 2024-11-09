import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d

from Finance.UtilityFunctions import UtilityFunction
from MacroModels.ExogenousVariables import ExogenousVariables
from MacroModels.ProductionFunctions import ProductionFunction


class RamseyModel:

    def __init__(self, depreciation_rate, discounting_rate, starting_capital, utility_function : UtilityFunction, production_function : ProductionFunction,
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

        n_steps = 5
        t= np.linspace(0.0001, self.time[-1], len(self.time)*n_steps)
        labor_function = interp1d(self.time,self.labor)

        # Define the function that returns the derivatives
        def fun(t, y):
            k, c = y
            l = labor_function(t)

            r = self.production_function.gradient([k, l])[0] -self.depreciation_rate # rate of return
            w =  self.production_function.gradient([k, l])[1]  # wage

            if c.any() == 0:
                print('consumption is 0, t=', t)
                pass
            e =  self.utility_function.absolute_risk_aversion(c) * c  # elasticity of substitution
            if e.any() == 0:
                print('elasticity is 0, t=', t)
                pass
            dc_dt = (r - self.discounting_rate) * c / e
            dk_dt = r * k + (w - c) * l

            return [dk_dt, dc_dt]

        # Define the boundary conditions
        def bc(ya, yb):
            return np.array([ya[0] - self.starting_capital,  # k(0) = self.starting_capital
                             yb[1]])  # c(time_horizon) = 0

        # Define the initial guess for the solution
        def init_guess(t):
            return np.ones((2, len(t)))*self.starting_capital

        sol = solve_bvp(fun, bc, t, init_guess(t))

        self.consumption[0] = sol.sol(0)[1]
        self.capital[0] = self.starting_capital

        for t in self.time[:-1]:

            self.income[t] = self.production_function([ self.capital[t], self.labor[t]])
            self.wages[t] = self.production_function.gradient([self.capital[t], self.labor[t]])[1]

            self.rental_rates[t] =  self.production_function.gradient([self.capital[t], self.labor[t]])[0]
            self.rates_of_return[t] = self.rental_rates[t]-self.depreciation_rate

            self.elasticity[t] = self.utility_function.absolute_risk_aversion(self.consumption[t]) * self.consumption[t]
            self.consumption[t+1] = self.consumption[t]*(1+(self.rates_of_return[t]-self.discounting_rate)/self.elasticity[t])
            self.capital[t+1]=self.rates_of_return[t]*self.capital[t]+self.wages[t]*self.labor[t]-self.consumption[t]*self.labor[t]

        self.income[-1] = self.production_function([ self.capital[-1], self.labor[-1]])
        self.wages[-1] = self.production_function.gradient([self.capital[-1], self.labor[-1]])[1]

        self.rental_rates[-1] =  self.production_function.gradient([self.capital[-1], self.labor[-1]])[0]
        self.rates_of_return[-1] = self.rental_rates[-1]-self.depreciation_rate

        self.elasticity[-1] = self.utility_function.absolute_risk_aversion(self.consumption[-1]) * self.consumption[-1]

        return pd.DataFrame({'time': self.time, 'capital': self.capital,  'consumption': self.consumption, 'labor': self.labor, 'income': self.income, 'rates': self.rates_of_return, 'wages': self.wages})


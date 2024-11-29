import numpy as np
import pandas as pd
import scipy
from Demos.SystemParametersInfo import new_value

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
        self.consumption =  0.0001*np.ones(time_horizon)
        self.time = range(time_horizon)
        self.elasticity = np.zeros(time_horizon)

        self.n_points = 3
        self.explore = 0.8

    def __call__(self):
        """
        Solves the Ramsey model using Value Function Iteration (VFI).

        Bellman equation:
        V(k)=max_{c} (f(k,c) + 1/(1+rho) * V(g(k,c))

        where:
        f(k_t,c_t) = L_t u(c_t)
        k_{t+1} = g(k_t,c_t) = (r+1)*k_t + (w_t- c_t)*l_t
        c_{t+1} = (1+(r_t-rho)/ e(c_t))*c_t

        with:
        r_t = F'_k(k_t,l_t) - delta
        w_t = F'_l(k_t,l_t)

        and:
        e elasticity
        rho discounting rate
        delta depreciation rate

        subject to:
        k_0 given
        c_T = 0

        so V(k_T)= max_c(L_T u(0) 1/(1+rho) * V(g(k,c))
        """

        def bellman_equation(c: float, k: float, v: callable, u: callable, params: list):
            # Model parameters
            beta, r, w, l = params

            # Calculate next period's utility
            V_prime = u(c) + beta * v(k * (1 - r) + l * (w - c))  # assume V is a function of k' and a

            # return discounted next period's utility

            return V_prime

        def vfi(params, grid_k, grid_c, max_iter=1000, tol=1e-6):

            grid_V = np.zeros(len(grid_k))  # initialize value function

            for i in range(max_iter):
                delta_V = 0
                for k_idx, k in enumerate(grid_k):

                    discounted_value = np.zeros(len(grid_c))
                    V = lambda k: np.interp(k, grid_k, grid_V)

                    for c_idx, c in enumerate(grid_c):
                        # Calculate expected value using Bellman equation
                        discounted_value[c_idx] = bellman_equation(k, c, V, self.utility_function, params)

                    v_max = np.max(discounted_value)

                    delta_V = np.max([delta_V, np.abs(v_max - grid_V[k_idx])])

                    grid_V[k_idx] = v_max  # update value function

                if delta_V < tol:
                    print("converged")
                    break

            return grid_V



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


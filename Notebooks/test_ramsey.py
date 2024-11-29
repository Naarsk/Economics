import numpy as np
import matplotlib.pyplot as plt

def production_function(k,l):
    return k**0.3*l**0.7

def utility_function(c):
    return np.log(c)

def bellman_equation(c : float, k : float, v : callable, u : callable, params :list):
    # Model parameters
    beta, r, w = params

    # Calculate next period's utility
    V_prime = u(c) + beta*v(k * (1 - r) + w - c)  # assume V is a function of k' and a

    # return discounted next period's utility

    return   V_prime


def vfi(params, grid_k, grid_c, max_iter=1000, tol=1e-6):


    grid_V = np.zeros(len(grid_k))  # initialize value function

    for i in range(max_iter):
        delta_V = 0
        for k_idx, k in enumerate(grid_k):

            discounted_value = np.zeros(len(grid_c))
            V = lambda k : np.interp(k, grid_k, grid_V)

            for c_idx, c in enumerate(grid_c):
                # Calculate expected value using Bellman equation
                discounted_value[c_idx] = bellman_equation(k, c, V, utility_function, params)

            v_max = np.max(discounted_value)

            delta_V = np.max([delta_V, np.abs(v_max - grid_V[k_idx])])

            grid_V[k_idx] = v_max  # update value function

        if delta_V < tol:
            print("converged")
            break

    return grid_V


# Example usage
params = [0.95, 0.05, 0.5]

grid_k = np.linspace(0.1, 100, 1000)
grid_c = np.linspace(0.1, 10, 1000)

grid_V = vfi(params, grid_k, grid_c)


plt.scatter(grid_k,grid_V)
plt.xlabel('k')
plt.ylabel('V')
plt.title('Value Function')
plt.show()


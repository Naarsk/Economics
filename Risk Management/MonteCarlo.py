#Chapter 10
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

n_paths = 10000
time_horizon = 21 # 1 trading month

zeta_vars = np.random.normal(0, 1, (n_paths, time_horizon))

def return_func(sigma, zeta, const):
    return const + sigma*zeta

def garch11_func(sigma, returns, omega, alpha, beta):
    return np.sqrt(omega + alpha*returns**2 + beta*sigma**2)


def simulation(zetas, sigma0, const, omega, alpha, beta):

    sigma = np.zeros((n_paths, time_horizon))
    returns = np.zeros((n_paths, time_horizon))

    sigma[:,0] = sigma0
    for j in range(time_horizon-1):
        returns[:,j+1] = return_func(sigma[:,j], zetas[:,j], const)
        sigma[:,j+1] = garch11_func(sigma[:,j], returns[:,j], omega, alpha, beta)

    return returns


returns_all_paths = simulation(zetas=zeta_vars, sigma0=10**(-2), const=10**(-4), omega=10**(-6), alpha=0.11, beta=0.86)
returns_average = np.mean(returns_all_paths, axis=0)

fig, axs = plt.subplots(2, figsize=(8, 6), dpi=100)

axs[0].hist(returns_all_paths[:,1], bins = 25, label="Day 1")
axs[0].hist(returns_all_paths[:,5], bins = 25, label = "Day 5", alpha = 0.8)
axs[0].hist(returns_all_paths[:,10], bins = 25, label = "Day 10", alpha = 0.5)
axs[0].hist(returns_all_paths[:,20], bins = 25, label = "Day 20", alpha = 0.2)
axs[0].legend()

axs[1].plot(np.arange(time_horizon), returns_average)
axs[1].plot(np.arange(time_horizon), returns_all_paths[0])
axs[1].plot(np.arange(time_horizon), returns_all_paths[1])
axs[1].plot(np.arange(time_horizon), returns_all_paths[2])

fig.tight_layout()

plt.show()

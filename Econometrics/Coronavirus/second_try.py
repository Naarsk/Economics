import numpy as np
from scipy.optimize import root

@np.vectorize
def g(n, alpha_1, alpha_2, epsilon):
    N_t = n[1:]
    N_t_1 = n[:-1]
    return np.array([   (N_t_1 - alpha_1 *N_t - alpha_2 *N_t ** (1 + epsilon)),
                        N_t * (N_t_1 - alpha_1 *N_t - alpha_2 *N_t ** (1 + epsilon)),
                        N_t ** epsilon * (N_t_1 - alpha_1 *N_t - alpha_2 *N_t ** (1 + epsilon))])

def m_hat(n,alpha_1, alpha_2, epsilon):
    return np.average(g(n, alpha_1, alpha_2, epsilon), axis=0)

def d_hat(n, alpha_2, epsilon):
    N_t = n[1:]
    d_alpha_1 = np.array([
        -N_t,
        -N_t**2,
        -N_t**(1+epsilon)
    ])
    d_alpha_2 = np.array([
        -N_t**(1+epsilon),
        -N_t**(2+epsilon),
        -N_t**(1+2*epsilon)
    ])
    d_epsilon = np.array([
        alpha_2 * N_t**(1+epsilon) * np.log(N_t),
        alpha_2 * N_t**(2+epsilon) * np.log(N_t),
        2 * alpha_2 * N_t**(1+2*epsilon) * np.log(N_t)
    ])
    jacobian = np.array([d_alpha_1, d_alpha_2, d_epsilon]).T
    return np.average(jacobian,axis=0)

def f(theta, n, S):
    alpha_1, alpha_2, epsilon = theta
    jacobian = d_hat(n, alpha_2, epsilon)
    function = m_hat(n, alpha_1, alpha_2, epsilon)
    return np.dot(jacobian, np.dot(S, function))


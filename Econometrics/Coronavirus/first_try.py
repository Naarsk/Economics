import numpy as np


def g(n, theta):
    alpha_1, alpha_2, epsilon = theta
    return np.array([1, n[1:], n[1:] ** epsilon ]) * (n[:-1] - alpha_1 * n[1:] - alpha_2 * n[1:] ** (1 + epsilon))

def compute_target_function(g, n, theta, S):
    # This function should compute the target function
    m=np.average(g(n, theta), axis=0)
    return m*S*m.T

def estimate_variance(g, n, theta):
    # This function should compute the sample equivalent of E g(N_t, N_{t+1}, theta) g^T(N_t, N_{t+1}, theta)
    # Replace this with your own implementation
    g=g(n, theta)
    return np.average(np.dot(g,g.T), axis=0)


    # Do something with theta_hat and variance
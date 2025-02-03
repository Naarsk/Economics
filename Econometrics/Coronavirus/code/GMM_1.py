import numpy as np

from Econometrics.Coronavirus.code.read_data import dta


def mean_func(params, data):
    N_t, N_t_1, N_t_2 = data
    alpha_1, alpha_2, epsilon = params

    if len(N_t) == len(N_t_1) == len(N_t_2):
        U_t = N_t - alpha_1 * N_t_1 - alpha_2 * N_t_1 ** (1 + epsilon)
        m1 = np.average(U_t)
        m2 = np.average(np.multiply(N_t_1, U_t))
        m3 = np.average(np.multiply(N_t_2, U_t))
    else:
        raise ValueError("N_t, N_t_1, N_t_2 must have the same length.")
    return np.asarray([m1, m2, m3])

def mean_var(params, data):
    N_t, N_t_1, N_t_2 = data
    alpha_1, alpha_2, epsilon = params

    if len(N_t) == len(N_t_1) == len(N_t_2):
        U_t = N_t - alpha_1 * N_t_1 - alpha_2 * N_t_1 ** (1 + epsilon)
        U_t_sq = np.multiply(U_t, U_t)
        N_t_1_sq = np.multiply(N_t_1, N_t_1)
        N_t_2_sq = np.multiply(N_t_2, N_t_2)
        N_t_cross = np.multiply(N_t_1,N_t_2)

        v11 = np.average(U_t_sq)
        v12 = np.average(np.multiply(U_t_sq, N_t_1))
        v13 = np.average(np.multiply(U_t_sq, N_t_2))
        v22 = np.average(np.multiply(U_t_sq, N_t_1_sq))
        v23 = np.average(np.multiply(U_t_sq, N_t_cross))
        v33 = np.average(np.multiply(U_t_sq, N_t_2_sq))
    else:
        raise ValueError("N_t, N_t_1, N_t_2 must have the same length.")
    return np.matrix([[v11, v12, v13],[v12, v22, v23],[v13, v23, v33]])

def mean_jac(params, data):
    N_t, N_t_1, N_t_2 = data
    alpha_1, alpha_2, epsilon = params

    if len(N_t) == len(N_t_1) == len(N_t_2):
        d11 = np.average(N_t_1)
        d12 = np.average(N_t_1**(1+epsilon))
        d13 = np.average(N_t_1**(1+epsilon) * alpha_2 * np.log(N_t_1))

        d21 = np.average(np.multiply(N_t_1, N_t_1))
        d22 = np.average(N_t_1**(2+epsilon))
        d23 = np.average(N_t_1**(2+epsilon) * alpha_2 * np.log(N_t_1))

        d31 = np.average(np.multiply(N_t_1, N_t_2))
        d32 = np.average(np.multiply(N_t_1**(1+epsilon), N_t_2))
        d33 = np.average(np.multiply(N_t_1**(1+epsilon), N_t_2) * alpha_2 * np.log(N_t_1))
    else:
        raise ValueError("N_t, N_t_1, N_t_2 must have the same length.")
    return np.matrix([[d11, d12, d13],[d21, d22, d23],[d31, d32, d33]])

data = np.asarray(dta[['Cumulative_cases', 'Cumulative_cases_lag_1', 'Cumulative_cases_lag_2']][10:])

theta = np.array([6.2, -5.4, 0.01])

m=mean_func(theta, data.T)
d=mean_jac(theta, data.T)
v=mean_var(theta, data.T)
s = np.linalg.inv(v)

alpha_1_grid = np.linspace(3, 5, 47)
alpha_2_grid = np.linspace(-1, -3, 47)
epsilon_grid = np.linspace(0.001, 0.1, 47)

delta_min = 10**(-3)
S = s
count = 0
delta = 1
min_delta = 100

while delta > delta_min:
    min_target = 10**20
    min_theta = theta
    old_theta = min_theta
    for alpha_1 in alpha_1_grid:
        for alpha_2 in alpha_2_grid:
            for epsilon in epsilon_grid:
                theta = np.array([alpha_1, alpha_2, epsilon])
                m=mean_func(theta, data.T)
                target = (m.T @ S) @ m
                if target < min_target:
                    min_target = target
                    min_theta = theta
    V = mean_var(min_theta, data.T)
    S = np.linalg.inv(V)
    delta = np.linalg.norm(old_theta - min_theta)
    if delta < min_delta:
        min_delta = delta
    print('count = ', count)
    print('delta = ', delta)
    print('theta = ', min_theta)
    if count > 4:
        break
    count=count+1

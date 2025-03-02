import numpy as np
from scipy.stats import chi2

from Econometrics.Coronavirus.code.read_data import dta
import json

def model_setup(data, params):
    # Unpack parameters
    alpha_1, alpha_2, epsilon = params
    # Check that all rows have the same length
    if not all(len(row) == len(data[0]) for row in data):
        raise ValueError("All rows in data must have the same length.")
    N_t = data[0]
    N_t_1 = data[1]    # Use the first instrument as the regressor in the structural equation.
    instruments = data[2:]  # This can be of arbitrary number (K instruments)

    U_t = error_equation(N_t, N_t_1, alpha_1, alpha_2, epsilon)
    return U_t, instruments

def error_equation(N_t, N_t_1, alpha_1, alpha_2, epsilon):
    U_t = N_t - alpha_1 * N_t_1 - alpha_2 * N_t_1 ** (1 + epsilon)
    # U_t = N_t - (alpha_1 + alpha_2) * N_t_1 - alpha_2 * epsilon * N_t_1 * np.log(N_t_1) + alpha_2*epsilon**2*N_t_1*(np.log(N_t_1))**2/2

    return U_t

# --- Functions for the moment conditions, their variance, and Jacobian ---
# The convention is that the input "data" is a 2D array (or matrix)
# where:
#    row 0 : dependent variable N_t
#    rows 1 to K : instruments (an arbitrary number)
#
# The moment conditions are defined as:
#    m_0    = E[ U_t ]
#    m_i    = E[ (instrument_i) * U_t ],  for i = 1,...,K
#
# with
#    U_t = N_t - alpha_1 * (N_t_1) - alpha_2 * (N_t_1)^(1+epsilon)
# Note: Here we assume that the first instrument (data row 1) is used as the regressor in the structural equation.
#       If you wish to use a different column for the regression (or add a constant), adjust accordingly.

def mean_func(params, data):
    """
    Computes the vector of moment conditions.

    Parameters:
        params : array-like, parameters [alpha_1, alpha_2, epsilon]
        data   : 2D array of shape ((K+1), n) where:
                 - data[0] is N_t (dependent variable)
                 - data[1:] are instruments (with the first instrument used in the regression)
    Returns:
        m : numpy array of moment conditions, with length (number of instruments + 1)
            m[0] = average(U_t)
            m[i] = average( (instrument_i) * U_t )  for i = 1,...,K.
    """
    U_t, instruments = model_setup(data, params)

    # Build the moment vector
    moments = [np.average(U_t)]     # Moment condition on constant

    # Moment conditions for each instrument
    for inst in instruments:
        moments.append(np.average(inst * U_t))
    return np.asarray(moments)


def mean_var(params, data):
    """
    Computes the variance-covariance matrix of the moment conditions.

    Parameters:
        params : array-like, parameters [alpha_1, alpha_2, epsilon]
        data   : 2D array with row0 = N_t and rows 1..K = instruments.
    Returns:
        V : Variance-covariance matrix (numpy array) of shape ((K+1) x (K+1))
    """
    U_t, instruments = model_setup(data, params)

    # Compute moment conditions for each observation
    # First moment: constant moment = U_t
    moment_array = [U_t]
    # Then each instrument's moment: instrument * U_t
    for inst in instruments:
        moment_array.append(inst * U_t)
    moment_array = np.array(moment_array)  # Shape: ((K+1), n)

    # Compute the covariance matrix by averaging the outer product over observations
    k = moment_array.shape[0]
    V = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            V[i, j] = np.average(moment_array[i] * moment_array[j])
    return V

def mean_jac(params, data):
    """
    Computes the Jacobian matrix (derivatives of moment conditions with respect to parameters).

    The moment conditions are:
       m0    = E[ U_t ]
       m_i   = E[ (instrument_i)*U_t ],  i = 1,...,K

    The derivative dU_t/dtheta is computed using the regressor (first instrument).

    Returns:
       Jacobian: A matrix of shape ((K+1) x 3)
    """
    alpha_1, alpha_2, epsilon = params

    if not all(len(row) == len(data[0]) for row in data):
        raise ValueError("All rows in data must have the same length.")

    instruments = data[2:]
    regressor = data[1]

    # Derivatives of U_t = N_t - alpha_1*regressor - alpha_2*regressor^(1+epsilon)
    dU_dalpha1 = -regressor
    dU_dalpha2 = -regressor ** (1 + epsilon)
    # dU_depsilon = -alpha_2 * (1+ epsilon) * regressor ** epsilon
    # dU_dalpha2 = -regressor * (1 + epsilon * np.log(regressor) - (epsilon * np.log(regressor))**2/2)
    dU_depsilon = -alpha_2 * regressor * (1+epsilon*np.log(regressor))

    # Jacobian for the constant moment condition m0 = average(U_t)
    jacobian_rows = []
    j0 = np.array([
        np.average(dU_dalpha1),
        np.average(dU_dalpha2),
        np.average(dU_depsilon)
    ])
    jacobian_rows.append(j0)

    # For each instrument moment: m_i = average( instrument_i * U_t )
    for inst in instruments:
        j_inst = np.array([
            np.average(inst * dU_dalpha1),
            np.average(inst * dU_dalpha2),
            np.average(inst * dU_depsilon)
        ])
        jacobian_rows.append(j_inst)

    return np.vstack(jacobian_rows)  # Shape: ((number of moments) x 3)

# --- Load and prepare the data ---
# For example, here we take three columns:
#   'Cumulative_cases' (as N_t),
#   'Cumulative_cases_lag_1',
#   'Cumulative_cases_lag_2'
#
# To use more instruments, simply include more columns in the list.
#
# We transpose the resulting array so that:
#   data[0] becomes N_t,
#   data[1] becomes first instrument, etc.

# Adjust the list of columns below to include as many instruments as you want.
# columns_to_use = ['Cumulative_cases', 'Cumulative_cases_lag_1', 'Cumulative_cases_lag_1_Hermite2', 'Cumulative_cases_lag_1_XlogX']

columns_to_use = ['Cumulative_cases',  'Cumulative_cases_lag_1', 'Average_temperature', 'Retail_and_recreation', 'Transit_stations', 'Parks', 'Cumulative_moving_average' ]
filtered_data = dta.loc[dta['Cumulative_cases_lag_1'] > 0, columns_to_use]
print(filtered_data)
data = np.asarray(filtered_data).T

# --- Grids for parameters ---
# alpha_1_grid = np.linspace(3, 4, 37)
# alpha_2_grid = np.linspace(-1.5, -2, 37)
# epsilon_grid = np.linspace(0.02, 0.03, 37)

alpha_1_grid = np.linspace(3.5, 4, 107)
alpha_2_grid = np.linspace(-2, -2.4, 97)
epsilon_grid = np.linspace(0.02, 0.025, 97)

# --- GMM Iteration ---
delta_min = 1e-3
count = 0
delta = 1
min_delta = 100

# --- Initial parameter values ---
theta = np.array([3.5, -1.8, 0.025])
V = mean_var(theta, data)
S = np.linalg.inv(V)  # Weighting matrix

while delta > delta_min:
    min_target = 1e20
    min_theta = theta.copy()
    old_theta = theta.copy()

    # Grid search over parameters
    for alpha_1 in alpha_1_grid:
        for alpha_2 in alpha_2_grid:
            for epsilon in epsilon_grid:
                theta_trial = np.array([alpha_1, alpha_2, epsilon])
                m_trial = mean_func(theta_trial, data)
                # Quadratic form: m' S m
                target = m_trial.T @ S @ m_trial
                if target < min_target:
                    min_target = target
                    min_theta = theta_trial.copy()

    # Update weighting matrix based on the new parameter estimates
    V = mean_var(min_theta, data)
    S = np.linalg.inv(V)
    delta = np.linalg.norm(old_theta - min_theta)
    if delta < min_delta:
        min_delta = delta

    print('Iteration count = ', count)
    print('delta = ', delta)
    print('theta = ', min_theta)

    theta = min_theta.copy()

    # For safety, break after a fixed number of iterations
    if count > 7:
        break
    count += 1

J = mean_jac(theta, data)
m = mean_func(theta, data)
Avar = np.linalg.inv(J.T @ S @ J)
N=len_data = len(data[0])
se = np.sqrt(np.diagonal(Avar)/N)

print('Var = ',Avar)

if len(columns_to_use) > 3:
    over_id = len(columns_to_use) - 5
    j_stat = N* m.T @ S @ m
else:
    over_id = 0
    j_stat = None

# create a dictionary with the model name and the results
results = {
    'N5*': {
        'regressors' : columns_to_use,
        'alpha1': theta[0],
        'alpha2': theta[1],
        'epsilon': theta[2],
        'se_alpha1': se[0],
        'se_alpha2': se[1],
        'se_epsilon': se[2],
        'sample size': N,
        'J-stat': j_stat,
        'Overid': over_id
    }
}

# open the file in write mode and write the dictionary to it
with open('../result_table/results_2.json', 'a') as f:
    json.dump(results, f, indent=4)

probability = 0.95
quantile = chi2.ppf(probability, over_id)

print("\nFinal parameter estimates:", theta)
print("\nFinal parameter se:", se)
print("\nFinal J-statistic:", j_stat)
print(f"The chi-squared quantile for probability {probability} and df {over_id} is {quantile}")
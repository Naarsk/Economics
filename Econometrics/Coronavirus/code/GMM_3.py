import numpy as np
from Econometrics.Coronavirus.code.read_data import dta


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
    # Unpack parameters
    alpha_1, alpha_2, epsilon = params

    # Check that all rows have the same length
    if not all(len(row) == len(data[0]) for row in data):
        raise ValueError("All rows in data must have the same length.")

    N_t = data[0]
    instruments = data[1:]  # This can be of arbitrary number (K instruments)

    # Use the first instrument as the regressor in the structural equation.
    # (If you wish to use a different column, change data index accordingly.)
    regressor = instruments[0]
    U_t = N_t - alpha_1 * regressor - alpha_2 * regressor ** (1 + epsilon)

    # Build the moment vector
    moments = []
    # Moment condition on constant
    moments.append(np.average(U_t))
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
    alpha_1, alpha_2, epsilon = params

    if not all(len(row) == len(data[0]) for row in data):
        raise ValueError("All rows in data must have the same length.")

    N_t = data[0]
    instruments = data[1:]
    regressor = instruments[0]
    U_t = N_t - alpha_1 * regressor - alpha_2 * regressor ** (1 + epsilon)

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

    N_t = data[0]
    instruments = data[1:]
    regressor = instruments[0]

    # Derivatives of U_t = N_t - alpha_1*regressor - alpha_2*regressor^(1+epsilon)
    dU_dalpha1 = -regressor
    dU_dalpha2 = -regressor ** (1 + epsilon)
    dU_depsilon = -alpha_2 * regressor ** (1 + epsilon) * np.log(regressor)

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
columns_to_use = ['Cumulative_cases', 'Cumulative_cases_lag_1', 'Cumulative_cases_lag_2','Cumulative_cases_lag_3','Cumulative_cases_lag_4','Cumulative_cases_lag_5']
filtered_data = dta.loc[dta['Cumulative_cases_lag_1'] > 0, columns_to_use]
data = np.asarray(filtered_data).T

# --- Initial parameter values ---
theta = np.array([3.5, -1.8, 0.025])

# Initial evaluations of moments, Jacobian, and variance
m = mean_func(theta, data)
J = mean_jac(theta, data)
V = mean_var(theta, data)
S = np.linalg.inv(V)  # Weighting matrix

# --- Grids for parameters ---
alpha_1_grid = np.linspace(3, 4, 67)
alpha_2_grid = np.linspace(-1.5, -2, 67)
epsilon_grid = np.linspace(0.02, 0.03, 67)

# --- GMM Iteration ---
delta_min = 1e-3
count = 0
delta = 1
min_delta = 100

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
    if count > 4:
        break
    count += 1

print("\nFinal parameter estimates:", theta)

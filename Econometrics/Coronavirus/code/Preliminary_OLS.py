# Load modules and data
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from Econometrics.Coronavirus.code.read_data import dta

epsilon_list = [1, 0.1, 0.01, 0.001, 0.0001]
params =[]


for epsilon in epsilon_list:
    data = np.matrix(dta[['New_cases', 'New_cases_lag_1']][10:]**(-epsilon)).T
    Y = np.asarray(data[0].T)
    X = np.asarray(data[1].T)
    X_const = sm.add_constant(X)
    # Fit and summarize OLS model
    mod = sm.OLS(Y,X_const)

    res = mod.fit()
    params.append(res.params)
    plt.scatter(X, Y, label='Actual Data')
    plt.plot(X, res.predict(X_const), label='Predicted Data', color='red')
    plt.title('Model with epsilon={}'.format(epsilon))
    plt.legend()
    plt.savefig('./plots/epsilon_{}.png'.format(epsilon))
    plt.close()
print(np.matrix(params))

# Load modules and data
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

epsilon_list = [1, 0.1, 0.01, 0.001, 0.0001]
params =[]

df= pd.read_csv('../WHO-COVID-19-global-daily-data.csv')
start_date = '2020-02-15'
end_date = '2020-07-15'
italy_df = df[(df['Country'] == 'Italy') &
              (df['Date_reported'] >= start_date) &
              (df['Date_reported'] <= end_date)]
dta=pd.DataFrame(italy_df['New_cases'].fillna(0))
dta.insert(1, 'New_cases_lag_1', dta['New_cases'].shift(1).fillna(0))

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

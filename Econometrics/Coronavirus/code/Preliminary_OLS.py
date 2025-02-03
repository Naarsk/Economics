# Load modules and data
import os

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

from Econometrics.Coronavirus.code.read_data import dta

epsilon_list = [1, 0.1, 0.01, 0.001, 0.0001]
params =[]

results_list = []
model_names = []

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
    plt.savefig('../plots/epsilon_{}.png'.format(epsilon))
    plt.close()

    # Save the result and a name for the column.
    results_list.append(res)
    model_names.append(rf"\varepsilon =: {epsilon}")

# Create a combined summary table with stars for significance levels.
combined_table = summary_col(results_list, stars=True, model_names=model_names,
                             info_dict={'N': lambda x: f"{int(x.nobs)}"})  # You can add more info as needed.

# Get the LaTeX string for the table.
latex_str = combined_table.as_latex()

# Specify your desired output directory (make sure it exists or create it if needed)
output_dir = '../result_table'  # Example: 'output' directory in the current working directory.
os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist.

# Define the full file path.
output_file = os.path.join(output_dir, 'combined_regressions.tex')

# Write the LaTeX table to the file.
with open(output_file, 'w') as f:
    f.write(latex_str)

print(np.matrix(params))

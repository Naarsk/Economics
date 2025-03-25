import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numerical_distributions.scipy_distribution import dump_distribution

# Alternatively, you can load all sheets at once
dfs = pd.read_excel('EU_Data.xlsx', sheet_name=None)

euronext100 = dfs['Euronext100'][['Date', 'Close']]
stoxx50e = dfs['Stoxx50E'][['Date', 'Close']]

data = stoxx50e

data['Returns'] = data['Close'].fillna(0).pct_change().fillna(0)

sample = (data['Returns'] - data['Returns'].mean())/data['Returns'].std()

kde = gaussian_kde(sample)

plot = False

if plot:
    # Plot the PDF
    x = np.linspace(sample.min(), sample.max(), 100)
    plt.plot(x, kde(x), label='PDF')
    plt.legend()
    plt.show()

    # Generate a new sample from the KDE
    new_sample = kde.resample(size=1000)[0]
    # print(new_sample)
    # Plot a histogram of the new sample
    plt.hist(new_sample, bins=30, density=True, label='Generated Sample')
    plt.legend()
    plt.show()

save = True


if save:
    x = np.linspace(-15,15,10000)
    y = kde(x)
    dump_distribution(x, y, 'kde_stoxx50e')
import numpy as np
from arch.univariate import arch_model, ConstantMean, GARCH
from numerical_distributions.arch_distribution import CustomDistribution
log_returns = np.random.normal(0, 1, 1000)  # Example data

garch = arch_model(log_returns, p=1, q=1, o=0, dist='normal')
tarch = arch_model(log_returns, mean='zero', p=1, o=1, q=1,power=1.0, dist='skewstudent')

g_results = garch.fit()
t_results = tarch.fit()

print("Garch \n",g_results.summary())
print("Tarch \n",t_results.summary())


am=ConstantMean(log_returns)
am.volatility = GARCH(1,0,1)
am.distribution = CustomDistribution()
import numpy as np
import matplotlib.pyplot as plt
from numerical_distributions.levy_distribution import _psi_tld
mu = 2.0
C = 1.0
alpha = 1.7
lam = 0.2
beta = 0.3

#Plotting
k_values = np.linspace(-5, 5, 400)
psi_values = _psi_tld(k_values, mu, C, alpha, lam, beta)
# Plot real and imaginary parts
plt.figure(figsize=(10, 5))
plt.plot(k_values, psi_values.real, label=r'$Re(\psi_{TL})$', color='b')
plt.plot(k_values, psi_values.imag, label=r'$Im(\psi_{TL})$', color='r',linestyle='dashed')
plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
plt.xlabel("k")
plt.ylabel(r"$\psi_{TL}$")
plt.legend()
plt.title(r"Characteristic function $\psi_{TL}$")
plt.grid()
plt.show()
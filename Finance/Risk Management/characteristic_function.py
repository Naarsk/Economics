import numpy as np
import matplotlib.pyplot as plt

def psi_TL(k, mu, c, alpha, lam, beta):

    term1 = 1j * mu * k
    term2 = c**alpha * ((lam**alpha - (k**2 + lam**2)**(alpha/2)) / np.cos(np.pi * alpha / 2)) * np.cos(alpha * np.arctan(np.abs(k) / lam))
    term3 = (1 + 1j * np.sign(k) * beta * np.tan(alpha * np.arctan(np.abs(k) /lam)))
    return np.exp(term1 + term2 * term3)

    #return np.exp(1j * mu * k - np.sqrt(-2j*c*k))
    #return np.exp( 1j * mu * k- c ** alpha / np.cos(np.pi * alpha / 2) * ((k ** 2 + lam ** 2) ** (alpha / 2) * np.cos(alpha * np.arctan(np.abs(k) / lam)) - lam ** alpha))


mu = 2.0
C = 1.0
alpha = 1.7
lam = 0.2
beta = 0.3

#Plotting
k_values = np.linspace(-5, 5, 400)
psi_values = psi_TL(k_values, mu, C, alpha, lam, beta)
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
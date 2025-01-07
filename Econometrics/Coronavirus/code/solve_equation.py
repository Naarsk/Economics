import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# define the model
def model(x, t, a, b, c):
    dxdt = a * x + b * x ** (1 + c)
    return dxdt

# parameters
a = 8.00
b = -6.95
c = 8.96*10**(-4)

# initial condition
x0 = 1.0

# time points
t = np.linspace(0, 150, 1000)

# solve ODE
x = odeint(model, x0, t, args=(a, b, c))

# compute derivative of solution
dxdt = a * x + b * x ** (1 + c)

# create two subplots that share the same x-axis
fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# plot the solution on the first subplot
axs[0].plot(t, x)
axs[0].set_ylabel('x(t)')
axs[0].set_title('Solution of the differential equation')
axs[0].grid()

# plot the derivative on the second subplot
axs[1].plot(t, dxdt)
axs[1].set_xlabel('t')
axs[1].set_ylabel("x'(t)")
axs[1].set_title("Derivative of the solution")
axs[1].grid()

# layout so plots do not overlap
fig.tight_layout()

plt.show()
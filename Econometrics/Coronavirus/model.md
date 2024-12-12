<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

My first try was with the following differential equation

$$\frac{dN}{dt} = r N \left(1-\frac{N}{K}\right), \quad N(0)=N_0$$

Which admits the following solution:

$$N(t) = \frac{K}{1+ \left(\frac{K-P_0}{P_0}\right) e^{-rt}}$$

That has a flex (i.e. the max of its derivative) in:
$$ t^* = \frac{1}{r}\ln \left(\frac{K-P_0}{P_0}\right)$$

To use a linear regression, the differential equation can be discretized into:
$$ N_{t+1} = \alpha_1 N_t + \alpha_2 N_{t}^2$$

So the parameter can be found using:
$$ r = \alpha_1-1, \quad K= \frac{1-\alpha_1}{\alpha_2}, \quad P_0 = 1$$

Recently, I found that:
$$ N_{t+1} = \alpha_1 N_t + \alpha_2 N_{t}^{1 + \varepsilon}$$

With $\varepsilon \in (0,1)$ would work much better to reproduce the right-hand skewdness of the contagion curve.

This equation too has a close form solution, being a Bernoulli differential equation:

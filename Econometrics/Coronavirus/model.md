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
$$V = N^{-\epsilon}, \qquad \frac{\dot V}{V} = -r \epsilon \frac{\dot N}{N}, \quad \Rightarrow \quad \dot V = -r\epsilon V + \frac{r\epsilon}{K}. $$
Which has close form solution:
$$ V(t) = K^{-1} + \left(V_0-K^{-1}\right)e^{-r\epsilon t} \quad \Rightarrow \quad N(t)=\frac{1}{\left(K^{-1} + \left(N_0^{-\epsilon}-K^{-1}\right)e^{-r\epsilon t}\right)^\epsilon}\,.$$

The maximum of its derivative is determined by solving:
$$ \ddot N = -\frac{V^{-\frac{1+\epsilon}{\epsilon}}}{\epsilon} \left(\ddot V -\frac{1+\epsilon}{\epsilon} \frac{\dot V ^2}{V}\right)$$

Where:
$$ \dot V = -r\epsilon\left(V_0-K^{-1}\right)e^{-r\epsilon t}, \qquad \ddot V = (r\epsilon)^2 \left(V_0-K^{-1}\right)e^{-r\epsilon t}. $$

In terms of finite differences:
$$ V_{t+1} = \beta_0 + \beta_1 V_t,$$
where:
$$\beta_0 = \frac{r \epsilon}{K}, \quad \beta_1=  r\epsilon.$$

Estimate 

% \dot V(t) = -r\epsilon\, \left(V_0-K^{-1}\right)e^{-r\epsilon t} 

For P-0 consider the stationary distribution as t_> infty and estimate t* as an average with that distribution
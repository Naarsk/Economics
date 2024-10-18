class ProductionFunction():
    def __init__(self, func, params):
        """
        Initialize a ProductionFunction object.

        Parameters
        ----------
        func : callable
            A function that takes two arguments (capital and labor) and returns
            the output of the production function.
        params : list
            A list of parameters of the production function.

        Returns
        -------
        None
        """
        self.func=func
        self.params=params

    def __call__(self, x):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        x : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return self.func(x, self.params)


class CobbDouglas(ProductionFunction):
    def __init__(self, alpha):

        """
        Initialize a CobbDouglas object.

        Parameters
        ----------
        alpha : float
            The exponent of capital in the Cobb-Douglas production function.

        Returns
        -------
        None
        """
        def cobb_douglas(x: list[2]):

            """
            Evaluate the Cobb-Douglas production function with the given variables and parameters.

            Parameters
            ----------
            x : list[2]
                A list of the variables to evaluate the production function with. The first element
                is capital, and the second element is labor.
            Returns
            -------
            float
                The output of the Cobb-Douglas production function.
            """
            k = x[0]
            l = x[1]

            return k ** alpha * l ** (1 - alpha)

        super().__init__(func=cobb_douglas, params=[alpha,])

    def __call__(self, x):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        x : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return super().__call__(x)


class CES(ProductionFunction):
    def __init__(self, sigma, gamma):
        """
        Initialize a CES object.

        Parameters
        ----------
        sigma : float
            The elasticity of substitution in the CES production function.
        gamma : float
            The distribution parameter in the CES production function.

        Returns
        -------
        None
        """
        def ces(x: list[5]):
            """
            The Constant Elasticity of Substitution (CES) production function.

            Parameters
            ----------
            x : list[5]
                A list with five elements, the first being the capital stock (k), the second being the labor (l), the third being the human capital (a_h), the fourth being the technological progress parameter for capital (a_k) and the fifth being the technological progress parameter for labor (a_l).

            Returns
            -------
            The output of the CES production function, which is given by a_h*(gamma*(a_k*k)**((sigma-1)/sigma) + a_l*(a_k*k)**((sigma-1)/sigma))**(sigma/(sigma-1)).
            """
            k=x[0]
            l=x[1]
            a_h=x[2]
            a_k=x[3]
            a_l=x[4]

            return a_h*(gamma*(a_k*k)**((sigma-1)/sigma) + (1-gamma)*(a_l*l)**((sigma-1)/sigma))**(sigma/(sigma-1))

        super().__init__(func=ces, params=[sigma,gamma])

    def __call__(self, x):
        """
        Evaluate the production function with the given variables and parameters.

        Parameters
        ----------
        x : list
            A list of the variables to evaluate the production function with.

        Returns
        -------
        float
            The output of the production function.
        """
        return super().__call__(x)

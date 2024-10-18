def logistic_map(x,r, n):
    return (1+r) * x * (1-x*n)

def compounded_growth(x,r):
    return (1+r) * x

def linear_growth(x,a):
    return x+a

def identity(x):
    return x
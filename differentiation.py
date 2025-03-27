import autograd as ad
import cvxpy as cp
import numpy as np
import pandas as pd


data = pd.read_csv("Data/returns_data.csv", index_col=0, parse_dates=True)

# compute expected returns and covariance matrix
mu_bar = data.mean().values
n = len(mu_bar)
Sigma = data.cov().values + 0.1*np.eye(n)
lambda_ = 0.1

# portfolio weight variable
w = cp.Variable(n)

#parameter is a special class that can be changed without changing the problem
b = cp.Parameter(n)
b.value = mu_bar

# objective: Maximize (w.T * mu_bar) - (lambda/2 * w.T * Sigma * w)
objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma))

constraints = [
    cp.sum(w) == 1,  
    w >= 0           
]

# solve the problem
problem = cp.Problem(objective, constraints)

#the requires_grad =True allows to call backward
problem.solve(requires_grad=  True)
# fills the parameter values with the gradient of 
# the optimum value w* wrt the parameter i.e. dw*/db
problem.backward()


# this is how you access the gradient of w* wrt mu_bar (as a parameter)
b.gradient

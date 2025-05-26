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


# # portfolio weight variable
# w = cp.Variable(n)

# #parameter is a special class that can be changed without changing the problem
# b = cp.Parameter(n)
# b.value = mu_bar

# # objective: Maximize (w.T * mu_bar) - (lambda/2 * w.T * Sigma * w)
# objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma))

# constraints = [
#     cp.sum(w) == 1,  
#     w >= 0           
# ]

# # solve the problem
# problem = cp.Problem(objective, constraints)

# #the requires_grad =True allows to call backwardIO hav
# problem.solve(requires_grad=  True)
# # fills the parameter values with the gradient of 
# # the optimum value w* wrt the parameter i.e. dw*/db
# problem.backward()



def g(w, mu, Sigma):
    return np.dot(w.T, mu) - (lambda_/2) * np.dot(w, Sigma @ w)

def grad_g(w_grad, mu):
    return w_grad @ mu - lambda_ * (Sigma @ w_grad)

def F_function(data:pd.DataFrame, lambda_, alpha):
    t, n = data.shape
    feature_matrix = np.vstack([data.values[:-1], np.ones(n)])
    target = data.values[-1]
    Sigma = data.cov().values + lambda_*np.eye(t)
    
    def F(theta: np.ndarray):
        w = cp.Variable(t)
        b = cp.Parameter(t)
        mu_hat = theta.T @ feature_matrix
        b.value = mu_hat
        objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma))
        constraints = [
        cp.sum(w) == 1,  
        w >= 0           
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(requires_grad = True)
        problem.backward()

        rms = np.linalg.norm(target - mu_hat)**2

        g1 = g(w.value, target, Sigma)
        b.value = target
        problem.solve(requires_grad = True)
        g2 = g(w.value, target, Sigma)

        regret = np.linalg.norm(g1 - g2)**2

        rmsgrad = 2*(mu_hat - target).T @ feature_matrix
        dwstar = feature_matrix @ b.gradient
        regretgrad = (2*(g1 - g2))* grad_g(dwstar, target)
        
        return rms + alpha*regret, rmsgrad + alpha*regretgrad
    
    return F





def G_function(theta, lambda_, alpha):
    def G(x):
        mu_hat = theta.T @ x
        (t,) =theta.shape
        w = cp.Variable(t)
        b = cp.Parameter(t)
        b.value = mu_hat
        objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma))
        constraints = [
        cp.sum(w) == 1,  
        w >= 0           
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(requires_grad = True)
        problem.backward()

        rms = np.linalg.norm(target - mu_hat)**2
    
    return G
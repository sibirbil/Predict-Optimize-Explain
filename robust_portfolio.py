
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.linalg import svd

# Our stock prices
df = pd.read_csv("Data/financial_data.csv", index_col=0, parse_dates=True)

# Here we compute simple returns
x = df.pct_change().dropna()

# mean of simple returns - we can also check log-mean
x_bar = x.mean().values
x_bar = x_bar.reshape(-1,1)

# Our covariance matrix for the returns
Sigma = x.cov().values

# uncertainty matrix is obtained as the diagonal matrix of variances
Omega = np.diag(x.var().values)


U, S, _ = svd(Omega)  # singular value decomp
S_sqrt = np.sqrt(np.diag(S)) 
GQ = U @ S_sqrt  # for robustness constraint

# number of assets
n = len(x_bar)

# Define decision variables
theta = cp.Variable(n) # weights
s = cp.Variable()  # risk variable
sq = cp.Variable()  # robustness term


kappa = 2.0  # robustness parameter
lambda_ = 0.2  # Risk aversion parameter

# constraints
constraints = [
    cp.sum(theta) == 1,  
    theta >= 0  
]

# robustness constraint using SVD
constraints.append(cp.SOC(sq, GQ.T @ theta))


objective = cp.Maximize(theta @ x_bar - t - (lambda_ / 2) * cp.quad_form(theta, Sigma))

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS, max_iters = 5000, abstol=1e-10, reltol=1e-10, feastol=1e-10)

# Extract results
optimal_weights = theta.value
optimized_objective = prob.value

# Let's compare with the analytical result:
theta_mu = optimal_weights.T @ x_bar  # theta^T x

# Compute sqrt(theta^T Omega theta)
sqrt_theta_Omega_theta = np.sqrt(optimal_weights.T @ Omega @ optimal_weights)

# Compute (lambda/2) * theta^T Sigma theta
quad_form = (lambda_ / 2) * (optimal_weights.T @ Sigma @ optimal_weights)

# Compute the theoretical objective value
computed_objective = theta_mu - kappa * sqrt_theta_Omega_theta - quad_form

# Compare the results
difference = np.abs(optimized_objective - computed_objective)
print("Optimized Objective Value from Solver:", optimized_objective)
print("Computed Objective Value from Given Equation:", computed_objective)
print("Difference:", difference)


np.save("optimal_weights.npy", optimal_weights)

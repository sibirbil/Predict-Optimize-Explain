import cvxpy as cp
import numpy as np
import pandas as pd

# Load financial data
data = pd.read_csv("Data/financial_data.csv", index_col=0, parse_dates=True)

# Compute expected returns and covariance matrix
mu0 = data.pct_change().mean().values
Sigma = data.pct_change().cov().values

# Fix non-positive definite covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
eigenvalues = np.maximum(eigenvalues, 1e-6)  # Replace small/negative eigenvalues
Sigma = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# Define problem parameters
n = len(mu0)  # Number of assets
delta = 0.5   # Risk aversion parameter
gamma = 0.2 # Robustness parameter

# Define the portfolio weight variable
x = cp.Variable(n)

# Define risk term: Portfolio variance
risk = cp.quad_form(x, Sigma)

# Define robustness term: L2-norm of x
robustness = cp.norm(x, 2)

# Define expected return
expected_return = mu0 @ x

# Objective function: Maximize worst-case return - risk penalty
objective = cp.Maximize(expected_return - gamma * robustness - delta * risk)

# Define constraints
constraints = [
    cp.sum(x) == 1,  # Budget constraint (sum of weights = 100%)
    x >= 0           # No short-selling constraint
]

# Solve the optimization problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)  # Use the free SCS solver

# Print the optimal weights
print("✅ Optimal Portfolio Weights:", x.value)

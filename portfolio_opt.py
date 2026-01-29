import cvxpy as cp
import numpy as np
import pandas as pd



data = pd.read_csv("Data/returns_data.csv", index_col=0, parse_dates=True)

# compute expected returns and covariance matrix
mu_bar = data.mean().values


n = len(mu_bar)  # num of assets
lambda_ = 0.1   # risk aversion parameter

# random symmetric positive semi-definite covariance matrix
A = np.random.randn(n, n)
Sigma = A @ A.T  # Ensures positive semi-definiteness
#Sigma = data.cov().values

"""
# Fix non-positive definite covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
eigenvalues = np.maximum(eigenvalues, 1e-6)  # Replace small/negative eigenvalues
Sigma = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
"""

# problem parameters



# portfolio weight variable
w = cp.Variable(n)

# objective: Maximize (w.T * mu_bar) - (lambda/2 * w.T * Sigma * w)
objective = cp.Maximize(w.T @ mu_bar - (lambda_ / 2) * cp.quad_form(w, Sigma))

constraints = [
    cp.sum(w) == 1,  
    w >= 0           
]

# solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal Portfolio Weights:", w.value)
print("Expected Return:", np.dot(w.value, mu_bar))
print("Portfolio Variance:", w.value.T @ Sigma @ w.value)



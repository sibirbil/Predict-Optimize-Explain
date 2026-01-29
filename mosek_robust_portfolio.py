import mosek.fusion as mf
import numpy as np
import pandas as pd
from scipy.linalg import svd


df = pd.read_csv("Data/financial_data.csv", index_col=0, parse_dates=True)

# compute simple returns  - we can also use logreturn
x = df.pct_change().dropna()

# expected returns
x_bar = x.mean().values.reshape(-1, 1)

# covariance matrix
Sigma = x.cov().values

# uncertainty matrix is set as the diagonal of variances
Omega = np.diag(x.var().values)

# SVD decomposition on Omega
U, S, _ = svd(Omega)  # singular value decomp
S_sqrt = np.sqrt(np.diag(S))  # sqrt of singular values
GQ = U @ S_sqrt  # transformed matrix for robustness constraint

n = len(x_bar) # num of assets
gamma = 0.2  # robustness param - size of uncertainty region

# to keep results for different lambda_
deltas = np.linspace(0.25, 1, 4)  # varying risk aversion values for analysis
columns = ["delta", "obj", "return", "risk"] + df.columns.tolist()
df_result = pd.DataFrame(columns=columns)

"""
We can model both terms using the second-order cones. 
For the term with square-root, the quadratic cone, 
while the portfolio variance term can be modeled using the rotated quadratic cone.
"""
df_weights = pd.DataFrame(index=df.columns)  # Create DataFrame to store weights
with mf.Model("Robust") as M:
    
    # variables
    theta = M.variable("theta", n, mf.Domain.greaterThan(0.0))  # portfolio weights
    s = M.variable("s", 1, mf.Domain.greaterThan(0.0))  # portfolio risk
    t = M.variable("t", 1, mf.Domain.greaterThan(0.0))  # robustness term

    # sum(theta) = 1
    M.constraint('budget', mf.Expr.sum(theta), mf.Domain.equalsTo(1.0))

    # objective func
    delta = M.parameter()
    worst_case_return = mf.Expr.sub(mf.Expr.dot(theta, x_bar.flatten()), mf.Expr.mul(gamma, t))
    M.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.sub(worst_case_return, mf.Expr.mul(delta, s)))

    # robustness constraint
    M.constraint("robustness", mf.Expr.vstack(t, mf.Expr.reshape(mf.Expr.mul(GQ.T, theta), n)), mf.Domain.inQCone())
     
    # risk constraint 
    G = np.linalg.cholesky(Sigma)  
    M.constraint("risk", mf.Expr.vstack(s, mf.Expr.constTerm(1.0), mf.Expr.reshape(mf.Expr.mul(G.T, theta), n)), mf.Domain.inRotatedQCone())

    # Solve for different risk parameters
    for d in deltas:
        delta.setValue(d)  # set risk tolerance parameter
        M.solve()
    
        # Check if the solution is feasible
        if M.getPrimalSolutionStatus() == mf.SolutionStatus.Optimal:
            df_weights[f"delta_{d}"] = theta.level()  # Store weights
        else:
            print(f"Warning: No optimal solution found for delta={d}")
        #results
        portfolio_return = np.dot(x_bar.T, theta.level())[0] - gamma * t.level()[0]
        portfolio_risk = np.sqrt(2 * s.level()[0])
        row = pd.Series([d, M.primalObjValue(), portfolio_return, portfolio_risk] + list(theta.level()), index=columns)
    
       
        df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
    

print(df_result)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:30:28 2025

@author: batuhanatas
"""

import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


def build_layer(n_assets: int, lambda_: float, preds: torch.Tensor, X_batch: torch.Tensor) -> CvxpyLayer:
    """
    Builds a CVXPY optimization layer for portfolio allocation using predicted returns.

    Args:
        n_assets (int): Number of assets in the current batch (B).
        lambda_ (float): Risk aversion parameter.
        preds (torch.Tensor): Predicted returns for the assets, shape (B,).
        X_batch (torch.Tensor): Feature matrix for assets, shape (B, F). 

    Returns:
        CvxpyLayer: A differentiable CVXPY layer solving the portfolio optimization problem.
    """
    w = cp.Variable(n_assets)  # Portfolio weights
    b = cp.Parameter(n_assets)  # Predicted returns
    b.value = preds.detach().cpu().numpy()

    # Diagonal covariance for simplicity (alternatively, use covariance matrix from X_batch)
    sigma = np.diag(preds.detach().cpu().numpy() ** 2).astype(np.float32)
    sigma_cvx = cp.Constant(sigma)

    # Mean-variance objective
    objective = cp.Maximize(w.T @ b - (lambda_ / 2) * cp.quad_form(w, sigma_cvx))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)

    assert problem.is_dpp()  # Ensure problem complies with DPP (disciplined parametric programming)

    return CvxpyLayer(problem, parameters=[b], variables=[w])

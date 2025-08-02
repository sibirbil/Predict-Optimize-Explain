#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:28:54 2025

@author: batuhanatas
"""

import torch
from torch.nn import Module
from torch import func as func
from cvxpylayers.torch import CvxpyLayer
from src.portfolio_layer import build_layer

def regret_loss(
    model: Module,
    lambda_: float,
    params: dict,
    x_t: torch.Tensor,  # shape (B, F)
    y_t: torch.Tensor   # shape (B,)
) -> torch.Tensor:
    """
    Computes the differentiable regret loss for a batch of assets.

    Args:
        model (nn.Module): Prediction model.
        lambda_ (float): Risk aversion parameter.
        params (dict): Model parameters.
        x_t (torch.Tensor): Features for B assets, shape (B, F).
        y_t (torch.Tensor): True returns for B assets, shape (B,).

    Returns:
        torch.Tensor: Scalar regret loss value.
    """
    B = len(y_t)
    preds = func.functional_call(model, params, x_t)  # Predicted returns, shape (B,)
    cvxpylayer = build_layer(B, lambda_, preds, x_t)

    y_t_tensor = y_t.detach().clone().requires_grad_(False)

    # Oracle solution (true returns)
    y_solution, = cvxpylayer(y_t_tensor)

    # Model-based portfolio (predicted returns)
    p_solution, = cvxpylayer(preds)

    # Regret: difference in portfolio returns
    regret = torch.dot(y_t, p_solution) - torch.dot(y_t, y_solution)
    return regret.pow(2)  # Squared regret loss

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:28:54 2025

@author: batuhanatas
"""

from typing import Optional

import torch
from torch.nn import Module
from torch import func as func
from cvxpylayers.torch import CvxpyLayer
from src.portfolio_layer import build_portfolio_layer, prepare_layer_inputs

def regret_loss(
    model: Module,
    lambda_: float,
    params: dict,
    x_t: torch.Tensor,  # shape (B, F)
    y_t: torch.Tensor,  # shape (B,)
    portfolio_layer: Optional[CvxpyLayer] = None,
    Sigma_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the differentiable regret loss for a batch of assets.

    Args:
        model (nn.Module): Prediction model.
        lambda_ (float): Risk aversion parameter.
        params (dict): Model parameters.
        x_t (torch.Tensor): Features for B assets, shape (B, F).
        y_t (torch.Tensor): True returns for B assets, shape (B,).

        Sigma_override (torch.Tensor, optional): Pre-computed covariance matrix for the batch.

    Returns:
        torch.Tensor: Scalar regret loss value.
    """
    B = len(y_t)
    layer = portfolio_layer or build_portfolio_layer(B, lambda_)
    preds = func.functional_call(model, params, x_t)

    sigma_source = None if Sigma_override is None else Sigma_override.detach()
    preds_cpu, L_cpu = prepare_layer_inputs(
        preds,
        x_t if Sigma_override is None else None,
        Sigma_override=sigma_source,
    )
    y_cpu = y_t.to(dtype=torch.double, device="cpu")

    y_solution, = layer(y_cpu, L_cpu)
    p_solution, = layer(preds_cpu, L_cpu)

    regret = torch.dot(y_cpu, p_solution) - torch.dot(y_cpu, y_solution)
    return regret.pow(2).to(x_t.device, dtype=x_t.dtype)

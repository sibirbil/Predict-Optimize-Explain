#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:52:14 2025

@author: batuhanatas
"""

import numpy as np
import torch
import torch.nn as nn
from torch import func as func

from src.portfolio_layer import build_portfolio_layer, prepare_layer_inputs

def G_function_benchmark(
    model               : nn.Module,
    params              : dict[str, torch.Tensor],
    x0                  : torch.Tensor,     #of shape (A, F) for A in the number of assets
    benchmark_return    : float,
    Sigma,
    lambda_
    ,
    centroids: torch.Tensor | None = None,
    centroid_weight: float = 0.0,
    *,
    feature_anchor: torch.Tensor | None = None,
    feature_reg: float = 0.0,
    return_reg: float = 0.0,
    pred_target: float = 0.0,
    **_unused,
    ):
    A, F = x0.shape         # A: number of assets in portfolio, F: number of features per asset
    portfolio_layer = build_portfolio_layer(A, lambda_)
    benchmark_target = torch.as_tensor(float(benchmark_return), dtype=torch.double)

    if isinstance(Sigma, torch.Tensor):
        Sigma_cpu = Sigma.detach().cpu().double()
    else:
        Sigma_cpu = torch.from_numpy(np.asarray(Sigma)).to(dtype=torch.double)

    def G(x: torch.Tensor) -> torch.Tensor:
        # Accept either full state (A, 1+F) or features-only (A, F).
        # If full, slice off the returns column before feeding the model.
        if x.ndim != 2 or x.shape[0] != A:
            raise ValueError(f"G expected x with shape (A, F) or (A, 1+F) where A={A}; got {tuple(x.shape)}")
        if x.shape[1] == F + 1:
            x_feats = x[:, 1:]
        elif x.shape[1] == F:
            x_feats = x
        else:
            raise ValueError(f"G got incompatible features dim {x.shape[1]}; expected F={F} or 1+F={F+1}")

        # functional_call requires args as a tuple
        pred = func.functional_call(model, params, (x_feats,))     # (A,)
        print(pred.shape)
        assert pred.shape == (A,), f"Model prediction shape mismatch: expected ({A},), got {pred.shape}"
        preds_cpu, Sigma_param = prepare_layer_inputs(pred, Sigma_override=Sigma_cpu)
        (w_opt,) = portfolio_layer(preds_cpu, Sigma_param)
        # Realized portfolio return evaluated on scenario returns (y)
        preds_cpu = preds_cpu.to(dtype=torch.double, device="cpu")
        portfolio_ret_preds_cpu = torch.dot(w_opt, preds_cpu)
        loss = (portfolio_ret_preds_cpu - benchmark_target) ** 2

        # centroid regularization: keep features near nearest centroid(s)
        if centroid_weight and centroids is not None and centroids.numel() > 0:
            # centroids: (K, F)
            # compute squared distance of each asset feature vector to nearest centroid
            # x_feats: (A, F)
            x_d = x_feats.double()
            # distances: (A, K)
            dists = torch.cdist(x_d, centroids.to(dtype=torch.double), p=2)
            
            min_dists, _ = torch.min(dists, dim=1)
            loss = loss + float(centroid_weight) * torch.sum(min_dists**2) # maybe different weights to different centroids?


        # feature-anchor regularization (optional): keep features near seed snapshot
        if feature_reg and feature_anchor is not None:
            feat_diff = x_feats.double() - feature_anchor.to(dtype=torch.double)
            loss = loss + float(feature_reg) * torch.sum(feat_diff ** 2)

        return loss
    
    def gradG(x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        val = G(x)
        (grad,) = torch.autograd.grad(val, x)
        return grad

    @torch.no_grad()
    def portfolio_eval(xfull: torch.Tensor) -> dict[str, float]:
        y, x = xfull[:, 0], xfull[:, 1:]
        pred = func.functional_call(model, params, (x,))
        preds_cpu, Sigma_param = prepare_layer_inputs(pred, Sigma_override=Sigma_cpu)
        (w_opt,) = portfolio_layer(preds_cpu, Sigma_param)
        return {
            "ret_port": float(torch.dot(w_opt, y.to(dtype=torch.double, device="cpu")).item()),
            "w_sum":    float(torch.sum(w_opt).item()),
            "w_min":    float(torch.min(w_opt).item()),
            "w_max":    float(torch.max(w_opt).item()),
            "w_nnz":    int((w_opt > 1e-8).sum().item()),
        }


    return G, gradG, portfolio_eval

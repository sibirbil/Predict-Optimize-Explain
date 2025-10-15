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
    feature_anchor: torch.Tensor | None = None,
    feature_reg: float = 0.0,
    ):
    A, F = x0.shape         # A: number of assets in portfolio, F: number of features per asset
    portfolio_layer = build_portfolio_layer(A, lambda_)
    benchmark_target = torch.as_tensor(float(benchmark_return), dtype=torch.double)

    if isinstance(Sigma, torch.Tensor):
        Sigma_cpu = Sigma.detach().cpu().double()
    else:
        Sigma_cpu = torch.from_numpy(np.asarray(Sigma)).to(dtype=torch.double)

    def G(xfull: torch.Tensor) -> torch.Tensor:
        # xfull is expected to be shape (A, 1+F) where [:,0]=returns, [:,1:]=features
        # model expects only features (A, F)
        y, x = xfull[:, 0], xfull[:, 1:]
        pred = func.functional_call(model, params, x)     # (A,)
        preds_cpu, Sigma_param = prepare_layer_inputs(pred, Sigma_override=Sigma_cpu)
        (w_opt,) = portfolio_layer(preds_cpu, Sigma_param)
        # scipy.stats.entropy(w_opt.detach().cpu().numpy()+1e-8, np.ones(A)/A) # encourage diversification
        # benchmark'a yakın olsun ve bütün weight'i aynı yere koymasın
        portfolio_ret = torch.dot(w_opt, preds_cpu)
        loss = (portfolio_ret - benchmark_target) ** 2

        # feature-anchor regularization: keep generated features near anchor (x0)
        if feature_reg and feature_anchor is not None:
            # feature_anchor expected shape (A, F)
            # penalize squared L2 distance between x and anchor (per-asset features)
            feat_diff = x.double() - feature_anchor.to(dtype=torch.double)
            loss = loss + float(feature_reg) * torch.sum(feat_diff ** 2)

        # centroid regularization: keep features near nearest centroid(s)
        if centroid_weight and centroids is not None and centroids.numel() > 0:
            # centroids: (K, F)
            # compute squared distance of each asset feature vector to nearest centroid
            # x: (A, F)
            x_d = x.double()
            # distances: (A, K)
            dists = torch.cdist(x_d, centroids.to(dtype=torch.double)) ** 2
            min_dists, _ = torch.min(dists, dim=1)
            loss = loss + float(centroid_weight) * torch.sum(min_dists)

        return loss

    @torch.no_grad()
    def portfolio_eval(xfull: torch.Tensor) -> dict[str, float]:
        y, x = xfull[:, 0], xfull[:, 1:]
        pred = func.functional_call(model, params, x)
        preds_cpu, Sigma_param = prepare_layer_inputs(pred, Sigma_override=Sigma_cpu)
        (w_opt,) = portfolio_layer(preds_cpu, Sigma_param)
        return {
            "ret_port": float(torch.dot(w_opt, y.to(dtype=torch.double, device="cpu")).item()),
            "w_sum":    float(torch.sum(w_opt).item()),
            "w_min":    float(torch.min(w_opt).item()),
            "w_max":    float(torch.max(w_opt).item()),
            "w_nnz":    int((w_opt > 1e-8).sum().item()),
        }

    def gradG(xfull: torch.Tensor) -> torch.Tensor:
        xfull = xfull.clone().requires_grad_(True)
        val = G(xfull)
        (grad,) = torch.autograd.grad(val, xfull)
        return grad

    return G, gradG, portfolio_eval

# original X'e yakın olsun
# langevin 

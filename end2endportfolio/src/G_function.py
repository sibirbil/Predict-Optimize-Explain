#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:52:14 2025

@author: batuhanatas
"""

import cvxpy as cp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
from typing import Callable

from torch import func as func

def G_function_benchmark(
    model               : nn.Module,
    params              : dict[str, torch.Tensor],
    x0                  : torch.Tensor,     #of shape (A, F) for A in the number of assets
    benchmark_return    : float,
    Sigma,
    lambda_
    ):
    A, F = x0.shape         # A: number of assets in portfolio, F: number of features per asset
    w = cp.Variable(A)  
    b = cp.Parameter(A) # predicted returns 
    objective_fn = w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma)
    objective = cp.Maximize(objective_fn)
    constraints = [
        cp.sum(w) == 1,  
        w >= 0           
    ]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[w])

    def G(xfull):
        y, x = xfull[:,0], xfull[:, 1:]
        returns = func.functional_call(model, params, x)
        solution, = cvxpylayer(returns)
        return (torch.dot(solution, y) - benchmark_return)**2
    
    def gradG(xfull:torch.Tensor):
        xfull.requires_grad_(True)
        value = G(xfull)
        value.backward()
        return xfull.grad

    return G, gradG
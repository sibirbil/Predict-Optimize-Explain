#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:47:21 2025

@author: batuhanatas
"""

# src/model.py

import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dim=46, hidden_dims=[32, 16, 8], dropout=0.5):
        super(FNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

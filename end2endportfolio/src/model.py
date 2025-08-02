#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:27:01 2025

@author: batuhanatas
"""

import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    Feedforward Neural Network for predicting expected returns.

    Args:
        input_dim (int): Number of input features per asset (default: 46).
        hidden_dims (List[int]): List specifying the number of neurons in each hidden layer.
        dropout (float): Dropout rate for regularization.

    Example:
        model = FNN(input_dim=46, hidden_dims=[32, 16, 8], dropout=0.5)
    """
    def __init__(self, input_dim: int = 46, hidden_dims: list = [32, 16, 8], dropout: float = 0.5):
        super(FNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))  # Final layer outputs scalar return per asset

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F), where B is number of assets, F is number of features.

        Returns:
            torch.Tensor: Predicted returns of shape (B,)
        """
        return self.net(x).squeeze(-1)

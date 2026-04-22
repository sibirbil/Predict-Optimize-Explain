from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class E2ERegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16, 8),
        dropout: float = 0.5,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if len(hidden_dims) != 3:
            raise ValueError("E2ERegressor currently expects exactly three hidden layers.")

        self.layer1 = self._build_block(input_dim, hidden_dims[0], dropout, batch_norm)
        self.layer2 = self._build_block(hidden_dims[0], hidden_dims[1], dropout, batch_norm)
        self.layer3 = self._build_block(hidden_dims[1], hidden_dims[2], dropout, batch_norm)
        self.output = nn.Linear(hidden_dims[2], 1)

    def _build_block(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        batch_norm: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(inputs)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        return self.output(hidden).squeeze(-1)

import torch
from torch import nn

Tensor = torch.Tensor

def check_traj(
    model   : nn.Module,    # inputs F features and outputs 1 return.
    traj    : Tensor,        # trajectory coming out of Langevin, shape (N, B, 1 + F)
):
    model.eval()
    x_traj, y_traj = traj[:, :, 1:], traj[:, :, 0]
    pred_rets = model(x_traj.flatten(end_dim = 1))     # input NxBxF, output  NxB, 
    return nn.functional.mse_loss(pred_rets, y_traj.flatten(end_dim=1))

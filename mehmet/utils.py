import numpy as np
import torch

def make_psd_np(Sigma: np.ndarray, eps=1e-10) -> np.ndarray:
    """Project symmetric matrix to PSD by eigenvalue clipping."""
    S = 0.5 * (Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T

def make_psd_torch(Sigma: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """Project symmetric matrix to PSD by eigenvalue clipping."""
    S = 0.5 * (Sigma + Sigma.T)
    vals, vecs = torch.linalg.eigh(S)
    vals = torch.maximum(vals, eps)
    return (vecs * vals) @ vecs.T

######################
## SCHEDULERS ET AL ##
######################

def as_scheduler(value):
    """
    Turns scalar into constant step-size function
    """
    if callable(value):
        return value
    return lambda step: value


def power_decay(
    init_lr : float,         # the starting learning rate 
    alpha   : float,         # decay rate exponent
    offset  : float  = 1.,   # in case step count starts from 0
    rate    : int | float = 100   # how many steps  
    ):
    """
    Returns a scheduler which decays by 1/(step/rate + 1)^alpha.
    The rate determines how many steps it takes to 
    """
    def schedule(step: int)-> float:
        return init_lr/ ((step/rate + offset)**alpha)
    
    return schedule

def sqrt_decay(init_lr):
    return power_decay(init_lr, 1/2)

def harmonic_decay(init_lr):
    return power_decay(init_lr, 1)


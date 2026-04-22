import torch
from dataclasses import dataclass, field
from pathlib import Path
from src.modules.pao_model_defs import PAOPortfolioModel

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from math import sqrt

import pandas as pd


import numpy as np
Tensor = torch.Tensor

_VAR_ARTIFACTS_ROOT = Path(__file__).resolve().parents[2] / "VAR1Regularizer" / "artifacts"
VAR_DIR = _VAR_ARTIFACTS_ROOT / "var1_standardized"
if not VAR_DIR.exists():
    VAR_DIR = _VAR_ARTIFACTS_ROOT / "var1"
A         = torch.tensor(np.load(VAR_DIR / "transition_A.npy"),                dtype=torch.float32)
c         = torch.tensor(np.load(VAR_DIR / "intercept_c.npy"),                 dtype=torch.float32)
Sigma     = torch.tensor(np.load(VAR_DIR / "innovation_covariance_Sigma.npy"), dtype=torch.float32)
Sigma_inv = torch.tensor(np.load(VAR_DIR / "innovation_covariance_inv.npy"),   dtype=torch.float32)


def mahalonobis_reg(
    A: torch.Tensor,
    c: torch.Tensor,
    Sigma_inv: torch.Tensor,
    anchor: torch.Tensor,
):
    anchor = torch.as_tensor(anchor, dtype=torch.float32)

    def regularizer(x: Tensor):
        diff = x - (A @ anchor + c)
        return diff @ (Sigma_inv @ diff)

    return regularizer


def _build_interactions(C_t, m: torch.Tensor) -> torch.Tensor:
    if hasattr(C_t, "build") and callable(getattr(C_t, "build")):
        return C_t.build(m)
    one = torch.ones((1,), dtype=m.dtype, device=m.device)
    mtilde = torch.cat([one, m])
    return (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)


@dataclass
class AllocationPipeline():
    model : PAOPortfolioModel #we need this so that the model has a predictor method as well as _transform_mu methods
    Sigma : torch.Tensor
    kappa : float = field(init= False)
    lambd : float = field(init= False)
    omega_mode : str = field(init = False)#diagSigma or identity
    Omega  : torch.Tensor = field(init=False)
    N  : int = field(init=False)
    problem : cp.Problem = field(init = False)

    def __post_init__(self):
        self.N = self.Sigma.shape[1]
        self.omega_mode = self.model.omega_mode
        self.kappa = self.model.kappa
        self.lambd = self.model.lambd

        diagS = torch.maximum(torch.diag(self.Sigma), torch.tensor(0.0))
        vol = torch.sqrt(torch.maximum(diagS, torch.tensor(1e-12)))
        
        if self.omega_mode=="diagSigma":
            self.Omega = torch.diag(vol)
        elif self.omega_mode=="identity":
            self.Omega = torch.eye(self.N)
        else:
            raise ValueError(f"Unknown omega_mode = {self.omega_mode}")
        
        self.problem = self._robust_optimization_problem()
        
    def _robust_optimization_problem(self):
        b = cp.Parameter(self.N)
        w = cp.Variable(self.N)
        obj = cp.Maximize(b @ w - self.kappa *cp.norm(self.Omega @ w, 2) - (self.lambd/2)*cp.quad_form(w, self.Sigma))
        cons = [cp.sum(w) ==1 , w>=0]
        problem = cp.Problem(obj, cons)
        assert problem.is_dpp(), "The optimization problem is not dpp"
        return problem




def G_function(
    pi  : AllocationPipeline,
    C_t   :torch.Tensor, #Firm characteristics
    rets_t : torch.Tensor,
    score_function : str = "PortfolioReturn", #PortfolioReturn or Sharpe or Benchmark or Entropy
    anchor      : torch.Tensor = torch.zeros((9)),
    l2reg       : float = 0.,
    macro_mean  : torch.Tensor | None = None,
    macro_std   : torch.Tensor | None = None,
):
    
    pi.model.eval()
    cvxpylayer = CvxpyLayer(pi.problem, parameters=pi.problem.parameters(), variables=pi.problem.variables())
    # scale so that the deviation in each coordinate can be measured in the same scale 
    # scale = torch.tensor([2.5856, 3.7924, 1.5339, 0.1413, 0.2326, 0.1126, 0.0372, 0.0844, 0.0414])
    anchor_in = (anchor - macro_mean) / macro_std if macro_mean is not None else anchor
    reg_fn = mahalonobis_reg(A, c, Sigma_inv, anchor_in)

    if score_function == "PortfolioReturn":
        def G(m:torch.Tensor):
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds) #standardize predictions with zscores (or model.mu_transform)
            w_star, = cvxpylayer(preds_std)
            #reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return - (w_star @ rets_t) + reg

    elif score_function == "Sharpe":
        def G(m:torch.Tensor):
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            returns = w_star @ rets_t
            vol = torch.sqrt(w_star @ pi.Sigma @ w_star)*sqrt(12)
            # reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return - returns/vol + reg

    elif isinstance(score_function,float): #how we detect the benchmark
        def G(m:torch.Tensor):
            b = score_function
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            returns = w_star @ rets_t
            #reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return (100*(returns - b))**2 + reg

    elif score_function=="Entropy": #encourages diverse networks
        def G(m: torch.Tensor):
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            # reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return -robust_entropy(w_star) + reg
        
    else:
        raise ValueError("score_function should be one of PortfolioReturn/Sharpe/Entropy or a float")

    def gradG(m:torch.Tensor):
        m.requires_grad_(True)
        value = G(m)
        value.backward()
        return m.grad
        
    return G, gradG


def G_contrast_function(
    pi1 : AllocationPipeline,
    pi2 : AllocationPipeline,
    C_t : torch.Tensor,
    rets_t :torch.Tensor,
    contrast_function : str = "distinct_return", #distinct_return or distinct_Sharpe or similar_return-distinct_Sharpe
    anchor      : torch.Tensor = torch.zeros((9)),
    l2reg       : float = 0.,
    gamma       : float = 0.5, # only for distinct_return; controls the strength of the contrast; higher gamma → stronger contrast
    macro_mean  : torch.Tensor | None = None,
    macro_std   : torch.Tensor | None = None,
    reg_fn      : "callable | None" = None,  # override regularizer; None → VAR(1) mahalonobis
):
    anchor_in = (anchor - macro_mean) / macro_std if macro_mean is not None else anchor
    if reg_fn is None:
        reg_fn = mahalonobis_reg(A, c, Sigma_inv, anchor_in)

    pi1.model.eval()
    pi2.model.eval()
    cvxpylayer1 = CvxpyLayer(pi1.problem, parameters=pi1.problem.parameters(), variables=pi1.problem.variables())
    cvxpylayer2 = CvxpyLayer(pi2.problem, parameters=pi2.problem.parameters(), variables=pi2.problem.variables())


    if contrast_function == "distinct_return":
        def G(m:torch.Tensor):
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            w1_star, = cvxpylayer1(preds1)
            w2_star, = cvxpylayer2(preds2)
            pret1 = w1_star @ rets_t
            pret2 = w2_star @ rets_t
            #reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return torch.exp(-gamma * pow((100*(pret1 - pret2)), 2)).div(gamma) + reg

    elif contrast_function == "similar_return-distinct_Sharpe":
        def G(m:torch.Tensor):
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            w1_star, = cvxpylayer1(preds1)
            w2_star, = cvxpylayer2(preds2)
            pret1 = w1_star @ rets_t
            pret2 = w2_star @ rets_t
            sharpe1 = sqrt(12)*pret1/torch.sqrt(w1_star@ pi1.Sigma @ w1_star)
            sharpe2 = sqrt(12)*pret2/torch.sqrt(w2_star @pi2.Sigma @ w2_star)
            #reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return (100*(pret1 - pret2))**2 + torch.exp(-gamma*(sharpe1 - sharpe2)**2).div(gamma) + reg

    elif contrast_function == "distinct_Sharpe":
        def G(m:torch.Tensor):
            m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
            interactions = _build_interactions(C_t, m_in)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            w_star1, = cvxpylayer1(preds1)
            w_star2, = cvxpylayer2(preds2)
            pret1 = w_star1 @ rets_t
            pret2 = w_star2 @ rets_t
            vol1 = w_star1 @ pi1.Sigma @ w_star1
            vol2 = w_star2 @ pi2.Sigma @ w_star2
            sharpe1 = sqrt(12)*pret1/torch.sqrt(vol1)
            sharpe2 = sqrt(12)*pret2/torch.sqrt(vol2)
            #reg = l2reg*((m - anchor).div(scale).square().sum())
            reg = l2reg*reg_fn(m_in)
            return torch.exp( -gamma* (pow(10*(sharpe1 - sharpe2), 2))).div(gamma) + reg
        

    def gradG(m:torch.Tensor):
        m.requires_grad_(True)
        value = G(m)
        value.backward()
        return m.grad
        
    return G, gradG



##########################################################
## EVALUATIONS OF THE TRAJECTORIES AND Macro Conditions ##
##########################################################

def evaluate(
    m       : torch.Tensor, #macro conditions
    C_t     : torch.Tensor,  # firm characteristics from time t
    rets_t  : torch.Tensor, # realized returns
    Sigma_t : torch.Tensor, # the covariance of the assets looking back with EWMA from time t
    pi  : AllocationPipeline,
    macro_mean : torch.Tensor | None = None,
    macro_std  : torch.Tensor | None = None,
):
    m_in = (m - macro_mean) / macro_std if macro_mean is not None else m
    interactions = _build_interactions(C_t, m_in)
    preds_raw = pi.model.predictor(interactions)
    cvxpylayer = CvxpyLayer(pi.problem, parameters=pi.problem.parameters(), variables=pi.problem.variables())
    preds_standardized = pi.model._transform_mu(preds_raw)
    w_star, = cvxpylayer(preds_standardized)
    portfolio_return = (w_star @ rets_t)
    portfolio_volatility = torch.sqrt(w_star @ Sigma_t @ w_star)
    portfolio_sharpe = portfolio_return*sqrt(12)/portfolio_volatility
    portfolio_entropy = robust_entropy(w_star)
    return torch.tensor([portfolio_return, portfolio_volatility, portfolio_sharpe, portfolio_entropy]), w_star

def robust_entropy(probs):
    """Cleanest implementation with explicit 0*log(0)=0"""
    # Only compute for non-zero probabilities
    mask = probs > 0
    log_vals = torch.where(mask, torch.log(probs), torch.tensor(0.0))
    entropy_vals = -probs * log_vals
    return torch.sum(entropy_vals)

def traj_outputs(
    m_traj  : torch.Tensor,
    C_t     : torch.Tensor,
    rets_t  : torch.Tensor,
    Sigma_t : torch.Tensor,
    pi      : AllocationPipeline,
    permnos : list,
    macro_mean : torch.Tensor | None = None,
    macro_std  : torch.Tensor | None = None,
):
    wstars = []
    reslts = []
    for m in m_traj:
        reslt , wstar = evaluate(m, C_t, rets_t, Sigma_t, pi, macro_mean, macro_std) 
        wstars.append(wstar)
        reslts.append(reslt)
    
    reslts = torch.vstack(reslts)
    wstars = torch.vstack(wstars)
    
    res_columns = ["excess_ret", "vol", "Sharpe", "entropy"]
    res_df = pd.DataFrame(reslts.detach().numpy(), columns = res_columns)
    w_df = pd.DataFrame(wstars.detach().numpy(), columns = permnos)
    return res_df, w_df
    

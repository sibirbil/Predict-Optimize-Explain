import torch
from dataclasses import dataclass, field
from src.modules.pao_model_defs import PAOPortfolioModel

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from math import sqrt

import pandas as pd

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
    l2reg       : float = 0.
):
    
    pi.model.eval()
    cvxpylayer = CvxpyLayer(pi.problem, parameters=pi.problem.parameters(), variables=pi.problem.variables())
    # scale so that the deviation in each coordinate can be measured in the same scale 
    scale = torch.tensor([2.5856, 3.7924, 1.5339, 0.1413, 0.2326, 0.1126, 0.0372, 0.0844, 0.0414])

    if score_function == "PortfolioReturn":
        def G(m:torch.Tensor):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds) #standardize predictions with zscores (or model.mu_transform)            
            w_star, = cvxpylayer(preds_std)
            reg = l2reg*((m - anchor).div(scale).square().sum())
            return - (w_star @ rets_t) + reg
        
    elif score_function == "Sharpe":
        def G(m:torch.Tensor):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            returns = w_star @ rets_t
            vol = torch.sqrt(w_star @ pi.Sigma @ w_star)*sqrt(12)
            reg = l2reg*((m - anchor).div(scale).square().sum())
            return - returns/vol + reg
        
    elif isinstance(score_function,float): #how we detect the benchmark
        def G(m:torch.Tensor):
            b = score_function
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            returns = w_star @ rets_t
            reg = l2reg*((m - anchor).div(scale).square().sum())
            return (100*(returns - b))**2 + reg
        
    elif score_function=="Entropy": #encourages diverse networks
        def G(m: torch.Tensor):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            reg = l2reg*((m - anchor).div(scale).square().sum())
            return -robust_entropy(w_star) + reg
        
    else:
        ValueError("score_function should be one of PortfolioReturn/Sharpe or a float")

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
    l2reg       : float = 0.
):
    
    scale = torch.tensor([2.5856, 3.7924, 1.5339, 0.1413, 0.2326, 0.1126, 0.0372, 0.0844, 0.0414])

    pi1.model.eval()
    pi2.model.eval()
    cvxpylayer1 = CvxpyLayer(pi1.problem, parameters=pi1.problem.parameters(), variables=pi1.problem.variables())
    cvxpylayer2 = CvxpyLayer(pi2.problem, parameters=pi2.problem.parameters(), variables=pi2.problem.variables())


    if contrast_function == "distinct_return":
        def G(m:torch.Tensor):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            preds1_std = pi1.model._transform_mu(preds1)
            preds2_std = pi2.model._transform_mu(preds2)
            w1_star, = cvxpylayer1(preds1_std)
            w2_star, = cvxpylayer2(preds2_std)
            reg = l2reg*((m - anchor).div(scale).square().sum())
            return torch.exp( - (w1_star @ rets_t - w2_star @ rets_t)**2) + reg

    elif contrast_function == "similar_return-distinct_Sharpe":
        def G(m:torch.Tensor):
            mtilde = torch.cat([torch.tensor([1]),m])   #prepend 1 to the macro variables
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            preds1_std = pi1.model._transform_mu(preds1)
            preds2_std = pi2.model._transform_mu(preds2)
            w1_star, = cvxpylayer1(preds1_std)
            w2_star, = cvxpylayer2(preds2_std)
            pret1 = w1_star @ rets_t
            pret2 = w2_star @ rets_t
            sharpe1 = sqrt(12)*pret1/torch.sqrt(w1_star@ pi1.Sigma @ w1_star) 
            sharpe2 = sqrt(12)*pret2/torch.sqrt(w2_star @pi2.Sigma @ w2_star)
            reg = l2reg*((m - anchor).div(scale).square().sum())
            alpha = 0.1
            return (100*(pret1 - pret2))**2 + torch.exp(-alpha*(sharpe1 - sharpe2)**2).div(alpha) + reg
        
    elif contrast_function == "distinct_Sharpe":
        def G(m:torch.Tensor):
            mtilde = torch.cat([1,m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            preds1_std = pi1.model._transform_mu(preds1)
            preds2_std = pi2.model._transform_mu(preds2)
            w_star1, = cvxpylayer1(preds1_std)
            w_star2, = cvxpylayer2(preds2_std)
            pret1 = w_star1 @ rets_t
            pret2 = w_star2 @ rets_t
            vol1 = w_star1 @ pi1.Sigma @ w_star1
            vol2 = w_star2 @ pi2.Sigma @ w_star2
            sharpe1 = pret1/torch.sqrt(vol1)
            sharpe2 = pret2/torch.sqrt(vol2)
            reg = l2reg*((m - anchor).div(scale).square().sum())
            return torch.exp(-(sharpe1 - sharpe2)**2) + reg
        

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
    pi  : AllocationPipeline
):
    mtilde = torch.cat([torch.tensor([1]),m])
    interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
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
    permnos : list
):
    wstars = []
    reslts = []
    for m in m_traj:
        reslt , wstar = evaluate(m, C_t, rets_t, Sigma_t, pi) 
        wstars.append(wstar)
        reslts.append(reslt)
    
    reslts = torch.vstack(reslts)
    wstars = torch.vstack(wstars)
    
    res_columns = ["excess_ret", "vol", "Sharpe", "entropy"]
    res_df = pd.DataFrame(reslts.detach().numpy(), columns = res_columns)
    w_df = pd.DataFrame(wstars.detach().numpy(), columns = permnos)
    return res_df, w_df
    
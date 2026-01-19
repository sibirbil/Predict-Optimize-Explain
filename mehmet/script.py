from mehmet.dataloaders import DataStorageEngine
from mehmet.dataloaders import TRAIN_END, VAL_END, strict_metadata_alignment

import torch
import pandas as pd

from dataclasses import dataclass, field
from mehmet.e2e_model_defs import E2EPortfolioModel, load_e2e_model_from_run

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

import mehmet.sigma as msig
from end2endportfolio.src import langevin
from mehmet import utils

READY_DATA_DIR = "./Data/final_data"

@dataclass
class AllocationPipeline():
    model : E2EPortfolioModel #we need this so that the model has a predictor method as well as _transform_mu methods
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
    score_function : str = "PortfolioReturn", #PortfolioReturn or Sharpe or Benchmark
    anchor      : torch.Tensor = torch.zeros((9)),
    l2reg       : float = 0.
):
    
    pi.model.eval()
    cvxpylayer = CvxpyLayer(pi.problem, parameters=pi.problem.parameters(), variables=pi.problem.variables())

    if score_function == "PortfolioReturn":
        def G(m:torch.Tensor):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds) #standardize predictions with zscores (or model.mu_transform)            
            w_star, = cvxpylayer(preds_std)
            reg = l2reg*((m - anchor).square().sum())
            return - (w_star @ rets_t) + reg
        
    elif score_function == "Sharpe":
        def G(m):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            returns = w_star @ rets_t
            vol = torch.sqrt(w_star @ pi.Sigma @ w_star)
            reg = l2reg*((m - anchor).square().sum())
            return - returns/vol + reg
        
    elif isinstance(score_function,float): #how we detect the benchmark
        def G(m):
            b = score_function
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds = pi.model.predictor(interactions)
            preds_std = pi.model._transform_mu(preds)
            w_star, = cvxpylayer(preds_std)
            returns = w_star @ rets_t
            reg = l2reg*((m - anchor).square().sum())
            return (1/2)*(returns - b)**2 + reg

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
    
    pi1.model.eval()
    pi2.model.eval()
    cvxpylayer1 = CvxpyLayer(pi1.problem, parameters=pi1.problem.parameters(), variables=pi1.problem.variables())
    cvxpylayer2 = CvxpyLayer(pi2.problem, parameters=pi2.problem.parameters(), variables=pi2.problem.variables())


    if contrast_function == "distinct_return":
        def G(m):
            mtilde = torch.cat([torch.tensor([1]),m])
            interactions = (C_t[:, None, :] * mtilde[None, :, None]).flatten(1)
            preds1 = pi1.model.predictor(interactions)
            preds2 = pi2.model.predictor(interactions)
            preds1_std = pi1.model._transform_mu(preds1)
            preds2_std = pi2.model._transform_mu(preds2)
            w1_star, = cvxpylayer1(preds1_std)
            w2_star, = cvxpylayer2(preds2_std)
            reg = l2reg*((m - anchor).square().sum())
            return torch.exp( - (w1_star @ rets_t - w2_star @ rets_t)**2) + reg

    elif contrast_function == "similar_return-distinct_Sharpe":
        def G(m):
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
            sharpe1 = pret1/torch.sqrt(w1_star@ pi1.Sigma @ w1_star) 
            sharpe2 = pret2/torch.sqrt(w2_star @pi2.Sigma @ w2_star)
            reg = l2reg*((m - anchor).square().sum())
            return (pret1 - pret2)**2 + 0.01*torch.exp(-(sharpe1 - sharpe2)**2) + reg
        
    elif contrast_function == "distinct_Sharpe":
        def G(m):
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
            reg = l2reg*((m - anchor).square().sum())
            return torch.exp(-(sharpe1 - sharpe2)**2) + reg
        

    def gradG(m:torch.Tensor):
        m.requires_grad_(True)
        value = G(m)
        value.backward()
        return m.grad
        
    return G, gradG


# --- 2. Execution & Inspection ---

# A. Load
# Change Path
storage = DataStorageEngine(storage_dir="./Data/final_data", load_train=False)
data = storage.load_dataset()

meta_train, meta_val, meta_test = strict_metadata_alignment(data['metadata'], train_end=TRAIN_END, val_end=VAL_END)


def construct_C(
    model   : E2EPortfolioModel,
    df      : pd.DataFrame, #Considered to contain all the interaction terms
    meta_df : pd.DataFrame,
    date    : int, #in yyyymm format
    K       : int, # the firm characteristics of top/bottom K predictions are given 
    bestK   : bool = True #otherwise we take the worst K
    ):
    dd = df[meta_df['yyyymm']==date] #date data
    dm = meta_df[meta_df['yyyymm']==date] #date meta, so that we capture firm ids
    dd_tensor = torch.tensor(dd.to_numpy())
    with torch.no_grad():
        predictions = model.predictor(dd_tensor)
    top3Kindices = torch.argsort(predictions, descending=bestK)[:3*K]
    firm_ids = dm.iloc[top3Kindices]['permno']
    firm_rets = dm.iloc[top3Kindices]['excess_ret']
    permnos = [int(a) for a in firm_ids]
    firm_chars = dd_tensor[top3Kindices][:, :140]
    Sigma, U, used_assets = msig.build_sigma_and_U_from_ready_data(
        ready_data_dir=READY_DATA_DIR,
        permnos=permnos,
        t=date,
        lookback=60,
        lam=0.94,
        shrink=0.10,
        ridge=1e-6,
        clip_lower=-0.99,
    )
    positions = [i for (i,val) in enumerate(firm_ids) if val in used_assets]
    C_t = firm_chars[positions[:K]]
    rets_t = firm_rets.iloc[positions].iloc[:K]
    return torch.tensor(Sigma[:K,:K]), C_t, rets_t, used_assets


run_dir1 = "./mehmet/e2e_state_dicts_bundle/runs/loss=return__gamma=5.0__kappa=1.0__omega=diagSigma__mu=zscore"
run_dir2 = "./mehmet/e2e_state_dicts_bundle/runs/loss=return__gamma=5.0__kappa=0.0__omega=identity__mu=zscore"
model1, cfg1 = load_e2e_model_from_run(run_dir=run_dir1)
model2, cfg2 = load_e2e_model_from_run(run_dir = run_dir2)

DATE = 202010
Sigma, C_t, rets_t, used_assets = construct_C(model1, data['X_test'], meta_test, DATE, 30)
macro_df = data['macro_final']
m0 = torch.tensor(macro_df[macro_df['yyyymm']==DATE].to_numpy())[0, 2:]


pi1 = AllocationPipeline(model1, Sigma)
pi2 = AllocationPipeline(model2, Sigma)

G, gradG = G_function(pi1, C_t, torch.tensor(rets_t.to_numpy()), "PortfolioReturn", anchor = m0, l2reg = 0.1)

betaG= 100.
etaG = 0.001/betaG
#etaG = utils.sqrt_decay(etaG)
N_m = 100

hypsG = G, gradG, etaG, betaG

G_cont, gradG_cont = G_contrast_function(pi1, pi2, C_t, torch.tensor(rets_t.to_numpy()), 'distinct_return')

betaG_cont = 100.
etaG_cont = 0.0001/betaG_cont
hypsG_cont= G_cont, gradG_cont, etaG_cont, betaG_cont


# Sigma, U, used_assets = msig.build_sigma_and_U_from_ready_data(
#         ready_data_dir=READY_DATA_DIR,
#         permnos=permnos,
#         t=DATE,
#         lookback=60,
#         lam=0.94,
#         shrink=0.10,
#         ridge=1e-6,
#         clip_lower=-0.99,
#     )
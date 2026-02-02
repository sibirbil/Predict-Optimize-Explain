from src.modules.dataloaders import TRAIN_END, VAL_END, strict_metadata_alignment

import torch
import pandas as pd
from typing import List, Optional
from src.modules.pao_model_defs import PAOPortfolioModel, load_fnn_from_dir, load_pao_model_from_run

from src.modules.probe_eval import AllocationPipeline, G_function, evaluate, robust_entropy, traj_outputs
from end2endportfolio.src import langevin
from src.modules.sigma import data, construct_C2
from src import utils

def print_traj(m_traj:torch.Tensor):
    df = pd.DataFrame(m_traj, columns=data["macro_final"].columns[2:])
    print(df.describe())
    return df

#EXPERIMENT 2
ASSET_SIZE = 60
DATE = 202404
KAPPA = 0.1
LAMBDA = 10.0
run_dir_pto = "./fnn_v1"

pto_model_fnn, pto_feature_cols, pto_cfg = load_fnn_from_dir(run_dir_pto)
pto_model = PAOPortfolioModel(
    input_dim=1400, n_assets=ASSET_SIZE, 
    lambd=LAMBDA, kappa=KAPPA, 
    omega_mode="identity", 
    hidden_dims=[32,16,8], 
    dropout = 0.5,
    mu_transform = "zscore", 
    mu_scale=1.0, mu_cap=1.0
)
pto_model.predictor = pto_model_fnn

meta_train, meta_val, meta_test = strict_metadata_alignment(data['metadata'], train_end=TRAIN_END, val_end=VAL_END)
Sigma, C_t, rets_t, permnos = construct_C2(data, DATE, ASSET_SIZE)

pi = AllocationPipeline(pto_model, Sigma)

macro_df :pd.DataFrame = data['macro_final']
hist_df = macro_df.drop(columns= ['yyyymm', 'Rfree'])

m0 = torch.tensor(macro_df[macro_df['yyyymm']==DATE].to_numpy())[0, 2:].detach()
a = torch.tensor((macro_df.min() - macro_df.std()).to_numpy())[2:]
b = torch.tensor((macro_df.max() + macro_df.std()).to_numpy())[2:]

results, w  = evaluate(m0, C_t, rets_t, Sigma, pi)

entropy = results[3].item()       

print(f"The PTO framework gives the following allocation on {DATE}:\n",
      w, 
      f"\nIt's return is {results[0].item():.2%} with Sharpe {results[2].item():.4f} \
and entropy is {entropy:.4f} we will try to increase it")

G, gradG = G_function(pi, C_t, rets_t, "Entropy", anchor = m0, l2reg= 0.1)

eta = 0.01
beta = 50.
hypsG = G, gradG, lambda t : (eta/beta)*utils.sqrt_decay(1.)(t), beta, a, b

def main_pto(
    n_seeds = 20, 
    lasts:Optional[List[torch.Tensor]] = None
    ):
    m_trajs_pto = []
    m_lasts_pto = []
    if lasts is not None:
        n_seeds = len(lasts)
    for i in range(n_seeds):
        if lasts is None:
            m_start = a + (b - a)*torch.rand((9,))
        else:
            m_start = lasts[i]
        print(f"evaluating trajectory {i + 1} for PTO")
        m_last, m_traj = langevin.torch_MALA_chain(m_start, hypsG, 500)
        m_lasts_pto.append(m_last)
        m_trajs_pto.append(m_traj)


    return {'date':DATE, 'gamma':LAMBDA, 'kappa':KAPPA, 'n_assets':ASSET_SIZE,
            'beta':beta,
            'm_trajs_pto':m_trajs_pto, 
            'm_lasts_pto':m_lasts_pto}

##### FOR THE PAO setting
run_dir_pao = f"./pao_state_dicts_bundle/runs/loss=utility__gamma={LAMBDA}__kappa={KAPPA}__omega=identity__mu=zscore"
pao_model, pao_cfg = load_pao_model_from_run(run_dir_pao)

pi_pao = AllocationPipeline(pao_model, Sigma)

results2, w2  = evaluate(m0, C_t, rets_t, Sigma, pi_pao)

entropy2 = results2[3].item()        

print(f"The PAO framework gives the following allocation on {DATE}:\n",
      w2, 
      f"\nIt's return is {results2[0].item():.2%} with Sharpe {results2[2].item():.4f} \
and entropy is {entropy2:.4f} we will try to increase it")

G2, gradG2 = G_function(pi_pao, C_t, rets_t, "Entropy", anchor = m0, l2reg= 0.1)

hypsG2 = G2, gradG2, lambda t : (eta/beta)*utils.sqrt_decay(1.)(t), beta, a, b

def main_pao(
    n_seeds : int = 20,
    lasts   : Optional[List[torch.Tensor]] = None
):
    m_trajs_pao = []
    m_lasts_pao = []
    if lasts is not None:
        n_seeds = len(lasts)
    for i in range(n_seeds):
        if lasts is None:
            m_start = a + (b - a)*torch.rand((9,))
        else:
            m_start = lasts[i]
        print(f"evaluating trajectory {i + 1} for PAO")
        m_last, m_traj = langevin.torch_MALA_chain(m_start, hypsG2, 500)
        m_lasts_pao.append(m_last)
        m_trajs_pao.append(m_traj)


    return {'date':DATE, 'gamma':LAMBDA, 'kappa':KAPPA, 
            'n_assets':ASSET_SIZE, 'beta':beta,
            'm_trajs_pao':m_trajs_pao, 
            'm_lasts_pao':m_lasts_pao}
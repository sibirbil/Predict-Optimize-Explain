from src.modules.dataloaders import TRAIN_END, VAL_END, strict_metadata_alignment

import torch
import pandas as pd

from typing import Optional, List
from src.modules.pao_model_defs import PAOPortfolioModel, load_pao_model_from_run, load_fnn_from_dir

from src.modules.probe_eval import AllocationPipeline, G_function, evaluate, robust_entropy, traj_outputs
from end2endportfolio.src import langevin
from src.modules.sigma import data, construct_C, construct_C2
from src.utils.helper_functions import sqrt_decay
from src.utils.plotting import pairplot
ASSET_SIZE = 60

## A FUNCTION TO TURN TENSORS TO DATAFRAMES in MACROVARS 
def print_traj(m_traj:torch.Tensor):
    df = pd.DataFrame(m_traj, columns=data["macro_final"].columns[2:])
    print(df.describe())
    return df


#EXPERIMENT 1
DATE = 202002
#DATE = 202404
KAPPA = 1.0
LAMBDA = 10.0


# download the PTO model, put it in the E2E format.
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

run_dir_pao = f"./pao_state_dicts_bundle/runs/loss=sharpe__gamma={LAMBDA}__kappa={KAPPA}__omega=diagSigma__mu=zscore"
pao_model, pao_cfg = load_pao_model_from_run(run_dir_pao)


meta_train, meta_val, meta_test = strict_metadata_alignment(data['metadata'], train_end=TRAIN_END, val_end=VAL_END)
#Sigma, C_t, rets_t, used_assets = construct_C(pao_model, data['X_test'], meta_test, DATE, ASSET_SIZE, bestK=True)
Sigma, C_t, rets_t, permnos = construct_C2(data, DATE, ASSET_SIZE)

macro_df :pd.DataFrame = data['macro_final']
hist_df = macro_df.drop(columns= ['yyyymm', 'Rfree'])
m0 = torch.tensor(macro_df[macro_df['yyyymm']==DATE].to_numpy())[0, 2:].detach()
a = torch.tensor((macro_df.min() - macro_df.std()).to_numpy())[2:]
b = torch.tensor((macro_df.max() + macro_df.std()).to_numpy())[2:]


pi_pto = AllocationPipeline(pto_model, Sigma) 
pi_pao = AllocationPipeline(pao_model, Sigma)

results_pto, w_pto  = evaluate(m0, C_t, rets_t, Sigma, pi_pto)
results_pao, w_pao  = evaluate(m0, C_t, rets_t, Sigma, pi_pao)

b_pto = results_pto[0].item() + 0.02       # benchmarks set at 2 percent hiher than
b_pao = results_pao[0].item() + 0.02       # actual return of the allocation pipeline
print(f"The PTO framework's performance on date {DATE} is {results_pto[0]:.2%} \
      and Sharpe is {results_pto[2]:.2f}  with entropy {results_pto[3]:.2f}\
        we will set benchmark at {b_pto:.2%}")
print(f"The PAO framework's performance on date {DATE} is {results_pao[0]:.2%} \
      and Sharpe is {results_pao[2]:.2f}  with entropy {results_pao[3]:.2f}\
        we will set benchmark at {b_pao:.2%}")


G_pto, gradG_pto = G_function(pi_pto, C_t, rets_t, b_pto, anchor=m0, l2reg=0.1)
G_pao, gradG_pao = G_function(pi_pao, C_t, rets_t, b_pao, anchor=m0, l2reg=0.1)

eta = 0.01
beta = 50.
hypsG_pto = G_pto, gradG_pto, lambda t: eta/beta*sqrt_decay(1.)(t), beta, a, b
hypsG_pao = G_pao, gradG_pao, lambda t: eta/beta*sqrt_decay(1.)(t), beta, a, b

def main_pto(
        n_seeds = 20, 
        lasts:Optional[List[torch.Tensor]] = None):
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
        m_last, m_traj = langevin.torch_MALA_chain(m_start, hypsG_pto, 500)
        m_lasts_pto.append(m_last)
        m_trajs_pto.append(m_traj)


    return {'date':DATE, 'gamma':LAMBDA, 'kappa':KAPPA, 
            'n_assets':ASSET_SIZE, 'beta':beta,
            'm_trajs_pto':m_trajs_pto, 
            'm_lasts_pto':m_lasts_pto}

def main_pao(
    n_seeds : int = 20,
    lasts    : Optional[List[torch.Tensor]] = None
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
        print(f"evaluating trajectory {i+1} for PAO")
        m_last, m_traj = langevin.torch_MALA_chain(m_start, hypsG_pao, 2000)
        m_lasts_pao.append(m_last)
        m_trajs_pao.append(m_traj)

        
    return {'date':DATE, 'gamma':LAMBDA, 'kappa':KAPPA, 
            'n_assets':ASSET_SIZE, 'beta':beta,
            'm_trajs_pao':m_trajs_pao,
            'm_lasts_pao':m_lasts_pao}



def analyze_traj(
    load_address    : str,
    savefig_address : str,
    pto_or_pao      : str, #'pto' or 'pao 
    small           : bool = False
):
    reslt_dict : dict = torch.load(load_address)
    if pto_or_pao == 'pto':
        pi = pi_pto
        b = b_pto
        start = 0
    elif pto_or_pao == 'pao':
        pi = pi_pao
        b = b_pao
        start = 1500
    traj_key = 'm_trajs_'+ pto_or_pao


    valid_trajs = []
    for traj in reslt_dict[traj_key]:
        ex_ret = traj_outputs(traj[start::100], C_t, rets_t, Sigma, pi, permnos)[0].mean()['excess_ret']
        if abs(ex_ret - b) <0.05:
            valid_trajs.append(traj)

    m_traj = torch.cat([traj[start::100] for traj in valid_trajs])
    m_traj_df = print_traj(m_traj)

    if small:
        cols = ['dp', 'ep', 'bm']
        pairplot(hist_df[cols], m_traj_df[cols], m0.detach()[:3], save_address=savefig_address)
    else:
        pairplot(hist_df, m_traj_df, m0.detach(), save_address=savefig_address)

    return m_traj_df


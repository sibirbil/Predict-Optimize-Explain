from src.modules.dataloaders import TRAIN_END, VAL_END, strict_metadata_alignment

import torch
import pandas as pd
from typing import Optional, List

from src.modules.pao_model_defs import PAOPortfolioModel, load_pao_model_from_run

from src.modules.probe_eval import AllocationPipeline, G_contrast_function, evaluate, robust_entropy, traj_outputs
from end2endportfolio.src import langevin
from src.modules.sigma import data, construct_C2
from src.utils.helper_functions import sqrt_decay

def print_traj(m_traj:torch.Tensor):
    df = pd.DataFrame(m_traj, columns=data["macro_final"].columns[2:])
    print(df.describe())
    return df


#EXPERIMENT 3
ASSET_SIZE = 60
DATE = 202001
LAMBDA = 10.
KAPPA = 1.0

run_dir_summer_child = "./pao_summerchild_vs_winterwolf_bundle/scenario=summer_child_no_dotcom_pre2007/runs_oneconfig"+\
f"/scenario=summer_child_no_dotcom_pre2007__topk=30__loss=utility__gamma={LAMBDA}__kappa={KAPPA}__omega=diagSigma__mu=zscore"
summer_model, summer_cfg = load_pao_model_from_run(run_dir_summer_child)

run_dir_winter_wolf = "./pao_summerchild_vs_winterwolf_bundle/scenario=winter_wolf_seen_2008_in_train/runs_oneconfig"+\
f"/scenario=winter_wolf_seen_2008_in_train__topk=30__loss=utility__gamma={LAMBDA}__kappa={KAPPA}__omega=diagSigma__mu=zscore"
winter_model, winter_cfg = load_pao_model_from_run(run_dir_winter_wolf)


meta_train, meta_val, meta_test = strict_metadata_alignment(data['metadata'], train_end=TRAIN_END, val_end=VAL_END)
Sigma, C_t, rets_t, permnos = construct_C2(data, DATE, ASSET_SIZE)

summer_pi = AllocationPipeline(summer_model, Sigma)
winter_pi = AllocationPipeline(winter_model, Sigma)

macro_df :pd.DataFrame = data['macro_final']
m0 = torch.tensor(macro_df[macro_df['yyyymm']==DATE].to_numpy())[0, 2:].detach()
a = torch.tensor((macro_df.min() - macro_df.std()).to_numpy())[2:]
b = torch.tensor((macro_df.max() + macro_df.std()).to_numpy())[2:]

results_summer, w_summer  = evaluate(m0, C_t, rets_t, Sigma, summer_pi)
results_winter, w_winter = evaluate(m0, C_t, rets_t, Sigma, winter_pi)

return_summer = results_summer[0].item()
sharpe_summer = results_summer[2].item()

return_winter = results_winter[0].item()
sharpe_winter = results_winter[2].item()

print(f"Summer Child's performance on {DATE} is {return_summer:.2%} with Sharpe ratio {sharpe_summer:.2f}")
print(f"Winter Wolf's performance on {DATE} is {return_winter:.2%} with Sharpe ratio {sharpe_winter:.2f}")


G, gradG = G_contrast_function(
    summer_pi,  winter_pi, C_t, rets_t, 
    "similar_return-distinct_Sharpe", 
    anchor = m0, l2reg=0.1
    )

eta = 0.01
beta = 50.
hypsG = G, gradG, lambda t: (eta/beta)*sqrt_decay(1.)(t), beta, a,b


def main(    
    n_seeds : int = 20,
    lasts   : Optional[List[torch.Tensor]] = None
):
    m_trajs_contrast = []
    m_lasts_contrast = []
    if lasts is not None:
        n_seeds = len(lasts)
    for i in range(n_seeds):
        if lasts is None:
            m_start = a + (b - a)*torch.rand((9,))
        else:
            m_start = lasts[i]
        print(f"evaluating trajectory {i + 1} contrasting two models")
        m_last, m_traj = langevin.torch_MALA_chain(m_start, hypsG, 500)
        m_lasts_contrast.append(m_last)
        m_trajs_contrast.append(m_traj)


    return {'date':DATE, 'gamma':LAMBDA, 'kappa':KAPPA, 
            'n_assets':ASSET_SIZE, 'beta':beta,
            'm_trajs_contrast':m_trajs_contrast, 
            'm_lasts_contrast':m_lasts_contrast}

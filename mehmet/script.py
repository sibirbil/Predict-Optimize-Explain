from mehmet.dataloaders import DataStorageEngine
from mehmet.dataloaders import TRAIN_END, VAL_END, strict_metadata_alignment

import torch
import pandas as pd

from mehmet.e2e_model_defs import E2EPortfolioModel, load_e2e_model_from_run, load_fnn_from_dir

import mehmet.sigma as msig
from mehmet.probe_eval import AllocationPipeline, G_contrast_function, G_function, evaluate, robust_entropy, traj_outputs
from end2endportfolio.src import langevin

READY_DATA_DIR = "./Data/final_data"

## A FUNCTION TO DECIDE YOUR 
def print_traj(m_traj:torch.Tensor):
    df = pd.DataFrame(m_traj, columns=data["macro_final"].columns[2:])
    print(df.describe())
    return df


# --- 2. LOAD THE DATA ---

# A. Load
# Change Path
storage = DataStorageEngine(storage_dir="./Data/final_data", load_train=False)
data = storage.load_dataset()



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
    rets_t = torch.tensor(firm_rets.iloc[positions].iloc[:K].to_numpy())
    return torch.tensor(Sigma[:K,:K], dtype = torch.float32), C_t, rets_t, used_assets[:K]


#run_dir1 = "./mehmet/e2e_state_dicts_bundle/runs/loss=return__gamma=5.0__kappa=1.0__omega=diagSigma__mu=zscore"
#run_dir2 = "./mehmet/e2e_state_dicts_bundle/runs/loss=return__gamma=5.0__kappa=0.0__omega=identity__mu=zscore"
#run_dir3 = "./mehmet/e2e_state_dicts_bundle/runs/loss=utility__gamma=20.0__kappa=1.0__omega=diagSigma__mu=zscore"
#run_dir4 = "./mehmet/e2e_state_dicts_bundle/runs/loss=utility__gamma=20.0__kappa=1.0__omega=identity__mu=zscore"

summer_dir = "./mehmet/e2e_kingslandingvswinterfell/scenario=summer_child_no_dotcom_pre2007/runs_universe_sweep/"
run_dir1 = summer_dir + "scenario=summer_child_no_dotcom_pre2007__topk=68__loss=utility" \
    + "__gamma=10.0__kappa=1.0__omega=diagSigma__mu=zscore"
winter_dir = "./mehmet/e2e_kingslandingvswinterfell/scenario=winter_wolf_seen_2008_in_train/runs_universe_sweep/"
run_dir2 = winter_dir + "scenario=winter_wolf_seen_2008_in_train__topk=68__loss=utility"\
    + "__gamma=10.0__kappa=1.0__omega=diagSigma__mu=zscore"

run_dir3 = "./mehmet/e2e_different_topks_warch/runs_universe_sweep/topk=21__loss=utility__gamma=10.0__kappa=1.0__omega=diagSigma__mu=zscore"
run_dir4 = "./mehmet/e2e_different_topks_warch/runs_universe_sweep/topk=94__loss=utility__gamma=10.0__kappa=1.0__omega=diagSigma__mu=zscore"

run_dir_pto = "./mehmet/fnn_v1"
pto_model, pto_feature_cols, pto_cfg = load_fnn_from_dir(run_dir_pto)

model1, cfg1 = load_e2e_model_from_run(run_dir = run_dir1)
model2, cfg2 = load_e2e_model_from_run(run_dir = run_dir2)
model3, cfg3 = load_e2e_model_from_run(run_dir = run_dir3)
model4, cfg4 = load_e2e_model_from_run(run_dir = run_dir4)

meta_train, meta_val, meta_test = strict_metadata_alignment(data['metadata'], train_end=TRAIN_END, val_end=VAL_END)

DATE = 201804
Sigma, C_t, rets_t, used_assets = construct_C(model1, data['X_test'], meta_test, DATE, 68, bestK=True)
macro_df :pd.DataFrame = data['macro_final']
m0 = torch.tensor(macro_df[macro_df['yyyymm']==DATE].to_numpy())[0, 2:]
a = torch.tensor((macro_df.min() - macro_df.std()).to_numpy())[2:]
b = torch.tensor((macro_df.max() + macro_df.std()).to_numpy())[2:]


pi1 = AllocationPipeline(model1, Sigma)
pi2 = AllocationPipeline(model2, Sigma)
pi3 = AllocationPipeline(model3, Sigma)
pi4 = AllocationPipeline(model4, Sigma)

G1, gradG1 = G_function(pi1, C_t, rets_t, "PortfolioReturn", anchor = m0, l2reg = 0.0)
G2, gradG2 = G_function(pi2, C_t, rets_t, "PortfolioReturn", anchor = m0, l2reg = 0.0)
betaG= 100.
etaG = 0.001/betaG
#etaG = utils.sqrt_decay(etaG)

hypsG1 = G1, gradG1, etaG, betaG,a, b
hypsG2 = G2, gradG2, etaG, betaG, a,b

G_cont, gradG_cont = G_contrast_function(pi1, pi2, C_t, rets_t, 'distinct_return')

betaG_cont = 100.
etaG_cont = 0.0001/betaG_cont
hypsG_cont= G_cont, gradG_cont, etaG_cont, betaG_cont


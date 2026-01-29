#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:01:44 2025

@author: batuhanatas
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


final_df =pd.read_csv("/content/drive/MyDrive/final_df_cleaned2.csv")
macro_df = pd.read_csv("/content/drive/MyDrive/PredictorData2024.xlsx - Monthly.csv")

# 1. Convert macro_df['yyyymm'] to datetime, then to month-end
macro_df['date'] = pd.to_datetime(macro_df['yyyymm'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)

# 2. Ensure final_df['date'] is datetime and also month-end
final_df['date'] = pd.to_datetime(final_df['date']) + pd.offsets.MonthEnd(0)

# 3. Drop yyyymm (optional)
macro_df = macro_df.drop(columns=['yyyymm'])

# 4. Merge
final_df = final_df.merge(macro_df, on='date', how='left')


# Firm characteristics Creation
final_df['absacc'] = ((final_df['ib'] - final_df['oancf']) / final_df['at']).abs()

final_df = final_df.sort_values(['PERMNO', 'date'])

# Calculate working capital components
final_df['delta_act_che'] = final_df.groupby('PERMNO')['act'].diff() - final_df.groupby('PERMNO')['che'].diff()
final_df['delta_lct_dlc'] = final_df.groupby('PERMNO')['lct'].diff() - final_df.groupby('PERMNO')['dlc'].diff()

# Compute accruals
final_df['acc'] = (final_df['delta_act_che'] - final_df['delta_lct_dlc']) / final_df['at']

final_df = final_df.sort_values(['PERMNO', 'date'])

# Compute trailing 3-month average volume (excluding current month)
final_df['vol_lag_mean'] = final_df.groupby('PERMNO')['VOL'].transform(lambda x: x.shift(1).rolling(3).mean())

# Abnormal volume proxy
final_df['aeavol'] = final_df['VOL'] / final_df['vol_lag_mean'] - 1

# Ensure datetime format
final_df['date'] = pd.to_datetime(final_df['date'])

# First appearance per PERMNO
first_appearance = final_df.groupby('PERMNO')['date'].transform('min')

# Calculate age in months
final_df['age'] = (
    (final_df['date'].dt.year - first_appearance.dt.year) * 12 +
    (final_df['date'].dt.month - first_appearance.dt.month)
)

final_df = final_df.sort_values(['PERMNO', 'date'])

# Compute 12-month lag of total assets
final_df['at_lag12'] = final_df.groupby('PERMNO')['at'].shift(12)

# Compute asset growth
final_df['agr'] = (final_df['at'] - final_df['at_lag12']) / final_df['at_lag12']

# Compute relative bid-ask spread
final_df['baspread'] = (final_df['ASK'] - final_df['BID']) / ((final_df['ASK'] + final_df['BID']) / 2)

final_df['mkt_ret'] = final_df['CRSP_SPvw']

def compute_beta(group):
    ret = group['RET']
    mkt = group['mkt_ret']
    cov = ret.rolling(window=36).cov(mkt)
    var = mkt.rolling(window=36).var()
    group['beta'] = cov / var
    return group

# Apply per firm (PERMNO)
final_df = final_df.groupby('PERMNO', group_keys=False).apply(compute_beta)


final_df['betasq'] = final_df['beta'].apply(lambda x: x**2 if pd.notnull(x) else np.nan)

# Step 1: Compute Book Equity (BE)
final_df['be'] = final_df['ceq'].fillna(0) + final_df['txditc'].fillna(0)
fallback_be = final_df['at'] - final_df['pstk'].fillna(0)

# Use fallback if 'be' is still zero or missing
final_df['be'] = final_df['be'].where(final_df['be'] > 0, fallback_be)

# Step 2: Compute Market Equity (ME)
final_df['me'] = final_df['PRC'].abs() * final_df['SHROUT']  # SHROUT is in thousands

# Step 3: Compute Book-to-Market (BM)
final_df['bm'] = final_df['be'] / final_df['me']

final_df['cash'] = final_df['che'] / final_df['at']

final_df['cashdebt'] = final_df['oancf'] / (final_df['dltt'].fillna(0) + final_df['dlc'].fillna(0))

final_df['me'] = final_df['PRC'].abs() * final_df['SHROUT']

final_df['cfp'] = final_df['oancf'] / final_df['me']

# Safe industry-adjusted cfp calculation
valid = final_df[['date', 'sic', 'cfp']].dropna()

# Compute industry median per (date, sic)
industry_medians = valid.groupby(['date', 'sic'])['cfp'].median().rename('cfp_median').reset_index()

# Merge back
final_df = final_df.merge(industry_medians, on=['date', 'sic'], how='left')

# Subtract median
final_df['cfp_ia'] = final_df['cfp'] - final_df['cfp_median']

final_df = final_df.drop(columns=['cfp_median'])

# ATO: Asset turnover
final_df['ato'] = final_df['sale'] / final_df['at']

# Sort
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag ATO by 12 months per firm
final_df['ato_lag12'] = final_df.groupby('PERMNO')['ato'].shift(12)

# chato: change in asset turnover
final_df['chato'] = final_df['ato'] - final_df['ato_lag12']

# Filter valid rows
valid = final_df[['date', 'sic', 'chato']].dropna()

# Compute industry median change in ATO
industry_medians = valid.groupby(['date', 'sic'])['chato'].median().rename('chato_median').reset_index()

# Merge into main df
final_df = final_df.merge(industry_medians, on=['date', 'sic'], how='left')

# Compute industry-adjusted change in ATO
final_df['chatoia'] = final_df['chato'] - final_df['chato_median']

# Cleanup
final_df = final_df.drop(columns=['chato_median'])


# Filter for non-missing values
valid = final_df[['date', 'sic', 'emp']].dropna()

# Compute median emp by (date, sic)
industry_medians = valid.groupby(['date', 'sic'])['emp'].median().rename('emp_median').reset_index()

# Merge back
final_df = final_df.merge(industry_medians, on=['date', 'sic'], how='left')

# Compute industry-adjusted employee count
final_df['chempia'] = final_df['emp'] - final_df['emp_median']

# Clean up
final_df = final_df.drop(columns=['emp_median'])

# Lag inventory by 12 months per firm
final_df['invt_lag12'] = final_df.groupby('PERMNO')['invt'].shift(12)

# Compute change in inventory scaled by total assets
final_df['chinv'] = (final_df['invt'] - final_df['invt_lag12']) / final_df['at']

# Sort for proper lagging
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag 12 months of common shares outstanding
final_df['csho_lag12'] = final_df.groupby('PERMNO')['csho'].shift(12)

# Compute percent change
final_df['chcsho'] = (final_df['csho'] - final_df['csho_lag12']) / final_df['csho_lag12']

# Step 1a: Compute Profit Margin
final_df['pm'] = final_df['ni'] / final_df['sale']

# Step 1b: Sort and compute lag
final_df = final_df.sort_values(['PERMNO', 'date'])
final_df['pm_lag12'] = final_df.groupby('PERMNO')['pm'].shift(12)

# Step 1c: Compute Change in PM
final_df['chpm'] = final_df['pm'] - final_df['pm_lag12']

# Drop missing chpm/sic/date rows
valid = final_df[['date', 'sic', 'chpm']].dropna()

# Industry median chpm per month
chpm_medians = valid.groupby(['date', 'sic'])['chpm'].median().rename('chpm_median').reset_index()

# Merge back
final_df = final_df.merge(chpm_medians, on=['date', 'sic'], how='left')

# Final calculation
final_df['chpmia'] = final_df['chpm'] - final_df['chpm_median']

# Clean up
final_df = final_df.drop(columns=['chpm_median'])

final_df['cogs_ratio'] = final_df['cogs'] / final_df['sale']

final_df['cp'] = final_df['gp'] / final_df['emp']

final_df['currat'] = final_df['act'] / final_df['lct']

final_df['depr'] = final_df['dp'] / final_df['at']

final_df['divi'] = (final_df['dvt'] > 0).astype(int)

final_df['divo'] = final_df['dvt'] / final_df['ni']

final_df['dy'] = final_df['dvt'] / final_df['me']

# Lag 12 months of net income
final_df['ni_lag12'] = final_df.groupby('PERMNO')['ni'].shift(12)

# Compute earnings growth rate
final_df['egr'] = (final_df['ni'] - final_df['ni_lag12']) / final_df['ni_lag12'].abs()

final_df['ep'] = final_df['ib'] / final_df['me']

final_df['gma'] = (final_df['revt'] - final_df['cogs']) / final_df['at']

# Sort first
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag 12 months of capex
final_df['capx_lag12'] = final_df.groupby('PERMNO')['capx'].shift(12)

# Compute capex growth
final_df['grcapx'] = (final_df['capx'] - final_df['capx_lag12']) / final_df['capx_lag12'].abs()

# Step 1: Construct total liabilities if missing
final_df['lt'] = final_df['at'] - final_df['ceq'].fillna(0) - final_df['pstk'].fillna(0)

# Step 2: Compute long-term NOA = (at - act) - (lt - lct)
final_df['ltnoa'] = (final_df['at'] - final_df['act']) - (final_df['lt'] - final_df['lct'])

# Step 3: Lag 12 months
final_df = final_df.sort_values(['PERMNO', 'date'])
final_df['ltnoa_lag12'] = final_df.groupby('PERMNO')['ltnoa'].shift(12)
final_df['at_lag12'] = final_df.groupby('PERMNO')['at'].shift(12)

# Step 4: Compute grltnoa
final_df['grltnoa'] = (final_df['ltnoa'] - final_df['ltnoa_lag12']) / final_df['at_lag12']

# Step 1: Compute market cap (if not already)
final_df['mktcap'] = final_df['PRC'].abs() * final_df['SHROUT']

# Step 2: Compute total industry market cap per month
industry_totals = final_df.groupby(['date', 'sic'])['mktcap'].transform('sum')

# Step 3: Compute market share
final_df['industry_share'] = final_df['mktcap'] / industry_totals

# Step 4: Square market share
final_df['industry_share_sq'] = final_df['industry_share'] ** 2

# Step 5: Compute Herfindahl index per (date, sic)
herf = final_df.groupby(['date', 'sic'])['industry_share_sq'].transform('sum')
final_df['herf'] = herf

# Sort by firm and date
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag 12-month employee count
final_df['emp_lag12'] = final_df.groupby('PERMNO')['emp'].shift(12)

# Compute net hiring
final_df['hire'] = (final_df['emp'] - final_df['emp_lag12']) / final_df['emp_lag12']

# Calculate dollar trading volume
final_df['dollar_vol'] = final_df['PRC'].abs() * final_df['VOL']

# Calculate Amihud illiquidity
final_df['ill'] = final_df['RET'].abs() / final_df['dollar_vol']

# Ensure proper sorting
final_df = final_df.sort_values(['sic', 'date'])

# Step 1: Compute industry average return per (date, sic)
final_df['ind_ret'] = final_df.groupby(['date', 'sic'])['RET'].transform('mean')

# Step 2: Shift by 2 to 12 months for each firm-industry
for lag in range(2, 13):
    final_df[f'ind_ret_lag{lag}'] = final_df.groupby('sic')['ind_ret'].shift(lag)

# Step 3: Compute industry momentum as average over those 10 months
lag_cols = [f'ind_ret_lag{lag}' for lag in range(2, 13)]
final_df['indmom'] = final_df[lag_cols].mean(axis=1)

# Sort by firm and date
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag 1 period (assumes monthly frequency, but same formula works if quarterly/annual too)
final_df['at_lag1'] = final_df.groupby('PERMNO')['at'].shift(1)

# Compute investment growth
final_df['invest'] = (final_df['at'] - final_df['at_lag1']) / final_df['at_lag1']


# Lag 12 months of sales
final_df['sale_lag12'] = final_df.groupby('PERMNO')['sale'].shift(12)

# Compute sales growth rate
final_df['lgr'] = (final_df['sale'] - final_df['sale_lag12']) / final_df['sale_lag12']

# Compute total industry sales per (date, sic)
industry_sales_total = final_df.groupby(['date', 'sic'])['sale'].transform('sum')

# Compute market share
final_df['ms'] = final_df['sale'] / industry_sales_total

# Compute industry median market cap per month
industry_median_me = final_df.groupby(['date', 'sic'])['me'].transform('median')

# Industry-adjusted market equity
final_df['mve_ia'] = final_df['me'] - industry_median_me

final_df['orgcap'] = 0.3 * final_df['xrd']

final_df['pctacc'] = (final_df['ib'] - final_df['oancf']) / final_df['sale']

final_df['ps'] = final_df['pstk'] / final_df['at']

final_df['quick'] = (final_df['act'] - final_df['invt']) / final_df['lct']

final_df['rd'] = final_df['xrd'] / final_df['at']

final_df['rd_mve'] = final_df['xrd'] / final_df['me']

# Compute invested capital (denominator)
final_df['invested_capital'] = final_df['ppent'] + final_df['act'] - final_df['che'] - final_df['lct']

# Compute ROIC
final_df['roic'] = final_df['ib'] / final_df['invested_capital']
final_df['roa'] = final_df['ib'] / final_df['at']

# Compute ROA
final_df['roa'] = final_df['ib'] / final_df['at']

# Sort first
final_df = final_df.sort_values(['PERMNO', 'date'])

# Pull ROA at annual intervals (every 12 months, past 5 years)
def compute_roavol(group):
    roa_annual = group['roa'].shift(12).rolling(window=5, step=12).apply(lambda x: np.std(x, ddof=1), raw=True)
    group['roavol'] = roa_annual
    return group

# Apply per firm
final_df = final_df.groupby('PERMNO', group_keys=False).apply(compute_roavol)

final_df['roeq'] = final_df['ib'] / final_df['ceq']

# Sort for lag
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag 12 months of revenue
final_df['revt_lag12'] = final_df.groupby('PERMNO')['revt'].shift(12)

# Compute approximate revenue surprise
final_df['rsup'] = (final_df['revt'] - final_df['revt_lag12']) / final_df['revt_lag12']

final_df['salecash'] = final_df['sale'] / final_df['che']

final_df['saleinv'] = final_df['sale'] / final_df['invt']

final_df['salerec'] = final_df['sale'] / final_df['rect']

# Define sin SIC ranges
sin_conditions = (
    final_df['sic'].between(2080, 2085) |  # Alcohol
    final_df['sic'].between(2100, 2199) |  # Tobacco
    final_df['sic'].between(7990, 7999)    # Gambling
)

# Apply indicator
final_df['sin'] = sin_conditions.astype(int)

final_df['sp'] = final_df['sale'] / final_df['me']

# Step 1: Compute log dollar volume
final_df['log_dolvol'] = np.log(final_df['PRC'].abs() * final_df['VOL'])

# Step 2: Compute 3-month rolling std deviation of log dollar volume
final_df = final_df.sort_values(['PERMNO', 'date'])
final_df['std_dolvol'] = final_df.groupby('PERMNO')['log_dolvol'].rolling(window=3).std().reset_index(level=0, drop=True)

# Step 1: Compute turnover (convert SHROUT to shares)
final_df['turnover'] = final_df['VOL'] / (final_df['SHROUT'] * 1000)

# Step 2: Compute 3-month rolling std dev of turnover
final_df = final_df.sort_values(['PERMNO', 'date'])
final_df['std_turn'] = final_df.groupby('PERMNO')['turnover'].rolling(window=3).std().reset_index(level=0, drop=True)

# 3-month rolling average of turnover
final_df = final_df.sort_values(['PERMNO', 'date'])
final_df['turn'] = final_df.groupby('PERMNO')['turnover'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Sort by firm and date
final_df = final_df.sort_values(['PERMNO', 'date'])

# Compute 1-month lagged return
final_df['mom1m'] = final_df.groupby('PERMNO')['RET'].shift(1)

# Compute 6-month momentum (lagged 1 month)
final_df['mom6m'] = (
    final_df.groupby('PERMNO')['RET']
    .transform(lambda x: x.shift(1).rolling(window=6).sum())
)

# Compute 12-month momentum (months t−2 to t−12)
final_df['mom12m'] = (
    final_df.groupby('PERMNO')['RET']
    .transform(lambda x: x.shift(2).rolling(window=11).sum())
)

# Compute 36-month momentum (excluding month t-1)
final_df['mom36m'] = (
    final_df.groupby('PERMNO')['RET']
    .transform(lambda x: x.shift(2).rolling(window=35).sum())
)

# Sort by firm and date
final_df = final_df.sort_values(['PERMNO', 'date'])

# Compute lagged mom6m (6 months ago)
final_df['mom6m_lag6'] = final_df.groupby('PERMNO')['mom6m'].shift(6)

# Compute change in momentum
final_df['chmom'] = final_df['mom6m'] - final_df['mom6m_lag6']

# Calculate log market cap
final_df['mvel1'] = np.log(np.abs(final_df['PRC']) * final_df['SHROUT'])

# Compute industry median BM per month
industry_median_bm = final_df.groupby(['date', 'sic'])['bm'].transform('median')

# Subtract to get industry-adjusted BM
final_df['bm_ia'] = final_df['bm'] - industry_median_bm

final_df['cashpr'] = final_df['che'] / final_df['PRC'].abs()

final_df['lev'] = (final_df['dltt'].fillna(0) + final_df['dlc'].fillna(0)) / final_df['at']

# Compute invested capital (denominator)
final_df['invested_capital'] = final_df['ppent'] + final_df['act'] - final_df['che'] - final_df['lct']

# Compute ROIC
final_df['roic'] = final_df['ib'] / final_df['invested_capital']

# Calculate numerator: operating income before taxes and special items
final_df['operprof'] = (
    (final_df['revt'] - final_df['cogs'] - final_df['xsga'] - final_df['xint']) / final_df['at']
)

# Lag 12 months of revenue
final_df['revt_lag12'] = final_df.groupby('PERMNO')['revt'].shift(12)

# Compute sustainable growth rate
final_df['sgr'] = (final_df['revt'] - final_df['revt_lag12']) / final_df['revt_lag12']

final_df['rd_sale'] = final_df['xrd'] / final_df['sale']

# Estimate pretax income = ib + txditc
final_df['pretax_income'] = final_df['ib'] + final_df['txditc'].fillna(0)

# Tax burden
final_df['tb'] = final_df['ni'] / final_df['pretax_income']

final_df['tang'] = final_df['ppent'] / final_df['at']

# Sort for proper lag
final_df = final_df.sort_values(['PERMNO', 'date'])

# Lag 12-month capx
final_df['capx_lag12'] = final_df.groupby('PERMNO')['capx'].shift(12)

# Compute capital investment ratio
final_df['cinvest'] = (final_df['capx'] - final_df['capx_lag12']) / final_df['capx_lag12']


# 1. Define firm characteristics (92 available in your data)
firm_chars = [
    'absacc', 'acc', 'aeavol', 'age', 'agr', 'ato', 'ato_lag12', 'baspread', 'be',
    'beta', 'betasq', 'bm', 'bm_ia', 'capx_lag12', 'cash', 'cashdebt', 'cashpr',
    'cfp', 'cfp_ia', 'chato', 'chatoia', 'chcsho', 'chempia', 'chinv', 'chmom',
    'chpm', 'chpmia', 'cinvest', 'cogs_ratio', 'cp', 'currat', 'depr', 'divi',
    'divo', 'dp', 'dy', 'egr', 'ep', 'gma', 'grcapx', 'grltnoa', 'herf', 'hire',
    'ill', 'indmom', 'invest', 'invt_lag12', 'lev', 'lgr', 'log_dolvol', 'ltnoa',
    'ltnoa_lag12', 'me', 'mktcap', 'mom12m', 'mom1m', 'mom36m', 'mom6m',
    'mom6m_lag6', 'mve_ia', 'mvel1', 'ni_lag12', 'operprof', 'orgcap', 'pctacc',
    'pm', 'pm_lag12', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale', 'revt_lag12',
    'roa', 'roavol', 'roeq', 'roic', 'rsup', 'sale_lag12', 'salecash', 'saleinv',
    'salerec', 'sgr', 'sin', 'sp', 'std_dolvol', 'std_turn', 'svar', 'tang', 'tb',
    'turn', 'turnover'
]
final_df['E12'] = pd.to_numeric(final_df['E12'], errors='coerce')
final_df['b/m'] = pd.to_numeric(final_df['b/m'], errors='coerce')
final_df['BAA'] = pd.to_numeric(final_df['BAA'], errors='coerce')
final_df['AAA'] = pd.to_numeric(final_df['AAA'], errors='coerce')
final_df['lty'] = pd.to_numeric(final_df['lty'], errors='coerce')
final_df['tbl'] = pd.to_numeric(final_df['tbl'], errors='coerce')

# Compute the rest safely
final_df['ep'] = final_df['E12']
final_df['bm'] = final_df['b/m']
final_df['tms'] = final_df['lty'] - final_df['tbl']
final_df['dfy'] = final_df['BAA'] - final_df['AAA']
# Ensure both columns are numeric
final_df['D12'] = pd.to_numeric(final_df['D12'], errors='coerce')
final_df['Index'] = pd.to_numeric(final_df['Index'], errors='coerce')

# Now compute dp
final_df['dp'] = final_df['D12'] / final_df['Index']

# 3. Define macro variables (add a constant = 1.0 for interaction)
macro_vars = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
macro_with_const = ['const'] + macro_vars
final_df['const'] = 1.0

interaction_frames = []

for c in firm_chars:
    for m in macro_with_const:
        col_name = f"{c}__x__{m}"
        interaction_frames.append(pd.DataFrame({col_name: final_df[c] * final_df[m]}))

interaction_df = pd.concat(interaction_frames, axis=1)
model_df = pd.concat([final_df[['PERMNO', 'date']], interaction_df], axis=1)

# Compute one-month-ahead excess return
final_df['target'] = final_df.groupby('PERMNO').apply(
    lambda df: df['RET'].shift(-1) - df['Rfree'].shift(-1)
).reset_index(drop=True)

model_df['target'] = final_df['target']

# Set a threshold: drop columns with more than 30% missing values (you can adjust this)
threshold = 0.8

# Calculate missing fraction for each column
missing_frac = model_df.isnull().mean()

# Keep only columns with missing rate <= threshold
columns_to_keep = missing_frac[missing_frac <= threshold].index

# Drop the rest
model_df_clean = model_df[columns_to_keep].copy()

print(f"Dropped {model_df.shape[1] - model_df_clean.shape[1]} columns with > {threshold*100:.0f}% missing values.")

# Fill NaNs using cross-sectional median at each month
model_df_filled = model_df.copy()

# Loop over columns except 'PERMNO', 'date', and 'target'
for col in model_df.columns:
    if col not in ['PERMNO', 'date', 'target']:
        model_df_filled[col] = model_df.groupby('date')[col].transform(lambda x: x.fillna(x.median()))
        
# Step 1: Find the top 200 firms with the most records
top_200_permnos = model_df_filled['PERMNO'].value_counts().nlargest(200).index

# Step 2: Filter the dataset to include only those firms
sub_df = model_df_filled[model_df_filled['PERMNO'].isin(top_200_permnos)].copy()

# Step 3 (Optional): Check how many rows you kept
print(f"Subset size: {sub_df.shape[0]} rows, {sub_df['PERMNO'].nunique()} firms")

sub_df = sub_df.dropna(subset=['target']).copy()

# Replace inf and -inf with NaN
sub_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Forward fill then backward fill across each column
sub_df.fillna(method='ffill', inplace=True)
sub_df.fillna(method='bfill', inplace=True)

sub_df['year'] = pd.to_datetime(sub_df['date']).dt.year

sub_df.to_csv('/content/drive/MyDrive/subdf_200firms.csv', index=False)





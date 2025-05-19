#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:08:14 2025

@author: batuhanatas
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


final_df =pd.read_csv("/Data/batuan_data/final_df_cleaned2.csv")
macro_df = pd.read_csv("/Data/batuhan_data/goyal-welch-a.csv")


# Filter for 1990–2020
macro_df_filtered = macro_df[macro_df['yyyy'].between(1990, 2020)].copy()

# Rename columns to be more intuitive
macro_df_renamed = macro_df_filtered.rename(columns={
    'yyyy': 'year',
    'infl': 'inflation',
    'tbill': 'tbill',
    'ltyld10': 'term',
    'ltrate': 'rate',
    'callmoney': 'short_rate',
    'aaa': 'aaa_yield',
    'baa': 'baa_yield',
    'corprate': 'corp_ret',
    'corprate.i': 'corp_ret_inf_adj',
    'sp500index': 'sp500',
    'sp500d12': 'dividend_12m',
    'sp500e12': 'earnings_12m',
    'vwm': 'vwret',
    'vwmx': 'vwexret',
    'svar': 'svar',
    'bkmk': 'book_to_market',
    'ntis': 'ntis',
    'eqis': 'eqis',
    'csp': 'csp',
    'cay': 'cay',
    'ik': 'inv_cap_ratio'
})



# Subset and normalize the macro variables (optional but common)
macro_xt = macro_df_renamed.copy()
# Final result
print("✅ macro_xt prepared with shape:", macro_xt.shape)
macro_xt.head()


# 🛠 1. Make sure date is datetime and extract year
final_df['date'] = pd.to_datetime(final_df['date'])
final_df['year'] = final_df['date'].dt.year

# 🛠 2. Clean macro_xt and ensure 'year' is int
macro_xt['year'] = macro_xt['year'].astype(int)

# 🛠 3. Merge firm panel with macro data by year
merged_df = final_df.merge(macro_xt, on='year', how='left')

# ✅ Check the result
print(f"✅ Merged dataset shape: {merged_df.shape}")
print("🧾 Columns added from macro:", list(macro_xt.columns.difference(['year'])))
merged_df.head()


# 1. Basic overview
print("🔢 Shape of merged_df:", merged_df.shape)
print("\n🧱 Columns and Data Types:")
print(merged_df.dtypes)

# 2. Nulls and missing values
print("\n🕳️ Missing Values Summary:")
missing_info = merged_df.isnull().sum()
missing_pct = 100 * missing_info / len(merged_df)
missing_df = pd.DataFrame({'missing_count': missing_info, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df.missing_count > 0].sort_values('missing_count', ascending=False)
print(missing_df)

# 3. Summary statistics for numerical columns
print("\n📊 Summary Statistics (Numeric Only):")
print(merged_df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)

# 4. Categorical column exploration
cat_cols = merged_df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    print(f"\n🔡 Unique values in '{col}':")
    print(merged_df[col].value_counts(dropna=False).head(10))

# 5. Duplicates
print("\n🧼 Duplicate Rows:", merged_df.duplicated().sum())
print("🧼 Duplicate Firm-Date Combos:", merged_df.duplicated(subset=['PERMNO', 'date']).sum())

# Convert to datetime
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

# Now you can extract periods safely
print("📆 Number of unique monthly periods:", merged_df['date'].dt.to_period("M").nunique())
print("🕰 Unique years:", merged_df['date'].dt.year.nunique())
print("🗓 Sample months:", merged_df['date'].dt.to_period("M").unique()[:5])

# 6. Date coverage and granularity
print("\n📅 Date Range:", merged_df['date'].min(), "→", merged_df['date'].max())
print("📆 Periods:", merged_df['date'].dt.to_period("M").nunique())

# 7. Firm universe
print("\n🏢 Unique firms (PERMNO):", merged_df['PERMNO'].nunique())


# Or: Encode them
merged_df['tic_code'] = merged_df['tic'].astype('category').cat.codes

# Example: Drop if not needed
merged_df.drop(columns=['tic', 'cusip', 'LINKDT', 'LINKENDDT', 'DLRET'], inplace=True, errors='ignore')

# Check for inf/-inf
print("♾️ Number of infinite values in dataset:", np.isinf(merged_df.select_dtypes(include=[np.number])).sum().sum())

# Replace inf with NaN if necessary
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)


# Unique PERMNOs and average number of records per firm
firm_counts = merged_df.groupby('PERMNO').size()
print("🏢 Number of unique firms:", merged_df['PERMNO'].nunique())
print("📊 Avg records per firm:", firm_counts.mean())
print("📊 Median records per firm:", firm_counts.median())
print("📈 Longest record count:", firm_counts.max())

# Check a sample firm over time
sample_PERMNO = merged_df['PERMNO'].value_counts().idxmax()
print(f"\n📘 Sample time series for PERMNO {sample_PERMNO}:")
print(merged_df[merged_df['PERMNO'] == sample_PERMNO][['date', 'RET']].sort_values('date').head(10))



# 8. Correlation heatmap (optional visualization)
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
corr = merged_df[num_cols].corr().round(2)

plt.figure(figsize=(15, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("📈 Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()


# Create forward 1-month return (target)
merged_df = merged_df.sort_values(['PERMNO', 'date'])
merged_df['RET_fwd1m'] = merged_df.groupby('PERMNO')['RET'].shift(-1)


# Sanity: Ensure PRC and SHROUT are floats
merged_df['PRC'] = pd.to_numeric(merged_df['PRC'], errors='coerce')
merged_df['SHROUT'] = pd.to_numeric(merged_df['SHROUT'], errors='coerce')

# Compute market equity (in $1,000 units because SHROUT is in thousands)
merged_df['me'] = merged_df['PRC'].abs() * merged_df['SHROUT']

# Optional: add log market equity
merged_df['log_me'] = np.log(merged_df['me'].replace(0, np.nan))

# Book-to-market
merged_df['bm'] = merged_df['ceq'] / merged_df['me']

# Earnings-to-price
merged_df['ep'] = merged_df['ib'] / merged_df['me']

# Cash earnings-to-price
merged_df['cash_ep'] = (merged_df['ib'] + merged_df['dp']) / merged_df['me']

# Sales-to-price
merged_df['sp'] = merged_df['sale'] / merged_df['me']

# Book equity to total assets
merged_df['be_at'] = merged_df['ceq'] / merged_df['at']

# Log size (already exists as 'log_me')


# Asset growth
merged_df['asset_growth'] = merged_df.groupby('PERMNO')['at'].pct_change()

# Capex to assets
merged_df['capx_at'] = merged_df['capx'] / merged_df['at']

# Change in PPE (investments in fixed assets)
merged_df['ppent_growth'] = merged_df.groupby('PERMNO')['ppent'].pct_change()

# Return on assets
merged_df['roa'] = merged_df['ib'] / merged_df['at']

# Return on equity
merged_df['roe'] = merged_df['ib'] / merged_df['ceq']

# Gross profitability
merged_df['gp_at'] = merged_df['gp'] / merged_df['at']

# Operating profitability
merged_df['op'] = (merged_df['revt'] - merged_df['cogs'] - merged_df['xsga']) / merged_df['at']


# Amihud illiquidity proxy
merged_df['amihud'] = merged_df['RET'].abs() / (merged_df['PRC'] * merged_df['VOL'])

# Turnover (volume relative to shares outstanding)
merged_df['turnover'] = merged_df['VOL'] / merged_df['SHROUT']


# Total leverage
merged_df['leverage'] = (merged_df['dltt'] + merged_df['dlc']) / merged_df['at']

# R&D to sales
merged_df['rd_sale'] = merged_df['xrd'] / merged_df['sale']

# R&D to assets
merged_df['rd_at'] = merged_df['xrd'] / merged_df['at']

# Intangible intensity
merged_df['intan_at'] = merged_df['intan'] / merged_df['at']


# R&D to sales
merged_df['rd_sale'] = merged_df['xrd'] / merged_df['sale']

# R&D to assets
merged_df['rd_at'] = merged_df['xrd'] / merged_df['at']

# Intangible intensity
merged_df['intan_at'] = merged_df['intan'] / merged_df['at']


merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
merged_df = merged_df.fillna(method='ffill')


# Make sure 'sic2' is integer type
merged_df['sic'] = pd.to_numeric(merged_df['sic'], errors='coerce').astype('Int64')

# Create one-hot encoded industry dummies
industry_dummies = pd.get_dummies(merged_df['sic'], prefix='ind')

# Concatenate dummies with the original dataframe
merged_df = pd.concat([merged_df, industry_dummies], axis=1)

# R&D to Market Equity
merged_df["rd_me"] = merged_df["xrd"] / merged_df["me"]

# Sales to Market Equity
merged_df["sale_me"] = merged_df["sale"] / merged_df["me"]

# Net Income to Assets
merged_df["ni_at"] = merged_df["ni"] / merged_df["at"]

# Operating Cash Flow to Assets
merged_df["ocf_at"] = merged_df["oancf"] / merged_df["at"]

# Investments to Assets
merged_df["inv_at"] = merged_df["invt"] / merged_df["at"]

# CapEx to Assets
merged_df["capx_at_2"] = merged_df["capx"] / merged_df["at"]

# CapEx to PPE
merged_df["capx_ppe"] = merged_df["capx"] / merged_df["ppent"]

# Intangible to Assets
merged_df["intan_at"] = merged_df["intan"] / merged_df["at"]

# SGA to Assets
merged_df["xsga_at"] = merged_df["xsga"] / merged_df["at"]

# Interest Expense to Assets
merged_df["xint_at"] = merged_df["xint"] / merged_df["at"]

# Dividend to Assets
merged_df["dvt_at"] = merged_df["dvt"] / merged_df["at"]

# R&D to Sales
merged_df["xrd_sale"] = merged_df["xrd"] / merged_df["sale"]

# PPE to Assets
merged_df["ppe_at"] = merged_df["ppent"] / merged_df["at"]

# Cash to Assets
merged_df["cash_at"] = merged_df["che"] / merged_df["at"]

# Sales to Assets
merged_df["sale_at"] = merged_df["sale"] / merged_df["at"]


# Leverage (Total Debt / Assets)
merged_df["leverage_2"] = (merged_df["dltt"] + merged_df["dlc"]) / merged_df["at"]

# Alternative Leverage (Debt / Debt + Equity)
merged_df["leverage_3"] = (merged_df["dltt"] + merged_df["dlc"]) / (merged_df["dltt"] + merged_df["dlc"] + merged_df["ceq"])

# Dividend Payout Ratio
merged_df["payout_ratio"] = merged_df["dvt"] / merged_df["ni"]


# First: make sure data is sorted properly
merged_df = merged_df.sort_values(['PERMNO', 'date'])

# Lagged values by 6 months
merged_df['at_lag6m'] = merged_df.groupby('PERMNO')['at'].shift(6)
merged_df['capx_lag6m'] = merged_df.groupby('PERMNO')['capx'].shift(6)
merged_df['sale_lag6m'] = merged_df.groupby('PERMNO')['sale'].shift(6)
merged_df['che_lag6m'] = merged_df.groupby('PERMNO')['che'].shift(6)
merged_df['ni_lag6m'] = merged_df.groupby('PERMNO')['ni'].shift(6)
merged_df['xrd_lag6m'] = merged_df.groupby('PERMNO')['xrd'].shift(6)

# Book Asset Growth
merged_df["growth_at"] = (merged_df["at"] - merged_df["at_lag6m"]) / merged_df["at_lag6m"]

# Capex Growth
merged_df["growth_capx"] = (merged_df["capx"] - merged_df["capx_lag6m"]) / merged_df["capx_lag6m"]

# Sales Growth
merged_df["growth_sale"] = (merged_df["sale"] - merged_df["sale_lag6m"]) / merged_df["sale_lag6m"]

# Cash Growth
merged_df["growth_che"] = (merged_df["che"] - merged_df["che_lag6m"]) / merged_df["che_lag6m"]

# Net Income Growth
merged_df["growth_ni"] = (merged_df["ni"] - merged_df["ni_lag6m"]) / merged_df["ni_lag6m"]

# R&D Growth
merged_df["growth_xrd"] = (merged_df["xrd"] - merged_df["xrd_lag6m"]) / merged_df["xrd_lag6m"]


# Log of Assets
merged_df["log_at"] = np.log(merged_df["at"])

# Log of Sales
merged_df["log_sale"] = np.log(merged_df["sale"])


# Sort by firm and date
merged_df = merged_df.sort_values(["PERMNO", "date"])

# 12-month lagged variables
merged_df["ni_lag12m"] = merged_df.groupby("PERMNO")["ni"].shift(12)
merged_df["sale_lag12m"] = merged_df.groupby("PERMNO")["sale"].shift(12)
merged_df["at_lag12m"] = merged_df.groupby("PERMNO")["at"].shift(12)
merged_df["oancf_lag12m"] = merged_df.groupby("PERMNO")["oancf"].shift(12)
merged_df["revt_lag12m"] = merged_df.groupby("PERMNO")["revt"].shift(12)


# 12-month rolling std of returns (volatility proxy)
merged_df["ret_std12m"] = merged_df.groupby("PERMNO")["RET"].rolling(window=12).std().reset_index(level=0, drop=True)

# 12-month rolling mean return (average past return)
merged_df["ret_mean12m"] = merged_df.groupby("PERMNO")["RET"].rolling(window=12).mean().reset_index(level=0, drop=True)

# 12-month rolling std of sales (sales volatility)
merged_df["sale_std12m"] = merged_df.groupby("PERMNO")["sale"].rolling(window=12).std().reset_index(level=0, drop=True)

# Rolling net income volatility
merged_df["ni_std12m"] = merged_df.groupby("PERMNO")["ni"].rolling(window=12).std().reset_index(level=0, drop=True)


# Acceleration in net income growth
merged_df["ni_growth_6m"] = (merged_df["ni"] - merged_df["ni_lag6m"]) / merged_df["ni_lag6m"]
merged_df["ni_growth_12m"] = (merged_df["ni"] - merged_df["ni_lag12m"]) / merged_df["ni_lag12m"]
merged_df["ni_acceleration"] = merged_df["ni_growth_6m"] - merged_df["ni_growth_12m"]

# Operating cash flow acceleration
merged_df["ocf_growth_6m"] = (merged_df["oancf"] - merged_df["oancf"].shift(6)) / merged_df["oancf"].shift(6)
merged_df["ocf_growth_12m"] = (merged_df["oancf"] - merged_df["oancf"].shift(12)) / merged_df["oancf"].shift(12)
merged_df["ocf_acceleration"] = merged_df["ocf_growth_6m"] - merged_df["ocf_growth_12m"]


# Ensure RET is float
merged_df['RET'] = pd.to_numeric(merged_df['RET'], errors='coerce')

# Sort for group operations
merged_df = merged_df.sort_values(by=['PERMNO', 'date'])

# Grouped object for firm-level operations
grouped = merged_df.groupby('PERMNO')

# MOMENTUM 1M: previous month return
merged_df['mom1m'] = grouped['RET'].shift(1)

# MOMENTUM 6M (t-6 to t-2): skip most recent month
merged_df['mom6m'] = (1 + grouped['RET'].shift(1)).rolling(window=5, min_periods=4).apply(np.prod, raw=True) - 1

# MOMENTUM 12M (t-12 to t-2)
merged_df['mom12m'] = (1 + grouped['RET'].shift(1)).rolling(window=11, min_periods=9).apply(np.prod, raw=True) - 1

# MOMENTUM 36M (t-36 to t-13)
merged_df['mom36m'] = (1 + grouped['RET'].shift(13)).rolling(window=24, min_periods=18).apply(np.prod, raw=True) - 1

# CHANGE IN MOMENTUM
merged_df['chmom'] = merged_df['mom12m'] - merged_df['mom36m']

# Basic info
print("🔢 Shape of merged_df:", merged_df.shape)

# Column data types
print("\n🧱 Column Types:")
print(merged_df.dtypes.sort_index())

# Missing values
print("\n🕳️ Missing Values (Top 20):")
print(merged_df.isna().sum().sort_values(ascending=False).head(20))

# Memory usage
print("\n📦 Memory Usage:")
print(merged_df.memory_usage(deep=True).sum() / 1e6, "MB")

# Unique dates and firms
print("\n📅 Date Range:", merged_df['date'].min(), "→", merged_df['date'].max())
print("📆 Unique months:", merged_df['date'].nunique())
print("🏢 Unique PERMNOs:", merged_df['PERMNO'].nunique())

# Industry distribution if industry dummies are included
industry_cols = [col for col in merged_df.columns if col.startswith("ind_")]
if industry_cols:
    print("\n🏭 Industry Dummy Distribution (Top 10):")
    print(merged_df[industry_cols].sum().sort_values(ascending=False).head(10))


# Replace inf/-inf with NaN
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Define clipping bounds for known erratic features
clip_bounds = {
    'mom1m': (-1, 1),
    'mom6m': (-1, 2),
    'mom12m': (-1, 3),
    'mom36m': (-1, 5),
    'chmom': (-5, 5),
    'amihud': (0, np.nanquantile(merged_df['amihud'], 0.99)),
    'turnover': (0, np.nanquantile(merged_df['turnover'], 0.99)),
}

for col, (low, high) in clip_bounds.items():
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].clip(lower=low, upper=high)


# % missing for each column
missing = merged_df.isna().mean().sort_values(ascending=False)
high_missing = missing[missing > 0.4]  # threshold can be adjusted

print("🔎 High-missing columns (40%+):")
print(high_missing)


cols_to_drop = ['gvkey', 'cusip', 'tic', 'datadate', 'LINKDT', 'LINKENDDT', 'LPERMNO', 'valid_me']
merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns], inplace=True)


# Drop industry dummies with very low frequency (e.g., < 1000 records)
ind_cols = [col for col in merged_df.columns if col.startswith('ind_')]
low_freq_ind = [col for col in ind_cols if merged_df[col].sum() < 1000]
merged_df.drop(columns=low_freq_ind, inplace=True)


core_features = ['RET_fwd1m', 'me', 'bm', 'ep', 'mom1m', 'mom12m']
merged_df = merged_df.dropna(subset=core_features)

firm_characteristics = [
    # Size / Market Structure
    'me', 'log_me', 'bm', 'sp', 'ep', 'cash_ep',

    # Profitability / Performance
    'roa', 'roe', 'op', 'gp_at', 'gp', 'ib', 'ni', 'ib', 'oibdp', 'dp', 'sale', 'revt', 'cogs',

    # Growth
    'asset_growth', 'ppent_growth', 'growth_ni', 'growth_sale', 'growth_at', 'growth_capx', 'growth_che', 'growth_xrd',

    # Investment / Capital Expenditures
    'capx', 'capx_at', 'capx_at_2', 'capx_ppe', 'ppe_at',

    # Cash Flow / Accruals
    'ocf_at', 'oancf', 'accruals', 'ocf_growth_6m', 'ocf_growth_12m', 'ocf_acceleration',

    # Liquidity / Leverage
    'che', 'cash_at', 'leverage', 'leverage_2', 'leverage_3', 'dltt', 'dlc', 'at', 'lct', 'act',

    # R&D and Intangibles
    'xrd', 'xrd_sale', 'rd_sale', 'rd_at', 'intan', 'intan_at',

    # Expenses
    'xsga', 'xsga_at', 'xint', 'xint_at', 'dvt', 'dvt_at',

    # Inventory, Receivables
    'invt', 'inv_at', 'rect',

    # Momentum
    'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom',

    # Volatility
    'ret_std12m',

    # Other price/valuation-based
    'prcc_f', 'sale_at', 'sale_me', 'ni_at',

    # Payouts
    'payout_ratio',

    # Misc Growth and Trends
    'ni_growth_6m', 'ni_growth_12m', 'ni_acceleration',

    # Misc
    'turnover', 'amihud', 'be_at', 'dp'
]

# Full list from the paper
expected_94_characteristics = [
    'me', 'log_me', 'bm', 'sp', 'ep', 'cash_ep',
    'roa', 'roe', 'op', 'gp_at', 'gp', 'ib', 'ni', 'oibdp', 'dp', 'sale', 'revt', 'cogs',
    'asset_growth', 'ppent_growth', 'growth_ni', 'growth_sale', 'growth_at', 'growth_capx', 'growth_che', 'growth_xrd',
    'capx', 'capx_at', 'capx_at_2', 'capx_ppe', 'ppe_at',
    'ocf_at', 'oancf', 'accruals', 'ocf_growth_6m', 'ocf_growth_12m', 'ocf_acceleration',
    'che', 'cash_at', 'leverage', 'leverage_2', 'leverage_3', 'dltt', 'dlc', 'at', 'lct', 'act',
    'xrd', 'xrd_sale', 'rd_sale', 'rd_at', 'intan', 'intan_at',
    'xsga', 'xsga_at', 'xint', 'xint_at', 'dvt', 'dvt_at',
    'invt', 'inv_at', 'rect',
    'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom',
    'ret_std12m',
    'prcc_f', 'sale_at', 'sale_me', 'ni_at',
    'payout_ratio',
    'ni_growth_6m', 'ni_growth_12m', 'ni_acceleration',
    'turnover', 'amihud', 'be_at', 'dp'
]

# Columns in your current dataframe
available_columns = merged_df.columns.tolist()

# Find missing ones
missing_characteristics = [col for col in expected_94_characteristics if col not in available_columns]

print(f"🧩 Missing Characteristics ({len(missing_characteristics)}):")
print(missing_characteristics)


# 2. Define the list of firm-level characteristics (the 82 you've validated earlier)
firm_chars = [
    'at', 'ceq', 'csho', 'sale', 'revt', 'ni', 'txditc', 'xint', 'xsga', 'xrd', 'emp', 'dltt', 'dlc',
    'act', 'lct', 'ppent', 'pstk', 'capx', 'che', 'cogs', 'oancf', 'rect', 'invt', 'dvt', 'gp', 'ib',
    'dp', 'oibdp', 'intan', 'pstkrv', 'prcc_f', 'me', 'log_me', 'bm', 'ep', 'cash_ep', 'sp', 'be_at',
    'asset_growth', 'capx_at', 'ppent_growth', 'roa', 'roe', 'gp_at', 'op', 'amihud', 'turnover',
    'leverage', 'rd_sale', 'rd_at', 'intan_at', 'ocf_at', 'accruals', 'rd_me', 'sale_me', 'ni_at',
    'inv_at', 'capx_at_2', 'capx_ppe', 'xsga_at', 'xint_at', 'dvt_at', 'xrd_sale', 'ppe_at', 'cash_at',
    'sale_at', 'leverage_2', 'leverage_3', 'payout_ratio', 'growth_at', 'growth_capx', 'growth_sale',
    'growth_che', 'growth_ni', 'growth_xrd', 'log_at', 'log_sale'
]


macro_cols = [
 'cpi',
 'gold',
 'inflation',
 'tbill',
 'term',
 'rate',
 'short_rate',
 'aaa_yield',
 'baa_yield',
 'corp_ret',
 'corp_ret_inf_adj',
 'sp500',
 'dividend_12m',
 'earnings_12m',
 'vwret',
 'vwexret',
 'svar',
 'book_to_market',
 'ntis',
 'eqis',
 'csp',
 'cay',
 'inv_cap_ratio']
firm_cols = firm_characteristics  # feature_cols should already be defined as your firm-level predictors

# Filter out any rows with missing macro or firm data
interaction_df = merged_df.copy()

# Initialize an empty DataFrame to hold interactions
interaction_features = []

# Loop over macro variables to create interaction terms
for macro_var in macro_cols:
    for firm_var in firm_cols:
        interaction_name = f'{macro_var}x{firm_var}'
        interaction_df[interaction_name] = interaction_df[macro_var] * interaction_df[firm_var]
        interaction_features.append(interaction_name)

print(f"✅ Created {len(interaction_features)} interaction features.")

# Optionally: combine with original firm + macro features if you'd like to include them too
full_feature_set = interaction_features + firm_cols + macro_cols

# 2. Nulls and missing values
print("\n🕳️ Missing Values Summary:")
missing_info = interaction_df.isnull().sum()
missing_pct = 100 * missing_info / len(interaction_df)
missing_df = pd.DataFrame({'missing_count': missing_info, 'missing_pct': missing_pct})

# Filter for columns with > 80% missing
high_missing_cols = missing_df[missing_df['missing_pct'] > 80].index

# Drop those columns from your main DataFrame
interaction_df = interaction_df.drop(columns=high_missing_cols)

# Print summary (optional)
print(f"🗑️ Dropped {len(high_missing_cols)} columns with > 80% missing values.")
print(missing_df[missing_df['missing_pct'] > 80])  # Show which were dropped


# 2. Nulls and missing values
print("\n🕳️ Missing Values Summary:")
missing_info = interaction_df.isnull().sum()
missing_pct = 100 * missing_info / len(interaction_df)
missing_df = pd.DataFrame({'missing_count': missing_info, 'missing_pct': missing_pct})

# Filter for columns with > 80% missing
high_missing_cols = missing_df[missing_df['missing_pct'] > 80].index

# Drop those columns from your main DataFrame
interaction_df = interaction_df.drop(columns=high_missing_cols)

# Print summary (optional)
print(f"🗑️ Dropped {len(high_missing_cols)} columns with > 80% missing values.")
print(missing_df[missing_df['missing_pct'] > 80])  # Show which were dropped

# Industry dummies
industry_cols = [col for col in merged_df.columns if col.startswith('ind_')]
feature_cols = firm_characteristics + industry_cols
meta_cols = ['PERMNO', 'date']
target_col = 'RET_fwd1m'
selected_cols = meta_cols + feature_cols + [target_col]
model_df = merged_df[selected_cols].dropna(subset=[target_col])  # drop if no target

target_col = 'RET_fwd1m'

model_df = interaction_df.dropna(subset=[target_col])  # drop if no targetmodel_df = interaction_df.dropna(subset=[target_col])  # drop if no target

model_df.replace([np.inf, -np.inf], np.nan, inplace=True)

model_df = model_df.fillna(method='ffill')

model_df = model_df.dropna()

feature_cols.remove('RET_fwd1m')
feature_cols.remove('date')
feature_cols.remove('PERMNO')


# Step 1: Count how many rows each PERMNO has
PERMNO_counts = interaction_df['PERMNO'].value_counts()

# Step 2: Select top 1000 PERMNOs with the most rows
top_1000_PERMNOs = PERMNO_counts.head(100).index.tolist()

model_df = model_df[interaction_df['PERMNO'].isin(top_1000_PERMNOs)].copy()

print(f"✅ Selected top 1000 PERMNOs with most data. Total rows: {len(model_df):,}")






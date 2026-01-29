import yfinance as yf
import pandas as pd
import os
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.expanduser(dir_path + '/Data')
os.makedirs(save_path, exist_ok=True)

# Define a diverse set of assets: Stocks, Bonds, ETFs, Commodities
assets = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'JNJ',
    'WMT', 'PG', 'DIS', 'MA', 'HD', 'BAC', 'XOM', 'PFE', 'KO', 'CSCO',
    'BND', 'AGG', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BNDX', 'EMB', 'MUB',
    'SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'VNQ', 'GLD', 'SLV', 'USO',
    'GLD', 'SLV', 'USO', 'DBA', 'DBC', 'UNG', 'PALL', 'CORN', 'WEAT'
]

# Download historical data 
data = yf.download(assets, start="2021-01-01", end="2024-12-1", interval="1mo")

# Ensure correct price column
if 'Adj Close' in data.columns:
    price_data = data['Adj Close']
elif 'Close' in data.columns:
    price_data = data['Close']
else:
    raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data")

# Check for missing data
print("assets with missing data:")
print(price_data.isnull().sum()[price_data.isnull().sum() > 0])

# Fill missing values using forward-fill
price_data.ffill(inplace=True)

# Drop assets that have missing values in all rows
price_data = price_data.dropna(axis=1, how="all")

# Calculate monthly returns
returns = price_data.pct_change().dropna()

# Compute the unified target return (mean of all assets)
target_return = returns.mean().mean()  # Average return across all assets

# Calculate expected returns and volatility
expected_returns = returns.mean()
volatility = returns.std()

# Save Data
price_data.to_csv(os.path.join(save_path, "financial_data.csv"))
returns.to_csv(os.path.join(save_path, "returns_data.csv"))

# Save summary data
summary_df = pd.DataFrame({
    'Expected Return': expected_returns,
    'Volatility': volatility
})

# Drop NaN values in summary
summary_df.dropna(inplace=True)
summary_df.to_csv(os.path.join(save_path, "summary_data15.csv"))

print(f"Data successfully saved in: {save_path}")
print(summary_df.head())  # Show preview of summary data

# Save target return
with open(os.path.join(save_path, "target_return.txt"), "w") as f:
    f.write(str(target_return))

print(f"✅ Financial data saved. Unified Target Return: {target_return:.6f}")

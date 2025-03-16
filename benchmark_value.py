
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

import nets, train, optax, utils

import logistic, langevin
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import pandas as pd


# Our stock prices
df = pd.read_csv("Data/financial_data.csv", index_col=0, parse_dates=True)

# Here we compute simple returns
x = df.pct_change().dropna()
returns_data = df.pct_change().dropna()

# mean of simple returns - we can also check log-mean
x_bar = x.mean().values.reshape(-1, 1)

# for dirac
theta_star = np.load("optimal_weights.npy").reshape(-1, 1)
   
n_time, n_sec = x.shape
n_col = int(n_sec*0.6)

# Select 60% of assets randomly
n_col = int(n_sec * 0.6)  # Number of selected assets
random_cols = np.random.choice(x.columns, n_col, replace=False)  # Select column names

# Convert DataFrame to JAX array (ensure it's 2D)
x = jnp.array(x.values)  # Full dataset
x = x.reshape(n_time, -1)  # Ensure it's (n_time, n_sec)

# Convert asset names to integer indices before JAX indexing
selected_indices = np.array([df.columns.get_loc(col) for col in random_cols], dtype=int)

# Convert full dataset to JAX array (Ensure it's 2D)
x = jnp.array(x)  # Remove .values, since Pandas DataFrame converts directly
x = x.reshape(n_time, -1)  # Ensure it's (n_time, n_sec)


# Debugging prints
print("Shape of x:", x.shape)  # Should be (n_time, n_sec)
print("Shape of selected_indices:", selected_indices.shape)  # Should be (n_col,)

# Correct JAX indexing
x_bm = jnp.take(x, selected_indices, axis=1)  # Select only the benchmark assets


# Benchmark value
xbar_bm = jnp.mean(x_bm, axis=0).reshape(-1, 1)

theta_bm = jnp.ones(x_bm.shape[1])/x_bm.shape[1]
theta_bm = theta_bm.reshape(-1,1)
v_bm = theta_bm.T @ xbar_bm  # scalar benchmark return


# Initialize the scaler
# scaler_X = MinMaxScaler()
# X = scaler_X.fit_transform(X)


key = random.key(42)
init_key, dropout_key, train_key, mala_key, x0_key = random.split(key, 5)


# Function G with regularization applied only to selected assets
def G_function(thetas: jax.Array, v_bm: jnp.float_, x_bm: jax.Array, x_full: jax.Array, selected_indices: jnp.array):
    def G(x):
        # Compute the first term: Portfolio return deviation
        return_term = jnp.mean(jnp.square((thetas.T @ x) - v_bm))
        
        D = np.zeros((n_col, n_sec)) 
        for i in range(n_col):
            D[i,selected_indices[i]] = 1
            
        D = jnp.array(D)

        x_filtered = jnp.dot(D, x)  

        
        regularization = jnp.sum(jnp.square(x_filtered- x_bm)) 

        # Extract only selected assets from `x`
        #x = x.reshape(1,-1)
        #x_selected = jnp.take(x, selected_indices, axis=1)
        #regularization = jnp.sum(jnp.square(x_selected - x_bm))  # Regularization

        return return_term + 10000*regularization

    grad_G = jax.grad(G)  # Compute gradient
    return G, grad_G

# Initialize Langevin MALA hyperparameters
beta = 1000.
eta = 0.01 / beta


# Compute function G and its gradient
G, gradG = G_function(theta_star, v_bm, xbar_bm, x, selected_indices)

# Set up Langevin sampling parameters
hypsG = (G, gradG, eta)
x0 = jax.random.uniform(x0_key, (int(x.shape[1]),))
state_x = (mala_key, x0)

# Run Langevin MALA chain
_, traj_x = langevin.MALA_chain(state_x, hypsG, 5000)

# Extract last 500 samples for synthetic data
synt_data_cnt = traj_x[-500:]

# Visualization of synthetic data
plotting_data = synt_data_cnt[-10:]

plt.figure(figsize=(8, 5))
for i in range(plotting_data.shape[0]):
    plt.plot(range(plotting_data.shape[1]), plotting_data[i], marker='o', label=f'Series {i+1}')
    
#plt.plot(range(plotting_data.shape[1]), df.iloc[5], marker='*', color='red', label="Real Data")

plt.xlabel("Securities")
plt.ylabel("Value")
plt.title("Line Chart for Matrix Data")
plt.legend()
plt.grid(True)
plt.show()


# Ensure synthetic_returns is available (Replace this with actual generated synthetic data)
synthetic_returns = np.array(synt_data_cnt[-100:])  # Last 10 synthetic generations



import seaborn as sns


# Convert to DataFrame
real_returns_df = pd.DataFrame(returns_data, columns=df.columns)  # Real data
synthetic_returns_df = pd.DataFrame(synthetic_returns, columns=df.columns)  # Synthetic data

num_stocks = real_returns_df.shape[1]  # Total stocks (46)
group_size = 8  # Number of stocks per subplot
num_groups = int(np.ceil(num_stocks / group_size))  # Number of groups

# Create grouped box plots
for i in range(num_groups):
    plt.figure(figsize=(10, 6))

    # Select 8 stocks for this group
    start_idx = i * group_size
    end_idx = min(start_idx + group_size, num_stocks)
    selected_stocks = real_returns_df.columns[start_idx:end_idx]

    # Prepare data for boxplot
    real_subset = real_returns_df[selected_stocks].melt(var_name="Stock", value_name="Returns")
    real_subset["Type"] = "Real"

    synthetic_subset = synthetic_returns_df[selected_stocks].melt(var_name="Stock", value_name="Returns")
    synthetic_subset["Type"] = "Synthetic"

    # Combine both datasets
    combined_df = pd.concat([real_subset, synthetic_subset])

    # Boxplot with Real (blue) & Synthetic (red) Overlapping
    sns.boxplot(data=combined_df, x="Stock", y="Returns", hue="Type", palette={"Real": "blue", "Synthetic": "red"})

    plt.xticks(rotation=90)
    plt.title(f"Real vs. Synthetic Returns (Stocks {start_idx+1} to {end_idx})")
    plt.xlabel("Stock")
    plt.ylabel("Returns")
    plt.legend(title="Data Type")
    plt.grid(True)

    plt.show()

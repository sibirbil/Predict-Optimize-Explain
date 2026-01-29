import pandas as pd
from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
import torch

# Example: 10 macroeconomic states, 5 variables each
states = torch.randn(10, 5)  # 10 states, 5 variables per state
states = torch.hstack([torch.ones((10,1)), states])

# Convert to DataFrame
df = pd.DataFrame(states.numpy(), 
                  columns=["name",'GDP', 'Inflation', 'Unemployment', 'Interest', 'Trade'])

plt.figure(figsize=(12, 6))
parallel_coordinates(df,class_column='name', colormap='viridis')  
# If you don't have classes, add a dummy class column
df['State'] = [f'State {i}' for i in range(len(df))]
parallel_coordinates(df, 'State', colormap='viridis')
plt.title('Macroeconomic States - Parallel Coordinates')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
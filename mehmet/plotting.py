import pandas as pd
from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
import torch
from typing import Optional

# # Example: 10 macroeconomic states, 5 variables each
# states = torch.randn(10, 5)  # 10 states, 5 variables per state
# states = torch.hstack([torch.ones((10,1)), states])

# # Convert to DataFrame
# df = pd.DataFrame(states.numpy(), 
#                   columns=["name",'GDP', 'Inflation', 'Unemployment', 'Interest', 'Trade'])

# plt.figure(figsize=(12, 6))
# parallel_coordinates(df,class_column='name', colormap='viridis')  
# # If you don't have classes, add a dummy class column
# df['State'] = [f'State {i}' for i in range(len(df))]
# parallel_coordinates(df, 'State', colormap='viridis')
# plt.title('Macroeconomic States - Parallel Coordinates')
# plt.xticks(rotation=45)
# plt.grid(True, alpha=0.3)
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("darkgrid")

# Combine both datasets with a label
def pairplot(
        historical_df   : pd.DataFrame, 
        generated_df    : pd.DataFrame, 
        m0              : torch.Tensor,
        lower_limits    : Optional[torch.Tensor] = None,
        upper_limits    : Optional[torch.Tensor] = None,
        save_address    : Optional[str] = None,
    ):

    if lower_limits is None:
        lower_limits = [-4.9313 - 0.05, -5.2638 - 0.05, -0.1035 - 0.05, -0.0758 - 0.01, -0.0348 - 0.01, 
                        -0.0517 - 0.01, 0.0010 - 0.005, -0.0056 - 0.005, -0.0227 - 0.001]
        upper_limits = [-2.3456 + 0.05, -1.4713 + 0.05,  1.4305 + 0.05,  0.0656 + 0.01,  0.1979 + 0.01,  
                        0.0608 +  0.01, 0.0383 + 0.005,  0.0789 + 0.005,  0.0188 + 0.001]
    else:
        distances = upper_limits - lower_limits

        lower_limits = [x.item()-0.05*distances[i] for (i,x) in enumerate(lower_limits)]
        upper_limits = [x.item()+0.05*distances[i] for (i,x) in enumerate(upper_limits)]

    limits = list(zip(lower_limits,upper_limits))
    
    col_names = list(historical_df.columns)
    start_pt_df = pd.DataFrame(m0.unsqueeze(0), columns = col_names)
    hist = historical_df.assign(source = 'historical')
    gen = generated_df.assign(source = 'generated')
    start  = start_pt_df.assign(source = 'current time')
    combined_df = pd.concat([hist, gen, start], ignore_index=True)


    palette_rgba = {
        'historical': (102/255, 51/255, 153/255, 0.25),      # RebeccaPurple, alpha=0.2
        'generated': (9/255, 121/255, 105/255, 0.25),       # Cadmium Green, alpha=0.1
        'current time': (1, 0., 0, 1.0)   # Red, alpha=1.0
    }
    # Create pairplot
    g = sns.pairplot(combined_df, 
            vars=col_names,
            hue='source',
            palette=palette_rgba,
            plot_kws={'s': 4, 'edgecolor': 'none'},
            diag_kind='kde',  # or 'kde' for density plots 'hist' for histograms
            diag_kws={'common_norm': False},
            height=1,
            aspect=1,
            corner=True)
    g.legend.remove()
    #g.figure.tight_layout(rect=[0, 0, 0.95, 1])
    
    for ax in g.axes.flat:
        if ax is not None:
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)
            ax.set_xticks([])      # Remove x-ticks
            ax.set_yticks([])      # Remove y-ticks

    n_vars = len(col_names)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = g.axes[i, j]
            if ax is not None and i >= j:
                # x-axis uses limits of column j
                ax.set_xlim(limits[j])
                # y-axis uses limits of column i
                ax.set_ylim(limits[i])


    g.figure.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce space between plots

    # Don't use tight_layout, set subplots_adjust manually
    g.figure.subplots_adjust(
        left=0.12,    # Enough space for y-labels
        right=0.98,   # Use most of width
        bottom=0.1,   # Enough space for x-labels
        top=0.95,     # Leave some top margin
        wspace=0.05,
        hspace=0.05
    )

    
    if save_address is not None:
        plt.savefig(save_address)
    plt.show()


def correlation_latex(m_df:pd.DataFrame):
    latex = m_df.corr().round(2).to_latex(
        caption='CAPTION',
        float_format = "%.3f",
        label="tab:label",
        position='h!',
        column_format='l' + 'c' * 9,  # Left-align first column
        escape=False  # Allow LaTeX commands     
        )
    print(latex)
    return latex

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_3x3_kde_grid(historical_df, generated_df, start_point):
    """
    3x3 grid of KDE plots comparing historical vs generated for each variable.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    hist_color = (102/255, 51/255, 153/255)  # RebeccaPurple
    gen_color = (9/255, 121/255, 105/255)    # Cadmium Green
    
    
    for idx, (ax, col) in enumerate(zip(axes, historical_df.columns)):
        # Plot KDEs
        sns.kdeplot(data=historical_df[col], ax=ax, 
                   color=hist_color, label='Historical', fill=True, alpha=0.25, 
                   common_norm=False, linewidth=1.5)
        sns.kdeplot(data=generated_df[col], ax=ax, 
                   color=gen_color, label='Generated', fill=True, alpha=0.25,
                   common_norm=False, linewidth=1.5)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Set gray background
        ax.set_facecolor('#f8f8f8')
        
        # Add title only
        ax.set_title(col, fontsize=22, pad=6)
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
 
        # Labels and title
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True, alpha=0.2)
        
        # Only show legend on first plot
        if idx == 0:
            ax.legend(fontsize=9)
        else:
            ax.legend().remove()

        ax.axvline(start_point[idx], color='red', linestyle='--', linewidth=2, label='m0')

    
    plt.suptitle('3x3 KDE Comparison: Historical vs Generated', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


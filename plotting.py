#############################
#############################
### FOR THE WINE DATSAET ####
#############################
#############################

from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def plot_wine_features(X_train, X_test, y_train, y_test, generated_samples, 
                       feature_idx1=0, feature_idx2=1, model=None, 
                       feature_names=None, figsize=(12, 8), alpha=0.7):
    """
    Plot wine dataset results with random forest decision boundaries.
    
    Parameters:
    -----------
    X_train : array-like, shape (n_train, n_features)
        Training features
    X_test : array-like, shape (n_test, n_features) 
        Test features
    y_train : array-like, shape (n_train,)
        Training labels
    y_test : array-like, shape (n_test,)
        Test labels
    generated_samples : array-like, shape (n_generated, n_features)
        Generated samples from MALA
    feature_idx1 : int, default=0
        Index of first feature to plot (x-axis)
    feature_idx2 : int, default=1
        Index of second feature to plot (y-axis)
    model : RandomForestClassifier, optional
        Trained random forest model for decision boundary
    feature_names : list, optional
        Names of features for axis labels
    figsize : tuple, default=(12, 8)
        Figure size
    alpha : float, default=0.7
        Transparency for points
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Color scheme as requested
    # Class 0: red (dark/light), Class 1: blue (dark/light), Class 2: orange (dark/light)
    # Generated: green
    colors = {
        'train': {0: '#8B0000', 1: '#00008B', 2: '#FF8C00'},  # Dark colors for training
        'test': {0: '#FF6B6B', 1: '#6B9BFF', 2: '#FFB366'},   # Light colors for test
        'generated': '#00B300'  # Green for generated samples
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract the two features for plotting
    X_train_2d = X_train[:, [feature_idx1, feature_idx2]]
    X_test_2d = X_test[:, [feature_idx1, feature_idx2]]
    generated_2d = generated_samples[:, [feature_idx1, feature_idx2]]
    
    # Plot training data (dark colors)
    for class_label in np.unique(y_train):
        mask = y_train == class_label
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
                  c=colors['train'][class_label], alpha=alpha, 
                  label=f'Train Class {class_label}', s=50, edgecolors='black', linewidth=0.5)
    
    # Plot test data (light colors)
    for class_label in np.unique(y_test):
        mask = y_test == class_label
        ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1], 
                  c=colors['test'][class_label], alpha=alpha, 
                  label=f'Test Class {class_label}', s=50, edgecolors='gray', linewidth=0.5)
    
    # Plot generated samples (green)
    ax.scatter(generated_2d[:, 0], generated_2d[:, 1], 
              c=colors['generated'], alpha=alpha*0.8, 
              label='Generated Samples', s=30, marker='^', edgecolors='darkgreen', linewidth=0.3)
    
    # Add decision boundary if RF model is provided
    if model is not None:
        # Create a mesh for decision boundary
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Extend limits slightly for better visualization
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        x_min, x_max = x_min - x_margin, x_max + x_margin
        y_min, y_max = y_min - y_margin, y_max + y_margin
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # For decision boundary, we need to create full feature vectors
        # We'll use the mean values for the other features
        mesh_points = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
        mesh_points[:, feature_idx1] = xx.ravel()
        mesh_points[:, feature_idx2] = yy.ravel()
        
        # Fill other features with training data means
        for i in range(X_train.shape[1]):
            if i not in [feature_idx1, feature_idx2]:
                mesh_points[:, i] = np.mean(X_train[:, i])
        
        # Predict on mesh
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary with light contours
        boundary_colors = ['#FFE4E4', '#E4E4FF', '#FFE4CC']  # Very light versions
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], 
                   colors=boundary_colors, extend='both')
    
    # Formatting
    if feature_names is not None:
        ax.set_xlabel(f'{feature_names[feature_idx1]}', fontsize=12)
        ax.set_ylabel(f'{feature_names[feature_idx2]}', fontsize=12)
    else:
        ax.set_xlabel(f'Feature {feature_idx1}', fontsize=12)
        ax.set_ylabel(f'Feature {feature_idx2}', fontsize=12)
    
    ax.set_title('Wine Dataset: Training, Test, and Generated Samples\nwith Decision Tree Boundaries', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

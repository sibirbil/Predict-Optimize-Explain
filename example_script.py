import cvxpy as cp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
from typing import Callable

from torch import func as func


def load_and_preprocess_data(train_path, valid_path, test_path):
    train_data = np.load(train_path)
    valid_data = np.load(valid_path)
    test_data = np.load(test_path)

    # Preprocess data (masking)
    X_train_full = train_data['data']
    X_valid_full = valid_data['data']
    X_test_full = test_data['data']

    mask_train = np.all(X_train_full != -99.99, axis=2)
    X_train_clean = X_train_full[mask_train]
    mask_valid = np.all(X_valid_full != -99.99, axis=2)
    X_valid_clean = X_valid_full[mask_valid]
    mask_test = np.all(X_test_full != -99.99, axis=2)
    X_test_clean = X_test_full[mask_test]

    y_train = X_train_clean[:, 0]
    X_train = X_train_clean[:, 1:]
    y_valid = X_valid_clean[:, 0]
    X_valid = X_valid_clean[:, 1:]
    y_test = X_test_clean[:, 0]
    X_test = X_test_clean[:, 1:]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_valid_scaled, y_valid, X_test_scaled, y_test

X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(
        './Data/datasets/char/Char_train.npz',
        './Data/datasets/char/Char_valid.npz',
        './Data/datasets/char/Char_test.npz'
    )


class FNN(nn.Module):
    def __init__(self, input_dim=46, hidden_dims=[32, 16, 8], dropout=0.5):
        super(FNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.net(x).squeeze(-1)

# The way we handle the data in this way is a little bit weird. 
# Originally the data comes to us with shape (240, 3685, 47) 
# where there are 240 timestamps, 3685 firms and 46 firm characteristics
# with the 0th column being the return (the target). 
# But after cleaning that information is gone and we have X_train.shape = (336113, 46)
# Each row is a single firm-timestamp pair with 46 features. In order to make a portfolio
# and decide on portfolio weights, I am going to batch this (into random sets) and the
# training is going to be done on these batches. What is weird is that each batch is going
# to try to find the weights for portfolio allocations given a different set of at different
# timepoints, or maybe even the same firm at different timepoints is going to show up as 
# different assets I could allocate to.

# I will still do this, and maybe we can fix the data 
# cleaning later, if this weirdness is undesirable.

BatchSize = 30  # batchsize
Nf = 46 # number of features
lambda_ = 0.1

def F_function(
    model       : nn.Module,
    x_batch     : torch.Tensor,
    y_batch     : torch.Tensor
):
    BatchSize = len(x_batch)
    Sigma = (torch.cov(x_batch) + 0.1*torch.eye(BatchSize)).numpy()
    lambda_ = 0.1

    w = cp.Variable(BatchSize)  
    b = cp.Parameter(BatchSize) # predicted returns 
    objective_fn = w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma)
    objective = cp.Maximize(objective_fn)
    constraints = [
        cp.sum(w) == 1,  
        w >= 0           
    ]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[w])

    y_solution, = cvxpylayer(y_batch)

    def F(params):
        returns = func.functional_call(model, params, x_batch)
        solution, = cvxpylayer(returns)
        return (returns.dot(solution) -  y_batch.dot(y_solution))**2
    
    return F

x_batch = torch.tensor(X_train[:30], dtype = torch.float)
y_batch = torch.tensor(y_train[:30], dtype = torch.float)
model = FNN()


def train(
    model   : nn.Module, 
    epochs  : int, 
    X : torch.Tensor, # Shape (T, B,F) where T = timestamps, B = portfolio size and F = 46 features
    y : torch.Tensor  # Shape (T, B,)
)-> None:

    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    params = dict(model.named_parameters())
    

    for epoch in range(epochs):
        for i in range(X.shape[0]):
            F = F_function(model, X[i], y[i])
            optimizer.zero_grad()
            loss = F(params)
            loss.backward()
            optimizer.step()



# permutation = torch.randperm(X_train.shape[0])
# for i in range(0, (X_train.shape[0]//BatchSize)*BatchSize, BatchSize):
#     indices = permutation[i:i + BatchSize]
#     x_batch = X_train[indices]
#     Sigma = np.cov(x_batch) + 0.1*np.eye(BatchSize)
#     y_batch = y_train[indices]

#     w = cp.Variable(BatchSize)  # predicted returns 
#     b = cp.Parameter(BatchSize)
#     objective_fn = w.T @ b - (lambda_ / 2) * cp.quad_form(w, Sigma)
#     objective = cp.Maximize(objective_fn)
#     constraints = [
#         cp.sum(w) == 1,  
#         w >= 0           
#     ]

#     problem = cp.Problem(objective, constraints)
#     assert problem.is_dpp()

#     cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[w])
#     b_torch = torch.tensor(y_batch, dtype = torch.float32, requires_grad=True)

#     solution, = cvxpylayer(b_torch)

#     torch.sum(b_torch*solution).backward()
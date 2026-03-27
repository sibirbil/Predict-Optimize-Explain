from torch import nn
import pandas as pd
import torch
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class AssetPricingFNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


def train_via_predictions(
    model: nn.Module,
    loader : DataLoader,
    epochs :int,
    lr = 0.001,
    device = 'mps'
    ):
    model.train()
    model.to(device)

    optimizer = SGD(model.parameters(), lr = lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 1e-8)
    
    for epoch in range(epochs):
        total_loss = 0.
        for idx, (x,y) in enumerate(loader):

            x,y = x.to(device = device), y.to(device = device)
            
            optimizer.zero_grad()
            out_batch = model(x)
            loss = torch.nn.functional.mse_loss(out_batch, y)
            loss.backward()
            optimizer.step()
        
            total_loss += loss
            if idx % 100 == 99:
                print(f"{idx + 1} batches with loss {total_loss/(idx + 1)}")
        print(f"Epoch {epoch+1}, loss: {total_loss/(idx + 1)}")
        scheduler.step()
        print(f"new step size is: {scheduler._last_lr}")


def eval_predictions(
    model :nn.Module,
    loader : DataLoader,
    device = 'mps'
    ):
    model.eval()
    model.to(device)
    ss_res = 0.0    # sum of squares of residuals
    sa_res = 0.0    # sum of absolute values
    sum_y = 0.0     # sum of values to calculate mean
    sum_y2 = 0.0
    n = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            yhat = model(x)
            err = yhat - y

            ss_res += err.square().sum().item()
            sa_res += err.abs().sum().item()
            sum_y += y.sum().item()
            sum_y2 += (y ** 2).sum().item()
            n += y.numel()
    
    y_bar = sum_y / n
    ss_tot = sum_y2 - n * y_bar**2
    R2 = 1.0 - ss_res / ss_tot
    MAE = sa_res / n
    MSE = ss_res/n

    print(f"MSE : {MSE}, MAE : {MAE}, R2 : {R2}")
    return MSE, MAE, R2



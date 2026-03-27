"""
E2E Portfolio Training Module.

Implements training loop with early stopping and checkpoint selection.
"""
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from ..utils.io import save_json
    from ..models.pao_portfolio import PAOPortfolioModel
except ImportError:
    from utils.io import save_json
    from models.pao_portfolio import PAOPortfolioModel


def selection_metric_for_loss(loss_type: str) -> Tuple[str, str]:
    """
    Returns (key_in_eval_dataset_output, human_label) for checkpoint selection.

    Args:
        loss_type: One of 'sharpe', 'return', or 'utility'

    Returns:
        Tuple of (metric_key, human_label)

    Raises:
        ValueError: If loss_type is unknown
    """
    if loss_type == "sharpe":
        return "sharpe_a", "ValSharpe"
    if loss_type == "return":
        return "mean_a", "ValAnnRet"
    if loss_type == "utility":
        return "util_a", "ValAnnUtility"
    raise ValueError(f"Unknown loss_type={loss_type}")


def train_one_run(
    run_dir: Path,
    train_ds,  # MonthCacheDataset
    val_ds,    # MonthCacheDataset
    model: PAOPortfolioModel,
    config,    # PAOConfig or config object with needed attributes
    loss_type: str,
    eval_fn=None  # Optional: pass eval_dataset function to avoid circular import
) -> Dict[str, Any]:
    """
    Train PAO portfolio model with early stopping and checkpoint selection.

    Training strategy depends on loss_type:
    - 'return': Maximize average monthly portfolio return
    - 'utility': Maximize average utility (return - lambda/2 * risk)
    - 'sharpe': Maximize Sharpe ratio over batches of months

    Args:
        run_dir: Directory to save checkpoints and logs
        train_ds: Training dataset (MonthCacheDataset)
        val_ds: Validation dataset (MonthCacheDataset)
        model: PAOPortfolioModel instance
        config: Configuration object with attributes:
            - lr: Learning rate
            - weight_decay: Weight decay for AdamW
            - epochs: Maximum number of epochs
            - patience: Early stopping patience
            - grad_clip: Gradient clipping max norm
            - month_batch: Batch size for return/utility loss
            - sharpe_batch: Batch size for Sharpe loss
            - device: Device to train on
        loss_type: One of 'return', 'utility', or 'sharpe'
        eval_fn: Optional evaluation function (to avoid circular imports)

    Returns:
        Dictionary with:
        - selection_metric: Metric used for checkpoint selection
        - best_val_metric: Best validation metric value
        - final_epoch: Final epoch number

    Side effects:
        - Saves train_log.csv with per-epoch metrics
        - Saves best_state.pt with best model checkpoint
        - Saves train_summary.json with training summary
    """
    # Import eval_dataset here if not provided (to avoid circular import)
    if eval_fn is None:
        try:
            from ..training.evaluation import eval_dataset
            eval_fn = eval_dataset
        except ImportError:
            from training.evaluation import eval_dataset
            eval_fn = eval_dataset

    run_dir.mkdir(parents=True, exist_ok=True)

    select_key, select_label = selection_metric_for_loss(loss_type)

    opt = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_score = -np.inf
    best_state = None
    bad_epochs = 0

    log_rows = []
    indices = np.arange(len(train_ds))

    for epoch in range(1, config.epochs + 1):
        model.train()
        np.random.shuffle(indices)

        if loss_type in ("return", "utility"):
            total_loss, n_blocks = 0.0, 0

            for b0 in range(0, len(indices), config.month_batch):
                block = indices[b0:b0 + config.month_batch]
                if len(block) == 0:
                    continue

                opt.zero_grad()
                block_losses = []

                for idx in block:
                    s = train_ds[int(idx)]
                    X = s["X"].to(config.device)
                    y = s["y"].to(config.device)
                    U = s["Sigma_factor"].to(config.device)
                    sigma_vol = s["sigma_vol"].to(config.device)

                    w, _ = model(X, U, sigma_vol)
                    port_ret = (w * y).sum()

                    if loss_type == "return":
                        # Maximize average monthly portfolio return over the block
                        block_losses.append(-port_ret)
                    else:  # utility
                        # Maximize average utility over the block: r - (lambda/2)*w'Σw
                        Uw = torch.mv(U, w)
                        risk = torch.sum(Uw * Uw)  # w'Σw = ||U w||^2
                        utility = port_ret - (model.lambda_ / 2.0) * risk
                        block_losses.append(-utility)

                loss = torch.stack(block_losses).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()

                total_loss += float(loss.item())
                n_blocks += 1

            train_loss = total_loss / max(n_blocks, 1)

        elif loss_type == "sharpe":
            total_loss, n_blocks = 0.0, 0

            for b0 in range(0, len(indices), config.sharpe_batch):
                block = indices[b0:b0 + config.sharpe_batch]
                if len(block) < max(4, config.sharpe_batch // 2):
                    continue

                opt.zero_grad()
                port_rets = []

                for idx in block:
                    s = train_ds[int(idx)]
                    X = s["X"].to(config.device)
                    y = s["y"].to(config.device)
                    U = s["Sigma_factor"].to(config.device)
                    sigma_vol = s["sigma_vol"].to(config.device)

                    w, _ = model(X, U, sigma_vol)
                    port_rets.append((w * y).sum())

                port_rets = torch.stack(port_rets)
                mean_ret = port_rets.mean()
                std_ret = port_rets.std(unbiased=True) + 1e-8
                sharpe = (mean_ret / std_ret) * math.sqrt(12.0)

                loss = -sharpe
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()

                total_loss += float(loss.item())
                n_blocks += 1

            train_loss = total_loss / max(n_blocks, 1)

        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

        # Validation
        val_metrics = eval_fn(model, val_ds, config.device)
        score = float(val_metrics[select_key])

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_sharpe": val_metrics["sharpe_a"],
            "val_mean_a": val_metrics["mean_a"],
            "val_vol_a": val_metrics["vol_a"],
            "val_util_a": val_metrics["util_a"],
            "val_risk_m": val_metrics["risk_m"],
            "val_max_w_med": val_metrics["max_w_med"],
            "val_n_eff_med": val_metrics["n_eff_med"],
            "val_ic_med": val_metrics["ic_med"],
            "val_mu_std_med": val_metrics["mu_std_med"],
            "val_select_score": score,
            "val_select_key": select_key,
        })

        improved = False
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = True

        bad_epochs = 0 if improved else (bad_epochs + 1)

        if epoch == 1 or epoch % 5 == 0 or improved:
            marker = " *" if improved else ""
            print(
                f"Epoch {epoch:3d} | TrainLoss={train_loss:+.6f} | "
                f"ValSharpe={val_metrics['sharpe_a']:.3f} | ValRet={val_metrics['mean_a']*100:.2f}% | "
                f"ValUtil={val_metrics['util_a']*100:.2f}% | {select_label}={score:+.4f}{marker} | "
                f"max_w_med={val_metrics['max_w_med']:.4f} | n_eff_med={val_metrics['n_eff_med']:.1f}"
            )

        if bad_epochs >= config.patience:
            print(f"Early stopping (no improvement in {select_label}).")
            break

    # Save logs + best checkpoint
    pd.DataFrame(log_rows).to_csv(run_dir / "train_log.csv", index=False)
    if best_state is not None:
        torch.save(best_state, run_dir / "best_state.pt")

    save_json(
        run_dir / "train_summary.json",
        {
            "loss_type": loss_type,
            "selection_metric": select_key,
            "best_val_metric": float(best_score),
            "final_epoch": int(len(log_rows))
        }
    )

    return {
        "selection_metric": select_key,
        "best_val_metric": float(best_score),
        "final_epoch": int(len(log_rows))
    }

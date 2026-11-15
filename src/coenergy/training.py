"""Training loop with per-target validation, early stopping and checkpointing."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from coenergy.data import SplitData
from coenergy.preprocessing import Scalers


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Maximum number of epochs.
        batch_size: Training batch size.
        lr: Peak learning rate of the one-cycle schedule.
        weight_decay: AdamW weight decay.
        patience: Stop after this many epochs without a better validation loss.
        eval_batch_size: Batch size for prediction.
        max_train: Use only the first `max_train` training rows (quick experiments), or None for all.
        data_on_device: Copy the training and validation tensors to the device once (faster if they fit).
        seed: Seed for the batch order.
    """

    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 4
    eval_batch_size: int = 4096
    max_train: int | None = None
    data_on_device: bool = True
    seed: int = 67


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """Predict in physical units, batch by batch. Returns a CPU tensor of shape (n, 6)."""
    device = next(model.parameters()).device
    model.eval()
    return torch.cat([model(x[i:i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)])


def normalized_mse(pred_phys: torch.Tensor, y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """MSE in z-scored target space, so all six targets weigh the same."""
    return F.mse_loss((pred_phys - mean) / std, y_norm)


def train_model(
    model: nn.Module,
    train: SplitData,
    val: SplitData,
    scalers: Scalers,
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_path: Path | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train with AdamW and a one-cycle schedule, keep the weights with the best validation loss.

    Args:
        model: Regressor returning targets in physical units.
        train: Training split (normalized inputs and targets).
        val: Validation split.
        scalers: Scalers used to normalize the targets.
        cfg: Training hyperparameters.
        device: Device to train on.
        checkpoint_path: If given, the best weights and the config are saved there.
        loss_fn: Training loss `loss_fn(pred_phys, y_norm)`. Defaults to the normalized MSE.
            The validation loss is always the normalized MSE, so runs stay comparable.
        verbose: Print one line per epoch.

    Returns:
        One row per epoch: train loss, val loss, val R2 per target, learning rate, time.
    """
    model.to(device)
    mean, std = scalers.tgt_mean.to(device), scalers.tgt_std.to(device)
    labels = scalers.target_labels

    n_train = len(train) if cfg.max_train is None else min(cfg.max_train, len(train))
    x_tr, y_tr = train.x[:n_train], train.y[:n_train]
    x_va, y_va = val.x, val.y
    if cfg.data_on_device:
        x_tr, y_tr, x_va, y_va = (t.to(device) for t in (x_tr, y_tr, x_va, y_va))

    steps_per_epoch = n_train // cfg.batch_size  # the last incomplete batch is dropped
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch)
    gen = torch.Generator(device=x_tr.device).manual_seed(cfg.seed)

    best_loss, best_state, stale = float('inf'), None, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, generator=gen, device=x_tr.device)
        running = torch.zeros((), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            xb, yb = x_tr[idx].to(device, non_blocking=True), y_tr[idx].to(device, non_blocking=True)
            pred = model(xb)
            loss = normalized_mse(pred, yb, mean, std) if loss_fn is None else loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach()
        train_loss = (running / steps_per_epoch).item()

        # Validation, in normalized space: per-target MSE, R2 = 1 - MSE / var
        pred = predict(model, x_va, cfg.eval_batch_size).to(device)
        z_err = ((pred - mean) / std - y_va) ** 2
        val_loss = z_err.mean().item()
        r2 = 1 - z_err.mean(dim=0) / y_va.var(dim=0)

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               **{f'R2_{l}': v for l, v in zip(labels, r2.tolist())},
               'lr': sched.get_last_lr()[0], 'time_s': time.time() - t0}
        history.append(row)
        if verbose:
            print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | R2 "
                  + ' '.join(f'{l} {v:.3f}' for l, v in zip(labels, r2.tolist()))
                  + f" | {row['time_s']:.0f}s", flush=True)

        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            if checkpoint_path is not None:
                torch.save({'state_dict': best_state, 'config': asdict(cfg), 'epoch': epoch}, checkpoint_path)
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f'Early stopping: no improvement for {cfg.patience} epochs')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history).set_index('epoch')

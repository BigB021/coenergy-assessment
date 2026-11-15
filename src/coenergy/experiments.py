"""Small controlled experiments: same budget for every variant, several seeds, validation metrics only.

Each (variant, seed) run is cached as a one-row CSV, so re-running a notebook skips finished runs.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from coenergy.data import SplitData
from coenergy.metrics import per_target_metrics
from coenergy.preprocessing import Scalers
from coenergy.training import TrainConfig, predict, train_model


def run_variant(
    name: str,
    make_model: Callable[[], nn.Module],
    train: SplitData,
    val: SplitData,
    scalers: Scalers,
    cfg: TrainConfig,
    seeds: tuple[int, ...],
    device: torch.device,
    results_dir: Path,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> pd.DataFrame:
    """Train one variant once per seed and score it on the validation split.

    The seed sets both the initial weights (`torch.manual_seed` before `make_model`) and the batch
    order (`cfg.seed`). The data split and the scalers never change.

    Args:
        name: Variant name, used for the cache files.
        make_model: Builds a freshly initialized model.
        train: Training split (possibly a subset, see `cfg.max_train`).
        val: Validation split; all metrics come from it, the test split is never used.
        scalers: Scalers used for the targets.
        cfg: Training configuration shared by all seeds.
        seeds: Seeds to run, fixed in advance.
        device: Device to train on.
        results_dir: Folder for the per-run CSV cache.
        loss_fn: Optional training loss, see `train_model`.

    Returns:
        One row per seed: validation loss, best epoch, time, parameters, and R2 / MAE per target.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    y_val = scalers.denorm_targets(val.y)
    rows = []
    for seed in seeds:
        cache = results_dir / f'{name}_s{seed}.csv'
        if cache.exists():
            rows.append(pd.read_csv(cache).iloc[0].to_dict())
            continue
        torch.manual_seed(seed)
        model = make_model()
        t0 = time.time()
        history = train_model(model, train, val, scalers, replace(cfg, seed=seed), device, loss_fn=loss_fn, verbose=False)
        metrics = per_target_metrics(predict(model, val.x, cfg.eval_batch_size), y_val, scalers.target_labels)
        row = {
            'variant': name,
            'seed': seed,
            'val_loss': history['val_loss'].min(),
            'best_epoch': int(history['val_loss'].idxmin()),
            'minutes': (time.time() - t0) / 60,
            'params': sum(p.numel() for p in model.parameters()),
            **{f'R2_{lbl}': metrics.loc[lbl, 'R2'] for lbl in scalers.target_labels},
            **{f'MAE_{lbl}': metrics.loc[lbl, 'MAE'] for lbl in scalers.target_labels},
        }
        pd.DataFrame([row]).to_csv(cache, index=False)
        rows.append(row)
        print(f"{name:14s} seed {seed} | val loss {row['val_loss']:.4f} | R2 HLC {row['R2_HLC']:.3f} | {row['minutes']:.1f} min", flush=True)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, reference: str, columns: list[str]) -> pd.DataFrame:
    """Mean and std over seeds for each variant, and the change against the reference variant.

    `z` is the change divided by the pooled seed std of the two variants: |z| above about 2 means the
    change is larger than run-to-run noise.

    Args:
        results: Concatenated outputs of `run_variant`.
        reference: Name of the reference variant.
        columns: Metric columns to summarize (e.g. 'val_loss', 'R2_HLC').
    """
    grouped = results.groupby('variant', sort=False)[columns]
    mean, std = grouped.mean(), grouped.std(ddof=1).fillna(0.0)
    out = {}
    for col in columns:
        out[(col, 'mean')] = mean[col]
        out[(col, 'std')] = std[col]
        delta = mean[col] - mean.loc[reference, col]
        pooled = ((std[col] ** 2 + std.loc[reference, col] ** 2) / 2) ** 0.5
        out[(col, 'delta')] = delta
        out[(col, 'z')] = delta / pooled.where(pooled > 0)
    return pd.DataFrame(out)


def make_log_tau_loss(
    scalers: Scalers, y_train_phys: torch.Tensor, device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Normalized MSE where the three time constants are compared in log space.

    The HLC family stays z-scored as usual; each tau is log-transformed, then z-scored with the
    training mean and std of its log. Predictions must be positive (physics head).

    Args:
        scalers: Scalers used for the targets.
        y_train_phys: Training targets in physical units, to fit the log statistics.
        device: Device the loss runs on.
    """
    mean, std = scalers.tgt_mean.to(device), scalers.tgt_std.to(device)
    log_tau = y_train_phys[:, 3:].log()
    log_mean, log_std = log_tau.mean(dim=0).to(device), log_tau.std(dim=0).to(device)

    def loss(pred_phys: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        y_phys = y_norm * std + mean
        z_pred = torch.cat([(pred_phys[:, :3] - mean[:3]) / std[:3],
                            (pred_phys[:, 3:].clamp_min(1e-6).log() - log_mean) / log_std], dim=1)
        z_true = torch.cat([y_norm[:, :3], (y_phys[:, 3:].log() - log_mean) / log_std], dim=1)
        return F.mse_loss(z_pred, z_true)

    return loss


def make_weighted_loss(
    scalers: Scalers, weights: list[float], device: torch.device,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Normalized MSE with one weight per target, rescaled so the weights average 1.

    Args:
        scalers: Scalers used for the targets.
        weights: One weight per target, in the order of `scalers.target_labels`.
        device: Device the loss runs on.
    """
    mean, std = scalers.tgt_mean.to(device), scalers.tgt_std.to(device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / w.mean()

    def loss(pred_phys: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        return ((((pred_phys - mean) / std - y_norm) ** 2) * w).mean()

    return loss


def heating_pattern(u_norm: torch.Tensor) -> pd.DataFrame:
    """Describe each scenario's heating signal from the scaled power (0 or 1).

    Args:
        u_norm: Scaled heating power, shape (n, T).

    Returns:
        Number of on/off switches and duty cycle per scenario.
    """
    on = u_norm > 0.5
    return pd.DataFrame({
        'switches': (on[:, 1:] != on[:, :-1]).sum(dim=1).numpy(),
        'duty_cycle': on.float().mean(dim=1).numpy(),
    })

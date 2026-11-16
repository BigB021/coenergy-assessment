"""Per-target evaluation metrics and physical-constraint checks, in physical units."""

from __future__ import annotations

import pandas as pd
import torch


def per_target_metrics(pred: torch.Tensor, true: torch.Tensor, labels: list[str]) -> pd.DataFrame:
    """MAE, RMSE, P95 absolute error, relative MAE and R2 for each target.

    Args:
        pred: Predictions in physical units, shape (n, n_targets).
        true: Ground truth in physical units, shape (n, n_targets).
        labels: Target names, in column order.

    Returns:
        One row per target.
    """
    err = pred - true
    abs_err = err.abs()
    sst = ((true - true.mean(dim=0)) ** 2).sum(dim=0)
    return pd.DataFrame(
        {
            'MAE': abs_err.mean(dim=0).numpy(),
            'RMSE': err.pow(2).mean(dim=0).sqrt().numpy(),
            'P95_abs': abs_err.quantile(0.95, dim=0).numpy(),
            'rel_MAE_%': (100 * (abs_err / true.abs()).mean(dim=0)).numpy(),
            'R2': (1 - err.pow(2).sum(dim=0) / sst).numpy(),
        },
        index=labels,
    )


def metrics_by_group(
    pred: torch.Tensor, true: torch.Tensor, labels: list[str], groups: pd.Series,
) -> pd.DataFrame:
    """`per_target_metrics` computed separately inside each group of scenarios.

    Args:
        pred: Predictions in physical units, shape (n, n_targets).
        true: Ground truth in physical units, shape (n, n_targets).
        labels: Target names, in column order.
        groups: Group of each scenario, length n (for example bins made with `pd.cut`).
            Groups are reported in their categorical order if they have one; empty groups are skipped.

    Returns:
        One row per (group, target), with the number of scenarios `n` and the metrics.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    keys = groups.cat.categories if isinstance(groups.dtype, pd.CategoricalDtype) else groups.dropna().unique()
    frames = {}
    for key in keys:
        mask = torch.from_numpy((groups == key).to_numpy().copy())
        if not mask.any():
            continue
        m = per_target_metrics(pred[mask], true[mask], labels)
        m.insert(0, 'n', int(mask.sum()))
        frames[key] = m
    return pd.concat(frames, names=['group', 'target'])


def constraint_violations(pred: torch.Tensor, sum_tol: float = 1e-3) -> dict[str, float]:
    """Share of predictions that break the physics of the targets.

    Args:
        pred: Predictions in physical units, columns HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf.
        sum_tol: Tolerance on |HLC - (HLC_mit + HLC_ext)| in W/K.
    """
    hlc, mit, ext, tau_b, tau_n, tau_inf = pred.unbind(dim=1)
    return {
        'sum_rule': ((hlc - mit - ext).abs() > sum_tol).float().mean().item(),
        'non_positive': (pred <= 0).any(dim=1).float().mean().item(),
        'tau_order': 1 - ((tau_b > tau_inf) & (tau_inf > tau_n)).float().mean().item(),
    }

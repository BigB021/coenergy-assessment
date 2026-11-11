"""Normalization of the model inputs and targets.

All statistics are fitted on the training split only, then applied to every split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Scalers:
    """Normalization statistics for temperature, heating power, and targets.

    Attributes:
        temp_mean: Global mean of the indoor temperature (degC).
        temp_std: Global standard deviation of the indoor temperature (degC).
        exc_scale: Heating power scale (W). Power is divided by it, so 0 W stays 0.
        tgt_mean: Per-target mean, shape (n_targets,), in physical units.
        tgt_std: Per-target standard deviation, shape (n_targets,), in physical units.
        target_labels: Target names, in column order.
        seed: Seed of the split the statistics were fitted on.
    """

    temp_mean: float
    temp_std: float
    exc_scale: float
    tgt_mean: torch.Tensor
    tgt_std: torch.Tensor
    target_labels: list[str]
    seed: int

    @classmethod
    def fit(
        cls,
        temperature: torch.Tensor,
        excitation: torch.Tensor,
        targets: torch.Tensor,
        target_labels: list[str],
        seed: int,
    ) -> Scalers:
        """Fit the statistics on training data.

        Args:
            temperature: Training indoor temperature, shape (n, T).
            excitation: Training heating power, shape (n, T).
            targets: Training targets, shape (n, n_targets).
            target_labels: Target names, in column order.
            seed: Seed of the split, stored for traceability.

        Returns:
            The fitted scalers.
        """
        return cls(
            temp_mean=temperature.mean().item(),
            temp_std=temperature.std().item(),
            exc_scale=excitation.max().item(),
            tgt_mean=targets.mean(dim=0),
            tgt_std=targets.std(dim=0),
            target_labels=list(target_labels),
            seed=seed,
        )

    def norm_temperature(self, t: torch.Tensor) -> torch.Tensor:
        """Global z-score of the indoor temperature."""
        return (t - self.temp_mean) / self.temp_std

    def norm_excitation(self, u: torch.Tensor) -> torch.Tensor:
        """Scale heating power to [0, 1] on the training range, keeping 0 W at 0."""
        return u / self.exc_scale

    def norm_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Per-column z-score of the targets."""
        return (y - self.tgt_mean) / self.tgt_std

    def denorm_targets(self, z: torch.Tensor) -> torch.Tensor:
        """Convert normalized targets back to physical units (W/K and h)."""
        return z * self.tgt_std + self.tgt_mean

    def save(self, path: Path) -> None:
        """Save the scalers as a plain dict with torch.save."""
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> Scalers:
        """Load scalers saved with `save`."""
        return cls(**torch.load(path, weights_only=False))

"""CNN-GRU regressor with a plain or a physics-constrained output head.

Every model returns predictions in physical units (W/K and h), in the column order
of `Scalers.target_labels`: HLC, HLC_mit, HLC_ext, tau_b, tau_n, tau_inf.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EXTRA_CHANNELS = ('dT', 'd2T', 'energy')


def derivative_std(x: torch.Tensor) -> float:
    """Std of the per-step change of normalized temperature, used to scale the dT/dt channel.

    Args:
        x: Training inputs, shape (n, 2, T), channel 0 is normalized temperature.
    """
    return x[:, 0].diff(dim=1).std().item()


def second_derivative_std(x: torch.Tensor) -> float:
    """Std of the per-step change of dT/dt (normalized temperature), used to scale the d2T/dt2 channel.

    Args:
        x: Training inputs, shape (n, 2, T), channel 0 is normalized temperature.
    """
    return x[:, 0].diff(n=2, dim=1).std().item()


class CNNGRU(nn.Module):
    """CNN feature extractor followed by a bidirectional GRU, with extra input channels derived from T and U.

    The extra channels are computed inside the model from the two inputs, so the model's interface
    stays (T(t), U(t)) -> targets:
    - 'dT': per-step temperature change, scaled by `dT_std`;
    - 'd2T': per-step change of dT, scaled by `d2T_std`;
    - 'energy': cumulative heating energy so far, as a fraction of the window (0 to 1).

    Shapes: (B, 2, T) -> add extra channels -> (B, 2 + k, T) -> CNN -> (B, width, T/8) -> BiGRU -> (B, T/8, 2*width) -> mean and last step -> (B, 4*width) -> MLP -> (B, n_out).
    """

    def __init__(
        self,
        dT_std: float,
        n_out: int,
        width: int = 64,
        gru_layers: int = 2,
        extra: tuple[str, ...] = ('dT',),
        d2T_std: float | None = None,
    ) -> None:
        super().__init__()
        unknown = set(extra) - set(EXTRA_CHANNELS)
        if unknown:
            raise ValueError(f'Unknown extra channels {unknown}, expected some of {EXTRA_CHANNELS}')
        if 'd2T' in extra and d2T_std is None:
            raise ValueError("The 'd2T' channel needs d2T_std")
        self.extra = tuple(extra)
        self.register_buffer('dT_std', torch.tensor(dT_std))
        if 'd2T' in extra:  # registered only when used, so older checkpoints still load
            self.register_buffer('d2T_std', torch.tensor(d2T_std))

        def block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=7, stride=stride, padding=3),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            )

        self.cnn = nn.Sequential(
            block(2 + len(self.extra), width, 1),
            block(width, width, 2),
            block(width, width, 2),
            block(width, width, 2),
        )
        self.gru = nn.GRU(width, width, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(4 * width, 128), nn.GELU(), nn.Linear(128, n_out))

    def add_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Append the extra channels computed from temperature (channel 0) and heating power (channel 1)."""
        channels = [x]
        diff = F.pad(x[:, 0].diff(dim=1), (1, 0))  # first step has no previous value
        if 'dT' in self.extra:
            channels.append((diff / self.dT_std).unsqueeze(1))
        if 'd2T' in self.extra:
            channels.append((F.pad(diff[:, 1:].diff(dim=1), (2, 0)) / self.d2T_std).unsqueeze(1))
        if 'energy' in self.extra:
            channels.append((x[:, 1].cumsum(dim=1) / x.shape[-1]).unsqueeze(1))
        return torch.cat(channels, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(self.cnn(self.add_channels(x)).transpose(1, 2))
        return self.mlp(torch.cat([seq.mean(dim=1), seq[:, -1]], dim=1))


class BaselineCNN(nn.Module):
    """Simple CNN baseline: 3 conv + max-pool blocks, then a fully connected layer.

    No dT/dt channel, no recurrence: the flattened features give every time step its own weights.
    Shapes: (B, 2, T) -> CNN -> (B, 64, T/8) -> flatten -> (B, 64*T/8) -> MLP -> (B, n_out).
    """

    def __init__(self, n_out: int, input_length: int = 480) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (input_length // 8), 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.cnn(x))


class PlainHead(nn.Module):
    """Unconstrained head: 6 raw outputs are read as normalized targets and converted back."""

    n_raw = 6

    def __init__(self, tgt_mean: torch.Tensor, tgt_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer('tgt_mean', tgt_mean.clone())
        self.register_buffer('tgt_std', tgt_std.clone())

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw * self.tgt_std + self.tgt_mean


class PhysicsHead(nn.Module):
    """Constrained head: 5 raw outputs give HLC_mit, HLC_ext, tau_n and two positive increments.

    By construction: HLC = HLC_mit + HLC_ext exactly, every target is positive,
    and tau_n < tau_inf < tau_b.
    """

    n_raw = 5

    def __init__(self, scales: torch.Tensor) -> None:
        """
        Args:
            scales: Shape (5,), [s_mit, s_ext, s_n, s_inf, s_b] from `physics_head_scales`.
        """
        super().__init__()
        self.register_buffer('scales', scales.clone())

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        mit, ext, tau_n, d_inf, d_b = (F.softplus(raw) * self.scales).unbind(dim=1)
        tau_inf = tau_n + d_inf
        tau_b = tau_inf + d_b
        return torch.stack([mit + ext, mit, ext, tau_b, tau_n, tau_inf], dim=1)


def physics_head_scales(y_phys: torch.Tensor) -> torch.Tensor:
    """Scales such that raw outputs of 0 predict the training mean (softplus(0) = ln 2).

    Args:
        y_phys: Training targets in physical units, shape (n, 6).
    """
    _, mit, ext, tau_b, tau_n, tau_inf = y_phys.unbind(dim=1)
    means = torch.stack([
        mit.mean(), ext.mean(), tau_n.mean(), (tau_inf - tau_n).mean(), (tau_b - tau_inf).mean(),
    ])
    return means / math.log(2)


class Regressor(nn.Module):
    """Backbone + head. Returns the 6 targets in physical units."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_cnn_gru(
    head: str,
    dT_std: float,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor,
    y_train_phys: torch.Tensor | None = None,
    width: int = 64,
    gru_layers: int = 2,
    extra: tuple[str, ...] = ('dT',),
    d2T_std: float | None = None,
) -> Regressor:
    """Build a CNN-GRU regressor.

    Args:
        head: 'plain' (6 free outputs) or 'physics' (constrained).
        dT_std: Scale of the dT/dt channel, from `derivative_std` on the training inputs.
        tgt_mean: Training target means (physical units), from the scalers.
        tgt_std: Training target stds (physical units), from the scalers.
        y_train_phys: Training targets in physical units, needed for the 'physics' head.
        width: Number of CNN channels and GRU hidden units.
        gru_layers: Number of stacked GRU layers.
        extra: Extra input channels computed from T and U (see `CNNGRU`).
        d2T_std: Scale of the d2T/dt2 channel, from `second_derivative_std`; needed if 'd2T' is in `extra`.
    """
    if head == 'plain':
        out: nn.Module = PlainHead(tgt_mean, tgt_std)
    elif head == 'physics':
        if y_train_phys is None:
            raise ValueError("The 'physics' head needs y_train_phys to set its scales")
        out = PhysicsHead(physics_head_scales(y_train_phys))
    else:
        raise ValueError(f"Unknown head {head!r}, expected 'plain' or 'physics'")
    backbone = CNNGRU(dT_std, n_out=out.n_raw, width=width, gru_layers=gru_layers, extra=extra, d2T_std=d2T_std)
    return Regressor(backbone, out)


def build_baseline_cnn(tgt_mean: torch.Tensor, tgt_std: torch.Tensor, input_length: int = 480) -> Regressor:
    """Build the baseline CNN with a plain head, returning targets in physical units.

    Args:
        tgt_mean: Training target means (physical units), from the scalers.
        tgt_std: Training target stds (physical units), from the scalers.
        input_length: Number of time steps per scenario.
    """
    head = PlainHead(tgt_mean, tgt_std)
    return Regressor(BaselineCNN(n_out=head.n_raw, input_length=input_length), head)

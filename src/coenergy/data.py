"""Loading the quentin_house simulations into normalized train / val / test tensors.

The batches are read one at a time and written straight into preallocated split
tensors, so the full raw dataset is never held in memory next to its normalized copy.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from coenergy.preprocessing import Scalers
from coenergy.util_assessement import load_single_batch

CDT_LABELS = ['tau_b', 'tau_n', 'tau_inf']
SPLIT_NAMES = ('train', 'val', 'test')


@dataclass
class SplitData:
    """One split, ready for the model.

    Attributes:
        x: Normalized inputs, shape (n, 2, T). Channel 0 is indoor temperature,
            channel 1 is heating power.
        y: Normalized targets, shape (n, 6), in the order of `Scalers.target_labels`.
        idx: Index of each row in the full dataset, shape (n,). Used to match
            rows with the flags and subsets from the EDA.
    """

    x: torch.Tensor
    y: torch.Tensor
    idx: torch.Tensor

    def __len__(self) -> int:
        return self.x.shape[0]


def batch_files(data_dir: Path) -> list[str]:
    """Batch files in the same order as `load_simulation_data` (lexicographic)."""
    files = sorted(glob.glob(str(Path(data_dir) / 'batch_*.pt')))
    if not files:
        raise FileNotFoundError(f'No batch_*.pt files found in {data_dir}')
    return files


def load_splits(
    data_dir: Path, split_path: Path, scalers: Scalers, round_inputs: bool = True,
) -> dict[str, SplitData]:
    """Load every split, apply the EDA rounding and the saved normalization.

    Args:
        data_dir: Folder with the batch_*.pt files.
        split_path: File saved by the EDA with 'train', 'val', 'test' index tensors and 'seed'.
        scalers: Scalers fitted on the training split of that same file.
        round_inputs: Round temperature to 0.01 degC and power to 0.1 W, as in the assessment's
            loading code. Set to False only for diagnostics.

    Returns:
        A dict mapping 'train', 'val', 'test' to their `SplitData`.
    """
    splits = torch.load(split_path, weights_only=True)
    if splits['seed'] != scalers.seed:
        raise ValueError(f"Split seed {splits['seed']} does not match scaler seed {scalers.seed}")

    files = batch_files(data_dir)
    n_total = sum(len(splits[name]) for name in SPLIT_NAMES)

    # For every scenario: which split it belongs to, and its row in that split
    split_id = torch.full((n_total,), -1, dtype=torch.long)
    row = torch.empty(n_total, dtype=torch.long)
    for s, name in enumerate(SPLIT_NAMES):
        split_id[splits[name]] = s
        row[splits[name]] = torch.arange(len(splits[name]))
    if (split_id < 0).any():
        raise ValueError('Split indices do not cover the dataset')

    out: dict[str, SplitData] | None = None
    offset = 0
    for path in files:
        batch = load_single_batch(path)
        n, T = batch.indoor_temperature.shape

        if out is None:
            labels = ['HLC'] + [lbl.strip() for lbl in batch.Hb_labels] + CDT_LABELS
            if labels != scalers.target_labels:
                raise ValueError(f'Target labels {labels} do not match scalers {scalers.target_labels}')
            out = {
                name: SplitData(
                    x=torch.empty(len(splits[name]), 2, T),
                    y=torch.empty(len(splits[name]), len(labels)),
                    idx=splits[name].clone(),
                )
                for name in SPLIT_NAMES
            }

        # Same rounding as the assessment example and the EDA (unless disabled for diagnostics)
        raw_temp, raw_exc = batch.indoor_temperature, batch.indoor_U
        if round_inputs:
            raw_temp, raw_exc = raw_temp.round(decimals=2), raw_exc.round(decimals=1)
        temp = scalers.norm_temperature(raw_temp)
        exc = scalers.norm_excitation(raw_exc)
        tgt = scalers.norm_targets(torch.cat([batch.HLC.unsqueeze(1), batch.Hb, batch.CDT], dim=1))
        del batch

        sid = split_id[offset:offset + n]
        pos = row[offset:offset + n]
        for s, name in enumerate(SPLIT_NAMES):
            mask = sid == s
            rows = pos[mask]
            out[name].x[rows, 0] = temp[mask]
            out[name].x[rows, 1] = exc[mask]
            out[name].y[rows] = tgt[mask]
        del temp, exc, tgt
        offset += n

    if offset != n_total:
        raise ValueError(f'Batches contain {offset} scenarios, splits expect {n_total}')
    assert out is not None
    return out


def load_solicitation(data_dir: Path) -> pd.DataFrame:
    """Parameters of the heating-signal generator for every scenario, in dataset order.

    These parameters describe how U(t) was generated (the 'recipe' of the heating signal).
    They are used for error analysis only, never as a model input: row i matches index i
    of the split tensors (`SplitData.idx`).

    Args:
        data_dir: Folder with the batch_*.pt files.

    Returns:
        One row per scenario, one column per generator parameter.
    """
    parts: list[torch.Tensor] = []
    names: list[str] = []
    for path in batch_files(data_dir):
        batch = load_single_batch(path)
        parts.append(batch.solicitation.clone())
        names = list(batch.solicitation_field_names)
        del batch
    return pd.DataFrame(torch.cat(parts).numpy(), columns=names)

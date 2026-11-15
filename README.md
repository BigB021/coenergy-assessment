# coEnergy assessment: predicting building indicators from indoor temperature

This project is my answer to the coEnergy internship assessment. The goal is to predict six energy-performance indicators of a building from two time series, using the `quentin_house` simulation dataset (1,000,000 scenarios).

- **Inputs:** indoor air temperature T_int(t) and heating power U(t), 480 one-minute steps (8 hours) per scenario.
- **Outputs:** HLC (W/K), its split into HLC_ext and HLC_mit (W/K), and three time constants tau_b, tau_n and tau_inf (hours).

The final model is an average of three CNN-GRU networks with a "physics head": an output layer that always respects HLC = HLC_ext + HLC_mit and tau_n < tau_inf < tau_b.

## Results (test set, 150,000 scenarios)

| Family | Target | R² | MAE | P95 abs. error |
|---|---|---|---|---|
| HLC | HLC | 0.440 | 18.7 W/K | 46.0 W/K |
| Hb | HLC_ext | 0.476 | 9.9 W/K | 24.1 W/K |
| | HLC_mit | 0.282 | 17.9 W/K | 43.5 W/K |
| CDT | tau_n | 0.269 | 1.35 h | 3.3 h |
| | tau_inf | 0.109 | 5.6 h | 13.7 h |
| | tau_b | 0.140 | 7.4 h | 18.1 h |

For comparison, always predicting the training mean gives an HLC MAE of 25.3 W/K, and a simple CNN gives HLC R² 0.37. The physics rules hold in 100% of the test predictions.

The scores are moderate, and I think the main reason is the data, not the model. In the EDA, scenarios with almost identical input curves often have very different targets, and in the experiments a 5× bigger model gained almost nothing. The heating pattern matters much more: scenarios with 1-2 long heating blocks are predicted clearly better (HLC R² 0.47) than scenarios with constant heating (0.32). The notebooks explain this in detail.

## Reading order

The notebooks are meant to be read in order. Together they form the report.

| Notebook | Question it answers |
|---|---|
| [`1_eda.ipynb`](notebooks/1_eda.ipynb) | What does the data look like, which physics rules hold, and how much can the inputs tell us? Also creates the train/val/test split and the normalization. |
| [`2_modeling.ipynb`](notebooks/2_modeling.ipynb) | First models: mean predictor, simple CNN, CNN-GRU with a plain head and with a physics head. Evaluation per target, physics checks, errors by subset. |
| [`3_improvements.ipynb`](notebooks/3_improvements.ipynb) | What limits the CNN-GRU? Short controlled experiments (3 seeds each, validation only): model size, amount of data, rounding, extra channels, other losses, heating pattern. |
| [`4_final_model.ipynb`](notebooks/4_final_model.ipynb) | Final model (3-seed ensemble), single evaluation on the test set, results by indicator family, and the project conclusion. |

## Project structure

```
coenergy-assessment/
├── notebooks/              the four notebooks above
├── src/coenergy/           code imported by the notebooks
│   ├── util_assessement.py   data loader provided with the assessment
│   ├── preprocessing.py      Scalers: normalization fitted on the training split
│   ├── data.py               load_splits: loads the batches into normalized train/val/test tensors
│   ├── models.py             CNN-GRU, baseline CNN, plain head, physics head
│   ├── training.py           training loop (AdamW, one-cycle, best-epoch checkpoint)
│   ├── metrics.py            per-target metrics and physics-rule checks
│   └── experiments.py        seeded experiment runner and summary used in notebook 3
├── models/                 trained checkpoints (.pt) and training histories (.csv)
├── reports/
│   ├── figures/            figures saved by the notebooks
│   ├── experiments/        one CSV per experiment run (notebook 3)
│   └── final/              final test metrics (notebook 4)
├── data/                   not in git: the dataset and the saved split
├── pyproject.toml, uv.lock
└── ssrn-6928008.pdf        reference paper
```

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/) and needs Python 3.11 or newer.

```bash
uv sync
```

This installs PyTorch 2.11 from the CUDA 12.8 wheel index (set in `pyproject.toml`). The same wheel also runs on CPU, but training on CPU would be very slow. If you have a different CUDA version, change the index URL in `pyproject.toml`.

if you have NVIDIA gpu run:
```bash 
nvidia-smi 
```
this displays your gpu info/specs, particularly the cuda version

**Data.** The dataset is not in the repository. Download the `quentin_house` folder (link in the assessment) and put the 20 batch files here:

```
data/quentin_house/batch_1.pt ... batch_20.pt
```

`1_eda.ipynb` creates `data/splits/split_70_15_15_seed67.pt` and `data/splits/scalers_seed67.pt`. The other notebooks load these files, so the EDA notebook has to be run first.

## Running the notebooks

Open the notebooks in VS Code or Jupyter with the `.venv` kernel created by uv, or run them from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/1_eda.ipynb
```

Training is skipped when the checkpoints already exist:
- `4_final_model.ipynb` has `RETRAIN = False` and loads `models/final_s{0,1,2}.pt`.
- `2_modeling.ipynb` has `RETRAIN = True` (the outputs show the training run). Set it to `False` to load `models/baseline_cnn.pt`, `cnn_gru_plain.pt` and `cnn_gru_physics.pt` instead.
- `3_improvements.ipynb` saves one CSV per (experiment, seed) in `reports/experiments/` and skips the runs that already have one.

**Hardware and run times.** I ran everything on an NVIDIA RTX 3060 (12 GB) with 32 GB of RAM. The EDA loads the full dataset into memory. With this setup, one CNN-GRU epoch on 700k scenarios takes about 36 s (about 12 min for 20 epochs), one baseline CNN epoch about 8 s, and one experiment run in notebook 3 about 1-4 min.

## Reproducibility

- The split and the normalization use seed 67 and are fitted on the training split only.
- Each model seed (0, 1, 2 in notebooks 3 and 4, 67 in notebook 2) sets both the initial weights and the batch order.
- The test split is used only for the final evaluation, in notebooks 2 and 4. All choices are made on the validation split.
- GPU training is not bit-for-bit deterministic, so a re-run can differ slightly. In my runs, R² changed by at most about 0.002.

## Assessment constraints

- The model input is only T_int(t) and U(t), loaded and rounded as in the assessment's example code. The extra channels (dT/dt and cumulative energy) are computed from them inside the model. The `solicitation` table is only used in the EDA, to explain how the heating signals were generated.
- All models are written in PyTorch. All the code lives in typed and documented modules in `src/coenergy/`, which the notebooks import.

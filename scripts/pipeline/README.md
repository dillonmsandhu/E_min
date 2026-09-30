# Modular Multi-Algorithm Sweep Pipeline

This folder contains the core **modular hyperparameter sweep and cross-algorithm comparison pipeline** ([`sweep_pipeline.py`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/pipeline/sweep_pipeline.py)).

For a comprehensive walkthrough of the design, see [sweep_walkthrough.md](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/pipeline/sweep_walkthrough.md).

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `sweep_pipeline.py` | Python Engine | Unified sweep engine supporting parallel vmapped sweeps, parameter grids, and cross-algorithm comparisons. |
| `sweep_walkthrough.md` | Documentation | Detailed architectural and usage walkthrough. |

---

## 🚀 How to Run from the Root

### 1. Sweep All Core Algorithms for Fixed Policy Evaluation
Evaluates `exact_td`, `exact_mc`, `exact_E_gd`, and `exact_td_lambda` on `FourRooms-misc`:

```bash
python scripts/pipeline/sweep_pipeline.py \
    --policy fixed \
    --env-name FourRooms-misc \
    --n-seeds 5 \
    --total-timesteps 1000
```

### 2. Sweep Specific Algorithms with Custom Learning Rate Grid
```bash
python scripts/pipeline/sweep_pipeline.py \
    --policy fixed \
    --algos exact_td exact_mc \
    --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001
```

### 3. Sweep Control Algorithms under PPO Policy
```bash
python scripts/pipeline/sweep_pipeline.py \
    --policy ppo \
    --env-name CartPole-v1 \
    --algos sampled_E sampled_td_lambda \
    --lr-grid 0.003 0.001 0.0003 \
    --value-lambda-grid 0.9 0.99 1.0 \
    --n-seeds 8 \
    --total-timesteps 2048000
```

---

## 📊 How to Run Analysis

`sweep_pipeline.py` automatically generates cross-algorithm comparisons in a `comparison/` directory when multiple algorithms are evaluated.

To perform custom post-hoc analysis on existing runs:

```bash
# Evaluate sweeps
python notebooks/evaluate_sweeps.py --policy fixed --env-name FourRooms-misc

# Analyze sweeps with geometric mean error bands
python notebooks/analyze_sweeps.py --policy fixed --env-name FourRooms-misc --use-geom-mean
```

---

## 📂 Where Output is Saved

All outputs are organized hierarchically under `results/<policy>/sweeps/<policy>_<env_name>_<timestamp>/`:

```
results/fixed/sweeps/fixed_FourRooms-misc_<timestamp>/
├── pipeline_config.json                 # Run specification
├── exact_td/
│   └── tuning/<timestamp>/FourRooms-misc/
│       ├── config.json                  # Base configuration
│       ├── best_config.json             # Winning hyperparameters & metadata
│       ├── tuning_summary.csv           # Sorted hyperparameter ranking
│       ├── tuning_summary.json
│       ├── out.pkl                      # Raw metrics tensor across combos & seeds
│       ├── hyperparameter_sweep_<metric>.png
│       ├── best_config_seeds.png        # Individual seed curves for #1 config
│       └── all_configs_seeds_grid.png   # Subplot grid of all evaluated configs
├── exact_mc/
│   └── tuning/...
├── exact_E_gd/
│   └── tuning/...
├── exact_td_lambda/
│   └── tuning/...
└── comparison/
    ├── comparison_summary.csv           # Multi-algorithm performance ranking
    ├── comparison_summary.json
    └── comparison_best_configs.png      # All best configurations on one graph
```

# E-Minimization Optimization & Architecture Sweeps

This folder contains scripts for **multi-dimensional hyperparameter optimization** of the `E_experimental` algorithm across Gymnax control tasks and MinAtar games. It explores:
- **Critic Epochs**: `[4, 16, 32]`
- **Critic Weight Decay**: `[0.001, 0.01]`
- **Value Heads**: `[1, 4]` (critic ensemble)
- **Loss Type**: `MSE` vs. `Huber`
- **Parallel Environments / Batch Size**: `[64, 512]` (in MinAtar optimization)

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm_gymnax_e_experimental.sh` | Shell / SLURM | Batch array launcher (0–11) across 12 non-MinAtar Gymnax tasks. |
| `sweep_gymnax_e_experimental.py` | Python Runner | Multi-dimensional grid sweep runner across epochs, weight decay, loss, and heads. |
| `run_slurm_minatar_e_opt.sh` | Shell / SLURM | Batch array launcher (0–3) across 4 MinAtar games (`Asterix`, `Breakout`, `Freeway`, `SpaceInvaders`). |
| `sweep_minatar_e_opt.py` | Python Runner | 8-condition sweep specifically tuning `NUM_ENVS`, `EPOCHS`, and `WEIGHT_DECAY` on MinAtar. |
| `visualize_multidim_sweep.py` | Python Analysis | Multi-dimensional post-processing suite (Parallel Coordinates, Main Effects, 2D Heatmaps, ANOVA variance decomposition, Ablation Curves). |

---

## 🚀 How to Run from the Root

### 1. Fast Local Smoke Test (CPU or Single GPU)
Test `CartPole-v1` with minimal steps and 1 seed:

```bash
python scripts/e_optimization/sweep_gymnax_e_experimental.py \
    --env-name CartPole-v1 \
    --n-seeds 1 \
    --total-timesteps 50000 \
    --epochs-grid 4 \
    --wd-grid 0.001 \
    --heads-grid 1
```

For MinAtar:

```bash
python scripts/e_optimization/sweep_minatar_e_opt.py \
    --env-name Asterix-MinAtar \
    --n-seeds 1 \
    --total-timesteps 50000
```

### 2. Run Locally via Shell Launcher
Run a single environment with default grid settings:

```bash
# Gymnax classic control
./scripts/e_optimization/run_slurm_gymnax_e_experimental.sh CartPole-v1

# MinAtar game
./scripts/e_optimization/run_slurm_minatar_e_opt.sh Asterix-MinAtar
```

### 3. Submit SLURM Batch Array (Cluster)

```bash
# Gymnax suite (12 environments)
sbatch scripts/e_optimization/run_slurm_gymnax_e_experimental.sh

# MinAtar suite (4 games)
sbatch scripts/e_optimization/run_slurm_minatar_e_opt.sh
```

---

## 📊 How to Run Analysis

Both `sweep_gymnax_e_experimental.py` and `sweep_minatar_e_opt.py` automatically invoke `visualize_multidim_sweep.py` to produce a 6-panel publication-ready PDF poster (`multidim_analysis.pdf` or `comparison_e_opt.pdf`) upon completion.

To re-run or perform advanced post-hoc multi-dimensional analysis on existing sweep data:

### Per-Environment Multi-Dimensional Breakdown
```bash
python scripts/e_optimization/visualize_multidim_sweep.py \
    --results-dir results/ppo/sweeps/e_experimental_gymnax_<suite_id>/CartPole-v1
```

### Suite-Wide Aggregated Analysis
```bash
python scripts/e_optimization/visualize_multidim_sweep.py \
    --suite-dir results/ppo/sweeps/e_experimental_gymnax_<suite_id>
```

Analysis outputs include:
1. **Parallel Coordinates Plot**: High-dimensional trajectories through hyperparameter space colored by performance.
2. **Marginal Main Effects & Significance**: Box distributions, 95% bootstrap CIs, Cohen's $d$, and Welch's $t$-test $p$-values.
3. **2D Interaction Heatmaps**: Pairwise synergies (e.g. Critic Epochs $\times$ Weight Decay).
4. **Variance Decomposition (ANOVA / fANOVA)**: Percentage of variance explained by each factor and two-way interaction.
5. **Ablation Learning Curves**: Paired comparisons (e.g. 16 Epochs vs. 4 Epochs, 4 Heads vs. 1 Head).

---

## 📂 Where Output is Saved

All outputs are saved under:
- Gymnax: `results/ppo/sweeps/e_experimental_gymnax_<suite_id>/<env_name>/`
- MinAtar: `results/ppo/sweeps/e_opt_<suite_id>/<env_name>/`

```
results/ppo/sweeps/e_experimental_gymnax_<suite_id>/<env_name>/
├── summary_ranking.csv             # Sorted table of all evaluated parameter configurations
├── metrics.pkl                     # Raw evaluation metric arrays
├── multidim_analysis.pdf           # 6-panel vector analysis poster
├── multidim_analysis.png           # Raster preview image
└── multidim_stats.json             # Computed ANOVA, effect sizes, and p-values
```

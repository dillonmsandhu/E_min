# Classic TD vs. Symmetrized E-Minimization Sweep

This folder contains scripts for the standard **head-to-head comparison sweep** evaluating **Classic TD Learning** (`algos/td.py`) against **Symmetrized E-Minimization** (`algos/E_experimental.py`) across a multi-dimensional grid:
- **Critic Learning Rate**: `[0.0003, 0.001, 0.003]`
- **Critic Epochs**: `[4, 16, 32]`
- **Weight Decay**: `[0.001, 0.01]`
- **Value Heads**: `[1, 4]`
- **Return Lambda (E)**: `[0.9, 0.99]`

Total evaluated configurations: 72 (36 per algorithm), vectorized over random seeds using JAX `vmap`.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm_td_vs_e_experimental.sh` | Shell / SLURM | SLURM batch array launcher across 16 Gymnax environments (indices 0–15). |
| `sweep_td_vs_e_experimental.py` | Python Runner | Runs the parallel vmapped grid sweep for both TD and E on an environment. |
| `compile_cmp_td_vs_e_master_pdf.py` | Python Analysis | Compiles all completed environment runs into a master executive dashboard and multi-page vector PDF report. |

---

## 🚀 How to Run from the Root

### 1. Fast Local Smoke Test (CPU or Single GPU)
Test a single environment (e.g. `CartPole-v1`) with reduced steps, 1 seed, and minimal grid:

```bash
python scripts/td_vs_e_experimental/sweep_td_vs_e_experimental.py \
    --env-name CartPole-v1 \
    --n-seeds 1 \
    --total-timesteps 50000 \
    --critic-lr-grid 0.001 \
    --epochs-grid 4 \
    --wd-grid 0.001 \
    --heads-grid 1
```

### 2. Run Locally via Shell Launcher
Run a specific environment using the full configuration grid:

```bash
./scripts/td_vs_e_experimental/run_slurm_td_vs_e_experimental.sh CartPole-v1
```

### 3. Submit Full SLURM Batch Array (Cluster)
Submits a job array spanning all 16 Gymnax environments:
- Classic Control: `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`
- MinAtar: `Asterix-MinAtar`, `Breakout-MinAtar`, `Freeway-MinAtar`, `SpaceInvaders-MinAtar`
- BSuite: `DeepSea-bsuite`, `DiscountingChain-bsuite`
- Misc / Continuous: `FourRooms-misc`, `PointRobot-misc`, `Reacher-misc`, `Swimmer-misc`, `Pong-misc`

```bash
sbatch scripts/td_vs_e_experimental/run_slurm_td_vs_e_experimental.sh
```

---

## 📊 How to Run Analysis

During training, each environment job automatically produces per-environment posters (`head_to_head_poster.pdf`), CSV tables, and multidimensional analysis reports.

Once all SLURM array jobs complete (or while jobs are in progress), compile the **suite-wide master PDF report**:

```bash
python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py \
    --suite-dir results/ppo/sweeps/cmp_td_vs_e_<suite_id> \
    --rank-by auc \
    --window-size 100
```

*Note: You can pass just the numeric job ID (e.g. `12724069`) or the full suite directory path.*

---

## 📂 Where Output is Saved

All outputs are saved under `results/ppo/sweeps/cmp_td_vs_e_<suite_id>/`:

```
results/ppo/sweeps/cmp_td_vs_e_<suite_id>/
├── <env_name>/                             # E.g. CartPole-v1, MountainCarContinuous-v0
│   ├── td/
│   │   ├── summary_td.csv                  # Ranked hyperparameter configs for TD
│   │   ├── metrics.pkl                     # Raw metric tensors
│   │   └── multidim_analysis.pdf           # 6-panel multi-dimensional analysis for TD
│   ├── E_experimental/
│   │   ├── summary_e_experimental.csv      # Ranked configs for E_experimental
│   │   ├── metrics.pkl                     # Raw metric tensors
│   │   └── multidim_analysis.pdf           # 6-panel multi-dimensional analysis for E
│   ├── head_to_head_summary.csv            # Pairwise matching table (E vs TD) with deltas
│   ├── head_to_head_poster.pdf             # 4-panel comparison poster (curves, epoch scaling, heads scaling, scatter)
│   └── head_to_head_poster.png
├── master_summary_table.csv                # Suite-wide comparison table (after running analysis)
└── master_td_vs_e_report.pdf               # Executive multi-page vector report (after running analysis)
```

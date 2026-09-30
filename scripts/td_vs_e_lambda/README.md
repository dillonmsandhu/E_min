# E(λ) vs. TD(λ) Head-to-Head Comparison Sweep

This folder contains scripts for the head-to-head comparison sweep evaluating **$E(\lambda)$** (`algos/E_lambda_experimental.py`) against **TD($\lambda$)** (`algos/td.py`) across a multi-dimensional grid:
- **$\lambda$ (Bootstrapping / Return parameter)**: `[0.0, 0.8, 0.95]`
- **Critic Epochs**: `[4, 16, 32]`
- **Critic Learning Rate**: `[0.0003, 0.001]`
- **Value Heads**: `[1, 4]`
- **Weight Decay**: `[0.001, 0.01]`

Total configurations evaluated: 72–144 configurations vectorized over random seeds using JAX `vmap`.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm_td_vs_e_lambda.sh` | Shell / SLURM | SLURM batch array launcher across 16 Gymnax environments (indices 0–15). |
| `sweep_td_vs_e_lambda.py` | Python Runner | Runs the parallel vmapped grid sweep for both TD($\lambda$) and E($\lambda$). |

---

## 🚀 How to Run from the Root

### 1. Fast Local Smoke Test (CPU or Single GPU)
Test a single environment (e.g. `CartPole-v1`) with reduced steps, 1 seed, and minimal grid:

```bash
python scripts/td_vs_e_lambda/sweep_td_vs_e_lambda.py \
    --env-name CartPole-v1 \
    --n-seeds 1 \
    --total-timesteps 50000 \
    --lambda-grid 0.0 0.8 \
    --critic-lr-grid 0.001 \
    --epochs-grid 4 \
    --wd-grid 0.001 \
    --heads-grid 1
```

### 2. Run Locally via Shell Launcher
Run a specific environment using the full configuration grid:

```bash
./scripts/td_vs_e_lambda/run_slurm_td_vs_e_lambda.sh CartPole-v1
```

### 3. Submit Full SLURM Batch Array (Cluster)
Submits a job array spanning all 16 Gymnax environments:
- Classic Control: `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`
- MinAtar: `Asterix-MinAtar`, `Breakout-MinAtar`, `Freeway-MinAtar`, `SpaceInvaders-MinAtar`
- BSuite: `DeepSea-bsuite`, `DiscountingChain-bsuite`
- Misc / Continuous: `FourRooms-misc`, `PointRobot-misc`, `Reacher-misc`, `Swimmer-misc`, `Pong-misc`

```bash
sbatch scripts/td_vs_e_lambda/run_slurm_td_vs_e_lambda.sh
```

---

## 📊 How to Run Analysis

During training, each environment job automatically produces:
1. `head_to_head_lambda_poster.pdf` & `.png`: 4-panel comparison poster (learning curves, lambda scaling, epoch scaling, scatter).
2. `head_to_head_lambda_summary.csv`: Matched pairwise comparisons with performance deltas.
3. `multidim_analysis.pdf`: 6-panel multi-dimensional analysis inside each algorithm's directory.

To compile all completed environment runs into a **suite-wide master PDF report**, run:

```bash
python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py \
    --suite-dir results/ppo/sweeps/cmp_td_vs_e_lambda_<suite_id> \
    --rank-by auc \
    --window-size 100
```

---

## 📂 Where Output is Saved

All outputs are saved under `results/ppo/sweeps/cmp_td_vs_e_lambda_<suite_id>/`:

```
results/ppo/sweeps/cmp_td_vs_e_lambda_<suite_id>/
├── <env_name>/                             # E.g. CartPole-v1, Asterix-MinAtar
│   ├── td/
│   │   ├── summary_td_lambda.csv           # Ranked hyperparameter configs for TD(lambda)
│   │   ├── metrics.pkl                     # Raw metric tensors
│   │   └── multidim_analysis.pdf           # 6-panel multi-dimensional analysis for TD
│   ├── E_lambda_experimental/
│   │   ├── summary_e_lambda.csv            # Ranked configs for E(lambda)
│   │   ├── metrics.pkl                     # Raw metric tensors
│   │   └── multidim_analysis.pdf           # 6-panel multi-dimensional analysis for E(lambda)
│   ├── head_to_head_lambda_summary.csv     # Pairwise matching table (E vs TD) with deltas
│   ├── head_to_head_lambda_poster.pdf      # 4-panel comparison poster
│   └── head_to_head_lambda_poster.png
├── master_summary_table.csv                # Suite-wide comparison table (after running analysis)
└── master_td_vs_e_report.pdf               # Executive multi-page vector report (after running analysis)
```

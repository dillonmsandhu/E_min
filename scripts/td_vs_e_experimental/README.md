# 4-Way Critic Algorithm Comparison Sweep: E(0), TD(0), E(λ), TD(λ)

This folder contains scripts for the standard **critic comparison sweep** evaluating 4 core critic learning formulations across an identical multi-dimensional hyperparameter grid:

1. **`E(0)`**: Symmetrized 1-step E-minimization ([`algos/E_experimental.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/E_experimental.py))
2. **`TD(0)`**: Classic 1-step online TD learning with minibatch updates ([`algos/td.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/td.py) with `TD_LAMBDA=0.0`)
3. **`E(λ)`**: Geometric jump sampled $E(\lambda)$ with $\lambda = 0.9$ ([`algos/E_lambda_experimental.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/E_lambda_experimental.py))
4. **`TD(λ)`**: Standard PPO fitted $T^\lambda$ value iteration with $\lambda = 0.9$ ([`algos/td.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/td.py) with `TD_LAMBDA=0.9`, static targets)

---

## 📐 Experimental Grid

All 4 algorithms are evaluated on the identical 36 hyperparameter configurations per environment:
- **Critic Learning Rate**: `[0.0003, 0.001, 0.003]` (3 values)
- **Critic Epochs**: `[4, 16, 32]` (3 values)
- **Weight Decay**: `[0.001, 0.01]` (2 values)
- **Value Heads**: `[1, 4]` (2 values)

**Held Constant across all runs:**
- Actor Learning Rate: `0.0003`
- Actor Epochs: `4`
- Multi-step $\lambda$: `0.9` (for $E(\lambda)$ and $TD(\lambda)$)
- Return Anchor $\lambda_{\text{ret}}$: `0.99` (for $E(0)$ and $E(\lambda)$)

**Total Evaluated per Environment:** $36 \times 4 = 144$ configurations, vectorized over random seeds (default 8) via JAX `vmap`.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| [`run_slurm_td_vs_e_experimental.sh`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/td_vs_e_experimental/run_slurm_td_vs_e_experimental.sh) | Shell / SLURM | SLURM batch array launcher across 16 Gymnax environments (indices 0–15). |
| [`sweep_td_vs_e_experimental.py`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/td_vs_e_experimental/sweep_td_vs_e_experimental.py) | Python Runner | Runs the parallel vmapped grid sweep for all 4 algorithms on an environment. Generates `auc_summary.md`, pairwise CSVs, and `head_to_head_poster.pdf`. |
| [`compile_cmp_td_vs_e_master_pdf.py`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py) | Python Analysis | Compiles all completed environment runs into a master Markdown AUC summary table, executive dashboard, and multi-page vector PDF report. |

---

## 🚀 How to Run

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
- **Classic Control**: `CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`
- **MinAtar**: `Asterix-MinAtar`, `Breakout-MinAtar`, `Freeway-MinAtar`, `SpaceInvaders-MinAtar`
- **BSuite**: `DeepSea-bsuite`, `DiscountingChain-bsuite`
- **Misc / Continuous**: `FourRooms-misc`, `PointRobot-misc`, `Reacher-misc`, `Swimmer-misc`, `Pong-misc`

```bash
sbatch scripts/td_vs_e_experimental/run_slurm_td_vs_e_experimental.sh
```

---

## 📊 Analysis & Outputs

### Per-Environment Outputs
During training, each environment job outputs:
- **`auc_summary.md`**: Markdown table showing AUC and Final Return for all 4 algorithms, plus matched-pair statistics for $E(0)$ vs. $TD(0)$ and $E(\lambda)$ vs. $TD(\lambda)$.
- **`head_to_head_poster.pdf` / `.png`**: 4-panel publication-grade comparison poster:
  1. Learning curves of Best-in-Class for all 4 algorithms (Mean ± 1 SEM)
  2. Critic epochs scaling ($4 \to 16 \to 32$) across all 4 algorithms
  3. Value heads scaling (1 vs. 4 heads) across all 4 algorithms
  4. Matched-pair scatter plots ($E(0)$ vs. $TD(0)$ and $E(\lambda)$ vs. $TD(\lambda)$)
- **`all_algos_summary.csv`**: Combined hyperparameter evaluation metrics across all 144 configurations.
- **`matched_pairs_e0_vs_td0.csv`**: 1-to-1 matched configuration comparisons for 1-step methods.
- **`matched_pairs_elambda_vs_tdlambda.csv`**: 1-to-1 matched configuration comparisons for $\lambda$-trace methods.

### Suite-Wide Master Compilation
To compile results across all 16 environments:

```bash
python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py \
    --suite-dir results/ppo/sweeps/cmp_td_vs_e_<suite_id> \
    --rank-by auc
```

This generates:
- **`suite_auc_summary.md`**: Markdown table of AUC across all tasks with pairwise win counts and statistical significance.
- **`master_summary_table.csv`**: Full suite metrics.
- **`master_td_vs_e_report.pdf`**: Multi-page vector report with executive dashboard, cross-task learning curves, and individual environment posters.

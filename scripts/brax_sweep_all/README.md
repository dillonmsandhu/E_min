# Brax 4-Way Continuous Control Critic Benchmark (`brax_sweep_all`)

Comprehensive hyperparameter sweep and empirical benchmark on the Brax Continuous Control suite comparing four critic formulations:

1. **`TD(0)`**: Online 1-step temporal difference error minimization (live bootstrap $r + \gamma (1-d) V(s')$ with MSE loss).
2. **`E(0)`**: Symmetrized Dirichlet error minimization on adjacent-state transitions ($K=0$).
3. **`TD(lambda)`**: Standard PPO Fitted Value Iteration MSE to $\lambda$-return targets ($G_t^\lambda$) with fixed $\lambda = 0.9$.
4. **`E(lambda)`**: Sampled geometric lookahead jump Dirichlet error minimization ($K \sim \text{Geom}(1 - \gamma \lambda)$) with fixed $\lambda = 0.9$.

---

## 1. Environments (11 Total, Excl. `fast`)

| Index | Environment Name | Physics Backend | Notes |
|:---:|:---|:---:|:---|
| `0` | `hopper` | Positional | Classic planar hopping robot |
| `1` | `walker2d` | Positional | Planar bipedal walking robot |
| `2` | `halfcheetah` | Positional | High-speed planar locomotion |
| `3` | `ant` | Positional | 3D quadrupedal locomotion |
| `4` | `swimmer` | Generalized | Multi-link viscous fluid swimmer |
| `5` | `humanoid` | Generalized | 3D high-DoF humanoid locomotion |
| `6` | `inverted_pendulum` | Positional | Classic pole-balancing benchmark |
| `7` | `inverted_double_pendulum` | Positional | Chaotic two-pole balancing |
| `8` | `reacher` | Positional | 2-DoF robotic reaching arm |
| `9` | `humanoidstandup` | Generalized | 3D humanoid standing-up control |
| `10` | `pusher` | Positional | Multi-joint manipulator pushing a puck to target |

---

## 2. Hyperparameter Grid

All 4 algorithms are evaluated on the exact same hyperparameter grid:

- **Critic Learning Rate**: `[1e-4, 3e-4, 1e-3]`
- **Critic Epochs**: `[4, 8, 16]`
- **Lambda Parameter**: Fixed $\lambda = 0.9$ for both `TD(lambda)` and `E(lambda)`
- **Return Target Lambda**: $\lambda_{\text{ret}} = 0.9$ for `E(0)` and `E(lambda)`
- **Total Configurations**: $3 \times 3 = 9$ per algorithm $\times 4$ algorithms = **36 configurations per environment**.
- **Vectorized Seeds**: 5 independent seeds evaluated via `jax.vmap`.

---

## 3. Directory Layout & Outputs

```
results/sweeps/<SWEEP_ID>/
  ├── <ENV_NAME>/
  │     ├── comparison_poster.pdf        # Publication-ready 4-panel poster
  │     ├── comparison_poster.png        # High-res PNG for presentations
  │     ├── metrics.pkl                  # Full rollout metrics & learning curves
  │     ├── summary_best_per_algo.csv    # Best config for each of the 4 algorithms
  │     ├── summary_all_algos.csv        # Detailed table of all 36 configurations
  │     ├── summary_td_0.csv             # TD(0) grid results
  │     ├── summary_e_0.csv              # E(0) grid results
  │     ├── summary_td_lambda.csv        # TD(lambda=0.9) grid results
  │     ├── summary_e_lambda.csv         # E(lambda=0.9) grid results
  │     ├── summary_comparison_0.csv     # E(0) vs TD(0) pairwise t-tests
  │     └── summary_comparison_lambda.csv # E(lambda) vs TD(lambda) pairwise t-tests
  ├── suite_summary_best.csv             # Suite-wide aggregation table across all 11 envs
  └── brax_sweep_all_suite_report.pdf    # Multi-page suite PDF report
```

---

## 4. Usage Instructions

### Submitting the Entire 11-Environment Array on SLURM Cluster:
```bash
sbatch scripts/brax_sweep_all/run_slurm_brax_all.sh
```

### Submitting a Single Environment Array Task:
```bash
# Task 9 is humanoidstandup; Task 10 is pusher
sbatch --array=9 scripts/brax_sweep_all/run_slurm_brax_all.sh
sbatch --array=10 scripts/brax_sweep_all/run_slurm_brax_all.sh
```

### Running Locally / CPU Testing:
```bash
./scripts/brax_sweep_all/run_slurm_brax_all.sh inverted_pendulum
```

### Generating Suite PDF Report across all Environments:
```bash
python scripts/brax_sweep_all/generate_brax_suite_all_pdf.py \
    --suite-dir results/sweeps/brax_sweep_all_<ARRAY_JOB_ID>
```

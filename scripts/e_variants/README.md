# E-Variants Suite (Comparison of 4 Formulations of E)

This folder contains scripts to evaluate and benchmark the **four mathematical formulations of the Symmetrized Value Objective $E$**:
1. **$E$**: 1-step $E(0)$ baseline (Tang & Munos).
2. **$E_{\lambda\text{-fixed}}$**: Method 1 — Fitted Value Iteration (FVI) stop-gradient form with forward/backward eligibility traces.
3. **$E_{\lambda\text{-diff}}$**: Method 2 — Full Autodiff Moment Scan.
4. **$E_{\lambda\text{-geom}}$**: Method 3 — Geometric Jump Sampling.

Evaluated alongside standard **PPO / TD($\lambda$)** across Gymnax classic control tasks and MinAtar games.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm_e_variants_suite.sh` | Shell / SLURM | Batch array launcher (0–11) across 19 Gymnax environments comparing all 4 variants. |
| `run_slurm_minatar_lambda_sweep.sh` | Shell / SLURM | Batch array launcher (0–3) for MinAtar comparing `RETURN_LAMBDA` vs. `VALUE_LAMBDA` / `E_LAMBDA`. |
| `run_slurm_minatar_e_vs_td.sh` | Shell / SLURM | Batch array launcher (0–3) comparing all 4 variants against TD($\lambda$) on MinAtar. |
| `slurm_diagnose_minatar.sh` | Shell / SLURM | Diagnostic sweep for MinAtar games with fine-grained logging. |
| `generate_e_variants_suite_pdf.py` | Python Analysis | Compiles all suite runs into a multi-page PDF report with an overview grid and per-env deep dives. |
| `plot_e_variants_comparison.py` | Python Analysis | Generates side-by-side 2-panel comparison figure (learning curves + $\lambda$ sensitivity curve). |

---

## 🚀 How to Run from the Root

### 1. Fast Local Smoke Test (Single Environment)
Run a fast smoke test on `CartPole-v1` for a single algorithm using the pipeline engine:

```bash
python scripts/pipeline/sweep_pipeline.py \
    --policy ppo \
    --env-name CartPole-v1 \
    --algos E E_lambda_fixed \
    --lr-grid 0.001 \
    --e-lambda-grid 0.0 0.5 \
    --n-seeds 1 \
    --total-timesteps 50000
```

### 2. Run Locally via Shell Launcher
Execute all 4 variants on a specific environment:

```bash
# Gymnax classic control
./scripts/e_variants/run_slurm_e_variants_suite.sh CartPole-v1

# MinAtar game
./scripts/e_variants/run_slurm_minatar_e_vs_td.sh Asterix-MinAtar
```

### 3. Submit SLURM Batch Array (Cluster)

```bash
# Gymnax suite (19 environments array)
sbatch scripts/e_variants/run_slurm_e_variants_suite.sh

# MinAtar 4 variants vs TD sweep
sbatch scripts/e_variants/run_slurm_minatar_e_vs_td.sh

# MinAtar lambda sweep
sbatch scripts/e_variants/run_slurm_minatar_lambda_sweep.sh
```

---

## 📊 How to Run Analysis

Each environment sweep automatically generates a 4-way comparison plot upon completion. You can also re-run or compile suite-level reports manually:

### 1. Single-Environment Comparison Plot
Generates side-by-side learning curves and $E(\lambda)$ performance scaling:

```bash
python scripts/e_variants/plot_e_variants_comparison.py \
    --sweep-dir results/ppo/sweeps/suite_e_variants_<suite_id>/CartPole-v1 \
    --metric returned_episode_returns \
    --rank-by final_window
```

### 2. Suite-Wide Multi-Page PDF Compiler
Compiles all environments in a suite into a publication-ready vector report:

```bash
python scripts/e_variants/generate_e_variants_suite_pdf.py \
    --suite-dir results/ppo/sweeps/suite_e_variants_<suite_id> \
    --metric returned_episode_returns \
    --rank-by final_window
```

Optional flags:
- `--output-pdf <path>`: Custom PDF file destination.
- `--window-size <int>`: Final evaluation window size (default: 100).
- `--email <address>`: Send PDF attachment upon completion.

---

## 📂 Where Output is Saved

All outputs are saved under `results/ppo/sweeps/suite_e_variants_<suite_id>/` (or `suite_minatar_e_vs_td_<suite_id>/`):

```
results/ppo/sweeps/suite_e_variants_<suite_id>/
├── <env_name>/                             # E.g. CartPole-v1, Asterix-MinAtar
│   ├── E/
│   │   └── tuning/.../best_config.json
│   ├── E_lambda_fixed/
│   │   └── tuning/...
│   ├── E_lambda_differentiable/
│   │   └── tuning/...
│   ├── E_lambda_geometric/
│   │   └── tuning/...
│   ├── ppo/
│   │   └── tuning/...
│   ├── comparison_e_variants.png           # 2-panel comparison figure
│   └── comparison_e_variants.pdf
├── suite_summary.csv                       # Summary ranking across all environments
└── e_variants_suite_report.pdf             # Comprehensive multi-page PDF report
```

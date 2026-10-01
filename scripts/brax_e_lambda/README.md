# Brax E(lambda) Sweep & Baseline Comparison

This folder contains scripts for sweeping **Symmetrized E(lambda)** (geometric lookahead jump Dirichlet error minimization) against a standard **PPO baseline** across continuous control environments in Brax.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm_brax_e_lambda_sweep.sh` | Shell / SLURM | SLURM batch array job launcher (tasks 0–8 across 9 Brax environments). |
| `sweep_brax_e_lambda_vs_baseline.py` | Python Runner | Evaluates E(lambda) across critic learning rates, epochs, and return lambdas ([0.9, 0.99]) with fixed $\lambda_{\text{dir}} = 0.9$ vs. baseline PPO. |
| `generate_brax_e_lambda_suite_pdf.py` | Python Analysis | Aggregates all environment runs into a suite summary CSV and multi-page vector PDF report. |

---

## ⚙️ Configuration Contract

- **Lambda Parameters**:
  - `E_LAMBDA = 0.9`: Geometric lookahead jump parameter for multi-step Dirichlet transition pairs $(s_t, s_{t+K})$.
  - `RETURN_LAMBDA = [0.9, 0.99]`: Target regression $\lambda$-return grid for value bootstrapping.
- **Critic Sweep Grid**:
  - Critic Learning Rate: `[1e-4, 3e-4, 1e-3]`
  - Critic Epochs: `[4, 8, 16]`
  - Return Lambda: `[0.9, 0.99]`
  (Total: 18 configurations for $E(\lambda)$ + 1 un-swept Baseline PPO reference per environment).
- **Rollout**:
  - 1024 parallel environments $\times$ 128 rollout steps (131,072 transitions per policy update).

---

## 🚀 How to Run from the Root

### 1. Fast Local Smoke Test (CPU or Single GPU)
To test execution on a single environment (e.g. `hopper`) with reduced steps and seeds:

```bash
python scripts/brax_e_lambda/sweep_brax_e_lambda_vs_baseline.py \
    --env-name hopper \
    --n-seeds 1 \
    --total-timesteps 100000 \
    --critic-lr-grid 0.0003 \
    --epochs-grid 4
```

### 2. Run Locally via Shell Launcher
Run a specific environment using the default parameter grid:

```bash
./scripts/brax_e_lambda/run_slurm_brax_e_lambda_sweep.sh hopper
```

### 3. Submit Full SLURM Batch Array (Cluster)
Submits a job array spanning all 9 Brax environments (`ant`, `halfcheetah`, `hopper`, `humanoid`, `pusher`, `reacher`, `walker2d`, `swimmer`, `inverted_pendulum`):

```bash
sbatch scripts/brax_e_lambda/run_slurm_brax_e_lambda_sweep.sh
```

---

## 📊 How to Run Analysis

SLURM array jobs run per-environment independently and generate per-environment posters automatically. To compile all completed environment runs into a unified **suite-wide master PDF and ranking CSV**, run:

```bash
python scripts/brax_e_lambda/generate_brax_e_lambda_suite_pdf.py --suite-dir results/sweeps/brax_e_lambda_<suite_id>
```

Optional arguments:
- `--output-pdf <path>`: Custom destination path for the PDF report (defaults to `<suite_dir>/brax_e_lambda_suite_report.pdf`).

---

## 📂 Where Output is Saved

All outputs are saved under `results/sweeps/brax_e_lambda_<suite_id>/` (or `results/ppo/sweeps/brax_e_lambda_<suite_id>/`):

```
results/sweeps/brax_e_lambda_<suite_id>/
├── <env_name>/                         # E.g., hopper, walker2d, ant
│   ├── summary_e_lambda.csv            # Ranked configurations for E(lambda)
│   ├── summary_baseline.csv            # Performance stats for baseline PPO
│   ├── summary_comparison.csv          # Head-to-head comparison and winning delta
│   ├── metrics.pkl                     # Raw learning curves and metrics dictionary
│   ├── comparison_poster.pdf           # Publication vector 4-panel poster for this env
│   └── comparison_poster.png           # High-resolution raster preview
├── suite_summary.csv                   # Aggregated suite-wide table (after running analysis)
└── brax_e_lambda_suite_report.pdf      # Multi-page executive PDF report (after running analysis)
```

# PPO Suite: Sampled E vs. TD(λ) Spectrum Sweeps

This folder contains scripts for comparing **Sampled E Minimization** against the full parameter spectrum of **TD($\lambda$)** under PPO control policies:
- **TD($\lambda$) $\lambda$ Spectrum**: `[0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]`
- **Sampled E Return $\lambda$**: `[0.9, 0.95, 0.98, 0.99, 1.0]`
- **Critic Learning Rate**: `[0.003, 0.001, 0.0003, 0.0001]`
- **Actor Training**: Held strictly constant (`ACTOR_LR=0.0003`, `GAE_LAMBDA=0.9`) to isolate critic value estimation differences.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm_gymnax_suite.sh` | Shell / SLURM | Batch array launcher (0–15) across 19 Gymnax tasks comparing E vs. full TD($\lambda$) spectrum. |
| `run_slurm_minatar_suite.sh` | Shell / SLURM | Batch array launcher (0–3) across 4 MinAtar games (10M timesteps per task). |
| `run_slurm_sweep_ppo.sh` | Shell / SLURM | PPO sweep comparing Sampled E, TD($\lambda$), and Sampled MC across 4 navigation tasks. |
| `generate_suite_pdf.py` | Python Analysis | Compiles all completed environment runs into a multi-page suite vector PDF report. |
| `plot_lambda_spectrum_vs_E.py` | Python Analysis | Generates side-by-side plot of learning curves across $\lambda$ values and $\lambda$-sensitivity curve. |
| `old_lambda_sweep_diagnostics.py` | Python Analysis | Legacy diagnostics tool with evaluation rollouts and animated GIF generation. |

---

## 🚀 How to Run from the Root

### 1. Fast Local Smoke Test (Single Environment)
Test `CartPole-v1` with 1 seed and small timesteps:

```bash
python scripts/pipeline/sweep_pipeline.py \
    --policy ppo \
    --env-name CartPole-v1 \
    --algos sampled_E sampled_td_lambda \
    --lr-grid 0.001 \
    --value-lambda-grid 0.9 1.0 \
    --n-seeds 1 \
    --total-timesteps 50000
```

### 2. Run Locally via Shell Launcher
Run a specific environment using default sweep settings:

```bash
# Gymnax task
./scripts/ppo_td_spectrum/run_slurm_gymnax_suite.sh CartPole-v1

# MinAtar game
./scripts/ppo_td_spectrum/run_slurm_minatar_suite.sh Asterix-MinAtar
```

### 3. Submit SLURM Batch Array (Cluster)

```bash
# Gymnax suite (19 environments array)
sbatch scripts/ppo_td_spectrum/run_slurm_gymnax_suite.sh

# MinAtar suite (4 games array)
sbatch scripts/ppo_td_spectrum/run_slurm_minatar_suite.sh

# Navigation suite
sbatch scripts/ppo_td_spectrum/run_slurm_sweep_ppo.sh
```

---

## 📊 How to Run Analysis

Each environment sweep automatically generates a `lambda_spectrum_vs_E.pdf` comparison upon completion. You can also re-run or compile suite-level reports manually:

### 1. Single-Environment Lambda Spectrum Plot
```bash
python scripts/ppo_td_spectrum/plot_lambda_spectrum_vs_E.py \
    --sweep-dir results/ppo/sweeps/suite_<suite_id>/CartPole-v1 \
    --metric returned_episode_returns
```

### 2. Suite-Wide Multi-Page PDF Compiler
Compiles all environment runs into a unified report:

```bash
python scripts/ppo_td_spectrum/generate_suite_pdf.py \
    --suite-dir results/ppo/sweeps/suite_<suite_id> \
    --metric returned_episode_returns \
    --rank-by final_window
```

Optional arguments:
- `--output-pdf <path>`: Custom destination PDF path.
- `--window-size <int>`: Final evaluation window size (default: 100).
- `--email <address>`: Email notification with PDF attachment upon completion.

---

## 📂 Where Output is Saved

All outputs are saved under `results/ppo/sweeps/suite_<suite_id>/` (or `suite_minatar_<suite_id>/`):

```
results/ppo/sweeps/suite_<suite_id>/
├── <env_name>/                             # E.g. CartPole-v1, Freeway-MinAtar
│   ├── sampled_E/
│   │   └── tuning/...
│   ├── sampled_td_lambda/
│   │   └── tuning/...
│   ├── comparison_sampled_td_lambda_vs_sampled_E.png
│   └── comparison_sampled_td_lambda_vs_sampled_E.pdf
├── suite_summary.csv                       # Summary ranking across all environments
└── suite_report.pdf                        # Multi-page executive PDF report
```

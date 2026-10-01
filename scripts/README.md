# Experiment & Sweep Scripts

All hyperparameter sweeps, cluster submission scripts, and post-experiment analysis tools are organized into self-contained modular directories below.

---

## 📁 Directory Structure & Experiment Workflows

| Folder | Focus / Workflow | Runner(s) | Analysis Tool(s) |
| :--- | :--- | :--- | :--- |
| [`brax_e0/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/brax_e0/README.md) | **Brax Continuous Control**: Adjacent-state Dirichlet error minimization ($E_0$) vs. PPO baseline | `sweep_brax_e0_vs_baseline.py`<br>`run_slurm_brax_e0_sweep.sh` | `generate_brax_e0_suite_pdf.py` |
| [`brax_e_lambda/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/brax_e_lambda/README.md) | **Brax Continuous Control**: Geometric jump Dirichlet error minimization ($E(\lambda)$ with $\lambda=0.9$) vs. PPO baseline | `sweep_brax_e_lambda_vs_baseline.py`<br>`run_slurm_brax_e_lambda_sweep.sh` | `generate_brax_e_lambda_suite_pdf.py` |
| [`td_vs_e_experimental/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/td_vs_e_experimental/README.md) | **Classic TD vs. E_experimental**: 36-config head-to-head comparison grid across 16 Gymnax environments | `sweep_td_vs_e_experimental.py`<br>`run_slurm_td_vs_e_experimental.sh` | `compile_cmp_td_vs_e_master_pdf.py` |
| [`td_vs_e_lambda/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/td_vs_e_lambda/README.md) | **E(λ) vs. TD(λ)**: Head-to-head lambda comparison across $\lambda \in [0.0, 0.8, 0.95]$, epochs, and heads | `sweep_td_vs_e_lambda.py`<br>`run_slurm_td_vs_e_lambda.sh` | Automatic posters + `compile_cmp_td_vs_e_master_pdf.py` |
| [`e_optimization/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/e_optimization/README.md) | **E Optimization & Architecture Tuning**: Tuning critic epochs, weight decay, MSE vs. Huber loss, and value heads | `sweep_gymnax_e_experimental.py`<br>`sweep_minatar_e_opt.py`<br>`run_slurm_gymnax_e_experimental.sh`<br>`run_slurm_minatar_e_opt.sh` | `visualize_multidim_sweep.py` (Parallel coords, ANOVA, main effects, heatmaps) |
| [`e_variants/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/e_variants/README.md) | **Four Formulations of E**: $E(0)$, $E_{\lambda\text{-fixed}}$, $E_{\lambda\text{-diff}}$, and $E_{\lambda\text{-geom}}$ vs. PPO/TD | `run_slurm_e_variants_suite.sh`<br>`run_slurm_minatar_lambda_sweep.sh`<br>`run_slurm_minatar_e_vs_td.sh`<br>`slurm_diagnose_minatar.sh` | `generate_e_variants_suite_pdf.py`<br>`plot_e_variants_comparison.py` |
| [`ppo_td_spectrum/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/ppo_td_spectrum/README.md) | **Sampled E vs. TD(λ) Spectrum**: Evaluates Sampled E against the spectrum of $\lambda \in [0.8, \dots, 1.0]$ | `run_slurm_gymnax_suite.sh`<br>`run_slurm_minatar_suite.sh`<br>`run_slurm_sweep_ppo.sh` | `generate_suite_pdf.py`<br>`plot_lambda_spectrum_vs_E.py`<br>`old_lambda_sweep_diagnostics.py` |
| [`pipeline/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/pipeline/README.md) | **Unified Pipeline Engine**: Core general multi-algorithm hyperparameter sweep and comparison framework | `sweep_pipeline.py`<br>`sweep_walkthrough.md` | Built-in `comparison/` ranking & plotting |
| [`single_run/`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/single_run/README.md) | **Single-Job Launcher**: Individual algorithm runs with checkpoints and videos | `run_slurm.sh` | Algorithm metrics |

---

## ⚡ Quick Start: Running from the Root

### Cluster SLURM Submission Examples
```bash
# Submit Brax E0 sweep across all 9 environments
sbatch scripts/brax_e0/run_slurm_brax_e0_sweep.sh

# Submit Brax E(lambda=0.9) sweep across all 9 environments
sbatch scripts/brax_e_lambda/run_slurm_brax_e_lambda_sweep.sh

# Submit TD vs. E_experimental comparison across 16 Gymnax environments
sbatch scripts/td_vs_e_experimental/run_slurm_td_vs_e_experimental.sh

# Submit E(lambda) vs TD(lambda) comparison across 16 Gymnax environments
sbatch scripts/td_vs_e_lambda/run_slurm_td_vs_e_lambda.sh

# Submit 4 E-variants suite across 19 Gymnax environments
sbatch scripts/e_variants/run_slurm_e_variants_suite.sh

# Submit MinAtar E-variants vs TD comparison
sbatch scripts/e_variants/run_slurm_minatar_e_vs_td.sh

# Submit Gymnax TD(lambda) spectrum sweep
sbatch scripts/ppo_td_spectrum/run_slurm_gymnax_suite.sh
```

### Local Execution / Single-Env Test Examples
```bash
# Run single environment locally
./scripts/brax_e0/run_slurm_brax_e0_sweep.sh hopper
./scripts/td_vs_e_experimental/run_slurm_td_vs_e_experimental.sh CartPole-v1
./scripts/td_vs_e_lambda/run_slurm_td_vs_e_lambda.sh MountainCarContinuous-v0
./scripts/e_optimization/run_slurm_gymnax_e_experimental.sh CartPole-v1
./scripts/e_variants/run_slurm_e_variants_suite.sh CartPole-v1
./scripts/ppo_td_spectrum/run_slurm_gymnax_suite.sh CartPole-v1
```

### Post-Experiment Suite Analysis Examples
```bash
# Compile Brax E0 master suite report
python scripts/brax_e0/generate_brax_e0_suite_pdf.py --suite-dir results/sweeps/brax_e0_<suite_id>

# Compile Classic TD vs E_experimental executive report
python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py --suite-dir results/ppo/sweeps/cmp_td_vs_e_<suite_id>

# Compile 4 E-variants suite report
python scripts/e_variants/generate_e_variants_suite_pdf.py --suite-dir results/ppo/sweeps/suite_e_variants_<suite_id>

# Compile PPO TD(lambda) spectrum suite report
python scripts/ppo_td_spectrum/generate_suite_pdf.py --suite-dir results/ppo/sweeps/suite_<suite_id>

# Run multi-dimensional analysis (parallel coordinates, ANOVA, heatmaps)
python scripts/e_optimization/visualize_multidim_sweep.py --suite-dir results/ppo/sweeps/e_experimental_gymnax_<suite_id>
```

---

## 🔄 Backward Compatibility
Root-level shims are maintained at [`scripts/sweep_pipeline.py`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/sweep_pipeline.py) and [`scripts/visualize_multidim_sweep.py`](file:///Users/dillonsandhu/Documents/Research/E_min/scripts/visualize_multidim_sweep.py) so legacy scripts, automated workflows, and cluster commands continue to run seamlessly.

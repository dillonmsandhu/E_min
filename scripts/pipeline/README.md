# ICML 2027 Two-Stage Tuning & Held-Out Evaluation Pipeline

This directory (`scripts/pipeline/`) implements the end-to-end **two-stage experimental pipeline** for all 5 benchmark suites—eliminating per-environment tuning, enforcing strict apples-to-apples critic parity, increasing held-out seed counts in Stage 2, and exporting a unified cross-suite CSV for all-environment diagnostic scatter plots.

---

## Scripts Map (Stage 1 vs. Stage 2)

| Stage | Python Script | SLURM Worker / Launcher | Purpose & Outputs |
| :--- | :--- | :--- | :--- |
| **Shared Registry** | `scripts/pipeline/suite_registry.py` | — | Suite specs (`brax`, `mjx_locomotion`, `octax`, `minatar`, `gymnax`), parity enforcement, chunked vmap runner, ablation inheritance. |
| **Stage 1 (Tune)** | `scripts/pipeline/stage1_tune.py` | `scripts/pipeline/run_stage1_worker.sh` | Grid search (`CRITIC_LR × CRITIC_EPOCHS × MAX_GRAD_NORM [× λ]`) on tuning seeds (`master_seed=42`, 3 seeds).<br>Writes `results/pipeline/<suite>/stage1_tune/<env>/<algo>/sweep_metrics_all.pkl`. |
| **Stage 1 $\to$ 2 (Select)** | `scripts/pipeline/select_configs.py` | *(chained CPU job in `submit_pipeline.sh`)* | Computes suite-wide normalized AUC IQM and exports global + per-$\lambda$ winners.<br>Writes `configs/tuned/<suite>/<algo>.json`, `<algo>_lam<λ>.json`, and `selection_summary.md`. |
| **Stage 2 (Held-Out Eval)** | `scripts/pipeline/stage2_eval.py` | `scripts/pipeline/run_stage2_worker.sh` | Loads cached JSONs from `configs/tuned/<suite>/` (zero per-env tuning) and runs disjoint held-out seeds (`master_seed=1000`, 6–10 seeds).<br>Writes `results/pipeline/<suite>/stage2_eval/{global,per_lambda}/<env>/data/sweep_metrics_all.pkl`. |
| **Stage 2 (Consolidate)** | `scripts/pipeline/stage2_consolidate.py` | *(chained CPU job in `submit_pipeline.sh`)* | Merges per-env Stage 2 pickles, generates championship profiles / per-$\lambda$ plots / diagnostics, and writes `results/pipeline/<suite>/stage2_eval/data/cross_suite_summary.csv`. |
| **All-Suite Paper Plots** | `scripts/pipeline/compile_cross_suite.py` | *(run locally or on CPU node)* | Compiles all suites into `results/pipeline/cross_suite/all_suites_summary.csv` and unified `r_rot,c` / `ρ_c` scatter PDFs. |
| **Full Orchestrator** | — | `scripts/pipeline/submit_pipeline.sh` | Submits the entire chained SLURM workflow (`Stage 1 -> Select -> Stage 2 -> Consolidate`) in one command. Supports `--stage2-only` and `--dry-run`. |

---

## Direct Stage 1 & Stage 2 Commands

### Run Full Chained Pipeline (Stage 1 $\to$ Select $\to$ Stage 2 $\to$ Consolidate)
```bash
./scripts/pipeline/submit_pipeline.sh <suite>          # brax | mjx_locomotion | octax | minatar | gymnax
./scripts/pipeline/submit_pipeline.sh <suite> --dry-run
```

### Run Stage 1 Directly (Single `(suite, env, algo)`)
```bash
# Via SLURM worker (local or sbatch):
SUITE=brax ./scripts/pipeline/run_stage1_worker.sh hopper td_lambda

# Or directly in Python:
python scripts/pipeline/stage1_tune.py --suite brax --envs hopper --algos td_lambda
```

### Run Config Selection Directly (Stage 1 $\to$ Stage 2)
```bash
JAX_PLATFORMS=cpu python scripts/pipeline/select_configs.py --suite brax
```

### Run Stage 2 Directly (Using Cached Configs in `configs/tuned/<suite>/`)
```bash
# Submit Stage 2 + Consolidate on SLURM (skipping Stage 1):
./scripts/pipeline/submit_pipeline.sh brax --stage2-only

# Single env via SLURM worker:
SUITE=brax ./scripts/pipeline/run_stage2_worker.sh hopper

# Or directly in Python:
python scripts/pipeline/stage2_eval.py --suite brax --envs hopper
JAX_PLATFORMS=cpu python scripts/pipeline/stage2_consolidate.py --suite brax
```

### Adding a New Algorithm Later (e.g. `tdrc` or New Inherited $E$ Variant)
```bash
# 1) New tuned baseline (runs Stage 1 + Select ONLY for tdrc, then Stage 2 caches/skips existing algos):
STAGE1_ALGOS="tdrc" STAGE2_ALGOS="td_0 e_0_v_aux td_lambda e_lambda_v_aux tdrc" \
  ./scripts/pipeline/submit_pipeline.sh brax

# 2) New inherited ablation (already mapped in ABLATION_INHERITANCE, zero Stage 1 needed):
STAGE2_ALGOS="td_0 e_0_v_aux td_lambda e_lambda_v_aux bsmc_v_aux e_0_rot_gated e_lambda_rot_gated" \
  ./scripts/pipeline/submit_pipeline.sh brax --stage2-only
```

---

## Architecture Overview

```
Stage 1 (GPU)          Select (CPU)              Stage 2 (GPU)                 Consolidate (CPU)
───────────────────    ──────────────────────    ──────────────────────────    ───────────────────────────
stage1_tune.py   ───►  select_configs.py   ───►  stage2_eval.py          ───►  stage2_consolidate.py
(tuning seeds 42,      (writes global &          (disjoint seeds 1000,         (championship profiles,
 3 seeds, full grid)    per-λ JSON configs        5–10 held-out seeds,          per-λ plots, diagnostics,
                        to configs/tuned/)        ZERO per-env tuning)          cross_suite_summary.csv)
                                                                                        │
                                                                                        ▼
                                                                               compile_cross_suite.py
                                                                               (ALL suites combined:
                                                                                r_rot,c & ρ_c scatters)
```

### Enforced Parity Invariants (`suite_registry.py`)
Every run across all 5 suites automatically enforces:
1. **`VF_COEF = 1.0`** at both top-level and `NETWORK_CONFIG["VF_COEF"]` (eliminates the `0.5` vs `1.0` mismatch).
2. **`VF_CLIP = None`** (standard fitted $\mathrm{TD}(\lambda)$ without heuristic PPO value clipping).
3. **`SPLIT_AUX_OPTIMIZER = True`** (dedicated optimizer & gradient clipping for `aux_loss` / $v_{\text{aux}}$).
4. **`MAX_GRAD_NORM ∈ {1.0, 10.0}`** swept identically across every tuned algorithm and synchronized into `NETWORK_CONFIG["MAX_GRAD_NORM"]` (and included in MJX's `static_compile_signature`).

### Algorithms
- **Separately Tuned in Stage 1 (`TUNED_BASE_ALGOS`):**
  1. `td_0` — Online $\mathrm{TD}(0)$
  2. `e_0_v_aux` — Sampled $E(0)$ with surgical $v_{\text{aux}}$
  3. `td_lambda` — Fitted $\mathrm{TD}(\lambda)$ (standard PPO critic)
  4. `e_lambda_v_aux` — Sampled $E(\lambda)$ with surgical $v_{\text{aux}}$
- **Inherited Ablations in Stage 2 (`ABLATION_INHERITANCE`, zero extra tuning):**
  - `bsmc_v_aux` $\leftarrow$ inherits `e_lambda_v_aux.json` (and per-$\lambda$ JSONs)
  - `e_0_rot_gated` $\leftarrow$ inherits `e_0_v_aux.json`
  - `e_lambda_rot_gated` $\leftarrow$ inherits `e_lambda_v_aux.json`

---

## Per-Suite Pipeline Reference

| Suite | Tune Envs (Stage 1) | Eval Envs (Stage 2) | Stage 1 Grid | Seeds (S1 $\to$ S2 [chunk]) | SLURM Defaults (S1 / S2, CPUs, RAM) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`brax`** | 10 classic continuous control envs | Same 10 envs | `clr ∈ {3e-4, 1e-3, 3e-3}` × `cep ∈ {4, 8}` × `gn ∈ {1, 10}` × `λ ∈ {0.4, 0.8, 0.95, 1.0}` | `3 -> 10 [5]` | `10h / 5h`, 2 CPUs, 24G, `a5000` |
| **`mjx_locomotion`** | 5 Playground locomotion envs (no Brax overlap, $\gamma=0.97$) | Same 5 envs | `clr ∈ {3e-4, 1e-3}` × `cep ∈ {4, 8}` × `gn ∈ {1, 10}` × `λ ∈ {0.6, 0.8, 0.95, 1.0}` | `3 -> 6 [3]` | `20h / 8h`, 2 CPUs, 32G, `a5000` (`a6000` for G1/Humanoid) |
| **`octax`** | 5 representative visual envs (`pong`, `brix`, `tetris`, `missile`, `tank`) | All 8 Octax envs (3 held-out envs!) | `clr ∈ {1e-4, 3e-4, 1e-3}` × `cep ∈ {4, 8}` × `gn ∈ {1, 10}` × `λ ∈ {0.4, 0.8, 0.95, 1.0}` | `3 -> 10 [5]` | `6h / 3h`, 2 CPUs, 20G, `a5000` |
| **`minatar`** | All 4 MinAtar envs (`Asterix`, `Breakout`, `Freeway`, `SpaceInvaders`) | Same 4 envs | `clr ∈ {1e-4, 3e-4, 1e-3}` × `cep ∈ {4, 8}` × `gn ∈ {1, 10}` × `λ ∈ {0.4, 0.8, 0.95, 1.0}` | `3 -> 10 [5]` | `6h / 3h`, 2 CPUs, 20G, `a5000` |
| **`gymnax`** *(diagnostic)* | 12 non-MinAtar Gymnax envs (`Acrobot`, `MountainCar`, etc.) | Same 12 envs | `clr ∈ {1e-4, 3e-4, 1e-3}` × `cep ∈ {4, 8}` × `gn ∈ {1, 10}` × `λ ∈ {0.4, 0.8, 0.95, 1.0}` | `3 -> 10 [10]` | `2h / 1h`, 2 CPUs, 16G, `a5000` |

---

## Common Workflows

### 1. Dry-Run (Verify Exact SLURM Dependency Chain Before Submitting)
```bash
./scripts/pipeline/submit_pipeline.sh brax --dry-run
```

### 2. Run Full Chained Pipeline (Stage 1 $\to$ Select $\to$ Stage 2 $\to$ Plots)
```bash
./scripts/pipeline/submit_pipeline.sh brax
./scripts/pipeline/submit_pipeline.sh mjx_locomotion
./scripts/pipeline/submit_pipeline.sh octax
./scripts/pipeline/submit_pipeline.sh minatar
./scripts/pipeline/submit_pipeline.sh gymnax
```

### 3. Re-Run Stage 2 Only (e.g. Testing New Ablations or More Seeds Using Cached Configs)
```bash
STAGE2_SEEDS=10 STAGE2_ALGOS="td_0 e_0_v_aux td_lambda e_lambda_v_aux bsmc_v_aux" \
  ./scripts/pipeline/submit_pipeline.sh brax --stage2-only
```

### 4. Local CPU Smoke Test (1 Env, 1 Seed, Short Timesteps)
```bash
python scripts/pipeline/stage1_tune.py \
  --suite gymnax --envs CartPole-v1 --algos td_0 e_0_v_aux td_lambda e_lambda_v_aux \
  --critic_lrs 0.0003 --critic_epochs 4 --max_grad_norms 1.0 --lambdas 0.8 \
  --n_seeds 1 --total_timesteps 16384 --out_root /tmp/test_pipe

python scripts/pipeline/select_configs.py \
  --suite gymnax --envs CartPole-v1 --lambdas 0.8 \
  --out_root /tmp/test_pipe --configs_dir /tmp/test_pipe/configs

python scripts/pipeline/stage2_eval.py \
  --suite gymnax --envs CartPole-v1 --lambdas 0.8 \
  --n_seeds 2 --seed_chunk_size 2 --total_timesteps 16384 \
  --configs_dir /tmp/test_pipe/configs --out_root /tmp/test_pipe

python scripts/pipeline/stage2_consolidate.py \
  --suite gymnax --out_root /tmp/test_pipe
```

### 5. Compile All-Suite Cross-Benchmark Scatters (Paper Figures)
Once two or more suites finish Stage 2, compile every environment across all suites into unified vector PDF scatter plots (`r_rot,c` vs. relative performance, `ρ_c` vs. relative performance, 1-step `E(0)` vs. `TD(0)`, and matched-$\lambda$):
```bash
python scripts/pipeline/compile_cross_suite.py --out_root results/pipeline
```
Outputs are written to `results/pipeline/cross_suite/`:
- `all_suites_summary.csv`
- `scatter_r_rot_vs_e_lambda_over_td_lambda.pdf`
- `scatter_rho_c_vs_e_lambda_over_td_lambda.pdf`
- `scatter_r_rot_vs_e0_over_td0.pdf`
- `scatter_r_rot_vs_e08_over_td08.pdf`

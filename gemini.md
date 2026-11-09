# RL Research Assistant Guidelines (JAX + Gymnax + Brax)

## 0. Project Architecture & Sibling R&D Repo (`bellman_error`)
- **Target Repository (`E_min`)**: Likely final repository for this project, hosting the pure JAX Gymnax and upcoming Brax implementations, benchmarking pipelines, and cluster sweeps.
- **R&D Sandbox (`bellman_error`)**: Located at `/Users/dillonsandhu/Documents/Research/bellman_error` (accessible via symlink `external_bellman_error`). Focuses on algorithmic formulation, mathematical proofs, tabular environments, and diagnostics.
- **Core Asset in `bellman_error`**: Contains the strong, verified implementation of the **E-minimization algorithm using `v_aux`** (auxiliary value network / debiased sampled E-min in `random_policy/`, `ppo/sampled_E.py`, `core/sampling_e_metrics.py`). Port and adapt algorithms from `bellman_error` into `E_min`.

## 1. Development & Execution Environment
- **Local machine**: Used ONLY for code editing, git commits, and fast CPU smoke tests.
- **Remote HPC Cluster**: Used for full Slurm GPU sweeps (`sbatch scripts/...`), which you don't have access to. The user will run code on remote cluster.
- **Safety Rule**: Avoid running large jobs locally - i.e. performing sweeps. However, you can run the algorithms with the standard config on most envs for testing purposes.

## 2. Configuration & Parameter Contract
- Base configuration lives in `core/config.py` as a dictionary.
- Config overrides are passed as JSON strings or kwargs and merged via `core.utils.merge_hparams`.
- Any new hyperparameter or flag MUST be added to `core/config.py` with sensible default types.
- Never hardcode environment step counts or network architectures inside training loops; always draw them from `config`.

## 3. JAX & Gymnax Standards
- Strictly adhere to functional purity: no in-place array mutation, no impure Python side effects in jitted loops.
- We want the code to run gymnax and RL as fast as possible on GPUs, with minimial memory overhead.
- Use `jax.lax.scan` for rollout loops and training epochs instead of Python `for` loops.
- Handle PRNG keys explicitly (`rng, _rng = jax.random.split(rng)`).
- Use standard Gymnax API conventions.
- Vectorize environments and seeds using `jax.vmap` over leading axes.

- This repo follows an existing pipeline for running sweeps. Try to match it as closely as possible.
- Sweep scripts are organized into modular folders under `scripts/` (e.g. `scripts/brax_e0/`, `scripts/td_vs_e_experimental/`, `scripts/td_vs_e_lambda/`, `scripts/pipeline/`).
- Sweeps follow the structure in `core/sweep.py` and `scripts/pipeline/sweep_pipeline.py` (with a root shim at `scripts/sweep_pipeline.py`).
- Sweep outputs (metrics, plots, CSV summaries) should always be written to `results/sweeps/<sweep_id>/` or `results/ppo/sweeps/<sweep_id>/`.

## 5. Coding & Style Rules
- Type annotate core runner functions and JAX state containers (use `flax.struct.dataclass` / NamedTuples for step states).
- Keep algorithm implementations modular and self-contained in `core/` or `algos/`.
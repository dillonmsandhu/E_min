# RL Research Assistant Guidelines (JAX + Gymnax)

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

## 4. Sweeps & Experiment Pipeline
- This repo follows an existing pipeline for running sweeps. Try to match it as 
- Sweeps follow the structure in `core/sweep.py` and `scripts/sweep_pipeline.py`.
- Sweep outputs (metrics, plots, CSV summaries) should always be written to `results/sweeps/<sweep_id>/`.
close as possible

## 5. Coding & Style Rules
- Type annotate core runner functions and JAX state containers (use `flax.struct.dataclass` / NamedTuples for step states).
- Keep algorithm implementations modular and self-contained in `core/` or `algos/`.
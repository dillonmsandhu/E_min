# Single-Run SLURM Launcher

This folder contains the script for launching **single-job training runs** for any individual algorithm in `algos/` on a given environment.

---

## 📁 Files in this Folder

| File | Type | Description |
| :--- | :--- | :--- |
| `run_slurm.sh` | Shell / SLURM | Submits or runs a single algorithm training job on a single GPU. |

---

## 🚀 How to Run from the Root

### Syntax
```bash
# SLURM
sbatch scripts/single_run/run_slurm.sh <ALGO_NAME> <RUN_SUFFIX> <ENV_NAME> [E_LAMBDA]

# Local execution
./scripts/single_run/run_slurm.sh <ALGO_NAME> <RUN_SUFFIX> <ENV_NAME> [E_LAMBDA]
```

Arguments:
1. `ALGO_NAME`: Filename inside `algos/` (without `.py`), e.g., `E`, `E_experimental`, `ppo`, `td`, `E_lambda_fixed`.
2. `RUN_SUFFIX`: Descriptive identifier tag for the run (e.g. `baseline_test`, `seed42`).
3. `ENV_NAME`: Gymnax or MinAtar environment name (e.g. `CartPole-v1`, `Asterix-MinAtar`).
4. `E_LAMBDA` *(optional, default 0.0)*: Value for `E_LAMBDA` and `VALUE_LAMBDA`.

### Examples

```bash
# Run 1-step E on Asterix-MinAtar
sbatch scripts/single_run/run_slurm.sh E test_run Asterix-MinAtar 0.0

# Run E_experimental on CartPole-v1 locally
./scripts/single_run/run_slurm.sh E_experimental smoke_test CartPole-v1 0.0

# Run TD(lambda=0.9) on MountainCarContinuous-v0
sbatch scripts/single_run/run_slurm.sh td lambda_run MountainCarContinuous-v0 0.9
```

---

## 📂 Where Output is Saved

Single-run outputs (checkpoints, videos, and metrics) are written according to the algorithm's output specification:
- Metrics: `results/<algo>/...`
- Checkpoints: `checkpoints/<algo>_<env>_<suffix>/`
- Videos: `videos/<algo>_<env>_<suffix>/`

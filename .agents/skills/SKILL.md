---
name: slurm-sweep
description: Prepare, validate, and launch Slurm sweeps for Gymnax JAX experiments.
---

# Slurm Sweep Workflow

When asked to configure, launch, or debug a hyperparameter sweep:

1. **Local Smoke Test First**:
   - Run a 1-step, 1-seed test on CPU before recommending or committing cluster script changes:
     ```bash
     JAX_PLATFORMS=cpu python scripts/sweep_pipeline.py \
       --policy random --env-name CartPole-v1 --algos td0 \
       --lr-grid "0.001" --n-seeds 1 \
       --config '{"NUM_ENVS": 2, "NUM_STEPS": 4, "TOTAL_TIMESTEPS": 8, "LIGHT_METRICS": true}'
     ```

2. **Cluster Script Generation / Modification**:
   - Ensure `scripts/run_slurm_*.sh` sets appropriate partition, GPU allocation, memory, and runtime limits.
   - Use array jobs or bash loops according to cluster queue limits.
   - Save output logs to `slurm/%DATE/%j.out`.

3. **Verify Git Sync Readiness**:
   - Ensure no large data files or local checkpoint paths are hardcoded.
   - Verify all paths in the Slurm scripts match the remote environment paths.
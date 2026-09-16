#!/bin/bash

# Submit sweeps for learning rate decay comparison
# TD (No Decay vs Decay) and MC (No Decay vs Decay)
# Uses the static constant memory optimization to safely run 10 seeds without OOM!

echo "Submitting Exact TD (No Decay) Sweep..."
sbatch slurm_lr_decay_sweeps.sh random_policy.exact_td cline_tune_no_decay null 5

echo "Submitting Exact TD (Decay) Sweep..."
sbatch slurm_lr_decay_sweeps.sh random_policy.exact_td cline_tune_decay 0.0 5

echo "Submitting Exact MC (No Decay) Sweep..."
sbatch slurm_lr_decay_sweeps.sh random_policy.exact_mc cline_tune_no_decay null 5

echo "Submitting Exact MC (Decay) Sweep..."
sbatch slurm_lr_decay_sweeps.sh random_policy.exact_mc cline_tune_decay 0.0 5

echo "All jobs submitted. Monitor with squeue."

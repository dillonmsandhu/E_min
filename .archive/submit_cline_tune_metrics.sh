#!/bin/bash

# Submit slurm_sweep.sh for random_policy.exact_td and random_policy.exact_mc with suffix cline_tune_metrics and LOG_FEATURE_METRICS=True

echo "Submitting exact_td with metrics logging and suffix cline_tune_metrics..."
sbatch slurm_sweep.sh random_policy.exact_td cline_tune_metrics True

echo "Submitting exact_mc with metrics logging and suffix cline_tune_metrics..."
sbatch slurm_sweep.sh random_policy.exact_mc cline_tune_metrics True

echo "All feature metric jobs submitted."

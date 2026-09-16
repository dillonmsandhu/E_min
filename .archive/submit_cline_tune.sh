#!/bin/bash

# Submit slurm_sweep.sh for random_policy.exact_td and random_policy.exact_mc with suffix cline_tune

echo "Submitting exact_td sweep with suffix cline_tune..."
sbatch slurm_sweep.sh random_policy.exact_td cline_tune

echo "Submitting exact_mc sweep with suffix cline_tune..."
sbatch slurm_sweep.sh random_policy.exact_mc cline_tune

echo "All jobs submitted."

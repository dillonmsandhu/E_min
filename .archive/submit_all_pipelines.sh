#!/bin/bash

# Launch pipeline slurm script for each algorithm

echo "Launching pipeline for random_policy.exact_td..."
sbatch /usr/xtmp/ds541/bellman_error/slurm_pipeline.sh random_policy.exact_td

echo "Launching pipeline for random_policy.exact_mc..."
sbatch /usr/xtmp/ds541/bellman_error/slurm_pipeline.sh random_policy.exact_mc

echo "Launching pipeline for random_policy.exact_E_gd..."
sbatch /usr/xtmp/ds541/bellman_error/slurm_pipeline.sh random_policy.exact_E_gd


echo "All algorithm pipelines submitted."

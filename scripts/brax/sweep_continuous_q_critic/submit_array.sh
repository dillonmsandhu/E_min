#!/bin/bash
# Submit SLURM array job across all 6 deterministic continuous Brax environments
# and schedule the post-processing plot/table consolidation upon array completion.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

mkdir -p slurm

OUT_DIR="${OUT_DIR:-results/sweep_continuous_q_critic_brax}"
export OUT_DIR

echo "Submitting Brax Continuous Q-Critic Sweep Array (0-5)..."
ARRAY_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/run_slurm_array.sh")
echo "Submitted array job: $ARRAY_JOB_ID"

echo "Scheduling consolidation post-processing dependency..."
sbatch --dependency=afterok:"$ARRAY_JOB_ID" \
       --job-name="cons_cont_q_brax" \
       --output="slurm/consolidate_%j.out" \
       --error="slurm/consolidate_%j.err" \
       --time=01:00:00 \
       --partition=compsci \
       --cpus-per-task=2 \
       --mem=16G \
       --wrap="python3 scripts/brax/sweep_continuous_q_critic/plot_consolidated_results.py --out_dir '$OUT_DIR'"

echo "All jobs queued successfully."

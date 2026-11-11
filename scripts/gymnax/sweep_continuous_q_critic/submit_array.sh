#!/bin/bash
# Submit SLURM array job across all 4 deterministic continuous Gymnax environments
# and schedule the post-processing plot/table consolidation upon array completion.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

mkdir -p slurm

OUT_DIR="${OUT_DIR:-results/sweep_continuous_q_critic_gymnax}"
export OUT_DIR

echo "Submitting Gymnax Continuous Q-Critic Sweep Array (0-3)..."
ARRAY_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/run_slurm_array.sh")
echo "Submitted array job: $ARRAY_JOB_ID"

echo "Scheduling consolidation post-processing dependency..."
sbatch --dependency=afterok:"$ARRAY_JOB_ID" \
       --job-name="cons_cont_q_gymnax" \
       --output="slurm/consolidate_%j.out" \
       --error="slurm/consolidate_%j.err" \
       --time=01:00:00 \
       --partition=compsci \
       --cpus-per-task=2 \
       --mem=16G \
       --wrap="python3 scripts/gymnax/sweep_continuous_q_critic/plot_consolidated_results.py --out_dir '$OUT_DIR'"

echo "All jobs queued successfully."

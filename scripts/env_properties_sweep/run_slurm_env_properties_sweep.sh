#!/bin/bash
#SBATCH --job-name=env_props_sweep
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1

# ==============================================================================
# Environmental Properties Sweep: E(0) vs. TD(0) vs. TD(lambda)
#
# Array Tasks:
#   0: MountainCar Family (Sparse Continuous, Dense Continuous, Dense Discrete)
#   1: PointRobot Family  (Sparse Continuous, Dense Continuous, Dense Discrete)
#
# Both evaluated under:
#   - Clean dynamics
#   - Noisy dynamics (5% tire slip, 50% force scale, transition noise)
#
# Usage:
#   sbatch scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh
#   ./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 0   # Run MountainCar locally
#   ./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 1   # Run PointRobot locally
# ==============================================================================

# Ensure we are in the repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../.."

REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm

# Python interpreter discovery with fallback chain
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/shims/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/shims/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Determine Task ID
TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"

FAMILIES=("mountain_car" "point_robot")
SELECTED_FAMILY="${FAMILIES[$TASK_ID]}"

# Unify Sweep ID to prevent divergent timestamp folders across nodes
SWEEP_ID="${SWEEP_ID:-${SLURM_ARRAY_JOB_ID:+env_props_${SLURM_ARRAY_JOB_ID}}}"
SWEEP_ID="${SWEEP_ID:-env_props_${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}}"

echo "======================================================================"
echo "Job ID:           ${SLURM_JOB_ID:-LOCAL}"
echo "Array Task:       $TASK_ID"
echo "Target Family:    $SELECTED_FAMILY"
echo "Sweep ID:         $SWEEP_ID"
echo "Python:           $PYTHON"
echo "Working Dir:      $(pwd)"
echo "======================================================================"

$PYTHON scripts/env_properties_sweep/sweep_env_properties.py \
    --env-family "$SELECTED_FAMILY" \
    --sweep-id "$SWEEP_ID" \
    --total-timesteps 2048000 \
    --n-seeds 8 \
    --num-epochs 16 \
    --slip-prob 0.05 \
    --slip-force-scale 0.5 \
    --transition-noise 0.001

echo "Task $TASK_ID for $SELECTED_FAMILY completed successfully!"

#!/bin/bash
# ==============================================================================
# scripts/jumanji/submit_tune_actor.sh
#
# Submits a Slurm batch job for PPO Actor hyperparameter tuning across discrete
# Jumanji tasks using an unbiased Monte Carlo critic (lambda=1.0).
#
# Target Partition & Resources:
#   - Partition: compsci-gpu
#   - GRES:      gpu:a5000:1 (RTX A5000 24GB)
#   - CPUs/Mem:  4 cores, 32GB
#
# Usage:
#   # Fast coordinate tuning (default: 8 runs per env, 2 seeds)
#   bash scripts/jumanji/submit_tune_actor.sh coordinate
#
#   # Full grid search (24 runs per env, 3 seeds)
#   bash scripts/jumanji/submit_tune_actor.sh grid 3
# ==============================================================================

set -e

# Ensure execution from repo root
[ -f "jumanji_rl/core/jumanji_config.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
MODE=${1:-coordinate}
N_SEEDS=${2:-2}
NUM_STEPS=${NUM_STEPS:-256}
NUM_ENVS=${NUM_ENVS:-128}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-500000}
OUT_JSON=${OUT_JSON:-"scripts/jumanji/jumanji_best_actor_config.json"}

LOG_DIR="slurm/${DATE}/tune_jumanji_actor"
mkdir -p "$LOG_DIR"

# Python interpreter discovery
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

echo "================================================================"
echo "Submitting Jumanji Actor Tuning (MC Critic Grounded)"
echo "  Mode:            $MODE"
echo "  Seeds:           $N_SEEDS"
echo "  Rollout Horizon: $NUM_STEPS steps (NUM_ENVS=$NUM_ENVS)"
echo "  Timesteps:       $TOTAL_TIMESTEPS per condition"
echo "  Target Config:   $OUT_JSON"
echo "  Log Directory:   $LOG_DIR"
echo "  Python:          $PYTHON"
echo "================================================================"

JOB_ID=$(sbatch --parsable \
    --job-name="tune_actor_jumanji" \
    --output="${LOG_DIR}/%j_${MODE}.out" \
    --error="${LOG_DIR}/%j_${MODE}.err" \
    --time=06:00:00 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --chdir="$REPO_ROOT" \
    --wrap="export PYTHONPATH='$REPO_ROOT:\${PYTHONPATH}' XLA_PYTHON_CLIENT_PREALLOCATE=false; $PYTHON -u scripts/jumanji/tune_actor_mc.py --mode $MODE --num_steps $NUM_STEPS --num_envs $NUM_ENVS --total_timesteps $TOTAL_TIMESTEPS --n_seeds $N_SEEDS --out_json '$OUT_JSON'")

echo "Submitted Batch Job ID: $JOB_ID"
echo "Log file:               ${LOG_DIR}/${JOB_ID}_${MODE}.out"

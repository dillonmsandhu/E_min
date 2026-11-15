#!/bin/bash
# ==============================================================================
# E_min/scripts/brax/sweep_q_ppo_vs_baselines/submit_array.sh
#
# Submits the 11-environment SLURM array comparing and tuning:
#   1. TD(lambda) (PPO Fitted)
#   2. Q-PPO E(0) (Bilinear + Hybrid GAE)
#   3. E(lambda) (v_aux PPO)
# Automatically chains plot_consolidated_results.py upon array completion.
#
# Usage:
#   bash scripts/brax/sweep_q_ppo_vs_baselines/submit_array.sh
#   bash scripts/brax/sweep_q_ppo_vs_baselines/submit_array.sh [MODE] [TOTAL_TIMESTEPS] [N_SEEDS] [OUT_DIR]
# ==============================================================================

set -e

[ -f "brax_rl/core/q_ppo_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sweep_q_ppo_vs_baselines_brax"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

MODE=${1:-"tune_coordinate"}
TOTAL_TIMESTEPS=${2:-"auto"}
N_SEEDS=${3:-3}
OUT_DIR=${4:-"results/sweep_q_ppo_vs_baselines_brax"}
MASTER_SEED=${MASTER_SEED:-42}
NUM_ENVS=${NUM_ENVS:-512}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}

export MODE TOTAL_TIMESTEPS N_SEEDS MASTER_SEED NUM_ENVS NUM_STEPS MINIBATCH_SIZE OUT_DIR REPO_ROOT

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
echo "Submitting Brax Q-PPO vs Baselines SLURM Array"
echo "  Mode:              $MODE"
echo "  Timesteps:         $TOTAL_TIMESTEPS"
echo "  Seeds:             $N_SEEDS (Master: $MASTER_SEED)"
echo "  Rollout:           $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Output Directory:  $OUT_DIR"
echo "  Log Directory:     $LOG_DIR"
echo "  Python:            $PYTHON"
echo "================================================================"

ARRAY_SCRIPT="scripts/brax/sweep_q_ppo_vs_baselines/run_slurm_array.sh"

ARRAY_JOB_ID=$(sbatch --parsable \
    --chdir="$REPO_ROOT" \
    --output="${LOG_DIR}/%A_%a.out" \
    --error="${LOG_DIR}/%A_%a.err" \
    --export=ALL,MODE="$MODE",TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS",N_SEEDS="$N_SEEDS",MASTER_SEED="$MASTER_SEED",NUM_ENVS="$NUM_ENVS",NUM_STEPS="$NUM_STEPS",MINIBATCH_SIZE="$MINIBATCH_SIZE",OUT_DIR="$OUT_DIR",REPO_ROOT="$REPO_ROOT" \
    "$ARRAY_SCRIPT")

echo "Submitted Array Job ID: $ARRAY_JOB_ID"

POST_SCRIPT="scripts/brax/sweep_q_ppo_vs_baselines/plot_consolidated_results.py"
POST_JOB_ID=$(sbatch --parsable \
    --job-name=plot_qppo_cmp \
    --output="${LOG_DIR}/%A_consolidated_plot.out" \
    --error="${LOG_DIR}/%A_consolidated_plot.err" \
    --time=01:00:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir="$REPO_ROOT" \
    --wrap="$PYTHON $POST_SCRIPT --out_dir '$OUT_DIR' --num_envs $NUM_ENVS --num_steps $NUM_STEPS")

echo "Submitted Consolidated Post-Processing Job ID: $POST_JOB_ID"
echo "Logs written to: $LOG_DIR/"

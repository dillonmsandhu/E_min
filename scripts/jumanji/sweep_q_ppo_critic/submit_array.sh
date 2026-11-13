#!/bin/bash
# ==============================================================================
# scripts/jumanji/sweep_q_ppo_critic/submit_array.sh
#
# Launches the 3-environment SLURM array job for the Q-PPO critic sweep
# across discrete Jumanji domains (Sokoban, Snake, Game2048).
# Automatically chains the consolidated post-processing job to run upon completion.
#
# Usage:
#   bash scripts/jumanji/sweep_q_ppo_critic/submit_array.sh
#   bash scripts/jumanji/sweep_q_ppo_critic/submit_array.sh [TOTAL_TIMESTEPS] [N_SEEDS] [OUT_DIR]
# ==============================================================================

set -e

# Ensure working directory is E_min repo root
[ -f "jumanji_rl/core/jumanji_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sweep_q_ppo_critic_jumanji"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-3}
OUT_DIR=${3:-"results/sweep_q_ppo_critic_jumanji"}
MASTER_SEED=${MASTER_SEED:-42}
CRITIC_LRS=${CRITIC_LRS:-"0.0001 0.0003 0.001"}
ACTOR_LR=${ACTOR_LR:-0.0003}
TD_LAMBDAS=${TD_LAMBDAS:-"0.0 0.4 0.8 0.95 1.0"}
NUM_ENVS=${NUM_ENVS:-256}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-512}

export TOTAL_TIMESTEPS N_SEEDS MASTER_SEED CRITIC_LRS ACTOR_LR TD_LAMBDAS NUM_ENVS NUM_STEPS MINIBATCH_SIZE OUT_DIR REPO_ROOT

# Resolve Python interpreter
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
echo "Submitting Jumanji Q-PPO Critic SLURM Array"
echo "  Timesteps:         $TOTAL_TIMESTEPS (auto per env)"
echo "  Seeds:             $N_SEEDS (Master: $MASTER_SEED)"
echo "  Critic LRs:        $CRITIC_LRS"
echo "  Actor LR:          $ACTOR_LR"
echo "  TD Lambdas:        $TD_LAMBDAS"
echo "  Rollout:           $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Output Directory:  $OUT_DIR"
echo "  Log Directory:     $LOG_DIR"
echo "  Python:            $PYTHON"
echo "  Repo Root:         $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/jumanji/sweep_q_ppo_critic/run_slurm_array.sh"

SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/array_%A_%a.out\" \
    --error=\"${LOG_DIR}/array_%A_%a.err\" \
    --export=ALL,TOTAL_TIMESTEPS=\"$TOTAL_TIMESTEPS\",N_SEEDS=\"$N_SEEDS\",MASTER_SEED=\"$MASTER_SEED\",CRITIC_LRS=\"$CRITIC_LRS\",ACTOR_LR=\"$ACTOR_LR\",TD_LAMBDAS=\"$TD_LAMBDAS\",NUM_ENVS=\"$NUM_ENVS\",NUM_STEPS=\"$NUM_STEPS\",MINIBATCH_SIZE=\"$MINIBATCH_SIZE\",OUT_DIR=\"$OUT_DIR\" \
    $ARRAY_SCRIPT"

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted SLURM Array Job ID: $ARRAY_JOB_ID"

# Chained post-processing consolidation job
POST_CMD="sbatch \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --job-name=post_q_ppo_jumanji \
    --partition=compsci-gpu \
    --gres=gpu:a5000:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/post_%j.out\" \
    --error=\"${LOG_DIR}/post_%j.err\" \
    --wrap=\"export PYTHONPATH=\\\"$REPO_ROOT:\\\$PYTHONPATH\\\"; $PYTHON scripts/jumanji/sweep_q_ppo_critic/plot_consolidated_results.py --out_dir \\\"$OUT_DIR\\\" --num_envs $NUM_ENVS --num_steps $NUM_STEPS\""

POST_JOB_ID=$(eval "$POST_CMD" | awk '{print $NF}')
echo "Submitted Post-Processing Job ID: $POST_JOB_ID (depends on afterok:$ARRAY_JOB_ID)"
echo "Logs saved to: $LOG_DIR"

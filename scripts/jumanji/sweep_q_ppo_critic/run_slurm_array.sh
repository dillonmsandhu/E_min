#!/bin/bash
#SBATCH --job-name=q_ppo_jumanji
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5

# ==============================================================================
# Q-PPO Critic Comparison Sweep: Jumanji Per-Environment Array Task
# Runs all 5 Q-PPO Critic analysis pipelines on the designated discrete environment:
#   1. E(0) Variants (Naive, v_aux, Forward-Only)
#   2. E(lambda=0.8) Variants (v_aux Debiased, Naive, Forward-Only)
#   3. TD(lambda) Fitted SARSA Grid (0.0, 0.4, 0.8, 0.95, 1.0)
#
# Environments:
#   0: Sokoban-v0 (2M steps)
#   1: Snake-v1 (2M steps)
#   2: Maze-v0 (2M steps)
#   3: SlidingTilePuzzle-v0 (2M steps)
#   4: RubiksCube-partly-scrambled-v0 (2M steps)
#   5: Game2048-v1 (2M steps)
# ==============================================================================

set -e

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "jumanji_rl/core/jumanji_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm

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

ALL_ENVS=(
    "Sokoban-v0"
    "Snake-v1"
    "Maze-v0"
    "SlidingTilePuzzle-v0"
    "RubiksCube-partly-scrambled-v0"
    "Game2048-v1"
)

# Select environment from CLI argument or SLURM_ARRAY_TASK_ID
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        ENV_NAME="${ALL_ENVS[$1]}"
    else
        ENV_NAME="$1"
    fi
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="Sokoban-v0"
fi

# Read runtime configurations
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"
N_SEEDS="${N_SEEDS:-3}"
MASTER_SEED="${MASTER_SEED:-42}"
CRITIC_LRS="${CRITIC_LRS:-0.0001 0.0003 0.001}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
TD_LAMBDAS="${TD_LAMBDAS:-0.0 0.4 0.8 0.95 1.0}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-128}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-512}"
OUT_DIR="${OUT_DIR:-results/sweep_q_ppo_critic_jumanji}"

echo "================================================================"
echo "Job ID:           ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Hostname:         $(hostname)"
echo "Environment:      $ENV_NAME"
echo "Seeds:            $N_SEEDS (Master: $MASTER_SEED)"
echo "Critic LRs:       $CRITIC_LRS"
echo "Actor LR:         $ACTOR_LR"
echo "TD Lambdas:       $TD_LAMBDAS"
echo "Rollout:          $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "Output Directory: $OUT_DIR/$ENV_NAME"
echo "Python:           $PYTHON"
echo "================================================================"

CMD=("$PYTHON" "scripts/jumanji/sweep_q_ppo_critic/run_single_env.py"
    --env_name "$ENV_NAME"
    --critic_lrs $CRITIC_LRS
    --actor_lr "$ACTOR_LR"
    --td_lambdas $TD_LAMBDAS
    --num_envs "$NUM_ENVS"
    --num_steps "$NUM_STEPS"
    --minibatch_size "$MINIBATCH_SIZE"
    --n_seeds "$N_SEEDS"
    --master_seed "$MASTER_SEED"
    --out_dir "$OUT_DIR"
)

if [ -n "$TOTAL_TIMESTEPS" ] && [ "$TOTAL_TIMESTEPS" != "auto" ]; then
    CMD+=(--total_timesteps "$TOTAL_TIMESTEPS")
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"

echo "================================================================"
echo "Completed worker task for $ENV_NAME"
echo "================================================================"

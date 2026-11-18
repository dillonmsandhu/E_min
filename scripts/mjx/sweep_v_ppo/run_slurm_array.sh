#!/bin/bash
#SBATCH --job-name=v_ppo_mjx
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-4

set -e

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/mjx_rl/core/mjx_config.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p slurm

if [ -x "$REPO_ROOT/.venv_mjx/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv_mjx/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/mjx/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/mjx/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/3.12/envs/mjx/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.12/envs/mjx/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ALL_ENVS=(
    "CheetahRun"
    "WalkerWalk"
    "HopperStand"
    "FishSwim"
    "FingerSpin"
)

if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        ENV_NAME="${ALL_ENVS[$1]}"
    else
        ENV_NAME="$1"
    fi
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ENV_NAME="${ALL_ENVS[$SLURM_ARRAY_TASK_ID]}"
else
    ENV_NAME="CheetahRun"
fi

OUT_DIR="${OUT_DIR:-results/mjx/sweep_v_ppo}"
IMPL="${IMPL:-jax}"
METHODS="${METHODS:-td_lambda e_lambda_v_aux}"
N_SEEDS="${N_SEEDS:-3}"
MASTER_SEED="${MASTER_SEED:-42}"
CRITIC_LRS="${CRITIC_LRS:-0.0003 0.001 0.003}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-8 16}"
LAMBDAS="${LAMBDAS:-0.8 0.95 0.99}"
ACTOR_LR="${ACTOR_LR:-0.001}"
ACTOR_EPOCHS="${ACTOR_EPOCHS:-16}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_EPS="${CLIP_EPS:-0.3}"
ENT_COEF="${ENT_COEF:-0.01}"
REWARD_SCALING="${REWARD_SCALING:-10.0}"
NUM_ENVS="${NUM_ENVS:-512}"
NUM_STEPS="${NUM_STEPS:-128}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-2048}"

TIMESTEP_FLAG=""
if [ -n "$TOTAL_TIMESTEPS" ] && [ "$TOTAL_TIMESTEPS" != "auto" ]; then
    TIMESTEP_FLAG="--total_timesteps $TOTAL_TIMESTEPS"
fi

$PYTHON -u scripts/mjx/sweep_v_ppo/run_single_env.py \
    --env_name "$ENV_NAME" \
    --impl "$IMPL" \
    --methods $METHODS \
    --critic_lrs $CRITIC_LRS \
    --critic_epochs $CRITIC_EPOCHS \
    --lambdas $LAMBDAS \
    --actor_lr "$ACTOR_LR" \
    --actor_epochs "$ACTOR_EPOCHS" \
    --gae_lambda "$GAE_LAMBDA" \
    --clip_eps "$CLIP_EPS" \
    --ent_coef "$ENT_COEF" \
    --reward_scaling "$REWARD_SCALING" \
    $TIMESTEP_FLAG \
    --num_envs "$NUM_ENVS" \
    --num_steps "$NUM_STEPS" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --n_seeds "$N_SEEDS" \
    --master_seed "$MASTER_SEED" \
    --out_dir "$OUT_DIR"

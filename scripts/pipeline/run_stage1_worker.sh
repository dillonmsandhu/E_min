#!/bin/bash
#SBATCH --job-name=pipe_s1
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -e
mkdir -p slurm

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

SUITE=${SUITE:?Must set SUITE (brax|mjx_locomotion|octax|minatar|gymnax)}
ENV_NAME=${1:?Must pass ENV_NAME as $1}
ALGO=${2:?Must pass ALGO as $2}
OUT_ROOT=${OUT_ROOT:-"results/pipeline"}

resolve_python() {
    if [ "$SUITE" = "mjx_locomotion" ]; then
        if [ -x "$REPO_ROOT/.venv_mjx/bin/python" ]; then
            echo "$REPO_ROOT/.venv_mjx/bin/python"
            return
        elif [ -f "/home/users/ds541/.pyenv/versions/mjx/bin/python" ]; then
            echo "/home/users/ds541/.pyenv/versions/mjx/bin/python"
            return
        fi
    fi
    if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
        echo "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
    elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
        echo "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
    elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
        echo "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        echo "$VIRTUAL_ENV/bin/python"
    else
        echo "python3"
    fi
}

PYTHON=$(resolve_python)

EXTRA_ARGS=()
[ -n "${STAGE1_SEEDS:-}" ] && EXTRA_ARGS+=(--n_seeds "$STAGE1_SEEDS")
[ -n "${STAGE1_MASTER_SEED:-}" ] && EXTRA_ARGS+=(--master_seed "$STAGE1_MASTER_SEED")
[ -n "${ACTOR_LR:-}" ] && EXTRA_ARGS+=(--actor_lr "$ACTOR_LR")
[ -n "${GAE_LAMBDA:-}" ] && EXTRA_ARGS+=(--gae_lambda "$GAE_LAMBDA")
[ -n "${TOTAL_TIMESTEPS:-}" ] && [ "$TOTAL_TIMESTEPS" != "auto" ] && EXTRA_ARGS+=(--total_timesteps "$TOTAL_TIMESTEPS")
[ -n "${CRITIC_LRS:-}" ] && EXTRA_ARGS+=(--critic_lrs $CRITIC_LRS)
[ -n "${CRITIC_EPOCHS:-}" ] && EXTRA_ARGS+=(--critic_epochs $CRITIC_EPOCHS)
[ -n "${MAX_GRAD_NORMS:-}" ] && EXTRA_ARGS+=(--max_grad_norms $MAX_GRAD_NORMS)
[ -n "${LAMBDAS:-}" ] && EXTRA_ARGS+=(--lambdas $LAMBDAS)

echo "[Stage 1 Worker] suite=$SUITE env=$ENV_NAME algo=$ALGO python=$PYTHON"
"$PYTHON" -u scripts/pipeline/stage1_tune.py \
    --suite "$SUITE" \
    --envs "$ENV_NAME" \
    --algos "$ALGO" \
    --out_root "$OUT_ROOT" \
    "${EXTRA_ARGS[@]}"

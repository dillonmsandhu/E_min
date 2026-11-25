#!/bin/bash
set -e

[ -f "octax_rl/core/octax_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sweep_v_ppo"
LOG_DIR="slurm/${DATE}/octax_${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-5}
OUT_DIR=${3:-"results/octax/sweep_v_ppo"}
MASTER_SEED=${MASTER_SEED:-42}
CRITIC_LRS=${CRITIC_LRS:-"0.0001 0.0003 0.001"}
CRITIC_EPOCHS=${CRITIC_EPOCHS:-"4 8"}
LAMBDAS=${LAMBDAS:-"0.4 0.8 0.95"}
ACTOR_LR=${ACTOR_LR:-0.0003}
GAE_LAMBDA=${GAE_LAMBDA:-0.8}
STICKY_ACTION_PROB=${STICKY_ACTION_PROB:-0.0}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}

export TOTAL_TIMESTEPS N_SEEDS MASTER_SEED CRITIC_LRS CRITIC_EPOCHS LAMBDAS ACTOR_LR GAE_LAMBDA STICKY_ACTION_PROB NUM_ENVS NUM_STEPS MINIBATCH_SIZE OUT_DIR REPO_ROOT

if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

ARRAY_JOB_ID=$(sbatch --parsable \
    --chdir="$REPO_ROOT" \
    --output="${LOG_DIR}/%A_%a.out" \
    --error="${LOG_DIR}/%A_%a.err" \
    --export=ALL \
    scripts/octax/sweep_v_ppo/run_slurm_array.sh)

POST_JOB_ID=$(sbatch --parsable \
    --job-name=plot_v_ppo_octax \
    --output="${LOG_DIR}/%A_consolidated_plot.out" \
    --error="${LOG_DIR}/%A_consolidated_plot.err" \
    --time=01:00:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir="$REPO_ROOT" \
    --wrap="$PYTHON scripts/octax/sweep_v_ppo/plot_consolidated_results.py --out_dir '$OUT_DIR'")

echo "Submitted Octax V-PPO Array Job $ARRAY_JOB_ID and Post Job $POST_JOB_ID -> $OUT_DIR"

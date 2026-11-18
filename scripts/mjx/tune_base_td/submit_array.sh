#!/bin/bash
set -e

[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="tune_base_td"
LOG_DIR="slurm/${DATE}/mjx_${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-3}
OUT_DIR=${3:-"results/mjx/tune_base_td"}
MODE=${MODE:-oat}
IMPL=${IMPL:-jax}
MASTER_SEED=${MASTER_SEED:-42}
ENVS=${ENVS:-"CheetahRun WalkerWalk HopperStand FishSwim FingerSpin"}
GPU_GRES=${GPU_GRES:-auto}

export TOTAL_TIMESTEPS N_SEEDS OUT_DIR MODE IMPL MASTER_SEED REPO_ROOT

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

get_env_gres() {
    local env="$1"
    if [ "$GPU_GRES" != "auto" ]; then
        echo "$GPU_GRES"
        return
    fi
    case "$env" in
        *Humanoid*)
            echo "gpu:a6000:1"
            ;;
        *)
            echo "gpu:a5000:1"
            ;;
    esac
}

JOB_IDS=()
for ENV_NAME in $ENVS; do
    GRES=$(get_env_gres "$ENV_NAME")
    JID=$(sbatch --parsable \
        --job-name="tune_td_${ENV_NAME}" \
        --gres="$GRES" \
        --array=0-0 \
        --chdir="$REPO_ROOT" \
        --output="${LOG_DIR}/%A_${ENV_NAME}.out" \
        --error="${LOG_DIR}/%A_${ENV_NAME}.err" \
        --export=ALL \
        scripts/mjx/tune_base_td/run_slurm_array.sh "$ENV_NAME")
    echo "  Submitted $ENV_NAME ($GRES) -> Job $JID"
    JOB_IDS+=("$JID")
done

DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

POST_JOB_ID=$(sbatch --parsable \
    --job-name=plot_tune_td_mjx \
    --output="${LOG_DIR}/%j_consolidated_plot.out" \
    --error="${LOG_DIR}/%j_consolidated_plot.err" \
    --time=00:30:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:${DEP_STR} \
    --chdir="$REPO_ROOT" \
    --wrap="$PYTHON scripts/mjx/tune_base_td/tune_base_td.py --consolidate_only --out_dir '$OUT_DIR'")

echo "Submitted MJX Base TD(λ) Tuning Jobs (${DEP_STR}) and Post Job $POST_JOB_ID -> $OUT_DIR"

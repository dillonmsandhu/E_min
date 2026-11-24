#!/bin/bash
set -e

[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="mjx_sweep_locomotion"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-3}
OUT_DIR=${3:-"results/mjx/sweep_locomotion"}
IMPL=${IMPL:-jax}
METHODS=${METHODS:-"td_lambda td_0 e_0_v_aux e_lambda_v_aux"}
ENVS=${ENVS:-"Go1JoystickFlatTerrain Go1Getup Go1Handstand G1JoystickFlatTerrain BerkeleyHumanoidJoystickFlatTerrain"}
GPU_GRES=${GPU_GRES:-auto}
MASTER_SEED=${MASTER_SEED:-42}
CRITIC_LRS=${CRITIC_LRS:-"0.0003 0.001"}
CRITIC_EPOCHS=${CRITIC_EPOCHS:-"4"}
LAMBDAS=${LAMBDAS:-"0.6 0.8 0.95"}
ACTOR_LR=${ACTOR_LR:-0.0003}
ACTOR_EPOCHS=${ACTOR_EPOCHS:-4}
GAE_LAMBDA=${GAE_LAMBDA:-0.8}
NUM_ENVS=${NUM_ENVS:-512}
NUM_STEPS=${NUM_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-2048}

export TOTAL_TIMESTEPS N_SEEDS MASTER_SEED IMPL METHODS CRITIC_LRS CRITIC_EPOCHS LAMBDAS \
    ACTOR_LR ACTOR_EPOCHS GAE_LAMBDA NUM_ENVS NUM_STEPS MINIBATCH_SIZE OUT_DIR REPO_ROOT

if [ -x "$REPO_ROOT/.venv_mjx/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv_mjx/bin/python"
elif [ -f "/home/users/ds541/.pyenv/versions/mjx/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/mjx/bin/python"
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
        *Humanoid*|G1*|T1*|H1*|Apollo*)
            echo "gpu:a6000:1"
            ;;
        *)
            echo "gpu:a5000:1"
            ;;
    esac
}

# Ensure mujoco_menagerie is downloaded once before parallel Slurm jobs start (avoids git clone race).
$PYTHON -c "from mujoco_playground._src.mjx_env import ensure_menagerie_exists; ensure_menagerie_exists()"

JOB_IDS=()
for ENV_NAME in $ENVS; do
    GRES=$(get_env_gres "$ENV_NAME")
    for METHOD in $METHODS; do
        JID=$(sbatch --parsable \
            --job-name="loco_${ENV_NAME}_${METHOD}" \
            --gres="$GRES" \
            --array=0-0 \
            --chdir="$REPO_ROOT" \
            --output="${LOG_DIR}/%A_${ENV_NAME}_${METHOD}.out" \
            --error="${LOG_DIR}/%A_${ENV_NAME}_${METHOD}.err" \
            --export=ALL,METHODS="$METHOD" \
            scripts/mjx/sweep_locomotion/run_slurm_array.sh "$ENV_NAME")
        echo "  Submitted $ENV_NAME [$METHOD] ($GRES) -> Job $JID"
        JOB_IDS+=("$JID")
    done
done

DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

POST_JOB_ID=$(sbatch --parsable \
    --job-name=plot_loco_mjx \
    --output="${LOG_DIR}/%j_consolidated_plot.out" \
    --error="${LOG_DIR}/%j_consolidated_plot.err" \
    --time=01:00:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:${DEP_STR} \
    --chdir="$REPO_ROOT" \
    --wrap="$PYTHON scripts/mjx/sweep_locomotion/plot_consolidated_results.py --out_dir '$OUT_DIR'")

echo "Submitted MJX Locomotion Jobs (${DEP_STR}) and Post Job $POST_JOB_ID -> $OUT_DIR"

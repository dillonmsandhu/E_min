#!/bin/bash
# scripts/pipeline/submit_pipeline.sh
#
# Submits the complete chained Two-Stage ICML 2027 Pipeline for any suite:
#   Stage 1 (GPU): Per-(env, algo) grid tune on tuning seeds
#     --dependency=afterok-->
#   Select  (CPU): Export global & per-λ JSON configs -> configs/tuned/<suite>/
#     --dependency=afterok-->
#   Stage 2 (GPU): Held-out comparative evaluation (zero per-env tuning, more seeds)
#     --dependency=afterok-->
#   Consolidate (CPU): Championship profiles, diagnostics & cross_suite_summary.csv
#
# Usage:
#   ./scripts/pipeline/submit_pipeline.sh <suite> [--stage2-only] [--dry-run]
#
# Suites: brax | mjx_locomotion | octax | minatar | gymnax

set -e

SUITE=${1:?Usage: $0 <brax|mjx_locomotion|octax|minatar|gymnax> [--stage2-only] [--dry-run]}
shift

STAGE2_ONLY=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --stage2-only) STAGE2_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

[ -f "scripts/pipeline/suite_registry.py" ] || cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

DATE=${DATE:-$(date +%Y-%m-%d)}
OUT_ROOT=${OUT_ROOT:-"results/pipeline"}
CONFIGS_DIR=${CONFIGS_DIR:-"configs/tuned"}
LOG_DIR="slurm/${DATE}/pipeline_${SUITE}"
mkdir -p "$LOG_DIR"

# Resolve Python interpreter for suite
if [ "$SUITE" = "mjx_locomotion" ]; then
    if [ -x "$REPO_ROOT/.venv_mjx/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv_mjx/bin/python"
    elif [ -f "/home/users/ds541/.pyenv/versions/mjx/bin/python" ]; then
        PYTHON="/home/users/ds541/.pyenv/versions/mjx/bin/python"
    else
        PYTHON="python3"
    fi
else
    if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
        PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
    elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
        PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
    elif [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
        PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
    else
        PYTHON="python3"
    fi
fi

# Query suite env lists from suite_registry.py (single source of truth)
TUNE_ENVS=${TUNE_ENVS:-$("$PYTHON" -c "from scripts.pipeline.suite_registry import get_suite_spec; print(' '.join(get_suite_spec('$SUITE').tune_envs))")}
EVAL_ENVS=${EVAL_ENVS:-$("$PYTHON" -c "from scripts.pipeline.suite_registry import get_suite_spec; print(' '.join(get_suite_spec('$SUITE').eval_envs))")}
STAGE1_ALGOS=${STAGE1_ALGOS:-"td_0 e_0_v_aux td_lambda e_lambda_v_aux"}
STAGE2_ALGOS=${STAGE2_ALGOS:-"td_0 e_0_v_aux td_lambda e_lambda_v_aux bsmc_v_aux e_0_rot_gated e_lambda_rot_gated"}
GPU_GRES=${GPU_GRES:-auto}
CPUS_GPU=${CPUS_GPU:-2}

# Suite-appropriate SLURM wall-clock & RAM defaults (overridable via env vars)
case "$SUITE" in
    gymnax)
        TIME_S1=${TIME_S1:-"02:00:00"}
        TIME_S2=${TIME_S2:-"01:00:00"}
        MEM_GPU=${MEM_GPU:-"16G"}
        ;;
    minatar|octax)
        TIME_S1=${TIME_S1:-"06:00:00"}
        TIME_S2=${TIME_S2:-"03:00:00"}
        MEM_GPU=${MEM_GPU:-"20G"}
        ;;
    brax)
        TIME_S1=${TIME_S1:-"10:00:00"}
        TIME_S2=${TIME_S2:-"05:00:00"}
        MEM_GPU=${MEM_GPU:-"24G"}
        ;;
    mjx_locomotion)
        TIME_S1=${TIME_S1:-"20:00:00"}
        TIME_S2=${TIME_S2:-"08:00:00"}
        MEM_GPU=${MEM_GPU:-"32G"}
        ;;
    *)
        TIME_S1=${TIME_S1:-"10:00:00"}
        TIME_S2=${TIME_S2:-"06:00:00"}
        MEM_GPU=${MEM_GPU:-"24G"}
        ;;
esac

export SUITE OUT_ROOT CONFIGS_DIR STAGE2_ALGOS REPO_ROOT

get_env_gres() {
    local env="$1"
    if [ "$GPU_GRES" != "auto" ]; then
        echo "$GPU_GRES"
        return
    fi
    if [ "$SUITE" = "mjx_locomotion" ]; then
        case "$env" in
            *Humanoid*|G1*|T1*|H1*|Apollo*) echo "gpu:a6000:1"; return ;;
        esac
    fi
    if [ "$SUITE" = "brax" ] && [ "$env" = "humanoid" ]; then
        echo "gpu:a6000:1"
        return
    fi
    echo "gpu:a5000:1"
}

run_sbatch() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "DRY_$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

# Pre-download mujoco_menagerie once on MJX to avoid parallel clone races
if [ "$SUITE" = "mjx_locomotion" ] && [ "$DRY_RUN" = "0" ]; then
    "$PYTHON" -c "from mujoco_playground._src.mjx_env import ensure_menagerie_exists; ensure_menagerie_exists()"
fi

SELECT_DEP_ARGS=()

if [ "$STAGE2_ONLY" = "0" ]; then
    echo "=== Submitting Stage 1 Tuning for '$SUITE' ==="
    S1_JIDS=()
    for ENV_NAME in $TUNE_ENVS; do
        GRES=$(get_env_gres "$ENV_NAME")
        for ALGO in $STAGE1_ALGOS; do
            JID=$(run_sbatch \
                --job-name="s1_${SUITE}_${ENV_NAME}_${ALGO}" \
                --time="$TIME_S1" \
                --cpus-per-task="$CPUS_GPU" \
                --mem="$MEM_GPU" \
                --gres="$GRES" \
                --chdir="$REPO_ROOT" \
                --output="${LOG_DIR}/%j_s1_${ENV_NAME}_${ALGO}.out" \
                --error="${LOG_DIR}/%j_s1_${ENV_NAME}_${ALGO}.err" \
                --export=ALL \
                scripts/pipeline/run_stage1_worker.sh "$ENV_NAME" "$ALGO")
            echo "  [Stage 1] $ENV_NAME [$ALGO] ($GRES) -> Job $JID"
            S1_JIDS+=("$JID")
        done
    done

    S1_DEP=$(IFS=:; echo "${S1_JIDS[*]}")

    SELECT_ALGOS=${SELECT_ALGOS:-"td_0 e_0_v_aux td_lambda e_lambda_v_aux"}
    echo "=== Submitting Config Selection (afterok:${S1_DEP}) ==="
    SELECT_JID=$(run_sbatch \
        --job-name="sel_${SUITE}" \
        --partition=compsci \
        --cpus-per-task=2 \
        --mem=16G \
        --time=00:30:00 \
        --dependency="afterok:${S1_DEP}" \
        --chdir="$REPO_ROOT" \
        --output="${LOG_DIR}/%j_select_configs.out" \
        --error="${LOG_DIR}/%j_select_configs.err" \
        --wrap="JAX_PLATFORMS=cpu $PYTHON scripts/pipeline/select_configs.py --suite '$SUITE' --algos $SELECT_ALGOS --out_root '$OUT_ROOT' --configs_dir '$CONFIGS_DIR'")
    echo "  [Select] Job $SELECT_JID -> ${CONFIGS_DIR}/${SUITE}/"
    SELECT_DEP_ARGS=(--dependency="afterok:${SELECT_JID}")
else
    echo "=== Skipping Stage 1 (--stage2-only); using existing configs in ${CONFIGS_DIR}/${SUITE}/ ==="
fi

echo "=== Submitting Stage 2 Held-Out Evaluation for '$SUITE' ==="
S2_JIDS=()
for ENV_NAME in $EVAL_ENVS; do
    GRES=$(get_env_gres "$ENV_NAME")
    JID=$(run_sbatch \
        --job-name="s2_${SUITE}_${ENV_NAME}" \
        --time="$TIME_S2" \
        --cpus-per-task="$CPUS_GPU" \
        --mem="$MEM_GPU" \
        --gres="$GRES" \
        "${SELECT_DEP_ARGS[@]}" \
        --chdir="$REPO_ROOT" \
        --output="${LOG_DIR}/%j_s2_${ENV_NAME}.out" \
        --error="${LOG_DIR}/%j_s2_${ENV_NAME}.err" \
        --export=ALL \
        scripts/pipeline/run_stage2_worker.sh "$ENV_NAME")
    echo "  [Stage 2] $ENV_NAME ($GRES) -> Job $JID"
    S2_JIDS+=("$JID")
done

S2_DEP=$(IFS=:; echo "${S2_JIDS[*]}")

echo "=== Submitting Stage 2 Consolidation & Cross-Suite CSV Export (afterany:${S2_DEP}) ==="
POST_JID=$(run_sbatch \
    --job-name="plot_${SUITE}" \
    --partition=compsci \
    --cpus-per-task=2 \
    --mem=16G \
    --time=00:45:00 \
    --dependency="afterany:${S2_DEP}" \
    --chdir="$REPO_ROOT" \
    --output="${LOG_DIR}/%j_stage2_consolidate.out" \
    --error="${LOG_DIR}/%j_stage2_consolidate.err" \
    --wrap="JAX_PLATFORMS=cpu $PYTHON scripts/pipeline/stage2_consolidate.py --suite '$SUITE' --algos $STAGE2_ALGOS --out_root '$OUT_ROOT'")

echo "=== Pipeline Submitted for '$SUITE' ==="
echo "  Stage 2 Consolidate Job: $POST_JID"
echo "  Outputs: ${OUT_ROOT}/${SUITE}/stage2_eval/"
echo "  Logs:    ${LOG_DIR}/"

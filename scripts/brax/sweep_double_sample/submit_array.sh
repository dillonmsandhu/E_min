#!/bin/bash
# ==============================================================================
# E_min/scripts/brax/sweep_double_sample/submit_array.sh
#
# Launches the 11-environment SLURM array job for the Double-Sampling Critic Sweep
# across deterministic continuous-control Brax domains:
#   1. Fitted Value Iteration (GAE baseline)
#   2. MSBE (Baird double-sampled Bellman error)
#   3. E-Min (Double-sampled bilinear energy minimization <e, bar_delta>_D)
#   4. MSPBE (Tangent space projection with CG & exact Hessian-vector products)
#
# Automatically chains the consolidated post-processing job to run upon completion.
#
# Usage:
#   bash scripts/brax/sweep_double_sample/submit_array.sh
#   bash scripts/brax/sweep_double_sample/submit_array.sh [TOTAL_TIMESTEPS] [N_SEEDS] [OUT_DIR]
# ==============================================================================

set -e

# Ensure working directory is E_min repo root
[ -f "brax_rl/core/brax_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sweep_double_sample_brax"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-5}
OUT_DIR=${3:-"results/sweep_double_sample_brax"}
MASTER_SEED=${MASTER_SEED:-42}
CRITIC_LRS=${CRITIC_LRS:-"0.0003 0.001 0.003"}
ACTOR_LR=${ACTOR_LR:-0.0003}
NUM_ENVS=${NUM_ENVS:-256}
NUM_STEPS=${NUM_STEPS:-256}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-1024}

export TOTAL_TIMESTEPS N_SEEDS MASTER_SEED CRITIC_LRS ACTOR_LR NUM_ENVS NUM_STEPS MINIBATCH_SIZE OUT_DIR REPO_ROOT

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
echo "Submitting Brax Double-Sampling Critic SLURM Array"
echo "  Timesteps:         $TOTAL_TIMESTEPS (auto per env)"
echo "  Seeds:             $N_SEEDS (Master: $MASTER_SEED)"
echo "  Algorithms:        Fitted, MSBE, E-Min, MSPBE"
echo "  Critic LRs:        $CRITIC_LRS"
echo "  Actor LR:          $ACTOR_LR"
echo "  Rollout:           $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Output Directory:  $OUT_DIR"
echo "  Log Directory:     $LOG_DIR"
echo "  Python:            $PYTHON"
echo "  Repo Root:         $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/brax/sweep_double_sample/run_slurm_array.sh"

SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS,N_SEEDS=$N_SEEDS,MASTER_SEED=$MASTER_SEED,CRITIC_LRS=\"$CRITIC_LRS\",ACTOR_LR=$ACTOR_LR,NUM_ENVS=$NUM_ENVS,NUM_STEPS=$NUM_STEPS,MINIBATCH_SIZE=$MINIBATCH_SIZE,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID (11 continuous Brax environments)"

POST_SCRIPT="scripts/brax/sweep_double_sample/plot_consolidated_results.py"

SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_double_s_brax \
    --output=\"${LOG_DIR}/%A_consolidated_plot.out\" \
    --error=\"${LOG_DIR}/%A_consolidated_plot.err\" \
    --time=01:00:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --out_dir '$OUT_DIR' --num_envs $NUM_ENVS --num_steps $NUM_STEPS\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID"
echo "  (Will automatically execute after Array Job $ARRAY_JOB_ID completes)"
echo "Logs written to:    $LOG_DIR/"
echo "Track status with:  squeue -u \$USER"

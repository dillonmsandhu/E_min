#!/bin/bash
# ==============================================================================
# external_E_min/scripts/gymnax/sweep_comprehensive_critic/submit_array.sh
#
# Launches the 12-environment SLURM array job for the comprehensive critic sweep.
# Automatically chains the consolidated post-processing job to run upon completion.
#
# Usage:
#   bash scripts/gymnax/sweep_comprehensive_critic/submit_array.sh
#   bash scripts/gymnax/sweep_comprehensive_critic/submit_array.sh [TOTAL_TIMESTEPS] [N_SEEDS] [OUT_DIR]
# ==============================================================================

set -e

# Ensure working directory is repo root
[ -f "core/config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sweep_comprehensive_critic"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-3}
MASTER_SEED=${3:-42}
LAMBDAS=${4:-"0.0 0.4 0.8 0.95"}
CRITIC_LRS=${5:-"0.0001 0.0003 0.001"}
ACTOR_LR=${6:-0.0003}
NUM_ENVS=${7:-32}
NUM_STEPS=${8:-256}
MINIBATCH_SIZE=${9:-512}
ABLATION_LAMBDA=${10:-0.8}
HUBER_DELTA=${11:-10.0}
TRUST_COEF=${12:-1.0}
OUT_DIR=${13:-"results/sweep_comprehensive_critic"}

export TOTAL_TIMESTEPS N_SEEDS MASTER_SEED LAMBDAS CRITIC_LRS ACTOR_LR NUM_ENVS NUM_STEPS MINIBATCH_SIZE ABLATION_LAMBDA HUBER_DELTA TRUST_COEF OUT_DIR REPO_ROOT

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
echo "Submitting Comprehensive Critic Comparison SLURM Array"
echo "  Timesteps:        $TOTAL_TIMESTEPS (auto: 1M for MountainCar envs, 300k for small envs)"
echo "  Seeds:            $N_SEEDS (Master: $MASTER_SEED)"
echo "  Lambdas:          $LAMBDAS"
echo "  Critic LRs:       $CRITIC_LRS"
echo "  Actor LR:         $ACTOR_LR"
echo "  Rollout:          $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Ablation Lambda:  $ABLATION_LAMBDA (Huber: $HUBER_DELTA, Trust: $TRUST_COEF)"
echo "  Output Directory: $OUT_DIR"
echo "  Log Directory:    $LOG_DIR"
echo "  Python:           $PYTHON"
echo "  Repo Root:        $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/gymnax/sweep_comprehensive_critic/run_slurm_array.sh"

SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS,N_SEEDS=$N_SEEDS,MASTER_SEED=$MASTER_SEED,LAMBDAS=\"$LAMBDAS\",CRITIC_LRS=\"$CRITIC_LRS\",ACTOR_LR=$ACTOR_LR,NUM_ENVS=$NUM_ENVS,NUM_STEPS=$NUM_STEPS,MINIBATCH_SIZE=$MINIBATCH_SIZE,ABLATION_LAMBDA=$ABLATION_LAMBDA,HUBER_DELTA=$HUBER_DELTA,TRUST_COEF=$TRUST_COEF,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID (12 environments)"

POST_SCRIPT="scripts/gymnax/sweep_comprehensive_critic/plot_consolidated_results.py"

SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_comp_critic \
    --output=\"${LOG_DIR}/%A_consolidated_plot.out\" \
    --error=\"${LOG_DIR}/%A_consolidated_plot.err\" \
    --time=01:00:00 \
    --partition=compsci \
    --cpus-per-task=4 \
    --mem=16G \
    --dependency=afterany:$ARRAY_JOB_ID \
    --chdir=\"$REPO_ROOT\" \
    --wrap=\"$PYTHON $POST_SCRIPT --out_dir '$OUT_DIR' --ablation_lambda $ABLATION_LAMBDA --huber_delta $HUBER_DELTA --trust_coef $TRUST_COEF --num_envs $NUM_ENVS --num_steps $NUM_STEPS\""

POST_JOB_ID=$(eval "$SBATCH_POST_CMD")
echo "Submitted Chained Post-Processing Job ID: $POST_JOB_ID"
echo "  (Will automatically execute after Array Job $ARRAY_JOB_ID completes)"
echo "Logs written to:    $LOG_DIR/"
echo "Track status with:  squeue -u \$USER"

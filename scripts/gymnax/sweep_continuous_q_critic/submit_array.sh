#!/bin/bash
# ==============================================================================
# E_min/scripts/gymnax/sweep_continuous_q_critic/submit_array.sh
#
# Launches the 4-environment SLURM array job for the continuous Q-critic (DDPG) sweep
# across deterministic continuous-control Gymnax domains.
# Automatically chains the consolidated post-processing job to run upon completion.
#
# Usage:
#   bash scripts/gymnax/sweep_continuous_q_critic/submit_array.sh
#   bash scripts/gymnax/sweep_continuous_q_critic/submit_array.sh [TOTAL_TIMESTEPS] [N_SEEDS] [OUT_DIR]
# ==============================================================================

set -e

# Ensure working directory is E_min repo root
[ -f "gymnax_rl/core/gymnax_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
DATE=${DATE:-$(date +%Y-%m-%d)}
SWEEP_NAME="sweep_continuous_q_critic_gymnax"
LOG_DIR="slurm/${DATE}/${SWEEP_NAME}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

TOTAL_TIMESTEPS=${1:-auto}
N_SEEDS=${2:-3}
OUT_DIR=${3:-"results/sweep_continuous_q_critic_gymnax"}
MASTER_SEED=${MASTER_SEED:-42}
CRITIC_LRS=${CRITIC_LRS:-"0.0003 0.001 0.003"}
ACTOR_LR=${ACTOR_LR:-0.0003}
SARSA_LAMBDAS=${SARSA_LAMBDAS:-"0.0 0.4 0.8 0.95 1.0"}
NUM_ENVS=${NUM_ENVS:-64}
NUM_STEPS=${NUM_STEPS:-64}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-512}
EXPLORATION_NOISE=${EXPLORATION_NOISE:-0.2}

export TOTAL_TIMESTEPS N_SEEDS MASTER_SEED CRITIC_LRS ACTOR_LR SARSA_LAMBDAS NUM_ENVS NUM_STEPS MINIBATCH_SIZE EXPLORATION_NOISE OUT_DIR REPO_ROOT

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
echo "Submitting Gymnax Continuous Q-Critic (DDPG) SLURM Array"
echo "  Timesteps:         $TOTAL_TIMESTEPS (auto: 500k MountainCarContinuous, 300k Pendulum/PointRobot)"
echo "  Seeds:             $N_SEEDS (Master: $MASTER_SEED)"
echo "  Critic LRs:        $CRITIC_LRS"
echo "  Actor LR:          $ACTOR_LR"
echo "  SARSA Lambdas:     $SARSA_LAMBDAS"
echo "  Rollout:           $NUM_ENVS envs x $NUM_STEPS steps | MB: $MINIBATCH_SIZE"
echo "  Exploration Noise: $EXPLORATION_NOISE"
echo "  Output Directory:  $OUT_DIR"
echo "  Log Directory:     $LOG_DIR"
echo "  Python:            $PYTHON"
echo "  Repo Root:         $REPO_ROOT"
echo "================================================================"

ARRAY_SCRIPT="scripts/gymnax/sweep_continuous_q_critic/run_slurm_array.sh"

SBATCH_ARRAY_CMD="sbatch --parsable \
    --chdir=\"$REPO_ROOT\" \
    --output=\"${LOG_DIR}/%A_%a.out\" \
    --error=\"${LOG_DIR}/%A_%a.err\" \
    --export=ALL,TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS,N_SEEDS=$N_SEEDS,MASTER_SEED=$MASTER_SEED,CRITIC_LRS=\"$CRITIC_LRS\",ACTOR_LR=$ACTOR_LR,SARSA_LAMBDAS=\"$SARSA_LAMBDAS\",NUM_ENVS=$NUM_ENVS,NUM_STEPS=$NUM_STEPS,MINIBATCH_SIZE=$MINIBATCH_SIZE,EXPLORATION_NOISE=$EXPLORATION_NOISE,OUT_DIR=\"$OUT_DIR\",REPO_ROOT=\"$REPO_ROOT\" \
    \"$ARRAY_SCRIPT\""

ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")
echo "Submitted Array Job ID: $ARRAY_JOB_ID (4 continuous Gymnax environments)"

POST_SCRIPT="scripts/gymnax/sweep_continuous_q_critic/plot_consolidated_results.py"

SBATCH_POST_CMD="sbatch --parsable \
    --job-name=plot_cont_q_gymnax \
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

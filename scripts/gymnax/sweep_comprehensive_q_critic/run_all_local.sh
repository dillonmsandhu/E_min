#!/bin/bash
# ==============================================================================
# external_E_min/scripts/gymnax/sweep_comprehensive_q_critic/run_all_local.sh
#
# Sequential local execution runner across the 6 deterministic discrete environments.
# ==============================================================================

set -e

[ -f "core/config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

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
    "CartPole-v1"
    "Acrobot-v1"
    "MountainCar-v0"
    "DeepSea-bsuite"
    "DiscountingChain-bsuite"
    "Pong-misc"
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

OUT_DIR="${1:-results/sweep_comprehensive_q_critic}"
TIMESTEPS="${2:-10000}"
N_SEEDS="${3:-1}"

echo "Running Sequential Local Q-PPO Sweep on: ${ALL_ENVS[*]}"
for ENV in "${ALL_ENVS[@]}"; do
    echo ">>> Running Environment: $ENV"
    $PYTHON scripts/gymnax/sweep_comprehensive_q_critic/run_single_env.py \
        --env_name "$ENV" \
        --total_timesteps "$TIMESTEPS" \
        --n_seeds "$N_SEEDS" \
        --out_dir "$OUT_DIR"
done

echo ">>> Running Consolidated Plotting"
$PYTHON scripts/gymnax/sweep_comprehensive_q_critic/plot_consolidated_results.py --out_dir "$OUT_DIR"

echo "Local Q-PPO sweep complete."

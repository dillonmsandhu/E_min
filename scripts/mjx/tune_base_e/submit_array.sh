#!/bin/bash
set -e

[ -f "mjx_rl/core/mjx_config.py" ] || cd "$(dirname "$0")/../../.."
export CRITIC_TYPE=e_lambda
export SWEEP_NAME="tune_base_e"
OUT_DIR=${3:-"results/mjx/tune_base_e"}
exec bash scripts/mjx/tune_base_td/submit_array.sh "${1:-auto}" "${2:-3}" "$OUT_DIR"

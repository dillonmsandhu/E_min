#!/bin/bash
set -e

[ -f "gymnax_rl/core/gymnax_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

OUT_DIR="${1:-results/gymnax/sweep_q_ppo}"
if [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
else
    PYTHON="python3"
fi

for idx in $(seq 0 13); do
    bash scripts/gymnax/sweep_q_ppo/run_slurm_array.sh "$idx"
done

$PYTHON scripts/gymnax/sweep_q_ppo/plot_consolidated_results.py --out_dir "$OUT_DIR"

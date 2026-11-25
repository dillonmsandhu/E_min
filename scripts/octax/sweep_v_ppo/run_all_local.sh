#!/bin/bash
set -e

[ -f "octax_rl/core/octax_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

OUT_DIR="${1:-results/octax/sweep_v_ppo}"
if [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
else
    PYTHON="python3"
fi

for idx in $(seq 0 7); do
    bash scripts/octax/sweep_v_ppo/run_slurm_array.sh "$idx"
done

$PYTHON scripts/octax/sweep_v_ppo/plot_consolidated_results.py --out_dir "$OUT_DIR"

#!/bin/bash
set -e

[ -f "jumanji_rl/core/jumanji_config.py" ] || cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

OUT_DIR="${1:-results/jumanji/sweep_q_ppo}"
if [ -f "/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/purejaxrl/bin/python"
else
    PYTHON="python3"
fi

for idx in $(seq 0 3); do
    bash scripts/jumanji/sweep_q_ppo/run_slurm_array.sh "$idx"
done

$PYTHON scripts/jumanji/sweep_q_ppo/plot_consolidated_results.py --out_dir "$OUT_DIR"

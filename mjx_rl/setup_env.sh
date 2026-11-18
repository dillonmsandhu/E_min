#!/bin/bash
# Creates <repo>/.venv_mjx for mjx_rl (Python 3.12, MuJoCo Playground rss2025).
# Usage (from E_min repo root):   bash mjx_rl/setup_env.sh
# On a Linux CUDA-12 node this additionally installs jax[cuda12] at the same version.
set -e
[ -f "mjx_rl/requirements_mjx.txt" ] || cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

uv venv --python 3.12 .venv_mjx
uv pip install --python .venv_mjx/bin/python -r mjx_rl/requirements_mjx.txt

if [ "$(uname -s)" = "Linux" ]; then
    JAX_VER=$(.venv_mjx/bin/python -c "import jax; print(jax.__version__)")
    uv pip install --python .venv_mjx/bin/python "jax[cuda12]==${JAX_VER}"
fi

.venv_mjx/bin/python -c "import jax, mujoco, brax, mujoco_playground; print('jax', jax.__version__, jax.default_backend(), '| mujoco', mujoco.__version__, '| brax', brax.__version__)"

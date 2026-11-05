# core/__init__.py
"""
Backward-compatibility root package.

The codebase is organized into:
- gymnax_rl: Gymnax-based reinforcement learning algorithms (PPO) and modular critics
- brax_rl: Brax-based continuous RL algorithms and critics
- common: Shared visualization, video, and notification tools
- analysis: Theoretical diagnostics, NTK, and Bellman error analysis
"""

from gymnax_rl.core.gymnax_config import config
from gymnax_rl.core.evaluator import tune, evaluate, train_and_evaluate_seeds, run_experiment_main
from gymnax_rl.core.gymnax_wrappers import make_env

__all__ = [
    "config",
    "tune",
    "evaluate",
    "train_and_evaluate_seeds",
    "run_experiment_main",
    "make_env",
]


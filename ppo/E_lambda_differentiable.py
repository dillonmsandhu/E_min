# Re-export from algos.E_lambda_differentiable to satisfy ppo/E_lambda_differentiable path conventions
from algos.E_lambda_differentiable import make_train, Transition, SAVE_DIR
from core.helpers import e_lambda_differentiable_critic_loss, e_lambda_differentiable_loss_fn
from core.runner import run_experiment_main

__all__ = [
    "make_train",
    "Transition",
    "SAVE_DIR",
    "e_lambda_differentiable_critic_loss",
    "e_lambda_differentiable_loss_fn",
]

if __name__ == "__main__":
    run_experiment_main(make_train, SAVE_DIR)

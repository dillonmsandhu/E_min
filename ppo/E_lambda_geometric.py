# Re-export from algos.E_lambda_geometric to satisfy ppo/E_lambda_geometric path conventions
from algos.E_lambda_geometric import make_train, Transition, SAVE_DIR
from core.helpers import e_lambda_geometric_critic_loss, e_lambda_geometric_loss_fn
from core.runner import run_experiment_main

__all__ = [
    "make_train",
    "Transition",
    "SAVE_DIR",
    "e_lambda_geometric_critic_loss",
    "e_lambda_geometric_loss_fn",
]

if __name__ == "__main__":
    run_experiment_main(make_train, SAVE_DIR)

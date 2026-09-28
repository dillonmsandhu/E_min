# Re-export from algos.E_lambda_fixed to satisfy ppo/E_lambda_fixed path conventions
from algos.E_lambda_fixed import make_train, Transition, SAVE_DIR
from core.helpers import calculate_e_lambda_targets
from core.runner import run_experiment_main

__all__ = ["make_train", "Transition", "SAVE_DIR", "calculate_e_lambda_targets"]

if __name__ == "__main__":
    run_experiment_main(make_train, SAVE_DIR)

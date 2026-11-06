import sys, os
_repo_root = '/Users/dillonsandhu/Documents/Research/E_min'
sys.path.insert(0, _repo_root)

import jax
import numpy as np
from gymnax_rl.algos.ppo import make_train

def run_test(critic_type, lr=3e-4, critic_lr=1e-3, lmb=0.8, use_huber=False, use_trust=False):
    cfg = {
        "ENV_NAME": "FourRooms-misc",
        "USE_VISUAL_OBS": True,
        "NETWORK_TYPE": "cnn",
        "SHARED_TORSO": False,
        "GAMMA": 0.99,
        "LR": lr,
        "ACTOR_LR": lr,
        "CRITIC_LR": critic_lr,
        "CRITIC_TYPE": critic_type,
        "VALUE_LAMBDA": lmb,
        "E_LAMBDA": lmb,
        "JUMP_MODE": "full_mc_invalid_jump",
        "GAE_LAMBDA": 0.8,
        "NUM_ENVS": 32,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 300000,
        "MINIBATCH_SIZE": 512,
        "ACTOR_MINIBATCH_SIZE": 512,
        "CRITIC_MINIBATCH_SIZE": 512,
        "NUM_EPOCHS": 4,
        "ACTOR_EPOCHS": 4,
        "CRITIC_EPOCHS": 4,
        "DEBUG": False,
    }
    if use_huber:
        cfg["V_AUX_LOSS_TYPE"] = "huber"
        cfg["V_AUX_HUBER_DELTA"] = 10.0
    if use_trust:
        cfg["V_AUX_TRUST_COEF"] = 1.0

    print(f"\n--- Testing FourRooms with {critic_type} (critic_lr={critic_lr}) ---")
    train_fn = make_train(cfg)
    rng = jax.random.PRNGKey(42)
    
    print("Compiling and training for 300k timesteps...")
    out = jax.jit(train_fn)(rng)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    
    metrics = out["metrics"]
    returns = np.array(metrics["returned_episode_returns"]) # shape (num_updates,)
    val_loss = np.array(metrics["value_loss"])
    
    batch_size = cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
    timesteps = np.arange(1, len(returns) + 1) * batch_size
    
    print(f"Total updates: {len(returns)} (~{timesteps[-1]} timesteps)")
    print("Learning progress (sample checkpoints):")
    step_indices = np.linspace(0, len(returns)-1, min(8, len(returns)), dtype=int)
    for idx in step_indices:
        t = timesteps[idx]
        ret = returns[idx]
        vl = val_loss[idx]
        print(f"  Step {t:6d} | Return: {ret:.4f} | Value Loss: {vl:.6f}")
        
    final_return = returns[-1]
    print(f"Final Return at 300k: {final_return:.4f}")
    return returns, val_loss

if __name__ == "__main__":
    # Test 1: E(0) with V_aux
    run_test("e_0", critic_lr=1e-3)
    # Test 2: E(lambda=0.8) with Huber and Trust
    run_test("e_lambda", critic_lr=1e-3, lmb=0.8, use_huber=True, use_trust=True)

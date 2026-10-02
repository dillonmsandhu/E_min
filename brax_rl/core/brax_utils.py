import jax
import jax.numpy as jnp
import yaml
import pickle
import cloudpickle
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json
import optax
import distrax
from typing import NamedTuple

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def load_config_dict(file_path: str) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, env_dir):
    config_path = os.path.join(env_dir, f"config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

def save_results(data, config, env_name, env_dir):
    # Create a subdirectory for the environment within the main run directory
    os.makedirs(env_dir, exist_ok=True)

    # Save the pickle file
    pickle_path = os.path.join(env_dir, "out.pkl")
    with open(pickle_path, 'wb') as f:
        cloudpickle.dump(data, f)
    print(f"Results saved to {pickle_path}")
        
    save_config(config, env_dir)
    print(f"Config saved to {os.path.join(env_dir, f'config.json')}")

    return env_dir

def save_plot(env_dir, env_name, steps_per_pi, episodic_return):
    plt.figure()
    plt.plot([i * steps_per_pi for i in range(len(episodic_return))], episodic_return, 'o-')
    plt.xlabel("Step")
    plt.ylabel("Return")
    plt.title(env_name)
    plt.legend()

    # Save plot as a .png file in the environment directory
    plot_path = os.path.join(env_dir, f"plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    return env_dir

def get_lr(config, total_grad_steps):
    if config['LR_SCHEDULE'] == 'linear':
        lr = optax.linear_schedule( # decays during FQE
            init_value=config["LR"],
            end_value=config.get('LR_END', 1e-15),
            transition_steps= total_grad_steps
        )
    elif config['LR_SCHEDULE'] == 'cosine':
        lr = optax.warmup_cosine_decay_schedule( # decays during FQE
            init_value=1e-20,
            peak_value = config["LR"], 
            warmup_steps = total_grad_steps // 20,
            end_value=config.get('LR_END', 1e-20),
            decay_steps = total_grad_steps
        )
    elif config['LR_SCHEDULE'] == 'warm_restarts': #This learning rate schedule applies multiple joined cosine decay cycles.
        def construct_sgdr_phases(config, total_grad_steps, num_phases=10):
            peak_lrs = np.linspace(config["LR"], config.get('LR_END', 1e-20), num_phases)  # Peaks decay linearly
            start_lrs = np.concatenate((peak_lrs[2:], [config.get('LR_END', 1e-20)] * 2))  # Start at peak after next
            decay_steps = total_grad_steps // num_phases
            cosine_kwargs = [
                {
                    'init_value': start,  # Starts at the peak after next
                    'peak_value': peak,   # Warms up to the current phase's peak
                    'warmup_steps': min(1, decay_steps // 10),
                    'end_value': end,     # Decays to the next peak
                    'decay_steps': decay_steps
                }
                for start, peak, end in zip(start_lrs, peak_lrs, jnp.roll(peak_lrs, -1))
            ]
            return optax.sgdr_schedule(cosine_kwargs)
        lr = construct_sgdr_phases(config, total_grad_steps)
    else:
        lr = config['LR']
    return lr

def shuffle_and_batch(rng, batch, n_minibatches):
    first_leaf = jax.tree_util.tree_leaves(batch)[0]
    batch_size = first_leaf.shape[0] * first_leaf.shape[1]
    flat_batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    permutation = jax.random.permutation(rng, batch_size)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), flat_batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [n_minibatches, -1] + list(x.shape[1:])),
        shuffled_batch,
    )
    return minibatches

# Scaling Functions
def scale_rms(x):
    """
    Scales input by its Root Mean Square (RMS) without centering.
    
    This normalization technique:
    1. Preserves the original sign of every element (unlike z-score which shifts the mean).
    2. Scales the magnitude so the "typical" value is around 1.0.
    3. Is robust for off-policy advantages which may be non-zero mean.
    
    Formula: x / sqrt(mean(x^2) + epsilon)
    """
    # Compute Root Mean Square (sqrt of the mean of squares)
    rms = jnp.sqrt(jnp.mean(jnp.square(x)))
    
    # Scale x by RMS. Add epsilon for numerical stability.
    scaled_x = x / (rms + 1e-8)
    
    return scaled_x

def warmup_and_reset_stats(rng, env, env_params, num_envs, warmup_steps = 1000):
    
    # 1. Standard Reset
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    obsv, env_state = env.reset(reset_rng, env_params)

    # 2. Determine "Stop Steps" for each env
    # Each env will run for a random number of steps between 0 and 1000, then freeze.
    rng, key_stops = jax.random.split(rng)
    stop_thresholds = jax.random.randint(key_stops, (num_envs,), 0, warmup_steps)

    # 3. Desync Phase (Run then Freeze)
    def _step(carrier, step_idx):
        state, rng = carrier
        rng, action_key, step_key = jax.random.split(rng, 3)
        
        # A. Random Action
        action = jax.random.uniform(
            action_key, 
            shape=(num_envs, env.action_size), 
            minval=-1.0, 
            maxval=1.0
        )
        
        # B. Step Everyone
        step_keys = jax.random.split(step_key, num_envs)
        _, next_state, _, _, _ = env.step(step_keys, state, action, env_params)
        
        # C. Apply Freeze Logic
        # Active if current step < assigned stop threshold
        is_active = step_idx < stop_thresholds
        
        # Update State: If active, take next_state. If frozen, keep state.
        def where_active(next_x, curr_x):
            # Safety: Skip Python scalars (like count=1e-4) to avoid 'ndim' crash
            if not hasattr(next_x, 'ndim'):
                return next_x
            
            # Safety: Skip Global Stats (Mean/Var) that don't match batch dim
            if next_x.ndim == 0 or next_x.shape[0] != num_envs:
                return next_x

            # Broadcast Mask: (N,) -> (N, 1, 1...)
            mask = is_active.reshape(-1, *([1] * (next_x.ndim - 1)))
            return jnp.where(mask, next_x, curr_x)

        state = jax.tree_map(where_active, next_state, state)

        return (state, rng), None

    # Run Scan
    (env_state, rng), _ = jax.lax.scan(
        _step, (env_state, rng), jnp.arange(warmup_steps)
    )

    # 4. Reset Running Stats Only
    # Physics states are now decorrelated (stopped at different times).
    # Stats are currently garbage (due to frozen data), so we wipe them.
    env_state = env.reset_stats(env_state)

    # 5. Quick Refill
    # Run briefly with valid, moving agents to populate variance.
    def _refill(carrier, _):
        state, rng = carrier
        rng, action_key, step_key = jax.random.split(rng, 3)
        
        action = jax.random.uniform(
            action_key, 
            shape=(num_envs, env.action_size), 
            minval=-1.0, 
            maxval=1.0
        )
        step_keys = jax.random.split(step_key, num_envs)
        _, next_s, _, _, _ = env.step(step_keys, state, action, env_params)
        return (next_s, rng), None

    (env_state, rng), _ = jax.lax.scan(_refill, (env_state, rng), jnp.arange(1000))
    
    # 6. Final Observation
    rng, step_key, action_key = jax.random.split(rng, 3)
    action = jax.random.uniform(
        action_key, 
        shape=(num_envs, env.action_size), 
        minval=-1.0, 
        maxval=1.0
    )
    step_keys = jax.random.split(step_key, num_envs)
    obsv, env_state, _, _, _ = env.step(step_keys, env_state, action, env_params)

    return env_state, obsv, rng

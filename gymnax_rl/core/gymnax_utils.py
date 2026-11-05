import os
import json
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Dict, Optional


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    aux_targets: Optional[jnp.ndarray] = None
    dest_obs: Optional[jnp.ndarray] = None
    dest_targets: Optional[jnp.ndarray] = None
    hit_true_terminal: Optional[jnp.ndarray] = None
    hit_timeout: Optional[jnp.ndarray] = None
    hit_horizon: Optional[jnp.ndarray] = None
    start_weight: Optional[jnp.ndarray] = None
    critic_mask: Optional[jnp.ndarray] = None
    v_aux: Optional[jnp.ndarray] = None


def scale_rms(x):
    """
    Scales input by its Root Mean Square (RMS) without centering.
    Formula: x / sqrt(mean(x^2) + epsilon)
    """
    rms = jnp.sqrt(jnp.mean(jnp.square(x)))
    return x / (rms + 1e-8)


def shuffle_and_batch(rng: jax.Array, batch: Any, n_minibatches: int):
    """
    Shuffles a batch (PyTree of arrays where leading dims are [T, B, ...] or [N, ...])
    and slices it into `n_minibatches` along the flattened sample axis.
    """
    first_leaf = jax.tree_util.tree_leaves(batch)[0]
    if first_leaf.ndim >= 2:
        batch_size = first_leaf.shape[0] * first_leaf.shape[1]
        flat_batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
        )
    else:
        batch_size = first_leaf.shape[0]
        flat_batch = batch

    permutation = jax.random.permutation(rng, batch_size)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), flat_batch
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [n_minibatches, -1] + list(x.shape[1:])),
        shuffled_batch,
    )
    return minibatches


def post_process_advantage(advantages, config, weights=None):
    """
    Standardizes and clips advantages for PPO policy optimization.
    """
    std_floor = config.get("ADV_STD_FLOOR", 0.1)
    adv_clip = config.get("ADV_CLIP", 3.0)

    if weights is not None:
        if weights.ndim == 1 and advantages.ndim == 2:
            weights = weights[:, None]
        w = weights / jnp.sum(weights)
        mean_adv = jnp.sum(w * advantages)
        var_adv = jnp.sum(w * (advantages - mean_adv) ** 2)
        std_adv = jnp.sqrt(var_adv + 1e-8)
    else:
        mean_adv = jnp.mean(advantages)
        var_adv = jnp.var(advantages)
        std_adv = jnp.sqrt(var_adv + 1e-8)

    if std_floor is not None and std_floor > 0.0:
        denom = jnp.maximum(std_adv, std_floor)
    else:
        denom = std_adv + 1e-8

    adv_norm = (advantages - mean_adv) / denom

    if adv_clip is not None and adv_clip > 0.0:
        adv_norm = jnp.clip(adv_norm, -adv_clip, adv_clip)

    return jax.lax.stop_gradient(adv_norm)


def merge_hparams(config: Dict[str, Any], hparams: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merges a PyTree/dictionary of hyperparameters into the config."""
    if hparams is None:
        return config
    merged = config.copy()
    for k, v in hparams.items():
        merged[k] = v
        if k == "LR" and "LR_END" not in hparams and config.get("LR_END") is None:
            merged["LR_END"] = v
        if k == "ACTOR_LR" and "ACTOR_LR_END" not in hparams and config.get("ACTOR_LR_END") is None:
            merged["ACTOR_LR_END"] = v
        if k == "CRITIC_LR" and "CRITIC_LR_END" not in hparams and config.get("CRITIC_LR_END") is None:
            merged["CRITIC_LR_END"] = v
    return merged


def parse_config_override(config_str: Optional[str]) -> Dict[str, Any]:
    """Parse config override from command line argument."""
    if config_str is None:
        return {}
    try:
        cfg = json.loads(config_str)
    except json.JSONDecodeError:
        try:
            import ast
            cfg = ast.literal_eval(config_str)
        except Exception as e:
            print(f"Error parsing config override: {e}")
            print("Config override should be valid JSON or Python dict, e.g.: '{\"LR\": 0.001, \"LAMBDA\": 0.0}'")
            exit(1)

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, str):
                if v.lower() == "false":
                    cfg[k] = False
                elif v.lower() == "true":
                    cfg[k] = True
    return cfg


def json_default(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def save_config(config, env_dir):
    config_path = os.path.join(env_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=json_default)
    print(f"Config saved to {config_path}")


def save_results(data, config, env_name, env_dir):
    os.makedirs(env_dir, exist_ok=True)
    pickle_path = os.path.join(env_dir, "out.pkl")
    with open(pickle_path, "wb") as f:
        cloudpickle.dump(data, f)
    print(f"Results saved to {pickle_path}")
    save_config(config, env_dir)
    return env_dir


def save_plot(env_dir, env_name, steps_per_pi, episodic_return, title, logscale=False):
    y = jnp.asarray(episodic_return)
    if y.ndim == 0:
        y = y[None]
    if y.ndim != 1 or y.shape[0] == 0:
        return
    plt.figure()
    x = [i * steps_per_pi for i in range(int(y.shape[0]))]
    plt.plot(x, y, "o-", label=title)
    if logscale:
        plt.yscale("log")
    plt.xlabel("Env. Step")
    plt.ylabel(f"{title}")
    plt.title(env_name)
    plt.legend()
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def save_multi_plot(env_dir, env_name, steps_per_pi, metrics_dict, title="Value_Errors", ylabel="Mean Squared Error (MSVE)", log_scale=False):
    plt.figure(figsize=(10, 6))
    lines_plotted = 0
    for label, data in metrics_dict.items():
        y = jnp.asarray(data)
        if y.ndim == 0:
            y = y[None]
        if y.ndim != 1 or y.shape[0] == 0:
            continue
        x = [i * steps_per_pi for i in range(y.shape[0])]
        plt.plot(x, y, "o-", label=label, markersize=3, alpha=0.8)
        lines_plotted += 1

    if lines_plotted == 0:
        plt.close()
        return

    plt.xlabel("Env. Step")
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    plt.title(f"{env_name} - {title}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def load_run_data(run_folder_name, env_name, results_base_path="results"):
    run_path = os.path.join(results_base_path, run_folder_name, env_name)
    if not os.path.exists(run_path):
        for candidate_base in [results_base_path, "../results", "results", os.path.join("..", results_base_path)]:
            test_path = os.path.join(candidate_base, run_folder_name, env_name)
            if os.path.exists(test_path):
                run_path = test_path
                break
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "out.pkl")
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    with open(results_path, "rb") as pkl_file:
        results = cloudpickle.load(pkl_file)
    return config, results


def load_run_data_from_path(run_path):
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "out.pkl")
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    with open(results_path, "rb") as pkl_file:
        results = cloudpickle.load(pkl_file)
    return config, results

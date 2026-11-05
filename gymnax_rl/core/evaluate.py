import os
import jax
import jax.numpy as jnp
from gymnax_rl.core.gymnax_utils import save_results, save_plot, save_multi_plot


def evaluate(run_config, make_train, run_dir, args, rng):
    steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]

    run_fn = jax.jit(jax.vmap(make_train(run_config)))
    rngs = jax.random.split(rng, run_config["N_SEEDS"])
    out = run_fn(rngs)
    metrics = out["metrics"]
    ret = metrics.get("returned_episode_returns", metrics.get("returned_discounted_episode_returns", 0.0))
    print(f"[{run_config['ENV_NAME']}] Mean return: {jnp.mean(ret):.4f}")
    print(f"[{run_config['ENV_NAME']}] Max return:  {jnp.max(ret):.4f}")

    base_env_name = run_config["ENV_NAME"]
    env_size = run_config.get("ENV_SIZE")
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name
    env_dir = os.path.join(run_dir, full_env_name)
    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {full_env_name} results to {env_dir}")

    if getattr(args, "save_checkpoint", False):
        save_results(out, run_config, full_env_name, env_dir)
    elif getattr(args, "save_metrics", False):
        save_results(metrics, run_config, full_env_name, env_dir)
    else:
        save_results(run_config, run_config, full_env_name, env_dir)

    def _mean_over_seeds(data):
        arr = jnp.asarray(data)
        if arr.ndim > 0 and arr.shape[0] == run_config["N_SEEDS"]:
            arr = arr.mean(0)
        return arr

    def _extract_series(data):
        arr = _mean_over_seeds(data)
        if arr.ndim == 0:
            return arr[None]
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=tuple(range(1, arr.ndim)))

    def get_metric(name, slice_idx=0):
        if name not in metrics:
            return None
        series = _extract_series(metrics[name])
        return series[slice_idx:]

    standard_plots = {
        "returned_episode_returns": "returned_episode_returns",
        "returned_discounted_episode_returns": "returned_discounted_episode_returns",
        "returned_episode_lengths": "returned_episode_lengths",
        "value_loss": "value_loss",
        "actor_loss": "actor_loss",
        "entropy": "entropy",
        "approx_kl": "approx_kl",
        "clip_fraction": "clip_fraction",
    }
    for title, metric_name in standard_plots.items():
        data = get_metric(metric_name)
        if data is not None:
            save_plot(env_dir, full_env_name, steps_per_pi, data, title, logscale=False)

    return out

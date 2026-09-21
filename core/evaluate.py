# This file is responsible for running a single training run, called by runner.py
from core.utils import save_results, save_plot, save_multi_plot
import os
import jax
import jax.numpy as jnp


def evaluate(run_config, make_train, run_dir, args, rng):
    # Setup specific to this run_config
    steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]

    # JIT the train function for this specific config (important if env changes)
    run_fn = jax.jit(jax.vmap(make_train(run_config)))

    rngs = jax.random.split(rng, run_config["N_SEEDS"])
    out = run_fn(rngs)
    metrics = out["metrics"]
    ret = metrics.get("returned_episode_returns", metrics.get("returned_discounted_episode_returns", 0.0))
    print(f"[{run_config['ENV_NAME']}] Mean return: {jnp.mean(ret):.4f}")
    print(f"[{run_config['ENV_NAME']}] Max return:  {jnp.max(ret):.4f}")

    # Directory structure: results/run_dir/timestamp/EnvName-Size/
    base_env_name = run_config["ENV_NAME"]
    env_size = run_config.get("ENV_SIZE")

    # Create the full name (e.g., DeepSea-bsuite-45)
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name

    env_dir = os.path.join(run_dir, full_env_name)

    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {full_env_name} results to {env_dir}")

    # Ensure save_results uses the full name for the filename
    if args.save_checkpoint:
        save_results(out, run_config, full_env_name, env_dir)
    elif args.save_metrics:
        save_results(metrics, run_config, full_env_name, env_dir)
    else:  # save config only
        save_results(run_config, run_config, full_env_name, env_dir)

    # --- Helper for Metrics extraction ---
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

    for m_key, save_name in standard_plots.items():
        data = get_metric(m_key, 0)
        if data is not None:
            try:
                save_plot(env_dir, run_config["ENV_NAME"], steps_per_pi, data, save_name, logscale=False)
            except Exception as e:
                print(f"Failed to save plot for {m_key}: {e}")

    # Plot: Episode Returns (undiscounted return only)
    undisc_returns = get_metric("returned_episode_returns", 0)
    if undisc_returns is not None:
        try:
            save_plot(
                env_dir,
                run_config["ENV_NAME"],
                steps_per_pi,
                undisc_returns,
                "Episode_Returns",
                logscale=False,
            )
        except Exception as e:
            print(f"Failed to save Episode_Returns plot: {e}")

    # Multi-plot: v_pred_start with discounted returns in v_start / V_start
    v_start_dict = {}
    if "v_pred_start" in metrics:
        v_start_dict["v_pred_start"] = get_metric("v_pred_start", 0)
    elif "V_start" in metrics:
        v_start_dict["v_pred_start"] = get_metric("V_start", 0)
    if "returned_discounted_episode_returns" in metrics:
        v_start_dict["discounted_return"] = get_metric("returned_discounted_episode_returns", 0)

    if len(v_start_dict) > 0:
        for v_name in ["V_start", "v_start"]:
            try:
                save_multi_plot(
                    env_dir=env_dir,
                    env_name=run_config["ENV_NAME"],
                    steps_per_pi=steps_per_pi,
                    metrics_dict=v_start_dict,
                    title=v_name,
                    ylabel="Value / Return",
                    log_scale=False,
                )
            except Exception as e:
                print(f"Failed to save {v_name} multiplot: {e}")

    # Multi-plot: Training Losses
    loss_dict = {}
    for k, name in [("total_loss", "Total Loss"), ("value_loss", "Value Loss"), ("actor_loss", "Actor Loss")]:
        if k in metrics:
            loss_dict[name] = get_metric(k, 0)
    if len(loss_dict) > 1:
        try:
            save_multi_plot(
                env_dir=env_dir,
                env_name=run_config["ENV_NAME"],
                steps_per_pi=steps_per_pi,
                metrics_dict=loss_dict,
                title="Training Losses",
                ylabel="Loss",
                log_scale=False,
            )
        except Exception as e:
            print(f"Failed to save losses multiplot: {e}")

    # Multi-plot: Policy Diagnostics
    diag_dict = {}
    for k, name in [("entropy", "Entropy"), ("approx_kl", "Approx KL"), ("clip_fraction", "Clip Fraction")]:
        if k in metrics:
            diag_dict[name] = get_metric(k, 0)
    if len(diag_dict) > 1:
        try:
            save_multi_plot(
                env_dir=env_dir,
                env_name=run_config["ENV_NAME"],
                steps_per_pi=steps_per_pi,
                metrics_dict=diag_dict,
                title="Policy Diagnostics",
                ylabel="Value",
                log_scale=False,
            )
        except Exception as e:
            print(f"Failed to save policy diagnostics multiplot: {e}")

    if hasattr(args, "save_video") and args.save_video:
        try:
            train_state_seed0 = jax.tree_util.tree_map(lambda x: x[0], out["runner_state"][0])
            from core.video import generate_policy_video
            generate_policy_video(
                run_config=run_config,
                train_state=train_state_seed0,
                env_dir=env_dir,
                seed=run_config.get("SEED", 42),
                save_name="policy.gif"
            )
        except Exception as e:
            print("Failed to generate policy video:", e)

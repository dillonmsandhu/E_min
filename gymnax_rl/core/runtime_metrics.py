from __future__ import annotations
import jax
import jax.numpy as jnp


def compute_runtime_metrics(
    train_state,
    network,
    traj_batch,
    loss_info,
    start_obs,
    config,
) -> dict[str, jax.Array]:
    """
    Computes runtime metrics for sampled RL algorithms (PPO, etc.).
    """
    # 1. Episode Returns and Durations from LogWrapper
    info = traj_batch.info
    if "returned_episode" in info and "returned_episode_returns" in info:
        done_mask = info["returned_episode"].astype(jnp.float32)
        n_done = jnp.sum(done_mask)
        has_done = n_done > 0

        returned_episode_returns = jnp.where(
            has_done,
            jnp.sum(info["returned_episode_returns"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_episode_returns"].mean(),
        )
        returned_discounted_episode_returns = jnp.where(
            has_done,
            jnp.sum(info["returned_discounted_episode_returns"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_discounted_episode_returns"].mean(),
        )
        returned_episode_lengths = jnp.where(
            has_done,
            jnp.sum(info["returned_episode_lengths"] * done_mask) / jnp.maximum(n_done, 1.0),
            info["returned_episode_lengths"].mean(),
        )
    elif "returned_episode_returns" in info:
        returned_episode_returns = info["returned_episode_returns"].mean()
        returned_discounted_episode_returns = info.get(
            "returned_discounted_episode_returns", returned_episode_returns
        ).mean()
        returned_episode_lengths = info.get("returned_episode_lengths", jnp.array(0.0)).mean()
    else:
        returned_episode_returns = traj_batch.reward.sum(axis=0).mean()
        returned_discounted_episode_returns = returned_episode_returns
        returned_episode_lengths = jnp.array(traj_batch.obs.shape[0], dtype=jnp.float32)

    # 2. Start-State Value Prediction (V_start)
    if start_obs is not None:
        v_start_pred = network.apply(train_state.params, start_obs, method=network.value).mean()
    else:
        v_start_pred = traj_batch.value[0].mean()

    # 3. Loss Info Handling (supports tuple or dict)
    if isinstance(loss_info, dict):
        total_loss = loss_info.get("total_loss", jnp.array(0.0)).mean()
        value_loss = loss_info.get("value_loss", loss_info.get("val_loss", jnp.array(0.0))).mean()
        actor_loss = loss_info.get("actor_loss", jnp.array(0.0)).mean()
        entropy = loss_info.get("entropy", jnp.array(0.0)).mean()
    else:
        total_loss = loss_info[0].mean()
        value_loss = loss_info[1].mean()
        actor_loss = loss_info[2].mean()
        entropy = loss_info[3].mean()

    # 4. Policy Diagnostics on Rollout Batch
    pi = network.apply(train_state.params, traj_batch.obs, method=network.policy)
    new_log_prob = pi.log_prob(traj_batch.action)
    log_ratio = new_log_prob - traj_batch.log_prob
    ratio = jnp.exp(log_ratio)

    # Approximate KL divergence: E[exp(log_ratio) - 1 - log_ratio]
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

    # Clip fraction
    clip_eps = config.get("CLIP_EPS", 0.2)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    n_terminations = jnp.sum(true_terminal.astype(jnp.float32))
    n_truncations = jnp.sum(traj_batch.is_timeout.astype(jnp.float32))

    metrics = {
        "returned_episode_returns": returned_episode_returns,
        "returned_discounted_episode_returns": returned_discounted_episode_returns,
        "returned_episode_lengths": returned_episode_lengths,
        "mean_rew": traj_batch.reward.mean(),
        "V_start": v_start_pred,
        "v_pred_start": v_start_pred,
        "v_pred": traj_batch.value.mean(),
        "total_loss": total_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "n_terminations": n_terminations,
        "n_truncations": n_truncations,
    }

    if hasattr(network, "both"):
        v_main, aux_head = network.apply(train_state.params, traj_batch.obs, method=network.both)
        if config["E_VARIANT"].lower() == "e_aux":
            # aux head is e_phi(s) ~ V(s) - v_theta(s); compare to empirical error G^{lambda_aux} - v_theta
            e_target = traj_batch.aux_targets - v_main
            metrics["mse_e_aux"] = jnp.mean((aux_head - e_target) ** 2)
            metrics["e_aux_mean_abs"] = jnp.mean(jnp.abs(aux_head))
            metrics["e_target_mean_abs"] = jnp.mean(jnp.abs(e_target))
        else:
            target = traj_batch.aux_targets if traj_batch.aux_targets is not None else traj_batch.targets
            if target is not None:
                v_aux = aux_head
                metrics["mse_v_main"] = jnp.mean((v_main - target) ** 2)
                metrics["mse_v_aux"] = jnp.mean((v_aux - target) ** 2)
                norm_main = jnp.linalg.norm(v_main)
                norm_aux = jnp.linalg.norm(v_aux)
                cos_sim = jnp.dot(v_main.flatten(), v_aux.flatten()) / (norm_main * norm_aux + 1e-8)
                metrics["v_similarity"] = cos_sim

    if isinstance(loss_info, dict):
        for k, v in loss_info.items():
            if k not in metrics:
                metrics[k] = v.mean()

    for k, v in info.items():
        if k not in [
            "real_next_obs",
            "real_next_state",
            "is_timeout",
            "truncation",
            "next_targets",
            "aux_targets",
            "is_start",
            "returned_episode",
            "returned_episode_returns",
            "returned_discounted_episode_returns",
            "returned_episode_lengths",
            "obs_jump",
            "targets_jump",
            "valid_jump",
            "is_absorbing",
        ]:
            metrics[k] = v.mean()

    return metrics

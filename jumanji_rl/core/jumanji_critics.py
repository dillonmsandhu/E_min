# jumanji_rl/core/jumanji_critics.py
# Q-function critic and actor losses for discrete Jumanji environments

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from jumanji_rl.core.jumanji_utils import post_process_advantage


def extract_q_taken(q_all: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """Extracts Q(s, a) from discrete Q(s, ·) tensor of shape (..., |A|)."""
    return jnp.take_along_axis(q_all, action[..., None], axis=-1).squeeze(-1)


def compute_aux_head_loss(q_aux: jnp.ndarray, targets: jnp.ndarray, aux_coef: float = 1.0) -> jnp.ndarray:
    """Pure MSE loss training the auxiliary Q-head toward stop-gradient targets."""
    err = jax.lax.stop_gradient(targets) - q_aux
    return aux_coef * jnp.mean(0.5 * jnp.square(err))


def calculate_gae(traj_batch, next_value, gamma: float, gae_lambda: float):
    """
    Computes Generalized Advantage Estimation (GAE) with closed-form marginalized values.
    Bootstraps through timeouts using true continuation next_value,
    and severs GAE accumulation at episode boundaries (done=True).
    """
    if next_value.ndim == 1:
        regular_next = jnp.concatenate([traj_batch.value[1:], next_value[None]], axis=0)
        next_val_arr = traj_batch.next_value if traj_batch.next_value is not None else regular_next
    else:
        next_val_arr = next_value

    def _get_advantages(gae, data):
        done, is_timeout, reward, value, next_val = data
        true_terminal = done & ~is_timeout

        bootstrap_mask = 1.0 - true_terminal
        boundary_mask = 1.0 - done

        delta = reward + gamma * next_val * bootstrap_mask - value
        gae = delta + (gamma * gae_lambda * boundary_mask * gae)
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    scan_data = (
        traj_batch.done,
        traj_batch.is_timeout,
        traj_batch.reward,
        traj_batch.value,
        next_val_arr,
    )
    _, advantages = jax.lax.scan(
        _get_advantages,
        initial_accs,
        scan_data,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    return advantages, targets


def calculate_targets(traj_batch, next_value, gamma: float, target_lambda: float):
    """Computes lambda-return regression targets G_t^lambda."""
    _, targets = calculate_gae(traj_batch, next_value, gamma=gamma, gae_lambda=target_lambda)
    return targets


def get_critic_target_lambda(config: dict) -> float:
    """
    Fail-fast dispatch returning target_lambda according to non-defensive invariants:
      - sarsa / td: config['VALUE_LAMBDA']
      - e_0 / e_lambda: config['RETURN_LAMBDA']
    """
    critic_type = config["CRITIC_TYPE"].lower()
    if critic_type in ["sarsa", "td", "td_0", "td_lambda", "fitted", "mc"]:
        return config["VALUE_LAMBDA"]
    elif critic_type in ["e_0", "e_0_q", "e0", "e", "e_lambda", "e_lambda_q", "elambda"]:
        return config["RETURN_LAMBDA"]
    else:
        raise ValueError(f"Unrecognized CRITIC_TYPE '{critic_type}' for target lambda resolution.")


def ppo_actor_loss_fn(actor_params, critic_params, network, traj_batch, gae, config):
    """PPO clipped surrogate actor loss on discrete policy heads."""
    full_params = {"params": {**actor_params, **critic_params}}
    pi = network.apply(full_params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    adv_norm = post_process_advantage(gae, config)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    total_actor_loss = config["POLICY_COEFF"] * actor_loss - config["ENT_COEF"] * entropy
    return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}


def sarsa_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Fitted on-policy SARSA(lambda) regression loss for discrete Q(s_t, a_t)."""
    full_params = {"params": {**actor_params, **critic_params}}
    q_all = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_taken = extract_q_taken(q_all, traj_batch.action)

    targets_i = targets if targets is not None else traj_batch.targets
    loss = 0.5 * jnp.mean((targets_i - q_taken) ** 2)
    delta_mag = jnp.abs(targets_i - q_taken).mean()
    scaled_loss = config["VF_COEF"] * loss
    return scaled_loss, {"val_loss": loss, "delta_mag": delta_mag}


def e_0_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Sampled E(0) Dirichlet error loss on discrete state-action pairs:
        e_i = G_t - Q(s_t, a_t)
        e_j = G_{t+1} - Q(s_{t+1}, a_{t+1})  (0.0 on terminals/timeouts)
        L = (1 - gamma) * E[e_i^2] + 0.5 * gamma * E[(e_i - e_j)^2]
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_i = extract_q_taken(q_all_i, traj_batch.action)

    q_all_j = network.apply(full_params, traj_batch.next_obs, method=network.q_values)
    q_j = extract_q_taken(q_all_j, traj_batch.next_action)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout

    gamma = config["GAMMA"]

    e_i = targets_i - q_i
    e_j = jnp.where(true_terminal | is_timeout, 0.0, targets_j - q_j)

    weight_mc = 1.0 - gamma
    weight_dir = 0.5 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    laplacian_loss = weight_dir * jnp.mean((e_i - e_j) ** 2)

    value_loss = magnitude_loss + laplacian_loss
    scaled_loss = config["VF_COEF"] * value_loss

    delta_mag = jnp.abs(targets_i - q_i).mean()
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


def e_lambda_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Sampled E(lambda) Geometric Jump critic loss on discrete state-action pairs:
    Dispatches on config['E_VARIANT'] in ('aux', 'fwd', 'naive').
    When E_VARIANT='aux': Debiased via auxiliary critic head e_aux.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    targets_i = targets if targets is not None else traj_batch.targets
    targets_jump = traj_batch.dest_targets

    gamma = config["GAMMA"]
    lmbda = config["E_LAMBDA"]
    variant = config["E_VARIANT"].lower()
    gl = gamma * lmbda

    is_absorbing = traj_batch.hit_true_terminal
    valid_jump = traj_batch.valid_jump

    if variant == "aux":
        q_i, q_aux_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.both_q)
        v_jump, v_aux_jump = network.apply(full_params, traj_batch.dest_obs, method=network.both)

        e_start = targets_i - q_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        e_aux_start = sg(q_aux_i) - q_i
        e_aux_jump = jnp.where(is_absorbing, 0.0, sg(v_aux_jump) - v_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2
        aux_mc_loss = compute_aux_head_loss(q_aux_i, targets_i, aux_coef=config.get("AUX_VF_COEF", 1.0))
        e_for_mag = e_start
    elif variant == "fwd":
        q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
        q_i = extract_q_taken(q_all_i, traj_batch.action)
        v_jump = network.apply(full_params, traj_batch.dest_obs, method=network.value)

        e_start = targets_i - q_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        inner_diff_sq = (e_start - sg(e_jump)) ** 2
        aux_mc_loss = 0.0
        e_for_mag = e_start
    elif variant == "naive":
        q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
        q_i = extract_q_taken(q_all_i, traj_batch.action)
        q_all_jump = network.apply(full_params, traj_batch.dest_obs, method=network.q_values)
        q_jump = extract_q_taken(q_all_jump, traj_batch.dest_action)

        e_i = targets_i - q_i
        e_jump = targets_jump - q_jump
        inner_diff_sq = (e_i - e_jump) ** 2
        aux_mc_loss = 0.0
        e_for_mag = e_i
    else:
        raise ValueError(f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'fwd', 'naive').")

    diff_sq = jnp.where(
        is_absorbing,
        e_for_mag ** 2,
        jnp.where(valid_jump, inner_diff_sq, e_for_mag ** 2),
    )

    weight_mag = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    weight_dir = (0.5 * gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)

    magnitude_loss = weight_mag * jnp.mean(e_for_mag ** 2)
    dirichlet_loss = weight_dir * jnp.mean(diff_sq)

    value_loss = magnitude_loss + dirichlet_loss + aux_mc_loss
    scaled_loss = config["VF_COEF"] * value_loss

    delta_mag = jnp.abs(targets_i - e_for_mag).mean()
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "aux_loss": aux_mc_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


Q_CRITIC_LOSS_FNS = {
    "sarsa": sarsa_critic_loss_fn,
    "td": sarsa_critic_loss_fn,
    "td_0": sarsa_critic_loss_fn,
    "td_lambda": sarsa_critic_loss_fn,
    "fitted": sarsa_critic_loss_fn,
    "mc": sarsa_critic_loss_fn,
    "e_0": e_0_q_critic_loss_fn,
    "e_0_q": e_0_q_critic_loss_fn,
    "e0": e_0_q_critic_loss_fn,
    "e": e_0_q_critic_loss_fn,
    "e_lambda": e_lambda_q_critic_loss_fn,
    "e_lambda_q": e_lambda_q_critic_loss_fn,
    "elambda": e_lambda_q_critic_loss_fn,
}


def get_q_critic_loss_fn(critic_type: str):
    key = critic_type.lower()
    if key not in Q_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown Q-critic type '{critic_type}'. Available: {list(Q_CRITIC_LOSS_FNS.keys())}")
    return Q_CRITIC_LOSS_FNS[key]


get_critic_loss_fn = get_q_critic_loss_fn


def ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Standard MSE value loss for scalar V(s) Actor-Critic."""
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    targets_i = targets if targets is not None else traj_batch.targets
    loss = 0.5 * jnp.mean((targets_i - value) ** 2)
    delta_mag = jnp.abs(targets_i - value).mean()
    scaled_loss = config["VF_COEF"] * loss
    return scaled_loss, {"val_loss": loss, "delta_mag": delta_mag}

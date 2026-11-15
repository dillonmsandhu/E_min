"""Minimal fresh-batch discrete/continuous SAC with soft TD(lambda) or E.

One actor, one Q critic, Adam, and fixed or optionally learned temperature.
No replay, target network, twin critic, reward/observation normalization, gradient
clipping, or learning-rate schedule. The actor is frozen during critic fitting;
rollout-snapshot lambda returns supply the target-policy value estimates. Optional
RECOMPUTE_TARGETS_EACH_EPOCH refreshes their critic bootstrap, reusing the same
rollout and bootstrap-action draws. DIAGNOSTICS observes fitting and policy
changes without changing training randomness. This is an
experimental soft policy iteration baseline, not the full standard SAC algorithm.

Discrete actions use a categorical policy and vector Q(s,.). Bounded continuous
actions use an affine-tanh Gaussian and scalar Q(s,a), with reparameterized actor
gradients. Observation inputs must be vectors.

CRITIC_TYPE="td_lambda" regresses Q against soft lambda returns (TD_LAMBDA).
CRITIC_TYPE="td_0" uses an online one-step semi-gradient backup, refreshed on
every minibatch (including every epoch). AUTOTUNE_ALPHA learns a positive alpha
through log_alpha, updated once per actor minibatch toward TARGET_ENTROPY.
ALPHA is the initial value in that mode. The critic phase holds alpha fixed.
CRITIC_TYPE="e" minimizes half the bilinear E estimate, (G-Q) * delta,
using RETURN_LAMBDA for G. E_VARIANT="auto" selects exact action marginalization
for discrete actions and an independent successor-action probe for continuous
actions. E_VARIANT="double_sample" also works for discrete actions. The return
branch and probe use separate RNG streams; no second physics step is needed.
The E path requires deterministic transitions/rewards conditional on (s,a).
E(0) keeps e_hat=G-Q; refreshing bootstrap values does not identify e with delta.
Finite bootstrapped returns still introduce truncation bias; neither loss is
advertised as exact policy evaluation with an approximate tail bootstrap.
"""
from __future__ import annotations

import copy
from typing import NamedTuple

import flax.linen as nn
from flax.training.train_state import TrainState
import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
import numpy as np
import optax

from envs.log_wrapper import LogWrapper
from envs.wrappers import FlattenObservationWrapper, TerminalInfoWrapper


DEFAULT_CONFIG = {
    "ENV_NAME": "CartPole-v1", "TOTAL_TIMESTEPS": 300_000,
    "NUM_ENVS": 32, "NUM_STEPS": 32, "GAMMA": 0.99, "TD_LAMBDA": 0.95,
    "CRITIC_TYPE": "td_lambda", "RETURN_LAMBDA": 1.0, "E_LAMBDA": 0.8,
    "E_VARIANT": "auto", "ENV_PARAMS": {},
    # None infers determinism for verified built-ins; other E envs must opt in.
    "DETERMINISTIC_DYNAMICS": None,
    "ALPHA": 1.0, "HIDDEN_DIMS": (64, 64),
    "AUTOTUNE_ALPHA": True, "ALPHA_LR": 3e-3, "TARGET_ENTROPY": None,
    "ACTOR_LR": 3e-4, "CRITIC_LR": 1e-3,
    "MAX_GRAD_NORM": 1.0, "WEIGHT_DECAY": 0.01, "ADAM_EPS": 1e-5,
    "ACTOR_EPOCHS": 4, "CRITIC_EPOCHS": 16,
    "ACTOR_MINIBATCH_SIZE": 256, "CRITIC_MINIBATCH_SIZE": 256,
    "KL_COEFF": 0.5, "KL_TARGET": 0.03,
    "DIAGNOSTICS": False,
    "RECOMPUTE_TARGETS_EACH_EPOCH": True,
}


class MLP(nn.Module):
    output_dim: int
    hidden_dims: tuple = (64, 64)

    @nn.compact
    def __call__(self, obs):
        for width in self.hidden_dims:
            obs = nn.relu(nn.Dense(width)(obs))
        return nn.Dense(self.output_dim)(obs)


class GaussianActor(nn.Module):
    action_dim: int
    hidden_dims: tuple = (64, 64)

    @nn.compact
    def __call__(self, obs):
        mean, log_std = jnp.split(MLP(2 * self.action_dim, self.hidden_dims)(obs), 2, axis=-1)
        return mean, jnp.clip(log_std, -5.0, 2.0)


class ContinuousQ(nn.Module):
    hidden_dims: tuple = (64, 64)

    @nn.compact
    def __call__(self, obs, action):
        action = action.reshape(obs.shape[:-1] + (-1,))
        return MLP(1, self.hidden_dims)(jnp.concatenate([obs, action], axis=-1)).squeeze(-1)


def sample_squashed_gaussian(mean, log_std, rng, low, high, greedy=False):
    """Flat actions and log density in physical units, including both Jacobians."""
    noise = jnp.zeros_like(mean) if greedy else jax.random.normal(rng, mean.shape)
    latent = mean + jnp.exp(log_std) * noise
    scale, midpoint = (high - low) / 2.0, (high + low) / 2.0
    action = midpoint + scale * jnp.tanh(latent)
    normal_logp = -0.5 * (noise**2 + jnp.log(2.0 * jnp.pi)) - log_std
    log_det = 2.0 * (jnp.log(2.0) - latent - jax.nn.softplus(-2.0 * latent))
    return action, jnp.sum(normal_logp - log_det - jnp.log(scale), axis=-1)


def sample_policy(actor, params, obs, rng, action_space, greedy=False):
    if isinstance(action_space, spaces.Discrete):
        logits = actor.apply(params, obs)
        action = jnp.argmax(logits, axis=-1) if greedy else jax.random.categorical(rng, logits)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], -1)[..., 0]
        return action, logp
    mean, log_std = actor.apply(params, obs)
    low = jnp.broadcast_to(jnp.asarray(action_space.low), action_space.shape).reshape(-1)
    high = jnp.broadcast_to(jnp.asarray(action_space.high), action_space.shape).reshape(-1)
    action, logp = sample_squashed_gaussian(mean, log_std, rng, low, high, greedy)
    return action.reshape(obs.shape[:-1] + action_space.shape), logp


def q_for_action(critic, params, obs, action, action_space):
    if isinstance(action_space, spaces.Discrete):
        q = critic.apply(params, obs)
        return jnp.take_along_axis(q, action[..., None], -1)[..., 0]
    return critic.apply(params, obs, action)


def _make_components(config):
    env, params = gymnax.make(config["ENV_NAME"])
    if config.get("ENV_PARAMS"):
        params = params.replace(**config["ENV_PARAMS"])
    needs_flatten = len(env.observation_space(params).shape) != 1
    action_space = env.action_space(params)
    hidden = tuple(config["HIDDEN_DIMS"])
    if isinstance(action_space, spaces.Discrete):
        actor, critic = MLP(action_space.n, hidden), MLP(action_space.n, hidden)
    elif isinstance(action_space, spaces.Box):
        low = np.broadcast_to(np.asarray(action_space.low), action_space.shape)
        high = np.broadcast_to(np.asarray(action_space.high), action_space.shape)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()):
            raise ValueError("Continuous SAC requires finite action bounds with high > low")
        actor = GaussianActor(int(np.prod(action_space.shape)), hidden)
        critic = ContinuousQ(hidden)
    else:
        raise ValueError("SAC supports Discrete or bounded Box actions")
    return env, params, action_space, actor, critic, needs_flatten


def initialize_actor_params(config, rng):
    cfg = {**DEFAULT_CONFIG, **config}
    env, params, _, actor, _, needs_flatten = _make_components(cfg)
    if needs_flatten:
        env = FlattenObservationWrapper(env)
    return actor.init(rng, jnp.zeros(env.observation_space(params).shape))


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    is_timeout: jax.Array
    next_obs: jax.Array
    log_prob: jax.Array
    episode_return: jax.Array


def soft_value(logits, q_values, alpha):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.sum(jnp.exp(logp) * (q_values - alpha * logp), axis=-1)


def temperature_loss(log_alpha, entropy, target_entropy):
    """Log-temperature proxy: raise alpha when entropy falls below its target."""
    return log_alpha * jax.lax.stop_gradient(entropy - target_entropy)


def soft_td0_loss(q_taken, next_soft_value, reward, done, is_timeout, gamma):
    """Online semi-gradient TD(0); next values are freshly computed but detached."""
    terminal = done & ~is_timeout
    target = jax.lax.stop_gradient(reward + gamma * jnp.where(terminal, 0.0, next_soft_value))
    return 0.5 * jnp.mean((q_taken - target)**2)


def soft_e_loss(q_taken, next_q_values, next_logits, reward, done, is_timeout,
                returns, gamma, alpha):
    """Half the bilinear E estimate for a fixed policy and deterministic physics.

    e_hat = sg(G) - Q(s,a)
    delta = r + gamma E_pi[Q(s',a') - alpha log pi(a'|s')] - Q(s,a)
    L = 0.5 mean(e_hat * delta)

    Both Q terms in delta stay live, and e_hat differentiates through current Q.
    Thus the gradient is 0.5 E[-(e_hat+delta) phi(s,a)
    + gamma e_hat E_pi[phi(s',a')]], the gradient of half E when G is unbiased.
    Exact action averaging makes delta's continuation deterministic conditional
    on (s,a); G can follow the same trajectory without successor covariance.
    The actor and return bootstrap are detached. Unlike the expected stationary
    objective, a finite-batch/truncated estimate of this loss may be negative.

    True terminals have zero continuation. Timeouts bootstrap the actual final
    observation (not an autoreset observation), consistently with the G targets.
    """
    next_value = soft_value(jax.lax.stop_gradient(next_logits), next_q_values, alpha)
    terminal = done & ~is_timeout
    delta = reward + gamma * jnp.where(terminal, 0.0, next_value) - q_taken
    error = jax.lax.stop_gradient(returns) - q_taken
    return 0.5 * jnp.mean(error * delta)


def sampled_soft_e_loss(q_taken, probe_q, probe_logp, reward, done, is_timeout,
                        returns, gamma, alpha):
    """Half E using a successor action independent of the return-error branch.

    Only the successor action is independently sampled: deterministic s' is
    shared. G is built from the primary rollout and independent tail samples.
    Probe Q and both current-Q occurrences stay live. Probe log pi and G are
    detached. Do not reuse the return branch's continuation action as the probe.
    """
    terminal = done & ~is_timeout
    continuation = probe_q - alpha * jax.lax.stop_gradient(probe_logp)
    delta = reward + gamma * jnp.where(terminal, 0.0, continuation) - q_taken
    error = jax.lax.stop_gradient(returns) - q_taken
    return 0.5 * jnp.mean(error * delta)


def soft_lambda_targets(reward, done, is_timeout, next_value, log_prob,
                        gamma, lambda_, alpha):
    """Mixture of soft n-step Q returns; time is axis zero.

    Q excludes the current action's entropy bonus. For an interior step,
      G_t = r_t + gamma [(1-lambda) Vsoft(s_{t+1})
                         + lambda (G_{t+1} - alpha log pi(a_{t+1}|s_{t+1}))].
    Vsoft integrates Q - alpha log pi over actions (or uses an independent
    policy sample when integration is unavailable or sampled E is selected).
    At the rollout boundary or
    a timeout, bootstrap Vsoft directly; terminate the trace at every reset.
    A true terminal has zero continuation. Targets are detached for regression.
    """
    next_logp = jnp.concatenate([log_prob[1:], jnp.zeros_like(log_prob[:1])])
    trace = jnp.where(done, 0.0, lambda_)
    trace = trace.at[-1].set(0.0)
    terminal = done & ~is_timeout

    def backward(carry, step):
        r, end, v, next_lp, weight = step
        continuation = (1 - weight) * v + weight * (carry - alpha * next_lp)
        target = r + gamma * jnp.where(end, 0.0, continuation)
        return target, target

    _, targets = jax.lax.scan(
        backward, jnp.zeros_like(reward[0]),
        (reward, terminal, next_value, next_logp, trace), reverse=True)
    return jax.lax.stop_gradient(targets)


def soft_e_lambda_loss(
    q_start, q_aux_start, v_jump, v_aux_jump,
    targets, dest_targets,
    hit_true_terminal, hit_timeout, hit_horizon, start_weight,
    gamma, e_lambda,
):
    """Sampled E(lambda) soft Q-critic loss debiased via auxiliary Q critic.

    Term (2) (forward) differentiates through q_start using empirical return errors.
    Term (1) (backward) differentiates through v_jump using deterministic v_aux proxy errors.
    The auxiliary critic is trained independently via MSE regression on targets.
    """
    sg = jax.lax.stop_gradient
    e_start = sg(targets) - q_start
    e_jump = jnp.where(hit_true_terminal, 0.0, sg(dest_targets) - v_jump)
    e_aux_start = sg(q_aux_start) - q_start
    e_aux_jump = jnp.where(hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

    inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2
    diff_sq = jnp.where(hit_true_terminal | hit_timeout | hit_horizon, e_start ** 2, inner_diff_sq)

    gl = gamma * e_lambda
    denom = jnp.maximum(1.0 - gl, 1e-8)
    tilde_gamma = (gamma * (1.0 - e_lambda)) / denom
    magnitude_weight = 0.5 * (1.0 - gamma) / denom
    dirichlet_weight = 0.5 * tilde_gamma

    main_loss = (
        magnitude_weight * jnp.mean(e_start ** 2)
        + dirichlet_weight * jnp.mean(diff_sq)
        + jnp.mean(start_weight * (e_start ** 2))
    )
    aux_loss = 0.5 * jnp.mean((q_aux_start - sg(targets)) ** 2)
    return main_loss + aux_loss


def _prepare_e_lambda_jumps(batch, gamma, e_lambda, rng, steps_since_start):
    """Sample geometric jump destinations j = min(t + m, T) and boundary flags."""
    T, B = batch.reward.shape
    gl = gamma * e_lambda
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    u = jax.random.uniform(rng, shape=(T, B))
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T)[:, None]
    j_idx = jnp.minimum(t_arr + m, T)

    is_timeout = batch.is_timeout
    true_terminals = batch.done & ~is_timeout
    term_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)], axis=0)
    timeout_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)], axis=0)
    hit_true_terminal = (jnp.take_along_axis(term_cumsum, j_idx, axis=0) - term_cumsum[:T]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum, j_idx, axis=0) - timeout_cumsum[:T]) > 0
    hit_horizon = (t_arr + m) > T

    def scan_steps(k, flags):
        term, timeout = flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    next_steps_since_start, all_steps = jax.lax.scan(scan_steps, steps_since_start, (true_terminals, is_timeout))
    tilde_gamma = (gamma * (1.0 - e_lambda)) / jnp.maximum(1.0 - gl, 1e-8)
    start_trace = jnp.where(gl < 1e-6, jnp.where(all_steps == 0, 1.0, 0.0), safe_gl ** all_steps)
    start_weight = 0.25 * tilde_gamma * start_trace

    obs_ext = jnp.concatenate([batch.obs, batch.next_obs[-1][None]], axis=0)
    j_obs = jnp.broadcast_to(j_idx.reshape(j_idx.shape + (1,) * (obs_ext.ndim - 2)), (T, B) + obs_ext.shape[2:])
    dest_obs = jnp.take_along_axis(obs_ext, j_obs, axis=0)
    return (dest_obs, j_idx, hit_true_terminal, hit_timeout, hit_horizon, start_weight), next_steps_since_start


def _minibatches(rng, batch, batch_size, minibatch_size):
    permutation = jax.random.permutation(rng, batch_size)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:])[permutation].reshape(
            (batch_size // minibatch_size, minibatch_size) + x.shape[2:]), batch)


def make_train(overrides=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(overrides or {})
    critic_aliases = {"td_lambda": "td_lambda", "td_0": "td_0", "td0": "td_0",
                      "online_td0": "td_0", "e": "e", "e_0": "e", "e_min": "e",
                      "e_lambda": "e_lambda", "elambda": "e_lambda", "e_geometric": "e_lambda"}
    critic_type = critic_aliases.get(str(config["CRITIC_TYPE"]).lower())
    if critic_type is None:
        raise ValueError("CRITIC_TYPE must be 'td_lambda', 'td_0', 'e', or 'e_lambda'")
    if config["E_VARIANT"] not in ("auto", "marginal", "double_sample"):
        raise ValueError("E_VARIANT must be 'auto', 'marginal', or 'double_sample'")
    for key in ("NUM_ENVS", "NUM_STEPS", "ACTOR_EPOCHS", "CRITIC_EPOCHS",
                "ACTOR_MINIBATCH_SIZE", "CRITIC_MINIBATCH_SIZE"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
    updates = config["TOTAL_TIMESTEPS"] // batch_size
    if updates < 1:
        raise ValueError("TOTAL_TIMESTEPS must cover at least one rollout")
    if (not 0 <= config["TD_LAMBDA"] <= 1 or not 0 <= config["RETURN_LAMBDA"] <= 1
            or not 0 <= config["E_LAMBDA"] <= 1 or not 0 <= config["GAMMA"] < 1):
        raise ValueError("Require TD_LAMBDA, RETURN_LAMBDA, E_LAMBDA in [0,1], GAMMA in [0,1)")
    if config["ALPHA"] < 0:
        raise ValueError("ALPHA must be nonnegative")
    if config["MAX_GRAD_NORM"] <= 0 or config["WEIGHT_DECAY"] < 0 or config["ADAM_EPS"] <= 0:
        raise ValueError("Require MAX_GRAD_NORM > 0, WEIGHT_DECAY >= 0, ADAM_EPS > 0")
    if config["KL_COEFF"] < 0:
        raise ValueError("KL_COEFF must be nonnegative")
    if config["KL_TARGET"] is not None:
        if config["KL_TARGET"] <= 0:
            raise ValueError("KL_TARGET must be positive when set")
        if config["KL_COEFF"] <= 0:
            raise ValueError("Adaptive KL requires positive initial KL_COEFF")
    if config["AUTOTUNE_ALPHA"] and (config["ALPHA"] <= 0 or config["ALPHA_LR"] <= 0):
        raise ValueError("Adaptive temperature requires positive initial ALPHA and ALPHA_LR")
    for role in ("ACTOR", "CRITIC"):
        key = f"{role}_MINIBATCH_SIZE"
        config[key] = min(config[key], batch_size)
        if batch_size % config[key]:
            raise ValueError(f"{key} must divide the rollout batch size")

    raw_env, env_params, action_space, actor, critic, needs_flatten = _make_components(config)
    discrete = isinstance(action_space, spaces.Discrete)
    action_dim = 1 if discrete else int(np.prod(action_space.shape))
    target_entropy = config["TARGET_ENTROPY"]
    if target_entropy is None:
        if discrete:
            target_entropy = 0.45 * np.log(action_space.n)
        else:
            # Target differential entropy in physical action units: H_max - 0.39 * dim.
            max_continuous_entropy = float(np.log(np.broadcast_to(
                np.asarray(action_space.high) - np.asarray(action_space.low),
                action_space.shape)).sum())
            target_entropy = max_continuous_entropy - 0.39 * int(np.prod(action_space.shape))
    if not np.isfinite(target_entropy):
        raise ValueError("TARGET_ENTROPY must be finite")
    if config["AUTOTUNE_ALPHA"]:
        max_entropy = (np.log(action_space.n) if discrete else np.log(np.broadcast_to(
            np.asarray(action_space.high) - np.asarray(action_space.low), action_space.shape)).sum())
        if target_entropy > max_entropy or (discrete and target_entropy < 0):
            raise ValueError("TARGET_ENTROPY is outside the action space's achievable range")
    e_variant = config["E_VARIANT"]
    if e_variant == "auto":
        e_variant = "marginal" if discrete else "double_sample"
    if critic_type == "e" and e_variant == "marginal" and not discrete:
        raise ValueError("Exact policy marginalization requires Discrete actions; use double_sample for Box")
    deterministic = config["DETERMINISTIC_DYNAMICS"]
    if deterministic is None:
        deterministic = config["ENV_NAME"] in (
            "CartPole-v1", "Pendulum-v1", "MountainCarContinuous-v0", "MountainCar-v0", "Acrobot-v1")
    if critic_type == "e" and not deterministic:
        raise ValueError(
            "Both E variants require deterministic transitions/rewards; two actions "
            "alone do not correct stochastic physics. "
            "For a verified deterministic environment, set DETERMINISTIC_DYNAMICS=True.")
    target_lambda = (config["RETURN_LAMBDA"] if critic_type in ("e", "e_lambda") else
                     0.0 if critic_type == "td_0" else config["TD_LAMBDA"])
    wrapped_env = TerminalInfoWrapper(raw_env)
    if needs_flatten:
        wrapped_env = FlattenObservationWrapper(wrapped_env)
    env = LogWrapper(wrapped_env, gamma=config["GAMMA"])
    obs_shape = env.observation_space(env_params).shape
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def _adamw_tx(lr):
        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(lr, weight_decay=config["WEIGHT_DECAY"], eps=config["ADAM_EPS"]),
        )

    def train(rng):
        if critic_type == "e_lambda":
            rng, ak, ck, aux_ck, ek = jax.random.split(rng, 5)
        else:
            rng, ak, ck, ek = jax.random.split(rng, 4)
            aux_ck = ck
        init_obs = jnp.zeros(obs_shape)
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor.init(ak, init_obs),
                                        tx=_adamw_tx(config["ACTOR_LR"]))
        critic_args = (init_obs,) if discrete else (init_obs, jnp.zeros(action_space.shape))
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic.init(ck, *critic_args),
                                         tx=_adamw_tx(config["CRITIC_LR"]))
        aux_critic_state = TrainState.create(apply_fn=critic.apply, params=critic.init(aux_ck, *critic_args),
                                             tx=_adamw_tx(config["CRITIC_LR"]))
        temperature_state = TrainState.create(
            apply_fn=None, params={"log_alpha": jnp.log(max(config["ALPHA"], 1e-8))},
            tx=optax.adam(config["ALPHA_LR"]))
        kl_coeff = jnp.asarray(config["KL_COEFF"], dtype=jnp.float32)
        steps_since_start = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        obs, env_state = reset(jax.random.split(ek, config["NUM_ENVS"]), env_params)

        def update(runner, _):
            actor_state, critic_state, env_state, obs, rng, temperature_state, kl_coeff, aux_critic_state, steps_since_start = runner
            alpha = (jnp.exp(temperature_state.params["log_alpha"])
                     if config["AUTOTUNE_ALPHA"] else config["ALPHA"])

            def collect(carry, _):
                env_state, obs, rng = carry
                rng, ak, ek = jax.random.split(rng, 3)
                action, logp = sample_policy(actor, actor_state.params, obs, ak, action_space)
                next_obs, env_state, reward, done, info = step(
                    jax.random.split(ek, config["NUM_ENVS"]), env_state, action, env_params)
                transition = Transition(obs, action, reward, done, info["is_timeout"],
                                        info["real_next_obs"], logp, info["returned_episode_returns"])
                return (env_state, next_obs, rng), transition

            (env_state, obs, rng), batch = jax.lax.scan(
                collect, (env_state, obs, rng), None, config["NUM_STEPS"])
            # The return/bootstrap branch never uses the critic probe's RNG key.
            if discrete and (critic_type != "e" or e_variant == "marginal"):
                next_logits = actor.apply(actor_state.params, batch.next_obs)
                next_value = soft_value(next_logits,
                                        critic.apply(critic_state.params, batch.next_obs), alpha)
            else:
                rng, return_key = jax.random.split(rng)
                return_action, return_logp = sample_policy(
                    actor, actor_state.params, batch.next_obs, return_key, action_space)
                next_value = q_for_action(critic, critic_state.params, batch.next_obs,
                                          return_action, action_space) - alpha * return_logp
                # Keep one array-only batch structure for both action-space types.
                next_logits = jnp.zeros(batch.reward.shape + (0,))
            targets = soft_lambda_targets(
                batch.reward, batch.done, batch.is_timeout, next_value,
                batch.log_prob, config["GAMMA"], target_lambda, alpha)

            last_val = jnp.where(batch.done[-1] & ~batch.is_timeout[-1], 0.0, next_value[-1])

            def recompute_targets(params):
                # Actor and bootstrap-action draws remain fixed for this batch.
                # Only the critic estimate in the return bootstrap is refreshed.
                if discrete and (critic_type != "e" or e_variant == "marginal"):
                    value = soft_value(next_logits, critic.apply(params, batch.next_obs), alpha)
                else:
                    value = q_for_action(critic, params, batch.next_obs,
                                          return_action, action_space) - alpha * return_logp
                new_targets = soft_lambda_targets(
                    batch.reward, batch.done, batch.is_timeout, value,
                    batch.log_prob, config["GAMMA"], target_lambda, alpha)
                new_last_val = jnp.where(batch.done[-1] & ~batch.is_timeout[-1], 0.0, value[-1])
                return new_targets, new_last_val

            if config["DIAGNOSTICS"]:
                q_before_fit = q_for_action(critic, critic_state.params, batch.obs, batch.action, action_space)
                fit_mse_before = jnp.mean((q_before_fit - targets)**2)

            if critic_type == "e_lambda":
                rng, jump_key = jax.random.split(rng)
                jump_data, steps_since_start = _prepare_e_lambda_jumps(
                    batch, config["GAMMA"], config["E_LAMBDA"], jump_key, steps_since_start)
                dest_obs, j_idx, hit_true_terminal, hit_timeout, hit_horizon, start_weight = jump_data

            def critic_epoch(carry, _):
                state, aux_state, rng = carry
                rng, key = jax.random.split(rng)
                if config["RECOMPUTE_TARGETS_EACH_EPOCH"]:
                    epoch_targets, epoch_last_val = recompute_targets(state.params)
                else:
                    epoch_targets, epoch_last_val = targets, last_val
                if critic_type == "e_lambda":
                    state_targets = jnp.concatenate(
                        [epoch_targets - alpha * batch.log_prob, epoch_last_val[None]], axis=0)
                    dest_targets = jnp.take_along_axis(state_targets, j_idx, axis=0)
                    mb_input = (batch, epoch_targets, next_logits, dest_obs, dest_targets,
                                hit_true_terminal, hit_timeout, hit_horizon, start_weight)
                else:
                    mb_input = (batch, epoch_targets, next_logits)
                minibatches = _minibatches(key, mb_input, batch_size, config["CRITIC_MINIBATCH_SIZE"])

                def fit(carry, mb):
                    state, aux_state, rng = carry
                    if critic_type == "e_lambda":
                        (traj, target, next_policy_logits, mb_dest_obs, mb_dest_target,
                         mb_hit_term, mb_hit_to, mb_hit_hor, mb_sw) = mb
                        if discrete:
                            mb_dest_logits = jax.lax.stop_gradient(actor.apply(actor_state.params, mb_dest_obs))
                        else:
                            rng, dest_key = jax.random.split(rng)
                            mb_dest_action, mb_dest_logp = sample_policy(
                                actor, actor_state.params, mb_dest_obs, dest_key, action_space)
                            mb_dest_action, mb_dest_logp = jax.tree_util.tree_map(
                                jax.lax.stop_gradient, (mb_dest_action, mb_dest_logp))
                    else:
                        traj, target, next_policy_logits = mb
                    if critic_type == "e" and e_variant == "double_sample":
                        rng, probe_key = jax.random.split(rng)
                        probe_action, probe_logp = sample_policy(
                            actor, actor_state.params, traj.next_obs, probe_key, action_space)
                        probe_action, probe_logp = jax.tree_util.tree_map(
                            jax.lax.stop_gradient, (probe_action, probe_logp))
                    if critic_type == "td_0" and not discrete:
                        rng, td_key = jax.random.split(rng)
                        td_action, td_logp = sample_policy(
                            actor, actor_state.params, traj.next_obs, td_key, action_space)

                    def loss_fn(params_pair):
                        p, aux_p = params_pair
                        taken = q_for_action(critic, p, traj.obs, traj.action, action_space)
                        if critic_type == "e_lambda":
                            aux_taken = q_for_action(critic, aux_p, traj.obs, traj.action, action_space)
                            if discrete:
                                v_jump = soft_value(mb_dest_logits, critic.apply(p, mb_dest_obs), alpha)
                                v_aux_jump = soft_value(mb_dest_logits, critic.apply(aux_p, mb_dest_obs), alpha)
                            else:
                                v_jump = critic.apply(p, mb_dest_obs, mb_dest_action) - alpha * mb_dest_logp
                                v_aux_jump = critic.apply(aux_p, mb_dest_obs, mb_dest_action) - alpha * mb_dest_logp
                            return soft_e_lambda_loss(
                                taken, aux_taken, v_jump, v_aux_jump, target, mb_dest_target,
                                mb_hit_term, mb_hit_to, mb_hit_hor, mb_sw,
                                config["GAMMA"], config["E_LAMBDA"])
                        if critic_type == "e":
                            if e_variant == "double_sample":
                                probe_q = q_for_action(critic, p, traj.next_obs, probe_action, action_space)
                                return sampled_soft_e_loss(
                                    taken, probe_q, probe_logp, traj.reward, traj.done,
                                    traj.is_timeout, target, config["GAMMA"], alpha)
                            next_q = critic.apply(p, traj.next_obs)
                            return soft_e_loss(
                                taken, next_q, next_policy_logits, traj.reward,
                                traj.done, traj.is_timeout, target,
                                config["GAMMA"], alpha)
                        if critic_type == "td_0":
                            if discrete:
                                value = soft_value(next_policy_logits, critic.apply(p, traj.next_obs), alpha)
                            else:
                                value = critic.apply(p, traj.next_obs, td_action) - alpha * td_logp
                            return soft_td0_loss(taken, value, traj.reward, traj.done,
                                                  traj.is_timeout, config["GAMMA"])
                        return 0.5 * jnp.mean((taken - target)**2)

                    loss, (grad, aux_grad) = jax.value_and_grad(loss_fn)((state.params, aux_state.params))
                    state = state.apply_gradients(grads=grad)
                    if critic_type == "e_lambda":
                        aux_state = aux_state.apply_gradients(grads=aux_grad)
                    return (state, aux_state, rng), loss

                (state, aux_state, rng), losses = jax.lax.scan(fit, (state, aux_state, rng), minibatches)
                return (state, aux_state, rng), losses.mean()

            (critic_state, aux_critic_state, rng), critic_losses = jax.lax.scan(
                critic_epoch, (critic_state, aux_critic_state, rng), None, config["CRITIC_EPOCHS"])
            if config["DIAGNOSTICS"]:
                q_after_fit = q_for_action(critic, critic_state.params, batch.obs, batch.action, action_space)
                fit_targets = (recompute_targets(critic_state.params)[0]
                               if config["RECOMPUTE_TARGETS_EACH_EPOCH"] or critic_type == "td_0" else targets)
                fit_mse_after = jnp.mean((q_after_fit - fit_targets)**2)
            if discrete:
                old_ref = jax.lax.stop_gradient(actor.apply(actor_state.params, batch.obs))
            else:
                old_mean, old_log_std = actor.apply(actor_state.params, batch.obs)
                old_ref = (jax.lax.stop_gradient(old_mean), jax.lax.stop_gradient(old_log_std))

            def actor_epoch(carry, _):
                state, rng, temperature_state = carry
                rng, key = jax.random.split(rng)
                minibatches = _minibatches(key, (batch.obs, old_ref), batch_size, config["ACTOR_MINIBATCH_SIZE"])

                def improve(carry, mb):
                    state, rng, temperature_state = carry
                    x, ref = mb
                    actor_alpha = (jnp.exp(temperature_state.params["log_alpha"])
                                   if config["AUTOTUNE_ALPHA"] else alpha)
                    if discrete:
                        fixed_q = critic.apply(critic_state.params, x)
                    else:
                        rng, actor_key = jax.random.split(rng)

                    def loss_fn(p):
                        if discrete:
                            logits = actor.apply(p, x)
                            loss = -jnp.mean(soft_value(logits, fixed_q, actor_alpha))
                            if config["KL_COEFF"] > 0 or config["KL_TARGET"] is not None:
                                lp = jax.nn.log_softmax(logits)
                                old_lp = jax.nn.log_softmax(ref)
                                kl = jnp.sum(jnp.exp(lp) * (lp - old_lp), axis=-1).mean()
                                loss = loss + kl_coeff * kl
                            return loss
                        action, logp = sample_policy(actor, p, x, actor_key, action_space)
                        q = critic.apply(critic_state.params, x, action)
                        loss = jnp.mean(actor_alpha * logp - q)
                        if config["KL_COEFF"] > 0 or config["KL_TARGET"] is not None:
                            m, ls = actor.apply(p, x)
                            om, ols = ref
                            kl = ols - ls + (jnp.exp(2 * ls) + (m - om)**2) / (2 * jnp.exp(2 * ols)) - 0.5
                            loss = loss + kl_coeff * jnp.sum(kl, axis=-1).mean()
                        return loss

                    loss, grad = jax.value_and_grad(loss_fn)(state.params)
                    state = state.apply_gradients(grads=grad)
                    alpha_loss = jnp.asarray(0.0)
                    if config["AUTOTUNE_ALPHA"]:
                        if discrete:
                            lp = jax.nn.log_softmax(actor.apply(state.params, x))
                            policy_entropy = -jnp.sum(jnp.exp(lp) * lp, axis=-1).mean()
                        else:
                            rng, temperature_key = jax.random.split(rng)
                            _, lp = sample_policy(actor, state.params, x, temperature_key, action_space)
                            policy_entropy = -lp.mean()

                        def alpha_loss_fn(p):
                            return temperature_loss(p["log_alpha"], policy_entropy, target_entropy)

                        alpha_loss, grad = jax.value_and_grad(alpha_loss_fn)(temperature_state.params)
                        temperature_state = temperature_state.apply_gradients(grads=grad)
                    return (state, rng, temperature_state), (loss, alpha_loss)

                (state, rng, temperature_state), (losses, alpha_losses) = jax.lax.scan(
                    improve, (state, rng, temperature_state), minibatches)
                return (state, rng, temperature_state), (losses.mean(), alpha_losses.mean())

            (actor_state, rng, temperature_state), (actor_losses, alpha_losses) = jax.lax.scan(
                actor_epoch, (actor_state, rng, temperature_state), None, config["ACTOR_EPOCHS"])
            if discrete:
                logp = jax.nn.log_softmax(actor.apply(actor_state.params, batch.obs))
                old_logp = jax.nn.log_softmax(old_ref)
                step_kl = jnp.sum(jnp.exp(logp) * (logp - old_logp), axis=-1).mean()
                entropy = -jnp.mean(jnp.sum(jnp.exp(logp) * logp, axis=-1))
                q_mean = critic.apply(critic_state.params, batch.obs).mean()
            else:
                rng, entropy_key = jax.random.split(rng)
                entropy_action, logp = sample_policy(actor, actor_state.params, batch.obs, entropy_key, action_space)
                mean, log_std = actor.apply(actor_state.params, batch.obs)
                step_kl = jnp.sum(
                    old_log_std - log_std + (jnp.exp(2 * log_std) + (mean - old_mean)**2) / (
                        2 * jnp.exp(2 * old_log_std)) - 0.5, axis=-1).mean()
                entropy = -logp.mean()
                q_mean = critic.apply(critic_state.params, batch.obs, batch.action).mean()
            if config["KL_TARGET"] is not None:
                kl_target = config["KL_TARGET"] * action_dim
                kl_coeff = jnp.where(
                    step_kl > 1.5 * kl_target,
                    jnp.minimum(kl_coeff * 2.0, 100.0),
                    jnp.where(step_kl < kl_target / 1.5, jnp.maximum(kl_coeff * 0.5, 1e-4), kl_coeff),
                )
            completed = batch.done.sum()
            metrics = {
                "critic_loss": critic_losses.mean(), "actor_loss": actor_losses.mean(),
                "entropy": entropy,
                "completed_episodes": completed,
                "episode_return_sum": jnp.sum(jnp.where(batch.done, batch.episode_return, 0.0)),
                "q_mean": q_mean,
                "alpha": (jnp.exp(temperature_state.params["log_alpha"])
                          if config["AUTOTUNE_ALPHA"] else jnp.asarray(alpha)),
                "alpha_loss": alpha_losses.mean(),
                "target_entropy": jnp.asarray(target_entropy),
                "actor_step_kl": step_kl,
                "kl_coeff": kl_coeff,
            }
            if config["DIAGNOSTICS"]:
                metrics.update({
                    "critic_fit_mse_before": fit_mse_before,
                    "critic_fit_mse_after": fit_mse_after,
                    "critic_fit_mse_ratio": fit_mse_after / (fit_mse_before + 1e-8),
                    "critic_target_mean": targets.mean(),
                    "critic_target_std": targets.std(),
                })
                if not discrete:
                    scale = (jnp.asarray(action_space.high) - jnp.asarray(action_space.low)) / 2
                    midpoint = (jnp.asarray(action_space.high) + jnp.asarray(action_space.low)) / 2
                    metrics.update({
                        "actor_log_std_mean": log_std.mean(),
                        "actor_mean_abs": jnp.abs(mean).mean(),
                        "action_saturation_fraction": (
                            jnp.abs((entropy_action - midpoint) / scale) > 0.95).mean(),
                    })
            return (actor_state, critic_state, env_state, obs, rng, temperature_state,
                    kl_coeff, aux_critic_state, steps_since_start), metrics

        runner, metrics = jax.lax.scan(
            update, (actor_state, critic_state, env_state, obs, rng, temperature_state,
                     kl_coeff, aux_critic_state, steps_since_start),
            None, updates)
        return {"runner_state": runner[:5], "temperature_state": runner[5],
                "kl_coeff": runner[6], "metrics": metrics}

    return train


def make_evaluate(config=None, num_episodes=32, greedy=False):
    """Fresh-start raw-return evaluation; no training rollouts are reused."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    env, params, action_space, actor, _, needs_flatten = _make_components(cfg)
    if needs_flatten:
        env = FlattenObservationWrapper(env)
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def evaluate(actor_params, rng):
        rng, key = jax.random.split(rng)
        obs, state = reset(jax.random.split(key, num_episodes), params)

        def tick(carry, _):
            state, obs, alive, returns, rng = carry
            rng, ak, ek = jax.random.split(rng, 3)
            action, _ = sample_policy(actor, actor_params, obs, ak, action_space, greedy)
            obs, state, reward, done, _ = step(
                jax.random.split(ek, num_episodes), state, action, params)
            returns = returns + jnp.where(alive, reward, 0.0)
            return (state, obs, alive & ~done, returns, rng), None

        carry = (state, obs, jnp.ones(num_episodes, dtype=bool), jnp.zeros(num_episodes), rng)
        carry, _ = jax.lax.scan(tick, carry, None, params.max_steps_in_episode)
        return carry[3]

    return evaluate

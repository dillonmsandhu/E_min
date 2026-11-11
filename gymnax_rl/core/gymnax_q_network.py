# gymnax_q_network.py
# Q-function Actor-Critic networks for:
# 1. Discrete action spaces (Q-PPO) with analytical marginalization: V(s) = sum_{a in A} pi(a | s) * Q(s, a)
# 2. Continuous action spaces (On-Policy DDPG) with deterministic actor mu(s) and state-action critic Q(s, a):
#    V(s) = Q(s, mu(s))
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant
import optax
import distrax
from typing import NamedTuple, Any
from gymnax.environments import spaces

from gymnax_rl.core.gymnax_network import (
    MLP,
    DualTrainState,
    initialize_split_flax_train_state,
    uses_aux_network,
)


class DiscretePolicyHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> distrax.Categorical:
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=logits)


class MLP_Q_AC(nn.Module):
    """
    Discrete Actor-Critic architecture outputting Q(s, ·) in R^{|A|} and pi(· | s).
    Marginalized state value V(s) is computed analytically:
        V(s) = sum_{a} pi(a | s) * Q(s, a)
    """
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = DiscretePolicyHead(self.action_dim)
        self.q_head = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi, V(s), Q(s, ·)) where V(s) is marginalized value."""
        pi = self.policy(x)
        q = self.q_values(x)
        v = jnp.sum(pi.probs * q, axis=-1)
        return pi, v, q

    def policy(self, x: jnp.ndarray):
        phi_a = self.actor_mlp(x)
        return self.actor_head(phi_a)

    def q_values(self, x: jnp.ndarray):
        phi_c = self.critic_mlp(x)
        return self.q_head(phi_c)

    def value(self, x: jnp.ndarray):
        """Analytical marginalization: V(s) = E_{a ~ pi}[Q(s, a)] = sum_a pi(a|s) Q(s, a)."""
        pi = self.policy(x)
        q = self.q_values(x)
        return jnp.sum(pi.probs * q, axis=-1)


class PQN_Q_AC(nn.Module):
    """CNN-based Q Actor-Critic for visual/grid observations (MinAtar)."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"

    def setup(self):
        from gymnax_rl.core.gymnax_network import PQN_CNN
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = DiscretePolicyHead(self.action_dim)
        self.q_head = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        v = jnp.sum(pi.probs * q, axis=-1)
        return pi, v, q

    def policy(self, x: jnp.ndarray):
        phi_a = self.actor_cnn(x)
        return self.actor_head(phi_a)

    def q_values(self, x: jnp.ndarray):
        phi_c = self.critic_cnn(x)
        return self.q_head(phi_c)

    def value(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        return jnp.sum(pi.probs * q, axis=-1)


# ==============================================================================
# Continuous On-Policy DDPG Networks (Deterministic Actor mu(s) + Critic Q(s, a))
# ==============================================================================

class ContinuousDDPGNetwork(nn.Module):
    """
    Continuous DDPG Actor-Critic network:
      - Actor: Deterministic policy mu(s) bounded to [action_low, action_high] via tanh.
               During rollout, policy(s) wraps mu(s) in N(mu(s), exploration_noise^2 I).
      - Critic: State-action value network Q(s, a) taking concatenated [s, a] as input.
      - Expected Value: V(s) = Q(s, mu(s)).
    """
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    action_low: float = -1.0
    action_high: float = 1.0
    exploration_noise: float = 0.1

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.q_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def actor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deterministic target policy action mu(s) in [action_low, action_high]."""
        phi_a = self.actor_mlp(x)
        raw = self.actor_head(phi_a)
        if x.ndim == 1:
            raw = raw.squeeze(0)
        mid = 0.5 * (self.action_low + self.action_high)
        half_range = 0.5 * (self.action_high - self.action_low)
        return mid + half_range * jnp.tanh(raw)

    def policy(self, x: jnp.ndarray) -> distrax.MultivariateNormalDiag:
        """Exploration policy distribution N(mu(s), sigma_expl^2 I) for on-policy rollouts."""
        mu = self.actor(x)
        scale = jnp.full_like(mu, self.exploration_noise)
        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)

    def q_value(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Evaluates scalar state-action value Q(s, a)."""
        if a.ndim == x.ndim - 1:
            a = a[..., None]
        sa = jnp.concatenate([x, a], axis=-1)
        phi_c = self.critic_mlp(sa)
        out = self.q_head(phi_c).squeeze(-1)
        if x.ndim == 1:
            out = out.squeeze(0)
        return out

    def value(self, x: jnp.ndarray) -> jnp.ndarray:
        """Target policy state value V(s) = Q(s, mu(s))."""
        mu = self.actor(x)
        return self.q_value(x, mu)

    def __call__(self, x: jnp.ndarray):
        """Returns (pi, V(s)) where V(s) = Q(s, mu(s))."""
        mu = self.actor(x)
        pi = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.full_like(mu, self.exploration_noise))
        v = self.q_value(x, mu)
        return pi, v


class ContinuousDDPGDualCritic(nn.Module):
    """
    Continuous DDPG Actor-Critic with dual critic heads (Q and Q_aux) for surgical E(0) and E(lambda).
    Supports shared or independent critic torsos taking [s, a] as input.
    """
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    action_low: float = -1.0
    action_high: float = 1.0
    exploration_noise: float = 0.1
    shared_torso: bool = False

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.q_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        if not self.shared_torso:
            self.aux_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.aux_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def actor(self, x: jnp.ndarray) -> jnp.ndarray:
        phi_a = self.actor_mlp(x)
        raw = self.actor_head(phi_a)
        if x.ndim == 1:
            raw = raw.squeeze(0)
        mid = 0.5 * (self.action_low + self.action_high)
        half_range = 0.5 * (self.action_high - self.action_low)
        return mid + half_range * jnp.tanh(raw)

    def policy(self, x: jnp.ndarray) -> distrax.MultivariateNormalDiag:
        mu = self.actor(x)
        scale = jnp.full_like(mu, self.exploration_noise)
        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)

    def both_q(self, x: jnp.ndarray, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (Q(s, a), Q_aux(s, a))."""
        if a.ndim == x.ndim - 1:
            a = a[..., None]
        sa = jnp.concatenate([x, a], axis=-1)
        phi_c = self.critic_mlp(sa)
        q = self.q_head(phi_c).squeeze(-1)
        phi_aux = phi_c if self.shared_torso else self.aux_mlp(sa)
        q_aux = self.aux_head(phi_aux).squeeze(-1)
        if x.ndim == 1:
            q = q.squeeze(0)
            q_aux = q_aux.squeeze(0)
        return q, q_aux

    def q_value(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        q, _ = self.both_q(x, a)
        return q

    def q_aux(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        _, q_aux = self.both_q(x, a)
        return q_aux

    def both(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (V(s), V_aux(s)) = (Q(s, mu(s)), Q_aux(s, mu(s)))."""
        mu = self.actor(x)
        return self.both_q(x, mu)

    def value(self, x: jnp.ndarray) -> jnp.ndarray:
        v, _ = self.both(x)
        return v

    def v_aux(self, x: jnp.ndarray) -> jnp.ndarray:
        _, v_aux = self.both(x)
        return v_aux

    def __call__(self, x: jnp.ndarray):
        mu = self.actor(x)
        pi = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.full_like(mu, self.exploration_noise))
        v, _ = self.both_q(x, mu)
        return pi, v


def initialize_q_actor_critic(rng, obs_shape, env, env_params, config):
    """Initializes Q-Actor-Critic for discrete action spaces (Q-PPO)."""
    act_space = env.action_space(env_params)
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError(f"Q-PPO requires discrete action space for closed-form marginalization, got {type(act_space)}")

    action_dim = act_space.n
    k = config["k"]
    layer_norm = config["LAYER_NORM"]
    norm_type = "layer_norm" if layer_norm else "none"
    use_mlp = len(obs_shape) < 3

    if use_mlp:
        model = MLP_Q_AC(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)
    else:
        model = PQN_Q_AC(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params


def initialize_ddpg_network(rng, obs_shape, env, env_params, config):
    """Initializes Continuous On-Policy DDPG network for Box action spaces."""
    import numpy as np
    act_space = env.action_space(env_params)
    if not isinstance(act_space, spaces.Box):
        raise ValueError(f"DDPG requires continuous Box action space, got {type(act_space)}")

    action_dim = act_space.shape[0] if len(act_space.shape) > 0 else 1
    action_low = float(np.asarray(act_space.low).flatten()[0])
    action_high = float(np.asarray(act_space.high).flatten()[0])

    k = config["k"]
    layer_norm = config["LAYER_NORM"]
    norm_type = "layer_norm" if layer_norm else "none"
    exploration_noise = config["EXPLORATION_NOISE"]
    uses_aux = uses_aux_network(config)
    shared_torso = config["SHARED_TORSO"]

    if uses_aux:
        model = ContinuousDDPGDualCritic(
            action_dim=action_dim,
            final_hidden_dim=k,
            norm_type=norm_type,
            action_low=action_low,
            action_high=action_high,
            exploration_noise=exploration_noise,
            shared_torso=shared_torso,
        )
    else:
        model = ContinuousDDPGNetwork(
            action_dim=action_dim,
            final_hidden_dim=k,
            norm_type=norm_type,
            action_low=action_low,
            action_high=action_high,
            exploration_noise=exploration_noise,
        )

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params

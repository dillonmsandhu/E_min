"""Tanh-squashed Gaussian actor and Q-critic networks matching Brax PPO architecture."""
from __future__ import annotations

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp


def _build_mlp(hidden_dims, output_dim, act_fn, layer_norm: bool, out_scale: float):
    layers = []
    for width in hidden_dims:
        layers.append(nn.Dense(width, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
        if layer_norm:
            layers.append(nn.LayerNorm())
        layers.append(act_fn)
    layers.append(nn.Dense(output_dim, kernel_init=orthogonal(out_scale), bias_init=constant(0.0)))
    return nn.Sequential(layers)


def _sample_squashed_gaussian(mean, log_std, rng, greedy: bool = False):
    """Reparameterized tanh-squashed Gaussian action in [-1, 1]^d and exact log density."""
    noise = jnp.zeros_like(mean) if greedy else jax.random.normal(rng, mean.shape)
    latent = mean + jnp.exp(log_std) * noise
    normal_logp = -0.5 * (noise**2 + jnp.log(2.0 * jnp.pi)) - log_std
    log_det = 2.0 * (jnp.log(2.0) - latent - jax.nn.softplus(-2.0 * latent))
    return jnp.tanh(latent), jnp.sum(normal_logp - log_det, axis=-1)


class BraxSACNetwork(nn.Module):
    """2x256 MLP tanh-policy SAC actor and single Q(s, a) critic."""

    action_dim: int
    activation: str = "tanh"
    layer_norm: bool = False
    hidden_dims: tuple = (256, 256)

    def setup(self):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        # Named actor_mean so initialize_split_train_state splits actor vs critic cleanly.
        self.actor_mean = _build_mlp(
            self.hidden_dims, 2 * self.action_dim, act_fn, self.layer_norm, out_scale=0.01)
        self.critic = _build_mlp(self.hidden_dims, 1, act_fn, self.layer_norm, out_scale=1.0)

    def actor_stats(self, obs: jnp.ndarray):
        mean, log_std = jnp.split(self.actor_mean(obs), 2, axis=-1)
        return mean, jnp.clip(log_std, -5.0, 2.0)

    def actor(self, obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = self.actor_stats(obs)
        return jnp.tanh(mean)

    def sample_action(self, obs: jnp.ndarray, rng: jax.Array, greedy: bool = False):
        mean, log_std = self.actor_stats(obs)
        return _sample_squashed_gaussian(mean, log_std, rng, greedy=greedy)

    def q_value(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.critic(jnp.concatenate([obs, action], axis=-1)).squeeze(-1)

    def __call__(self, obs: jnp.ndarray):
        action = self.actor(obs)
        return action, self.q_value(obs, action)


class BraxSACDualCritic(nn.Module):
    """2x256 MLP tanh-policy SAC actor with Q(s, a) and auxiliary Q_aux(s, a) heads."""

    action_dim: int
    activation: str = "tanh"
    layer_norm: bool = False
    shared_torso: bool = False
    hidden_dims: tuple = (256, 256)

    def setup(self):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        self.actor_mean = _build_mlp(
            self.hidden_dims, 2 * self.action_dim, act_fn, self.layer_norm, out_scale=0.01)

        def _torso():
            layers = []
            for width in self.hidden_dims:
                layers.append(nn.Dense(width, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
                if self.layer_norm:
                    layers.append(nn.LayerNorm())
                layers.append(act_fn)
            return nn.Sequential(layers)

        self.critic_torso = _torso()
        self.q_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        if not self.shared_torso:
            self.aux_torso = _torso()
        self.aux_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def actor_stats(self, obs: jnp.ndarray):
        mean, log_std = jnp.split(self.actor_mean(obs), 2, axis=-1)
        return mean, jnp.clip(log_std, -5.0, 2.0)

    def actor(self, obs: jnp.ndarray) -> jnp.ndarray:
        mean, _ = self.actor_stats(obs)
        return jnp.tanh(mean)

    def sample_action(self, obs: jnp.ndarray, rng: jax.Array, greedy: bool = False):
        mean, log_std = self.actor_stats(obs)
        return _sample_squashed_gaussian(mean, log_std, rng, greedy=greedy)

    def q_value(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        sa = jnp.concatenate([obs, action], axis=-1)
        return self.q_head(self.critic_torso(sa)).squeeze(-1)

    def q_aux(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        sa = jnp.concatenate([obs, action], axis=-1)
        phi = self.critic_torso(sa) if self.shared_torso else self.aux_torso(sa)
        return self.aux_head(phi).squeeze(-1)

    def both_q(self, obs: jnp.ndarray, action: jnp.ndarray):
        sa = jnp.concatenate([obs, action], axis=-1)
        phi_main = self.critic_torso(sa)
        phi_aux = phi_main if self.shared_torso else self.aux_torso(sa)
        return self.q_head(phi_main).squeeze(-1), self.aux_head(phi_aux).squeeze(-1)

    def __call__(self, obs: jnp.ndarray):
        action = self.actor(obs)
        q_val, _ = self.both_q(obs, action)
        return action, q_val

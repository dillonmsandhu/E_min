"""Octax CNN actor-critic networks and split TrainState initialization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

from gymnax_rl.core.gymnax_network import (
    DualTrainState,
    PQN_AC,
    PQN_AC_DualCritic,
    PolicyHead,
    initialize_split_flax_train_state,
    uses_aux_network,
)


class OctaxCNN(nn.Module):
    """CNN from the Octax paper for (4, 64, 32) stacked observations.

    Casts to float32 and transposes (..., C, W, H) -> (..., W, H, C) only at
    forward-pass time so rollout buffers store compact 1-byte bool arrays.
    """

    norm_type: str = "none"
    final_hidden_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim >= 3, f"Input shape should be at least (C, W, H), got {x.shape}"
        x = jnp.moveaxis(x.astype(jnp.float32), -3, -1)
        batch_dims = x.shape[:-3]

        if len(batch_dims) == 0:
            x = x[None, ...]
            batch_dims = (1,)
        elif len(batch_dims) > 1:
            x = x.reshape(-1, *x.shape[-3:])

        for features, kernel_size, strides in (
            (32, (8, 4), (4, 2)),
            (64, (4, 4), (2, 2)),
            (64, (3, 3), (1, 1)),
        ):
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=strides,
                padding="VALID",
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            if self.norm_type == "layer_norm":
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        x = x.reshape(*batch_dims, -1)
        x = nn.Dense(
            self.final_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        if self.norm_type == "layer_norm":
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return x


class Octax_AC(PQN_AC):
    """Actor-critic with separate OctaxCNN trunks for actor and critic."""

    def setup(self):
        self.actor_cnn = OctaxCNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = OctaxCNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        if self.n_value_heads > 1:
            self.critic_head = nn.Dense(self.n_value_heads, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        else:
            self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))


class Octax_AC_DualCritic(PQN_AC_DualCritic):
    """Actor-critic with separate OctaxCNN trunks for actor, v, and v_aux."""

    def setup(self):
        self.actor_cnn = OctaxCNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = OctaxCNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.critic_aux_cnn = OctaxCNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.critic_aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))


def initialize_actor_critic(rng, obs_shape, env, env_params, config):
    """Initializes Octax actor-critic network and parameters based on config."""
    rng, init_rng = jax.random.split(rng)

    action_dim = env.action_space(env_params).n
    k = config["k"]
    n_value_heads = config["NUM_VALUE_HEADS"]
    norm_type = "layer_norm" if config["LAYER_NORM"] else "none"
    uses_aux = uses_aux_network(config)
    shared_torso = config["SHARED_TORSO"]

    if uses_aux:
        model = Octax_AC_DualCritic(
            action_dim=action_dim,
            is_continuous=False,
            final_hidden_dim=k,
            norm_type=norm_type,
            shared_torso=shared_torso,
        )
    else:
        model = Octax_AC(
            action_dim=action_dim,
            is_continuous=False,
            final_hidden_dim=k,
            norm_type=norm_type,
            n_value_heads=n_value_heads,
        )

    params = model.init(init_rng, jnp.zeros(obs_shape, dtype=jnp.bool_))
    return model, params


initialize_network = initialize_actor_critic

__all__ = [
    "OctaxCNN",
    "Octax_AC",
    "Octax_AC_DualCritic",
    "DualTrainState",
    "initialize_actor_critic",
    "initialize_network",
    "initialize_split_flax_train_state",
    "uses_aux_network",
]

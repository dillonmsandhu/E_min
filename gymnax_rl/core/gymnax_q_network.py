# gymnax_q_network.py
# Q-function Actor-Critic networks for discrete action spaces with analytical marginalization:
# V(s) = sum_{a in A} pi(a | s) * Q(s, a)
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
    Actor-Critic architecture outputting Q(s, ·) in R^{|A|} and pi(· | s).
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


def initialize_q_actor_critic(rng, obs_shape, env, env_params, config):
    """Initializes Q-Actor-Critic for discrete action spaces."""
    act_space = env.action_space(env_params)
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError(f"Q-PPO requires discrete action space for closed-form marginalization, got {type(act_space)}")

    action_dim = act_space.n
    k = config.get("k", 16)
    layer_norm = config.get("LAYER_NORM", False)
    norm_type = "layer_norm" if layer_norm else "none"
    use_mlp = len(obs_shape) < 3

    if use_mlp:
        model = MLP_Q_AC(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)
    else:
        model = PQN_Q_AC(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params

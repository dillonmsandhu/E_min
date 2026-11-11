# brax_q_network.py
# Continuous On-Policy DDPG Actor-Critic networks for Brax environments:
#   - Actor: Deterministic policy mu(s) in [-1, 1]^{d_a} via tanh(actor_mean(s)).
#            During rollouts, policy(s) returns N(mu(s), exploration_noise^2 I).
#   - Critic: State-action value network Q(s, a) taking [s, a] in R^{d_s + d_a}.
#   - Expected Value: V(s) = Q(s, mu(s)).
from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from brax_rl.core.brax_network import DualTrainState, initialize_split_train_state


class BraxDDPGNetwork(nn.Module):
    """
    Continuous DDPG Actor-Critic for Brax (256-256 MLPs matching brax_network.py).
    """
    action_dim: int
    activation: str = "tanh"
    layer_norm: bool = False
    exploration_noise: float = 0.1

    def setup(self):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        actor_layers = [
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
        ]
        if self.layer_norm:
            actor_layers.append(nn.LayerNorm())
        actor_layers.append(act_fn)
        actor_layers.append(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
        if self.layer_norm:
            actor_layers.append(nn.LayerNorm())
        actor_layers.append(act_fn)
        actor_layers.append(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))
        self.actor_mean = nn.Sequential(actor_layers)

        critic_layers = [
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
        ]
        if self.layer_norm:
            critic_layers.append(nn.LayerNorm())
        critic_layers.append(act_fn)
        critic_layers.append(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
        if self.layer_norm:
            critic_layers.append(nn.LayerNorm())
        critic_layers.append(act_fn)
        self.critic_torso = nn.Sequential(critic_layers)
        self.q_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def actor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deterministic target policy action mu(s) in [-1, 1]."""
        return jnp.tanh(self.actor_mean(x))

    def policy(self, x: jnp.ndarray) -> distrax.MultivariateNormalDiag:
        """Exploration policy N(mu(s), exploration_noise^2 I) for on-policy rollouts."""
        mu = self.actor(x)
        scale = jnp.full_like(mu, self.exploration_noise)
        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)

    def q_value(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Evaluates scalar state-action value Q(s, a)."""
        sa = jnp.concatenate([x, a], axis=-1)
        phi_c = self.critic_torso(sa)
        return self.q_head(phi_c).squeeze(-1)

    def value(self, x: jnp.ndarray) -> jnp.ndarray:
        """Target policy state value V(s) = Q(s, mu(s))."""
        mu = self.actor(x)
        return self.q_value(x, mu)

    def __call__(self, x: jnp.ndarray):
        mu = self.actor(x)
        pi = distrax.MultivariateNormalDiag(loc=mu, scale_diag=jnp.full_like(mu, self.exploration_noise))
        v = self.q_value(x, mu)
        return pi, v


class BraxDDPGDualCritic(nn.Module):
    """
    Continuous DDPG Actor-Critic with dual critic heads (Q and Q_aux) for surgical E(0) and E(lambda).
    """
    action_dim: int
    activation: str = "tanh"
    layer_norm: bool = False
    exploration_noise: float = 0.1
    shared_torso: bool = True

    def setup(self):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        actor_layers = [
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
        ]
        if self.layer_norm:
            actor_layers.append(nn.LayerNorm())
        actor_layers.append(act_fn)
        actor_layers.append(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
        if self.layer_norm:
            actor_layers.append(nn.LayerNorm())
        actor_layers.append(act_fn)
        actor_layers.append(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))
        self.actor_mean = nn.Sequential(actor_layers)

        critic_layers = [
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
        ]
        if self.layer_norm:
            critic_layers.append(nn.LayerNorm())
        critic_layers.append(act_fn)
        critic_layers.append(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
        if self.layer_norm:
            critic_layers.append(nn.LayerNorm())
        critic_layers.append(act_fn)
        self.critic_torso = nn.Sequential(critic_layers)
        self.q_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        if not self.shared_torso:
            aux_layers = [
                nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            ]
            if self.layer_norm:
                aux_layers.append(nn.LayerNorm())
            aux_layers.append(act_fn)
            aux_layers.append(nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
            if self.layer_norm:
                aux_layers.append(nn.LayerNorm())
            aux_layers.append(act_fn)
            self.aux_torso = nn.Sequential(aux_layers)

        self.aux_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def actor(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(self.actor_mean(x))

    def policy(self, x: jnp.ndarray) -> distrax.MultivariateNormalDiag:
        mu = self.actor(x)
        scale = jnp.full_like(mu, self.exploration_noise)
        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)

    def both_q(self, x: jnp.ndarray, a: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([x, a], axis=-1)
        phi_c = self.critic_torso(sa)
        q = self.q_head(phi_c).squeeze(-1)
        phi_aux = phi_c if self.shared_torso else self.aux_torso(sa)
        q_aux = self.aux_head(phi_aux).squeeze(-1)
        return q, q_aux

    def q_value(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        q, _ = self.both_q(x, a)
        return q

    def q_aux(self, x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        _, q_aux = self.both_q(x, a)
        return q_aux

    def both(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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

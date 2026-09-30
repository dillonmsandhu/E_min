import flax.linen as nn
import jax.numpy as jnp
import distrax
import optax
from typing import Sequence, NamedTuple
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    log_std_init: float = -0.5
    n_value_heads: int = 1

    def setup(self):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        self.actor_mean = nn.Sequential([
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)),
        ])

        self.actor_logstd = self.param(
            "log_std",
            lambda key, shape: jnp.full(shape, self.log_std_init),
            (self.action_dim,),
        )

        self.critic = nn.Sequential([
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            activation,
            nn.Dense(self.n_value_heads, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
        ])

    def __call__(self, x):
        actor_mean = self.actor_mean(x)
        clamped_logstd = jnp.clip(self.actor_logstd, a_min=-5.0, a_max=2.0)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(clamped_logstd))
        critic_out = self.critic(x)
        if self.n_value_heads == 1:
            val = jnp.squeeze(critic_out, axis=-1)
        else:
            val = jnp.mean(critic_out, axis=-1)
        return pi, val

    def policy(self, x):
        actor_mean = self.actor_mean(x)
        clamped_logstd = jnp.clip(self.actor_logstd, a_min=-5.0, a_max=2.0)
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(clamped_logstd))

    def value(self, x):
        critic_out = self.critic(x)
        if self.n_value_heads == 1:
            return jnp.squeeze(critic_out, axis=-1)
        return critic_out

    def act(self, x, key):
        policy = self.policy(x)
        return policy.sample(seed=key).squeeze()


class DualTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState

    @property
    def params(self):
        return {"params": {**self.actor.params, **self.critic.params}}

    def apply_critic_gradients(self, grads):
        return self._replace(critic=self.critic.apply_gradients(grads=grads))

    def apply_actor_gradients(self, grads):
        return self._replace(actor=self.actor.apply_gradients(grads=grads))


def initialize_split_train_state(config, network, network_params):
    """
    Initializes independent TrainStates for Actor and Critic, with separate
    learning rates, schedules, and optimizers.
    """
    p = network_params.get("params", network_params)
    actor_params = {k: v for k, v in p.items() if k in ["actor_mean", "log_std"]}
    critic_params = {k: v for k, v in p.items() if k == "critic"}

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    n_updates = config["TOTAL_TIMESTEPS"] // batch_size
    net_cfg = config.get("NETWORK_CONFIG", {})

    actor_epochs = config.get("ACTOR_EPOCHS", net_cfg.get("NUM_EPOCHS", 4))
    critic_epochs = config.get("CRITIC_EPOCHS", net_cfg.get("NUM_EPOCHS", 4))

    actor_minibatch_size = config.get("ACTOR_MINIBATCH_SIZE", net_cfg.get("MINIBATCH_SIZE", 1024))
    critic_minibatch_size = config.get("CRITIC_MINIBATCH_SIZE", net_cfg.get("MINIBATCH_SIZE", 1024))

    actor_num_minibatches = max(1, batch_size // min(batch_size, actor_minibatch_size))
    critic_num_minibatches = max(1, batch_size // min(batch_size, critic_minibatch_size))

    actor_total_steps = n_updates * actor_num_minibatches * actor_epochs
    critic_total_steps = n_updates * critic_num_minibatches * critic_epochs

    actor_lr = config.get("ACTOR_LR", net_cfg.get("LR", 3e-4))
    critic_lr = config.get("CRITIC_LR", net_cfg.get("LR", 3e-4))

    actor_lr_end = config.get("ACTOR_LR_END", net_cfg.get("LR_END", 1e-5))
    critic_lr_end = config.get("CRITIC_LR_END", net_cfg.get("LR_END", 1e-5))

    schedule_type = net_cfg.get("LR_SCHEDULE", "linear")
    if schedule_type == "linear":
        actor_schedule = optax.linear_schedule(actor_lr, actor_lr_end, max(1, actor_total_steps))
        critic_schedule = optax.linear_schedule(critic_lr, critic_lr_end, max(1, critic_total_steps))
    elif schedule_type == "cosine":
        actor_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-20,
            peak_value=actor_lr,
            warmup_steps=max(1, actor_total_steps // 20),
            end_value=actor_lr_end,
            decay_steps=max(1, actor_total_steps),
        )
        critic_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-20,
            peak_value=critic_lr,
            warmup_steps=max(1, critic_total_steps // 20),
            end_value=critic_lr_end,
            decay_steps=max(1, critic_total_steps),
        )
    else:
        actor_schedule = actor_lr
        critic_schedule = critic_lr

    max_grad_norm = net_cfg.get("MAX_GRAD_NORM", 1.0)
    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=actor_schedule, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=critic_schedule, eps=1e-5),
    )

    actor_state = TrainState.create(
        apply_fn=network.apply,
        params=actor_params,
        tx=actor_tx,
    )
    critic_state = TrainState.create(
        apply_fn=network.apply,
        params=critic_params,
        tx=critic_tx,
    )
    return DualTrainState(actor=actor_state, critic=critic_state)

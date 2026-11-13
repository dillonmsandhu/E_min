# jumanji_rl/core/jumanji_network.py
# Neural network architectures and TrainState initializers for Jumanji RL

from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant
import optax
import distrax
from typing import Tuple, Dict, Any, Callable, NamedTuple
from gymnax.environments import spaces

AUX_ELIGIBLE_CRITICS = (
    "e_0",
    "e_0_q",
    "e0",
    "e",
    "e_lambda",
    "e_lambda_q",
    "elambda",
)
NON_AUX_CRITICS = (
    "sarsa",
    "td",
    "td_0",
    "td_lambda",
    "fitted",
    "mc",
)

VALID_E_VARIANTS = ("aux", "fwd", "naive")
AUX_HEAD_E_VARIANTS = ("aux",)


def uses_aux_network(config: dict) -> bool:
    """
    Determines from config whether the auxiliary critic head (q_aux_head) should be
    initialized and optimized separately. Fails fast on unrecognized CRITIC_TYPE or E_VARIANT.
    """
    critic_type = config["CRITIC_TYPE"].lower()
    e_variant = config["E_VARIANT"].lower()
    if e_variant not in VALID_E_VARIANTS:
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of {VALID_E_VARIANTS}."
        )
    if critic_type in AUX_ELIGIBLE_CRITICS:
        return e_variant in AUX_HEAD_E_VARIANTS
    elif critic_type in NON_AUX_CRITICS:
        return False
    else:
        raise ValueError(
            f"Unrecognized CRITIC_TYPE '{config['CRITIC_TYPE']}'. "
            f"Expected one of {AUX_ELIGIBLE_CRITICS + NON_AUX_CRITICS}."
        )


class DiscretePolicyHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> distrax.Categorical:
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=logits)


class MLP(nn.Module):
    norm_type: str
    final_hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.norm_type == "layer_norm":
            normalize = lambda tensor: nn.LayerNorm()(tensor)
        else:
            normalize = lambda tensor: tensor

        x = nn.Dense(self.final_hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Dense(self.final_hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class PQN_CNN(nn.Module):
    """Convolutional torso for visual/grid observations (Sokoban, Snake, Game2048)."""
    norm_type: str
    final_hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.norm_type == "layer_norm":
            normalize = lambda tensor: nn.LayerNorm()(tensor)
        else:
            normalize = lambda tensor: tensor

        assert x.ndim >= 3, f"Input shape should be at least (H, W, C), got {x.shape}"
        batch_dims = x.shape[:-3]

        if len(batch_dims) == 0:
            x = x[None, ...]
            batch_dims = (1,)
        elif len(batch_dims) > 1:
            x = x.reshape(-1, *x.shape[-3:])

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=1,
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        x = x.reshape(*batch_dims, -1)

        x = nn.Dense(
            self.final_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class MLP_Q_AC(nn.Module):
    """
    MLP Discrete Actor-Critic with Q(s, ·) outputs and closed-form marginalization:
        V(s) = sum_{a} pi(a | s) * Q(s, a)
    """
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    use_aux: bool = False

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = DiscretePolicyHead(self.action_dim)
        self.q_head = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        if self.use_aux:
            self.q_aux_head = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        v = jnp.sum(pi.probs * q, axis=-1)
        if self.use_aux:
            _ = self.q_aux_values(x)
        return pi, v, q

    def policy(self, x: jnp.ndarray):
        phi_a = self.actor_mlp(x)
        return self.actor_head(phi_a)

    def q_values(self, x: jnp.ndarray):
        phi_c = self.critic_mlp(x)
        return self.q_head(phi_c)

    def q_aux_values(self, x: jnp.ndarray):
        phi_c = self.critic_mlp(x)
        return self.q_aux_head(phi_c)

    def both_q(self, x: jnp.ndarray, a: jnp.ndarray):
        q = self.q_values(x)
        q_taken = jnp.take_along_axis(q, a[..., None], axis=-1).squeeze(-1)
        if self.use_aux:
            q_aux = self.q_aux_values(x)
            q_aux_taken = jnp.take_along_axis(q_aux, a[..., None], axis=-1).squeeze(-1)
            return q_taken, q_aux_taken
        return q_taken, q_taken

    def both(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        v = jnp.sum(pi.probs * q, axis=-1)
        if self.use_aux:
            q_aux = self.q_aux_values(x)
            v_aux = jnp.sum(pi.probs * q_aux, axis=-1)
            return v, v_aux
        return v, v

    def value(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        return jnp.sum(pi.probs * q, axis=-1)


class PQN_Q_AC(nn.Module):
    """CNN-based Q Actor-Critic for visual/grid observations in Jumanji."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    use_aux: bool = False

    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = DiscretePolicyHead(self.action_dim)
        self.q_head = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        if self.use_aux:
            self.q_aux_head = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        v = jnp.sum(pi.probs * q, axis=-1)
        if self.use_aux:
            _ = self.q_aux_values(x)
        return pi, v, q

    def policy(self, x: jnp.ndarray):
        phi_a = self.actor_cnn(x)
        return self.actor_head(phi_a)

    def q_values(self, x: jnp.ndarray):
        phi_c = self.critic_cnn(x)
        return self.q_head(phi_c)

    def q_aux_values(self, x: jnp.ndarray):
        phi_c = self.critic_cnn(x)
        return self.q_aux_head(phi_c)

    def both_q(self, x: jnp.ndarray, a: jnp.ndarray):
        q = self.q_values(x)
        q_taken = jnp.take_along_axis(q, a[..., None], axis=-1).squeeze(-1)
        if self.use_aux:
            q_aux = self.q_aux_values(x)
            q_aux_taken = jnp.take_along_axis(q_aux, a[..., None], axis=-1).squeeze(-1)
            return q_taken, q_aux_taken
        return q_taken, q_taken

    def both(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        v = jnp.sum(pi.probs * q, axis=-1)
        if self.use_aux:
            q_aux = self.q_aux_values(x)
            v_aux = jnp.sum(pi.probs * q_aux, axis=-1)
            return v, v_aux
        return v, v

    def value(self, x: jnp.ndarray):
        pi = self.policy(x)
        q = self.q_values(x)
        return jnp.sum(pi.probs * q, axis=-1)


class ActorCritic(nn.Module):
    """Standard Discrete Actor-Critic with scalar state-value head V(s) for V-PPO."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"

    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = DiscretePolicyHead(self.action_dim)
        self.v_head = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        pi = self.policy(x)
        v = self.value(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        phi_a = self.actor_cnn(x)
        return self.actor_head(phi_a)

    def value(self, x: jnp.ndarray):
        phi_c = self.critic_cnn(x)
        return self.v_head(phi_c).squeeze(-1)


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

    def apply_gradients(self, grads):
        p = grads["params"] if "params" in grads else grads
        a_grads = {k: v for k, v in p.items() if "actor" in k}
        c_grads = {k: v for k, v in p.items() if "actor" not in k}
        return self._replace(
            actor=self.actor.apply_gradients(grads=a_grads),
            critic=self.critic.apply_gradients(grads=c_grads),
        )


def make_warmup_linear_schedule(init_lr: float, end_lr: float, total_steps: int, warmup_ratio: float = 0.1):
    """Linear warmup from 0.0 to init_lr, followed by linear decay to end_lr."""
    if warmup_ratio is None or warmup_ratio <= 0.0 or total_steps <= 1:
        return optax.linear_schedule(
            init_value=init_lr, end_value=end_lr, transition_steps=total_steps
        )
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = max(total_steps - warmup_steps, 1)
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=init_lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=init_lr, end_value=end_lr, transition_steps=decay_steps
    )
    return optax.join_schedules([warmup_fn, decay_fn], [warmup_steps])


def _build_adamw_tx(lr_schedule, max_grad_norm: float, weight_decay: float, adam_eps: float):
    """Constructs a single global-norm-clipped AdamW transformation."""
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(
            lr_schedule,
            weight_decay=weight_decay,
            eps=adam_eps,
        ),
    )


def initialize_split_train_state(config: dict, network: nn.Module, params: dict) -> DualTrainState:
    """Separate TrainStates for actor and critic with independent AdamW optimizers."""
    p = params["params"] if "params" in params else params
    actor_params = {k: v for k, v in p.items() if "actor" in k}
    critic_params = {k: v for k, v in p.items() if "actor" not in k}

    actor_epochs = config.get("ACTOR_EPOCHS") or config["NUM_EPOCHS"]
    critic_epochs = config.get("CRITIC_EPOCHS") or config["NUM_EPOCHS"]

    actor_mb = config.get("ACTOR_MINIBATCH_SIZE") or config["MINIBATCH_SIZE"]
    critic_mb = config.get("CRITIC_MINIBATCH_SIZE") or config["MINIBATCH_SIZE"]

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    n_updates = config["NUM_UPDATES"]

    actor_n_mb = max(1, batch_size // min(batch_size, actor_mb))
    critic_n_mb = max(1, batch_size // min(batch_size, critic_mb))

    actor_total_steps = n_updates * actor_n_mb * actor_epochs
    critic_total_steps = n_updates * critic_n_mb * critic_epochs

    actor_lr = config["ACTOR_LR"] if config["ACTOR_LR"] is not None else config["LR"]
    critic_lr = config["CRITIC_LR"] if config["CRITIC_LR"] is not None else config["LR"]
    actor_lr_end = config["ACTOR_LR_END"] if config["ACTOR_LR_END"] is not None else actor_lr
    critic_lr_end = config["CRITIC_LR_END"] if config["CRITIC_LR_END"] is not None else critic_lr

    warmup_ratio = config.get("WARMUP_RATIO", 0.0)
    max_grad_norm = config["MAX_GRAD_NORM"]
    adam_eps = config["ADAM_EPS"]
    actor_wd = config.get("ACTOR_WEIGHT_DECAY") or config.get("WEIGHT_DECAY", 0.0)
    critic_wd = config.get("CRITIC_WEIGHT_DECAY") or config.get("WEIGHT_DECAY", 0.0)

    actor_lr_scheduler = make_warmup_linear_schedule(
        init_lr=actor_lr,
        end_lr=actor_lr_end,
        total_steps=actor_total_steps,
        warmup_ratio=warmup_ratio,
    )
    critic_lr_scheduler = make_warmup_linear_schedule(
        init_lr=critic_lr,
        end_lr=critic_lr_end,
        total_steps=critic_total_steps,
        warmup_ratio=warmup_ratio,
    )

    actor_tx = _build_adamw_tx(actor_lr_scheduler, max_grad_norm, actor_wd, adam_eps)
    main_critic_tx = _build_adamw_tx(critic_lr_scheduler, max_grad_norm, critic_wd, adam_eps)

    uses_aux = uses_aux_network(config)
    has_aux_params = any("aux" in k for k in critic_params)
    if uses_aux != has_aux_params:
        raise ValueError(
            f"Config/network mismatch: uses_aux_network(config)={uses_aux}, "
            f"but critic_params keys={list(critic_params.keys())}."
        )

    if uses_aux:
        aux_critic_tx = _build_adamw_tx(critic_lr_scheduler, max_grad_norm, critic_wd, adam_eps)

        def critic_param_labels(path, val):
            is_aux = any("aux" in getattr(p, "key", "") or "aux" in str(p) for p in path)
            return "critic_aux" if is_aux else "critic"

        critic_tx = optax.multi_transform(
            {"critic": main_critic_tx, "critic_aux": aux_critic_tx},
            jax.tree_util.tree_map_with_path(critic_param_labels, critic_params),
        )
    else:
        critic_tx = main_critic_tx

    actor_ts = TrainState.create(apply_fn=network.apply, params=actor_params, tx=actor_tx)
    critic_ts = TrainState.create(apply_fn=network.apply, params=critic_params, tx=critic_tx)
    return DualTrainState(actor=actor_ts, critic=critic_ts)


def initialize_q_actor_critic(rng: jax.Array, obs_shape: tuple, env: Any, env_params: Any, config: dict):
    """Initializes Q-Actor-Critic for discrete Jumanji environments."""
    act_space = env.action_space(env_params)
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError(f"Q-PPO requires discrete action space for closed-form marginalization, got {type(act_space)}")

    action_dim = act_space.n
    k = config["k"]
    layer_norm = config["LAYER_NORM"]
    norm_type = "layer_norm" if layer_norm else "none"
    use_mlp = len(obs_shape) < 3
    uses_aux = uses_aux_network(config)

    if use_mlp:
        model = MLP_Q_AC(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type, use_aux=uses_aux)
    else:
        model = PQN_Q_AC(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type, use_aux=uses_aux)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params


def initialize_actor_critic(rng: jax.Array, obs_shape: tuple, env: Any, env_params: Any, config: dict):
    """Initializes standard Actor-Critic with scalar value head V(s) for Jumanji environments."""
    act_space = env.action_space(env_params)
    action_dim = act_space.n
    k = config["k"]
    layer_norm = config["LAYER_NORM"]
    norm_type = "layer_norm" if layer_norm else "none"

    model = ActorCritic(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    return model, params

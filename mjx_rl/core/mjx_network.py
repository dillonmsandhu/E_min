"""
Actor-Critic network architectures for MuJoCo Playground (MJX).

Supports:
  - Configurable hidden layer sizes for policy and value networks independently
    (e.g. policy=(32, 32, 32, 32), value=(256, 256, 256, 256, 256) for Brax DMC parity)
  - Configurable activations ('swish'/'silu', 'tanh', 'relu', 'elu')
  - Both 'tanh_normal' (Brax NormalTanhDistribution parity, operating on pre-tanh raw actions)
    and 'normal' (Diagonal Gaussian with learned state-independent log_std)
  - Asymmetric observations via `policy_obs_key` and `value_obs_key` when obs is a dict
  - Dual critic heads (`ActorCriticDualCritic`) for debiased E(0) / E(lambda) with `v_aux`.
"""

from typing import Any, Mapping, NamedTuple, Sequence
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, lecun_uniform, orthogonal
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax


def _get_activation(name: str):
    key = name.lower()
    if key in ("swish", "silu"):
        return nn.swish
    if key == "tanh":
        return nn.tanh
    if key == "relu":
        return nn.relu
    if key == "elu":
        return nn.elu
    raise ValueError(f"Unsupported activation '{name}'.")


def _extract_obs(x: Any, obs_key: str) -> jax.Array:
    if isinstance(x, Mapping):
        if obs_key not in x:
            raise KeyError(f"Observation key '{obs_key}' not in observation dict keys {list(x.keys())}.")
        return x[obs_key]
    return x


class BraxTanhNormalDistribution:
    """
    Matches `brax.training.distribution.NormalTanhDistribution`:
      - Network outputs `2 * action_dim` parameters `(loc, raw_scale)`.
      - `scale = softplus(raw_scale) + min_std`.
      - `sample_raw_and_postprocess(seed)` samples pre-tanh `u ~ N(loc, scale)`,
        returns `(action = tanh(u), raw_action = u, log_prob(u))`.
      - `log_prob(raw_action)` evaluates the squashed log density at `raw_action`
        (matching Brax PPO, which stores `raw_action` in the trajectory buffer to
        avoid `arctanh` saturation).
    """

    def __init__(self, loc: jax.Array, raw_scale: jax.Array, min_std: float = 0.001, var_scale: float = 1.0):
        self.loc = loc
        self.scale = (jax.nn.softplus(raw_scale) + min_std) * var_scale

    def _normal_sample(self, seed: jax.Array) -> jax.Array:
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def _normal_log_prob(self, u: jax.Array) -> jax.Array:
        log_unnorm = -0.5 * jnp.square(u / self.scale - self.loc / self.scale)
        log_norm = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return log_unnorm - log_norm

    def _normal_entropy(self) -> jax.Array:
        log_norm = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return (0.5 + log_norm) * jnp.ones_like(self.loc)

    @staticmethod
    def _tanh_log_det_jacobian(u: jax.Array) -> jax.Array:
        return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))

    def sample_raw(self, seed: jax.Array) -> jax.Array:
        return self._normal_sample(seed)

    def postprocess(self, raw_action: jax.Array) -> jax.Array:
        return jnp.tanh(raw_action)

    def sample(self, seed: jax.Array) -> jax.Array:
        return jnp.tanh(self._normal_sample(seed))

    def log_prob(self, raw_action: jax.Array) -> jax.Array:
        lp = self._normal_log_prob(raw_action) - self._tanh_log_det_jacobian(raw_action)
        return jnp.sum(lp, axis=-1)

    def entropy(self, seed: jax.Array | None = None) -> jax.Array:
        if seed is None:
            u = self.loc
        else:
            u = self._normal_sample(seed)
        ent = self._normal_entropy() + self._tanh_log_det_jacobian(u)
        return jnp.sum(ent, axis=-1)


class NormalDiagDistributionWrapper:
    """
    Wraps `distrax.MultivariateNormalDiag` so it exposes the same
    `sample_raw`, `postprocess`, `log_prob`, and `entropy` interface as
    `BraxTanhNormalDistribution`.
    """

    def __init__(self, mean: jax.Array, std: jax.Array):
        self._dist = distrax.MultivariateNormalDiag(mean, std)

    def sample_raw(self, seed: jax.Array) -> jax.Array:
        return self._dist.sample(seed=seed)

    def postprocess(self, raw_action: jax.Array) -> jax.Array:
        return raw_action

    def sample(self, seed: jax.Array) -> jax.Array:
        return self._dist.sample(seed=seed)

    def log_prob(self, raw_action: jax.Array) -> jax.Array:
        return self._dist.log_prob(raw_action)

    def entropy(self, seed: jax.Array | None = None) -> jax.Array:
        del seed
        return self._dist.entropy()


def _build_mlp(
    layer_sizes: Sequence[int],
    out_dim: int,
    activation_fn,
    kernel_init_type: str,
    final_ortho_scale: float,
    layer_norm: bool,
) -> nn.Sequential:
    layers = []
    for sz in layer_sizes:
        ki = lecun_uniform() if kernel_init_type == "lecun_uniform" else orthogonal(jnp.sqrt(2))
        layers.append(nn.Dense(sz, kernel_init=ki, bias_init=constant(0.0)))
        layers.append(activation_fn)
        if layer_norm:
            layers.append(nn.LayerNorm())
    final_ki = lecun_uniform() if kernel_init_type == "lecun_uniform" else orthogonal(final_ortho_scale)
    layers.append(nn.Dense(out_dim, kernel_init=final_ki, bias_init=constant(0.0)))
    return nn.Sequential(layers)


def _build_torso(
    layer_sizes: Sequence[int],
    activation_fn,
    kernel_init_type: str,
    layer_norm: bool,
) -> nn.Sequential:
    layers = []
    for sz in layer_sizes:
        ki = lecun_uniform() if kernel_init_type == "lecun_uniform" else orthogonal(jnp.sqrt(2))
        layers.append(nn.Dense(sz, kernel_init=ki, bias_init=constant(0.0)))
        layers.append(activation_fn)
        if layer_norm:
            layers.append(nn.LayerNorm())
    return nn.Sequential(layers)


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "swish"
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32)
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256)
    distribution_type: str = "tanh_normal"
    kernel_init_type: str = "lecun_uniform"
    log_std_init: float = -0.5
    n_value_heads: int = 1
    layer_norm: bool = False
    policy_obs_key: str = "state"
    value_obs_key: str = "state"

    def setup(self):
        act_fn = _get_activation(self.activation)
        out_dim = 2 * self.action_dim if self.distribution_type == "tanh_normal" else self.action_dim
        self.actor_net = _build_mlp(
            self.policy_hidden_layer_sizes,
            out_dim=out_dim,
            activation_fn=act_fn,
            kernel_init_type=self.kernel_init_type,
            final_ortho_scale=0.01,
            layer_norm=self.layer_norm,
        )
        if self.distribution_type == "normal":
            self.log_std = self.param(
                "log_std",
                lambda key, shape: jnp.full(shape, self.log_std_init),
                (self.action_dim,),
            )
        self.critic_net = _build_mlp(
            self.value_hidden_layer_sizes,
            out_dim=self.n_value_heads,
            activation_fn=act_fn,
            kernel_init_type=self.kernel_init_type,
            final_ortho_scale=1.0,
            layer_norm=self.layer_norm,
        )

    def _make_dist(self, actor_out: jax.Array):
        if self.distribution_type == "tanh_normal":
            loc, raw_scale = jnp.split(actor_out, 2, axis=-1)
            return BraxTanhNormalDistribution(loc, raw_scale)
        clamped_logstd = jnp.clip(self.log_std, -20.0, 2.0)
        return NormalDiagDistributionWrapper(actor_out, jnp.exp(clamped_logstd))

    def __call__(self, x):
        pi = self.policy(x)
        v = self.value(x)
        return pi, v

    def policy(self, x):
        obs_p = _extract_obs(x, self.policy_obs_key)
        actor_out = self.actor_net(obs_p)
        return self._make_dist(actor_out)

    def value(self, x):
        obs_v = _extract_obs(x, self.value_obs_key)
        v = self.critic_net(obs_v)
        if self.n_value_heads == 1:
            return jnp.squeeze(v, axis=-1)
        return v

    def act(self, x, key):
        pi = self.policy(x)
        return pi.sample(seed=key).squeeze()


class ActorCriticDualCritic(nn.Module):
    """
    Actor-Critic with primary value head `value` and auxiliary value head `v_aux`
    for debiased E(0) / E(lambda) minimization.
    """

    action_dim: int
    activation: str = "swish"
    policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32)
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256)
    distribution_type: str = "tanh_normal"
    kernel_init_type: str = "lecun_uniform"
    log_std_init: float = -0.5
    layer_norm: bool = False
    shared_torso: bool = False
    policy_obs_key: str = "state"
    value_obs_key: str = "state"

    def setup(self):
        act_fn = _get_activation(self.activation)
        out_dim = 2 * self.action_dim if self.distribution_type == "tanh_normal" else self.action_dim
        self.actor_net = _build_mlp(
            self.policy_hidden_layer_sizes,
            out_dim=out_dim,
            activation_fn=act_fn,
            kernel_init_type=self.kernel_init_type,
            final_ortho_scale=0.01,
            layer_norm=self.layer_norm,
        )
        if self.distribution_type == "normal":
            self.log_std = self.param(
                "log_std",
                lambda key, shape: jnp.full(shape, self.log_std_init),
                (self.action_dim,),
            )

        head_ki = lecun_uniform() if self.kernel_init_type == "lecun_uniform" else orthogonal(1.0)
        self.critic_torso = _build_torso(
            self.value_hidden_layer_sizes,
            activation_fn=act_fn,
            kernel_init_type=self.kernel_init_type,
            layer_norm=self.layer_norm,
        )
        self.critic_head = nn.Dense(1, kernel_init=head_ki, bias_init=constant(0.0))

        if not self.shared_torso:
            self.aux_torso = _build_torso(
                self.value_hidden_layer_sizes,
                activation_fn=act_fn,
                kernel_init_type=self.kernel_init_type,
                layer_norm=self.layer_norm,
            )
        self.aux_head = nn.Dense(1, kernel_init=head_ki, bias_init=constant(0.0))

    def _make_dist(self, actor_out: jax.Array):
        if self.distribution_type == "tanh_normal":
            loc, raw_scale = jnp.split(actor_out, 2, axis=-1)
            return BraxTanhNormalDistribution(loc, raw_scale)
        clamped_logstd = jnp.clip(self.log_std, -20.0, 2.0)
        return NormalDiagDistributionWrapper(actor_out, jnp.exp(clamped_logstd))

    def __call__(self, x):
        pi = self.policy(x)
        v, v_aux = self.both(x)
        return pi, v, v_aux

    def policy(self, x):
        obs_p = _extract_obs(x, self.policy_obs_key)
        actor_out = self.actor_net(obs_p)
        return self._make_dist(actor_out)

    def value(self, x):
        obs_v = _extract_obs(x, self.value_obs_key)
        phi_v = self.critic_torso(obs_v)
        return jnp.squeeze(self.critic_head(phi_v), axis=-1)

    def v_aux(self, x):
        obs_v = _extract_obs(x, self.value_obs_key)
        phi = self.critic_torso(obs_v) if self.shared_torso else self.aux_torso(obs_v)
        return jnp.squeeze(self.aux_head(phi), axis=-1)

    def both(self, x):
        obs_v = _extract_obs(x, self.value_obs_key)
        phi_v = self.critic_torso(obs_v)
        v = jnp.squeeze(self.critic_head(phi_v), axis=-1)
        phi_aux = phi_v if self.shared_torso else self.aux_torso(obs_v)
        v_aux = jnp.squeeze(self.aux_head(phi_aux), axis=-1)
        return v, v_aux

    def act(self, x, key):
        pi = self.policy(x)
        return pi.sample(seed=key).squeeze()


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
    Initializes independent TrainStates for Actor and Critic, supporting both
    standard Adam (Brax PPO default) and AdamW, as well as constant/linear/cosine LR schedules.
    """
    p = network_params.get("params", network_params)
    actor_keys = {"actor_net", "actor_mean", "log_std"}
    actor_params = {k: v for k, v in p.items() if k in actor_keys}
    critic_params = {k: v for k, v in p.items() if k not in actor_keys}

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    n_updates = max(1, int(config["TOTAL_TIMESTEPS"] // batch_size))
    net_cfg = config["NETWORK_CONFIG"]

    actor_epochs = config.get("ACTOR_EPOCHS", net_cfg["NUM_EPOCHS"])
    critic_epochs = config.get("CRITIC_EPOCHS", net_cfg["NUM_EPOCHS"])

    actor_num_minibatches = config.get(
        "ACTOR_NUM_MINIBATCHES",
        max(1, batch_size // min(batch_size, config.get("ACTOR_MINIBATCH_SIZE", net_cfg["MINIBATCH_SIZE"]))),
    )
    critic_num_minibatches = config.get(
        "CRITIC_NUM_MINIBATCHES",
        max(1, batch_size // min(batch_size, config.get("CRITIC_MINIBATCH_SIZE", net_cfg["MINIBATCH_SIZE"]))),
    )

    actor_total_steps = n_updates * actor_num_minibatches * actor_epochs
    critic_total_steps = n_updates * critic_num_minibatches * critic_epochs

    actor_lr = float(config["ACTOR_LR"])
    critic_lr = float(config["CRITIC_LR"])
    actor_lr_end = float(config.get("ACTOR_LR_END", net_cfg.get("LR_END", 1e-5)))
    critic_lr_end = float(config.get("CRITIC_LR_END", net_cfg.get("LR_END", 1e-5)))

    schedule_type = net_cfg.get("LR_SCHEDULE", "constant").lower()
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
    elif schedule_type in ("constant", "none"):
        actor_schedule = actor_lr
        critic_schedule = critic_lr
    else:
        raise ValueError(f"Unsupported LR_SCHEDULE '{schedule_type}'.")

    max_grad_norm = net_cfg.get("MAX_GRAD_NORM", 1.0)
    opt_name = net_cfg.get("OPTIMIZER", "adam").lower()
    adam_eps = float(net_cfg.get("ADAM_EPS", 1e-8))
    if opt_name == "adam":
        actor_opt = optax.adam(learning_rate=actor_schedule, eps=adam_eps)
        critic_opt = optax.adam(learning_rate=critic_schedule, eps=adam_eps)
    elif opt_name == "adamw":
        actor_opt = optax.adamw(learning_rate=actor_schedule, eps=adam_eps)
        critic_opt = optax.adamw(learning_rate=critic_schedule, eps=adam_eps)
    else:
        raise ValueError(f"Unsupported OPTIMIZER '{opt_name}'.")

    if max_grad_norm is not None and max_grad_norm > 0:
        actor_tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), actor_opt)
        critic_tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), critic_opt)
    else:
        actor_tx = actor_opt
        critic_tx = critic_opt

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

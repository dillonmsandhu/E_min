from core.imports import *
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from flax.training.train_state import TrainState
from gymnax.environments import spaces


class PQN_CNN(nn.Module):
    norm_type: str
    final_hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
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


class MLP(nn.Module):
    norm_type: str
    final_hidden_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.norm_type == "layer_norm":
            normalize = lambda tensor: nn.LayerNorm()(tensor)
        else:
            normalize = lambda tensor: tensor

        assert x.ndim in (1, 2, 3), f"Input shape should be (D) or (B, D) or (L, B, D) got {x.shape}"

        batch_dims = (x.shape[0],) if x.ndim == 2 else (1,)

        if x.ndim == 1:
            x = x[None, ...]

        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        x = nn.Dense(
            self.final_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        return x


class PolicyHead(nn.Module):
    action_dim: int
    is_continuous: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.relu(x)
        if not self.is_continuous:
            # Discrete: Output Logits
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            return distrax.Categorical(logits=logits)
        else:
            # Continuous: Output Mean and Log Std
            loc = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            return distrax.MultivariateNormalDiag(loc=loc, scale_diag=jnp.exp(log_std))


class PQN_AC(nn.Module):
    """Actor critic with separate small CNNs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    n_value_heads: int = 1

    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        if self.n_value_heads > 1:
            self.critic_head = nn.Dense(self.n_value_heads, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        else:
            self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s)) where V(s) is scalar consensus value for policy/rollouts."""
        v = self.value(x)
        v_pi = v.mean(axis=-1) if self.n_value_heads > 1 else v
        pi = self.policy(x)
        return pi, v_pi

    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns features for V(s)"""
        return self.critic_cnn(x)

    def value_from_features(self, phi):
        out = self.critic_head(phi)
        if self.n_value_heads == 1:
            return out.squeeze(-1)
        return out

    def value(self, x: jnp.ndarray):
        """Returns V(s) as scalar (if n_value_heads==1) or (..., n_value_heads)."""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)

    def value_all_heads(self, x: jnp.ndarray):
        """Always returns shape (..., n_value_heads) even when n_value_heads == 1."""
        v = self.value(x)
        if self.n_value_heads == 1:
            return v[..., None]
        return v

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


class MLP_AC(nn.Module):
    """Actor critic with separate small MLPs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    n_value_heads: int = 1

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        if self.n_value_heads > 1:
            self.critic_head = nn.Dense(self.n_value_heads, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        else:
            self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s)) where V(s) is scalar consensus value for policy/rollouts."""
        v = self.value(x)
        v_pi = v.mean(axis=-1) if self.n_value_heads > 1 else v
        pi = self.policy(x)
        return pi, v_pi

    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_mlp(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns features for V(s)"""
        return self.critic_mlp(x)

    def value_from_features(self, phi):
        out = self.critic_head(phi)
        if self.n_value_heads == 1:
            return out.squeeze(-1)
        return out

    def value(self, x: jnp.ndarray):
        """Returns V(s) as scalar (if n_value_heads==1) or (..., n_value_heads)."""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)

    def value_all_heads(self, x: jnp.ndarray):
        """Always returns shape (..., n_value_heads) even when n_value_heads == 1."""
        v = self.value(x)
        if self.n_value_heads == 1:
            return v[..., None]
        return v

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


def make_warmup_linear_schedule(init_lr, end_lr, total_steps, warmup_ratio=0.1):
    """Linear warmup from 0.0 to init_lr over warmup_steps, followed by linear decay to end_lr."""
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


def initialize_flax_train_state(config, network, params):
    """PPO / E-minimization optimizer with separate actor and critic learning rates."""
    total_grad_steps = (
        config["NUM_UPDATES"] * config.get("NUM_MINIBATCHES", 1) * config["NUM_EPOCHS"]
    )

    actor_lr = config.get("ACTOR_LR", config["LR"])
    critic_lr = config["LR"]
    actor_lr_end = config.get("ACTOR_LR_END")
    if actor_lr_end is None:
        actor_lr_end = actor_lr
    critic_lr_end = config.get("LR_END")
    if critic_lr_end is None:
        critic_lr_end = critic_lr

    warmup_ratio = config.get("WARMUP_RATIO", 0.1)

    actor_lr_scheduler = make_warmup_linear_schedule(
        init_lr=actor_lr,
        end_lr=actor_lr_end,
        total_steps=total_grad_steps,
        warmup_ratio=warmup_ratio,
    )
    critic_lr_scheduler = make_warmup_linear_schedule(
        init_lr=critic_lr,
        end_lr=critic_lr_end,
        total_steps=total_grad_steps,
        warmup_ratio=warmup_ratio,
    )

    actor_wd = config.get("ACTOR_WEIGHT_DECAY")
    if actor_wd is None:
        actor_wd = config.get("WEIGHT_DECAY", 1e-2)
    critic_wd = config.get("CRITIC_WEIGHT_DECAY")
    if critic_wd is None:
        critic_wd = config.get("WEIGHT_DECAY", 1e-2)

    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            actor_lr_scheduler,
            weight_decay=actor_wd,
            eps=config.get("ADAM_EPS", 1e-5),
        ),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            critic_lr_scheduler,
            weight_decay=critic_wd,
            eps=config.get("ADAM_EPS", 1e-5),
        ),
    )

    def param_labels(path, val):
        is_actor = any("actor" in getattr(p, "key", "") or "actor" in str(p) for p in path)
        return "actor" if is_actor else "critic"

    tx = optax.multi_transform(
        {"actor": actor_tx, "critic": critic_tx},
        jax.tree_util.tree_map_with_path(param_labels, params),
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return train_state


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
        p = grads.get("params", grads)
        a_grads = {k: v for k, v in p.items() if "actor" in k}
        c_grads = {k: v for k, v in p.items() if "critic" in k}
        return self._replace(
            actor=self.actor.apply_gradients(grads=a_grads),
            critic=self.critic.apply_gradients(grads=c_grads),
        )


def initialize_split_flax_train_state(config, network, params):
    """Separate TrainStates for actor and critic, isolating gradient steps and weight decay."""
    p = params.get("params", params)
    actor_params = {k: v for k, v in p.items() if "actor" in k}
    critic_params = {k: v for k, v in p.items() if "critic" in k}

    actor_epochs = config.get("ACTOR_EPOCHS")
    if actor_epochs is None:
        actor_epochs = config.get("NUM_EPOCHS", 4)

    critic_epochs = config.get("CRITIC_EPOCHS")
    if critic_epochs is None:
        critic_epochs = config.get("NUM_EPOCHS", 4)

    n_minibatches = config.get("NUM_MINIBATCHES", 1)
    batch_size = config.get("NUM_STEPS", 64) * config.get("NUM_ENVS", 256)
    n_updates = config.get("NUM_UPDATES", max(1, config.get("TOTAL_TIMESTEPS", 1000000) // max(1, batch_size)))

    actor_total_steps = n_updates * n_minibatches * actor_epochs
    critic_total_steps = n_updates * n_minibatches * critic_epochs

    actor_lr = config.get("ACTOR_LR", config["LR"])
    critic_lr = config["LR"]
    actor_lr_end = config.get("ACTOR_LR_END")
    if actor_lr_end is None:
        actor_lr_end = actor_lr
    critic_lr_end = config.get("LR_END")
    if critic_lr_end is None:
        critic_lr_end = critic_lr

    warmup_ratio = config.get("WARMUP_RATIO", 0.1)

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

    actor_wd = config.get("ACTOR_WEIGHT_DECAY")
    if actor_wd is None:
        actor_wd = config.get("WEIGHT_DECAY", 1e-2)

    critic_wd = config.get("CRITIC_WEIGHT_DECAY")
    if critic_wd is None:
        critic_wd = config.get("WEIGHT_DECAY", 1e-2)

    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            actor_lr_scheduler,
            weight_decay=actor_wd,
            eps=config.get("ADAM_EPS", 1e-5),
        ),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            critic_lr_scheduler,
            weight_decay=critic_wd,
            eps=config.get("ADAM_EPS", 1e-5),
        ),
    )

    actor_ts = TrainState.create(
        apply_fn=network.apply,
        params=actor_params,
        tx=actor_tx,
    )
    critic_ts = TrainState.create(
        apply_fn=network.apply,
        params=critic_params,
        tx=critic_tx,
    )
    return DualTrainState(actor=actor_ts, critic=critic_ts)


def initialize_actor_critic(rng, obs_shape, env, env_params, config, iv=False):
    """Initializes actor-critic network and parameters supporting discrete and continuous actions."""
    rng, init_rng = jax.random.split(rng)

    # Detect if continuous
    is_continuous = isinstance(env.action_space(env_params), spaces.Box)
    action_dim = (
        env.action_space(env_params).shape[0]
        if is_continuous
        else env.action_space(env_params).n
    )

    k = config.get("k", 16)
    n_value_heads = config.get("NUM_VALUE_HEADS", 1)
    layer_norm = config.get("LAYER_NORM", False)
    norm_type = "layer_norm" if layer_norm else "none"
    use_mlp = len(obs_shape) < 3

    print("- obs shape for initialization is ", obs_shape)

    if use_mlp:
        model = MLP_AC(
            action_dim=action_dim,
            is_continuous=is_continuous,
            final_hidden_dim=k,
            norm_type=norm_type,
            n_value_heads=n_value_heads,
        )
    else:
        model = PQN_AC(
            action_dim=action_dim,
            is_continuous=is_continuous,
            final_hidden_dim=k,
            norm_type=norm_type,
            n_value_heads=n_value_heads,
        )

    params = model.init(init_rng, jnp.zeros(obs_shape))
    print("number of features is ", model.final_hidden_dim)
    return model, params


def initialize_network(rng, obs_shape, env, env_params, k=16, n_heads=2, layer_norm=False, n_value_heads=1):
    """Convenience wrapper for initialize_actor_critic."""
    config = {"k": k, "LAYER_NORM": layer_norm, "NUM_VALUE_HEADS": n_value_heads}
    return initialize_actor_critic(rng, obs_shape, env, env_params, config)

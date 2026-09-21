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

        assert x.ndim in (3, 4), f"Input shape should be (H, W, C) or (B, H, W, C), got {x.shape}"
        batch_dims = (x.shape[0],) if x.ndim == 4 else (1,)

        if x.ndim == 3:
            x = x[None, ...]
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

    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s))"""
        v = self.value(x)
        pi = self.policy(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns features for V(s)"""
        return self.critic_cnn(x)

    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)

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

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s))"""
        v = self.value(x)
        pi = self.policy(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_mlp(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns features for V(s)"""
        return self.critic_mlp(x)

    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)

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

    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            actor_lr_scheduler,
            weight_decay=config.get("WEIGHT_DECAY", 1e-2),
            eps=config.get("ADAM_EPS", 1e-5),
        ),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            critic_lr_scheduler,
            weight_decay=config.get("WEIGHT_DECAY", 1e-2),
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
        )
    else:
        model = PQN_AC(
            action_dim=action_dim,
            is_continuous=is_continuous,
            final_hidden_dim=k,
            norm_type=norm_type,
        )

    params = model.init(init_rng, jnp.zeros(obs_shape))
    print("number of features is ", model.final_hidden_dim)
    return model, params


def initialize_network(rng, obs_shape, env, env_params, k=16, n_heads=2, layer_norm=False):
    """Convenience wrapper for initialize_actor_critic."""
    config = {"k": k, "LAYER_NORM": layer_norm}
    return initialize_actor_critic(rng, obs_shape, env, env_params, config)

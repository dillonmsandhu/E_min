import jax
import jax.numpy as jnp
import optax
import distrax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import spaces
from typing import NamedTuple


AUX_ELIGIBLE_CRITICS = (
    "e_0",
    "e0",
    "e",
    "sampled_e",
    "surgical_v_aux",
    "surgical",
    "e_lambda",
    "elambda",
    "e_geometric",
)

NON_AUX_CRITICS = (
    "e_0_basic",
    "e_basic",
    "e_0_biased",
    "e_biased",
    "fitted",
    "ppo",
    "td_0",
    "td",
    "td0",
    "classic_td",
    "td_lambda",
    "sampled_td_lambda",
    "mc",
    "sampled_mc",
)

VALID_E_VARIANTS = ("aux", "fwd", "naive")


def uses_aux_network(config: dict) -> bool:
    """
    Determines from config whether the auxiliary critic network (v_aux) should be
    initialized and optimized separately. Fails fast on unrecognized CRITIC_TYPE or E_VARIANT.
    """
    critic_type = config["CRITIC_TYPE"].lower()
    e_variant = config["E_VARIANT"].lower()
    if e_variant not in VALID_E_VARIANTS:
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of {VALID_E_VARIANTS}."
        )
    if critic_type in AUX_ELIGIBLE_CRITICS:
        return e_variant == "aux"
    elif critic_type in NON_AUX_CRITICS:
        return False
    else:
        raise ValueError(
            f"Unrecognized CRITIC_TYPE '{config['CRITIC_TYPE']}'. "
            f"Expected one of {AUX_ELIGIBLE_CRITICS + NON_AUX_CRITICS}."
        )


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
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            return distrax.Categorical(logits=logits)
        else:
            loc = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            return distrax.MultivariateNormalDiag(loc=loc, scale_diag=jnp.exp(log_std))


class PQN_AC(nn.Module):
    """Actor critic with separate small CNNs for actor and critic (for MinAtar)."""
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
        v = self.value(x)
        v_pi = v.mean(axis=-1) if self.n_value_heads > 1 else v
        pi = self.policy(x)
        return pi, v_pi

    def policy(self, x: jnp.ndarray):
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        return self.critic_cnn(x)

    def value_from_features(self, phi):
        out = self.critic_head(phi)
        if self.n_value_heads == 1:
            return out.squeeze(-1)
        return out

    def value(self, x: jnp.ndarray):
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)

    def value_all_heads(self, x: jnp.ndarray):
        v = self.value(x)
        if self.n_value_heads == 1:
            return v[..., None]
        return v

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


class MLP_AC(nn.Module):
    """Actor critic with separate MLPs for actor and critic (for vector / classic control)."""
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
        v = self.value(x)
        v_pi = v.mean(axis=-1) if self.n_value_heads > 1 else v
        pi = self.policy(x)
        return pi, v_pi

    def policy(self, x: jnp.ndarray):
        actor_features = self.actor_mlp(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        return self.critic_mlp(x)

    def value_from_features(self, phi):
        out = self.critic_head(phi)
        if self.n_value_heads == 1:
            return out.squeeze(-1)
        return out

    def value(self, x: jnp.ndarray):
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)

    def value_all_heads(self, x: jnp.ndarray):
        v = self.value(x)
        if self.n_value_heads == 1:
            return v[..., None]
        return v

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


class MLP_AC_DualCritic(nn.Module):
    """Actor critic with separate MLPs for actor, main critic (v), and auxiliary critic (v_aux)."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    shared_torso: bool = False

    def setup(self):
        self.actor_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.critic_aux_mlp = MLP(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.critic_aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s)) by default for standard AC interface."""
        v, _ = self.both(x)
        pi = self.policy(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        actor_features = self.actor_mlp(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        return self.critic_mlp(x)

    def aux_features(self, x: jnp.ndarray):
        if self.shared_torso:
            return self.critic_mlp(x)
        return self.critic_aux_mlp(x)

    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def aux_from_features(self, phi):
        return self.critic_aux_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        return self.value_from_features(self.value_features(x))

    def v_aux(self, x: jnp.ndarray):
        return self.aux_from_features(self.aux_features(x))

    def both(self, x: jnp.ndarray):
        if self.shared_torso:
            phi = self.value_features(x)
            return self.value_from_features(phi), self.aux_from_features(phi)
        else:
            phi_c = self.value_features(x)
            phi_a = self.aux_features(x)
            return self.value_from_features(phi_c), self.aux_from_features(phi_a)

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


class PQN_AC_DualCritic(nn.Module):
    """Actor critic with separate CNNs for actor, main critic (v), and auxiliary critic (v_aux)."""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    shared_torso: bool = False

    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        if not self.shared_torso:
            self.critic_aux_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim=self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))
        self.critic_aux_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), V(s)) by default for standard AC interface."""
        v, _ = self.both(x)
        pi = self.policy(x)
        return pi, v

    def policy(self, x: jnp.ndarray):
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        return self.critic_cnn(x)

    def aux_features(self, x: jnp.ndarray):
        if self.shared_torso:
            return self.critic_cnn(x)
        return self.critic_aux_cnn(x)

    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def aux_from_features(self, phi):
        return self.critic_aux_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        return self.value_from_features(self.value_features(x))

    def v_aux(self, x: jnp.ndarray):
        return self.aux_from_features(self.aux_features(x))

    def both(self, x: jnp.ndarray):
        if self.shared_torso:
            phi = self.value_features(x)
            return self.value_from_features(phi), self.aux_from_features(phi)
        else:
            phi_c = self.value_features(x)
            phi_a = self.aux_features(x)
            return self.value_from_features(phi_c), self.aux_from_features(phi_a)

    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()


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


def initialize_flax_train_state(config, network, params):
    """Unified TrainState with separate actor, critic, and (when uses_aux=True) critic_aux optimizers."""
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    num_minibatches = max(1, batch_size // min(batch_size, config["MINIBATCH_SIZE"]))
    total_grad_steps = config["NUM_UPDATES"] * num_minibatches * config["NUM_EPOCHS"]

    actor_lr = config["ACTOR_LR"] if config["ACTOR_LR"] is not None else config["LR"]
    critic_lr = config["CRITIC_LR"] if config["CRITIC_LR"] is not None else config["LR"]
    actor_lr_end = config["ACTOR_LR_END"] if config["ACTOR_LR_END"] is not None else actor_lr
    critic_lr_end = config["CRITIC_LR_END"] if config["CRITIC_LR_END"] is not None else critic_lr

    warmup_ratio = config["WARMUP_RATIO"]
    max_grad_norm = config["MAX_GRAD_NORM"]
    adam_eps = config["ADAM_EPS"]

    actor_wd = config["ACTOR_WEIGHT_DECAY"] if config["ACTOR_WEIGHT_DECAY"] is not None else config["WEIGHT_DECAY"]
    critic_wd = config["CRITIC_WEIGHT_DECAY"] if config["CRITIC_WEIGHT_DECAY"] is not None else config["WEIGHT_DECAY"]

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

    actor_tx = _build_adamw_tx(actor_lr_scheduler, max_grad_norm, actor_wd, adam_eps)
    critic_tx = _build_adamw_tx(critic_lr_scheduler, max_grad_norm, critic_wd, adam_eps)

    uses_aux = uses_aux_network(config)
    if uses_aux:
        aux_critic_tx = _build_adamw_tx(critic_lr_scheduler, max_grad_norm, critic_wd, adam_eps)

        def param_labels(path, val):
            is_actor = any("actor" in getattr(p, "key", "") or "actor" in str(p) for p in path)
            if is_actor:
                return "actor"
            is_aux = any("aux" in getattr(p, "key", "") or "aux" in str(p) for p in path)
            return "critic_aux" if is_aux else "critic"

        tx = optax.multi_transform(
            {"actor": actor_tx, "critic": critic_tx, "critic_aux": aux_critic_tx},
            jax.tree_util.tree_map_with_path(param_labels, params),
        )
    else:
        def param_labels(path, val):
            is_actor = any("actor" in getattr(p, "key", "") or "actor" in str(p) for p in path)
            return "actor" if is_actor else "critic"

        tx = optax.multi_transform(
            {"actor": actor_tx, "critic": critic_tx},
            jax.tree_util.tree_map_with_path(param_labels, params),
        )

    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )


def initialize_split_flax_train_state(config, network, params):
    """
    Separate TrainStates for actor and critic.
    Reads config via uses_aux_network(config) to check if v_aux is initialized; if so,
    partitions critic parameters into 'critic' and 'critic_aux' with independent global
    gradient norm clipping and AdamW states.
    """
    p = params["params"] if "params" in params else params
    actor_params = {k: v for k, v in p.items() if "actor" in k}
    critic_params = {k: v for k, v in p.items() if "actor" not in k}

    actor_epochs = config["ACTOR_EPOCHS"] if config["ACTOR_EPOCHS"] is not None else config["NUM_EPOCHS"]
    critic_epochs = config["CRITIC_EPOCHS"] if config["CRITIC_EPOCHS"] is not None else config["NUM_EPOCHS"]

    actor_minibatch_size = (
        config["ACTOR_MINIBATCH_SIZE"] if config["ACTOR_MINIBATCH_SIZE"] is not None else config["MINIBATCH_SIZE"]
    )
    critic_minibatch_size = (
        config["CRITIC_MINIBATCH_SIZE"] if config["CRITIC_MINIBATCH_SIZE"] is not None else config["MINIBATCH_SIZE"]
    )

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    n_updates = config["NUM_UPDATES"]

    actor_n_mb = max(1, batch_size // min(batch_size, actor_minibatch_size))
    critic_n_mb = max(1, batch_size // min(batch_size, critic_minibatch_size))

    actor_total_steps = n_updates * actor_n_mb * actor_epochs
    critic_total_steps = n_updates * critic_n_mb * critic_epochs

    actor_lr = config["ACTOR_LR"] if config["ACTOR_LR"] is not None else config["LR"]
    critic_lr = config["CRITIC_LR"] if config["CRITIC_LR"] is not None else config["LR"]
    actor_lr_end = config["ACTOR_LR_END"] if config["ACTOR_LR_END"] is not None else actor_lr
    critic_lr_end = config["CRITIC_LR_END"] if config["CRITIC_LR_END"] is not None else critic_lr

    warmup_ratio = config["WARMUP_RATIO"]
    max_grad_norm = config["MAX_GRAD_NORM"]
    adam_eps = config["ADAM_EPS"]

    actor_wd = config["ACTOR_WEIGHT_DECAY"] if config["ACTOR_WEIGHT_DECAY"] is not None else config["WEIGHT_DECAY"]
    critic_wd = config["CRITIC_WEIGHT_DECAY"] if config["CRITIC_WEIGHT_DECAY"] is not None else config["WEIGHT_DECAY"]

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


# Alias matching brax naming
initialize_split_train_state = initialize_split_flax_train_state


def initialize_actor_critic(rng, obs_shape, env, env_params, config):
    """Initializes actor-critic network and parameters based on config."""
    rng, init_rng = jax.random.split(rng)

    is_continuous = isinstance(env.action_space(env_params), spaces.Box)
    action_dim = (
        env.action_space(env_params).shape[0]
        if is_continuous
        else env.action_space(env_params).n
    )

    k = config["k"]
    n_value_heads = config["NUM_VALUE_HEADS"]
    layer_norm = config["LAYER_NORM"]
    norm_type = "layer_norm" if layer_norm else "none"
    use_mlp = len(obs_shape) < 3

    uses_aux = uses_aux_network(config)
    shared_torso = config["SHARED_TORSO"]

    if uses_aux:
        if use_mlp:
            model = MLP_AC_DualCritic(
                action_dim=action_dim,
                is_continuous=is_continuous,
                final_hidden_dim=k,
                norm_type=norm_type,
                shared_torso=shared_torso,
            )
        else:
            model = PQN_AC_DualCritic(
                action_dim=action_dim,
                is_continuous=is_continuous,
                final_hidden_dim=k,
                norm_type=norm_type,
                shared_torso=shared_torso,
            )
    else:
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
    return model, params


initialize_network = initialize_actor_critic


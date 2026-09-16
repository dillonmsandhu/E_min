# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import (NormalizeObservationWrapper, NormalizeRewardWrapper, 
AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, 
TerminalInfoWrapper, MountainCarNormalizeWrapper, MountainCarSparseRewardWrapper)
from envs.boyan_chain import MatrixMockEnv, BoyanParams
from envs.whirlpool import WhirlpoolExactValue
from envs.whirlpool_env import Whirlpool
from gymnax.environments import spaces
from flax.core import unfreeze, freeze

def create_evaluator(config, env=None, env_params=None):
    from envs.fourrooms import FourRoomsExactValue
    from envs.fourrooms_continuing import ContinuingFourRooms
    from envs.eightrooms import EightRoomsExactValue, ContinuingEightRooms
    from envs.boyan_chain import ContinuingBoyanRing
    from envs.whirlpool import WhirlpoolExactValue, ContinuingWhirlpool
    from envs.mountaincar_exact import MountainCarExactValue

    env_name = config['ENV_NAME'].lower()

    if env_name == 'fourrooms-misc':
        return FourRoomsExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name == 'fourrooms-cont':
        return ContinuingFourRooms(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (11, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms', 'eightrooms-misc']:
        return EightRoomsExactValue(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['eightrooms-cont']:
        return ContinuingEightRooms(
            start_pos=getattr(env, 'pos_fixed', (3, 1)),
            goal_pos=getattr(env, 'goal_fixed', (23, 11)),
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.25)),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name == 'boyan':
        return ContinuingBoyanRing(
            size=config.get('ENV_SIZE', 21),
            gamma=config['GAMMA'],
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['whirlpool', 'whirlpool-misc']:
        return WhirlpoolExactValue(
            size=config.get('ENV_SIZE', 21),
            gamma=config['GAMMA'],
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.95)),
            start_pos=getattr(env, 'pos_fixed', None),
            goal_pos=getattr(env, 'goal_fixed', None),
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name in ['whirlpool-cont']:
        return ContinuingWhirlpool(
            size=config.get('ENV_SIZE', 20),
            gamma=config['GAMMA'],
            fail_prob=getattr(env_params, 'fail_prob', config.get('FAIL_PROB', 0.95)),
            start_pos=getattr(env, 'pos_fixed', None),
            goal_pos=getattr(env, 'goal_fixed', None),
            use_visual_obs=config.get('USE_VISUAL_OBS', True),
        )
    elif env_name == 'mountaincar-v0':
        return MountainCarExactValue(gamma=config['GAMMA'])
    return None

def initialize_evaluator(config, env, env_params):
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    if hasattr(env, "evaluator") and env.evaluator is not None:
        return env.evaluator
    return create_evaluator(config, env, env_params)

def make_env(config):
    env_name = config['ENV_NAME'].lower()
    use_tabular = config.get("USE_TABULAR_SIMULATOR", True)

    tabular_env_names = [
        'whirlpool', 'whirlpool-misc', 'whirlpool-cont',
        'fourrooms-misc', 'fourrooms-cont',
        'eightrooms', 'eightrooms-misc', 'eightrooms-cont',
        'boyan'
    ]

    if use_tabular and env_name in tabular_env_names:
        from envs.tabular_matrix_env import TabularMatrixEnv, TabularParams
        evaluator = create_evaluator(config)
        env = TabularMatrixEnv(evaluator, name=config['ENV_NAME'])
        env_params = TabularParams(
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
            fail_prob=getattr(evaluator, 'fail_prob', 0.0),
        )
        env = TerminalInfoWrapper(env)
        if '-cont' in env_name:
            from envs.wrappers import ContinuingWrapper
            env = ContinuingWrapper(env)

    elif config['ENV_NAME'] == 'MountainCar-v0':
        env, env_params = gymnax.make(config["ENV_NAME"])
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE']
        )
        env = TerminalInfoWrapper(env)
        env = MountainCarNormalizeWrapper(env)
        env = MountainCarSparseRewardWrapper(env)
        config["NETWORK_TYPE"] = 'mlp'

    elif config['ENV_NAME'] == 'FourRooms-misc':
        env, env_params = gymnax.make(config["ENV_NAME"], use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config.get('FAIL_PROB', 0.1)
        )
        env = TerminalInfoWrapper(env)
        
    elif config['ENV_NAME'] == 'FourRooms-cont':
        from envs.wrappers import ContinuingWrapper
        env, env_params = gymnax.make('FourRooms-misc', use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config.get('FAIL_PROB', 0.1)
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms', 'eightrooms-misc']:
        from envs.eightrooms import EightRooms, EightRoomsParams
        env = EightRooms(use_visual_obs=True, goal_fixed=(23, 11), pos_fixed=(3, 1))
        env_params = EightRoomsParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01)
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['eightrooms-cont']:
        from envs.eightrooms import EightRooms, EightRoomsParams
        from envs.wrappers import ContinuingWrapper
        env = EightRooms(use_visual_obs=True, goal_fixed=(23, 11), pos_fixed=(3, 1))
        env_params = EightRoomsParams(
            max_steps_in_episode=config.get('MAX_STEPS_IN_EPISODE', 1e6),
            fail_prob=config.get('FAIL_PROB', 0.01)
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)
        
    elif config['ENV_NAME'] == 'boyan':
        # Create our lightweight mock primitives right here
        env = MatrixMockEnv(size=20, use_visual_obs=config.get("USE_VISUAL_OBS", True))
        env_params = BoyanParams(
            fail_prob=0.0, 
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE']
        )
    elif config['ENV_NAME'].lower() in ['whirlpool', 'whirlpool-misc']:
        from envs.whirlpool_env import Whirlpool, EnvParams
        env = Whirlpool(size=config.get('ENV_SIZE', 21), use_visual_obs=True)
        env_params = EnvParams(
            fail_prob=config.get('FAIL_PROB', 0.95),
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
        )
        env = TerminalInfoWrapper(env)

    elif config['ENV_NAME'].lower() in ['whirlpool-cont']:
        from envs.whirlpool_env import Whirlpool, EnvParams
        from envs.wrappers import ContinuingWrapper
        env = Whirlpool(size=config.get('ENV_SIZE', 21), use_visual_obs=True)
        env_params = EnvParams(
            fail_prob=config.get('FAIL_PROB', 0.95),
            max_steps_in_episode=int(config.get('MAX_STEPS_IN_EPISODE', 1e6)),
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)

    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    
    print('Env:', config['ENV_NAME'])
    print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    env = LogWrapper(env)
    
    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env)
    
    if config["NETWORK_TYPE"] == "mlp":
        if len(env.observation_space(env_params).shape) > 1:
            env = FlattenObservationWrapper(env)
    if config["NETWORK_TYPE"] == "cnn":
        if len(env.observation_space(env_params).shape) < 3:
            env = AddChannelWrapper(env)
    if config["NORMALIZE_OBS"]:
        env = NormalizeObservationWrapper(env) 
    
    print('Obs Shape:', env.observation_space(env_params).shape)
    print('Action Shape:', env.action_space(env_params).shape)
    return env, env_params
    
def _loss_fn(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get('POLICY_COEFF', 1.0) * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)

def _loss_fn_no_w(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = no_w_v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)    

def post_process_advantage(advantages, config, weights=None):
    """
    Standardizes and clips advantages for PPO policy optimization.
    
    If weights are provided (e.g. w = mu[:-1, None] * old_pi for exact methods),
    computes the weighted mean and weighted standard deviation over state-action visitation.
    Otherwise (e.g. sampled rollouts in standard PPO and hybrid scripts),
    computes the unweighted sample mean and standard deviation.
    
    Supports soft floor (ADV_STD_FLOOR) to prevent noise explosion near initialization,
    and outlier clipping (ADV_CLIP).
    """
    std_floor = config.get("ADV_STD_FLOOR", 0.1)
    adv_clip = config.get("ADV_CLIP", 3.0)

    if weights is not None:
        if weights.ndim == 1 and advantages.ndim == 2:
            weights = weights[:, None]
        w = weights / jnp.sum(weights)
        mean_adv = jnp.sum(w * advantages)
        var_adv = jnp.sum(w * (advantages - mean_adv) ** 2)
        std_adv = jnp.sqrt(var_adv + 1e-8)
    else:
        mean_adv = jnp.mean(advantages)
        var_adv = jnp.var(advantages)
        std_adv = jnp.sqrt(var_adv + 1e-8)

    if std_floor is not None and std_floor > 0.0:
        denom = jnp.maximum(std_adv, std_floor)
    else:
        denom = std_adv + 1e-8

    adv_norm = (advantages - mean_adv) / denom

    if adv_clip is not None and adv_clip > 0.0:
        adv_norm = jnp.clip(adv_norm, -adv_clip, adv_clip)

    return jax.lax.stop_gradient(adv_norm)


def compute_exact_advantage(P, R, P_pi, R_pi, v, γ, λ):
    """Computes exact GAE advantages across all non-terminal states and actions."""
    # δ_gae = (I - γ * λ * P_pi)^(-1) δ is the discounted sum of TD errors from step 1 onward.
    # At step 0, action a has immediate TD error δ(s, a) = R(s, a) + γ * v(s') - v(s).
    # Future TD errors from step 1 onward are discounted by γ * λ:
    # A^GAE(s, a) = δ + γ * λ * E_{s'}[δ_gae(s')]
    #             = R(s, a) + γ * E_{s'}[v(s') + λ * δ_gae(s')] - v(s)
    I = jnp.eye(P_pi.shape[0])
    L_pi = jnp.linalg.inv(I - γ * λ * P_pi)
    δ = R_pi + γ * (P_pi @ v) - v
    δ_gae = L_pi @ δ

    R_sa = jnp.einsum("sam,sam->sa", P[:-1], R[:-1])
    Q_sa = R_sa + γ * jnp.einsum("sam,m->sa", P[:-1], v + λ * δ_gae)
    return Q_sa - v[:-1, None]


def pi_loss_fn(params, network, traj_batch, gae, config):
    pi = network.apply(params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = post_process_advantage(gae, config)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor, entropy


def ppo_clipped_v_loss(traj_batch, value_pred, targets, config):
    e = config["VF_CLIP"]
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value).clip(-e,e)
    value_losses = jnp.square(value_pred - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
def v_loss_fn(params, network, traj_batch, gae, targets, config):
    # VALUE LOSS
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def v_loss_fn_laplacian_smoothing(params, network, traj_batch, targets, config):
    gamma = config["GAMMA"]
    c = config["VF_CLIP"]
    # 1. Current State Predictions & Errors (e_i)
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value).clip(-c,c)
    e_i = targets - value_pred_clipped 
    base_ve_loss = 0.5 * jnp.mean(e_i ** 2)
    # 2. Next State Predictions & Errors (e_j)
    # Requires traj_batch to contain the adjacent (s, G) pairs
    next_value_pred = network.apply(params, traj_batch.next_obs, method=network.value)
    next_v_fixed = jax.lax.stop_gradient(next_value_pred)
    next_value_pred_clipped = next_v_fixed + (
        next_value_pred - next_v_fixed).clip(-c,c)
    e_j = traj_batch.next_target - next_value_pred_clipped
    valid_mask = 1.0 - traj_batch.done
    n_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    laplacian_loss = 0.5 * jnp.sum(valid_mask * (e_i - e_j) ** 2) / n_valid
    # Combine using the exact Dirichlet expansion weights
    weight_laplacian = gamma * config['LAPLACE_SMOOTHING_COEFF']
    dirichlet_value_loss = (1 - weight_laplacian) * base_ve_loss + weight_laplacian * laplacian_loss
    total_loss = config["VF_COEF"] * dirichlet_value_loss
    return total_loss, {"base_ve_loss": base_ve_loss, "laplacian_loss": laplacian_loss, "total_loss": total_loss}

def no_w_v_loss_fn(params, network, traj_batch, gae, targets, config):
    # ---------------------------------------------------------
    # Firewalled Parameters
    # ---------------------------------------------------------
    def freeze_w_map(path, val):
        is_w = any(getattr(p, 'key', None) in ('w_layer', 'critic_head') or 
                   'w_layer' in str(p) or 'critic_head' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_w else val
    
    params_w_frozen = jax.tree_util.tree_map_with_path(freeze_w_map, params)
    value_for_phi = network.apply(params_w_frozen, traj_batch.obs, method = network.value)
    loss_phi = ppo_clipped_v_loss(traj_batch, value_for_phi, targets, config)
    return config["VF_COEF"] * loss_phi

def v_loss_fn_no_grad(params, network, traj_batch, gae, targets, config):
    "No update to phi."
    # 1. Forward pass through the CNN to get the features
    phi = network.apply(params, traj_batch.obs, method=network.value_features)
    
    # 2. SEVER THE GRAPH: Gradients from the value loss cannot pass this point.
    # The CNN weights will receive zero gradient from this loss function.
    phi_freeze = jax.lax.stop_gradient(phi)
    
    # 3. Forward pass through ONLY the linear head using the frozen features
    value_pred = network.apply(params, phi_freeze, method=network.value_from_features)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches


def add_values_to_metric(config, metric, evaluator, network, train_state, traj_batch, compute_true_vals = True):
    """Uses evaluator to compute the per-state quantities and append them to metric."""
    if evaluator:
        pi, v_pred = network.apply(train_state.params, evaluator.obs_stack)
        pi = jnp.vstack([pi, jnp.zeros((1, pi.shape[-1]))]) # assumes terminal state.
        
        Φ = network.apply(train_state.params, evaluator.obs_stack, method=network.value_features)
        Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))])  # assumes terminal state.
        
        v_pred = network.apply(train_state.params, Φ, method=network.value_from_features)
    
    # True value
    if compute_true_vals:
        # The evaluator dictates the exact ground truth shapes here
        v = evaluator.compute_true_values(pi)

    # 4. Visitation Logic
    obs = jnp.asarray(traj_batch.obs)
    next_obs = jnp.asarray(traj_batch.next_obs)
    env_name = config.get("ENV_NAME", "")
    
    if env_name in {"FourRooms-misc", "FourRoomsCustom-v0"} or "SparseMaze" in env_name:
        if obs.ndim >= 5:
            metric['visitation_count'] = next_obs[..., 1].sum(axis=(0, 1))
        elif obs.ndim >= 3 and obs.shape[-1] >= 2:
            size = traj_batch.reward.shape[0] 
            pos = next_obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, size - 1).reshape(-1)
            x = jnp.clip(pos[..., 1], 0, size - 1).reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
    
    # 5. Error Metrics (Perfect shape alignment guaranteed by the evaluator)    
    metric.update({
        "v": v,
        "v_pred": v_pred,
        "pi": pi,
        "Empirical MSVE": jnp.mean((v - v_pred)**2) ,
    })
    
    return metric


# def calculate_gae(traj_batch, γ, λ,):

#     def _get_advantages(gae, transition):
#         done = transition.done

#         delta = transition.reward + γ * transition.next_value * (1 - done) - transition.value
#         gae = delta + (γ * λ * (1 - done) * gae)
        
#         return gae, gae

#     initial_accs = jnp.zeros_like(traj_batch.value[0])
#     _, advantages = jax.lax.scan(
#         _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
#     )
    
#     return (advantages, advantages + traj_batch.value)

def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation (Trajectory Boundary)
        # Sever the GAE chain if the environment reset for ANY reason (terminal or timeout).
        # The 'gae' variable coming from the future belongs to a different episode.
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = transition.reward + γ * transition.next_value * bootstrap_mask - transition.value
        
        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)
        
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    return (advantages, advantages + traj_batch.value)


def find_closest_divisor(total, requested):
    for n in range(requested, 0, -1):
        if total % n == 0:
            return n
    return 1

def inject_weights(train_state, w):
    """Overwrites the critic_head weights, preserving the original PyTree type."""
    # 1. Slice the weights (last dim is bias)
    kernel_weights = jnp.expand_dims(w[:-1], axis=-1)
    bias_weight = w[-1:]
    
    # 2. Define the new layer dictionary
    new_head = {
        'kernel': kernel_weights,
        'bias': bias_weight
    }
    
    # 3. Inject it while preserving the container type (dict vs frozendict)
    params = train_state.params
    if not isinstance(params, dict):
        params = unfreeze(params)
    
    new_params = dict(params)
    new_params['params'] = dict(new_params['params'])
    if 'critic_head' in new_params['params']:
        new_params['params']['critic_head'] = new_head
    if 'w_layer' in new_params['params']:
        new_params['params']['w_layer'] = new_head

    if isinstance(train_state.params, dict):
        return train_state.replace(params=new_params)
    else:
        return train_state.replace(params=freeze(new_params))

def get_evaluation_policies(base_config, evaluator):
    """Gets a target policy for evaluation. Returns (policy_fn, policy_matrix):
        - policy_fn: a function from obs to action distribution
        - policy_matrix: a matrix of size |S| x |A| of the policy
        This function constructs these functions for either an epsilon-optimal policy or a trained actor network.
    """
    if base_config.get("USE_GREEDY_POLICY", False):
        import core.bellman_error as bellman_error
        import distrax
        if hasattr(evaluator, "get_optimal_value_function"):
            V_star = evaluator.get_optimal_value_function()
        else:
            V_star = jnp.zeros(evaluator.num_total_states)
        
        greedy_actions = bellman_error.compute_greedy_policy(evaluator.P, evaluator.R, evaluator.gamma, V_star)
        pi_greedy = jax.nn.one_hot(greedy_actions, evaluator.num_actions)
        epsilon = base_config.get("POLICY_EPSILON", 0.0)
        pi_eps = (1 - epsilon) * pi_greedy + (epsilon / evaluator.num_actions) * jnp.ones_like(pi_greedy)
        
        def policy_fn(obs):
            obs_flat = obs.reshape((obs.shape[0], -1)) if obs.ndim > 1 else obs.flatten()[None, :]
            stack_flat = evaluator.obs_stack.reshape((evaluator.obs_stack.shape[0], -1))
            diffs = jnp.sum((obs_flat[:, None, :] - stack_flat[None, :, :])**2, axis=-1)
            state_indices = jnp.argmin(diffs, axis=-1)
            probs = pi_eps[state_indices]
            return distrax.Categorical(probs=probs)
            
        # pi_eps already has shape (num_total_states, A), we just need to ensure the terminal state is uniform
        pi_eps = pi_eps.at[-1, :].set(jnp.ones(evaluator.num_actions) / evaluator.num_actions)
        return policy_fn, pi_eps
    
    else:
        import core.utils as utils
        model_dir = 'ppo/' + base_config['MODEL_LOAD_DIR']
        print(model_dir)
        _, out = utils.load_run_data(model_dir, base_config['ENV_NAME'], 'results') 
        policy_train_state = out['runner_state'][0]
        policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
        
        def policy_fn(obs):
            pi, _ = policy_train_state.apply_fn(policy_params, obs)
            # handle cases where apply_fn returns a tuple (pi, value) or just pi
            if isinstance(pi, tuple):
                pi = pi[0]
            return pi
            
        # build the matrix
        pi_dist = policy_fn(evaluator.obs_stack)
        pi_probs = pi_dist.probs
        terminal_policy = jnp.ones([1, evaluator.num_actions], dtype=pi_probs.dtype) / evaluator.num_actions
        policy_matrix = jnp.vstack([pi_probs, terminal_policy])
        
        return policy_fn, policy_matrix

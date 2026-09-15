# REINFORCE / GRPO-style policy gradient (for intrinsic value)
# uses a timestep dependent variant, based on batch index i.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.networks import nn
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "random/policy_trained_z_and_phi"

class IVActorCritic(nn.Module):
    action_dim: int
    is_continuous: bool
    norm_type: str = 'None'
    z_dim: int = 32
    phi_dim: int = 32

    def setup(self):
        self.actor_torso = networks.PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.phi_dim)
        self.phi_net = networks.PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.phi_dim)
        self.z_net = networks.PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.phi_dim)
        
        self.pi_head = networks.PolicyHead(action_dim=self.action_dim, is_continuous=self.is_continuous)
        self.w_layer = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = nn.initializers.zeros)
        
        # g_A will automatically size its input kernel to (z_dim + action_dim) 
        # on the first forward pass.
        self.g_A_dense1 = nn.Dense(32)
        self.g_A_dense2 = nn.Dense(self.phi_dim)
        self.g_A_dense_done = nn.Dense(1)
        self.g_A_z_to_phi = nn.Dense(self.phi_dim)

    def w(self, phi_features):
        return self.w_layer(phi_features)
    
    def value(self, obs):
        phi_curr = self.phi(obs)
        value = self.w(phi_curr).squeeze(-1)
        return value

    def policy(self, x):
        return self.pi_head(self.actor_torso(x))
    
    def value_features(self, obs):
        return self.phi(obs)
    
    def phi(self, obs):
        return self.phi_net(obs)

    def z(self, obs):
        return self.z_net(obs)

    def g_A(self, z_and_a):
        # Extract the pure instrument z
        z = z_and_a[..., :self.z_dim] 
        
        # 1. Project Z into the feature space
        phi_hat = self.g_A_z_to_phi(z)
        
        # 2. Predict the discounted expected next features
        x = self.g_A_dense1(z_and_a)
        x = jax.nn.leaky_relu(x)
        gamma_phi_prime = self.g_A_dense2(x)
        
        # 3. The true Residual Connection
        x_hat = phi_hat - gamma_phi_prime 
        
        done_logit = self.g_A_dense_done(x)
        return x_hat, done_logit

    # --- Action optional ---
    def __call__(self, obs, action=None):
        pi = self.policy(obs)
        phi_curr = self.phi(obs)
        value = self.w(phi_curr).squeeze(-1)
        
        # for initialization only...
        if action is not None:
            _z = self.z(obs)
            if _z.ndim > action.ndim:
                _z = _z[0]
            _z_and_a = jnp.concatenate([_z, action], axis=-1)
            _g_a_out = self.g_A(_z_and_a)
            
        # return pi, value
        return value

def one_hot_action(traj_batch, is_continuous, action_dim):
    action_features = traj_batch.action
    
    if not is_continuous: # e.g., CartPole
        action_features = jax.nn.one_hot(traj_batch.action, num_classes=action_dim)
    
    return action_features

def iv_loss_fn(params, network, traj_batch, advantages, targets, config):
    # =========================================================
    # STAGE 1
    # =========================================================
    z = network.apply(params, traj_batch.obs, method=network.z) # z is seperate from phi. it is differentiable for next feature prediction.
    
    action_features = one_hot_action(traj_batch, config['IS_CONTINUOUS'], network.action_dim)
    z = jnp.concatenate([z, action_features], axis=-1)
    z_frozen = jax.lax.stop_gradient(z)
    
    x_hat, done_logit = network.apply(params, z, method=network.g_A)
    
    phi_next = network.apply(params, traj_batch.next_obs, method=network.phi)
    not_done = (1 - jnp.expand_dims(traj_batch.done, -1))

    phi = network.apply(params, traj_batch.obs, method=network.phi)

    x_target = jax.lax.stop_gradient(
        phi - config['GAMMA'] * phi_next * not_done
    )
    
    forward_loss = jnp.mean((x_target - x_hat) ** 2)
    done_loss = 0
    # done_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
    #     logits=done_logit, 
    #     labels=jnp.expand_dims(traj_batch.done, -1)
    # ))
    
    # =========================================================
    # STAGE 2
    # =========================================================

    def firewall_map(path, val):
        is_g_A = any('g_A' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_g_A else val

    params_frozen_A = jax.tree_util.tree_map_with_path(firewall_map, params)

    # 2. Re-run the forward pass using the frozen A, but LIVE Z!
    # Because 'z' is live in 'params', JAX will flow gradients right through 
    # the frozen 'params_frozen_A' and directly into 'z_net'.
    x_hat_freeze, done_logit_freeze = network.apply(params_frozen_A, z, method=network.g_A)
    
    # 3. w is live. Predict immediate reward.
    reward_pred = network.apply(params, x_hat_freeze, method=network.w).squeeze(-1)
    reward_loss = jnp.mean((reward_pred - traj_batch.reward) ** 2)

    # =========================================================
    # PPO Actor Loss (Standard)
    # =========================================================
    # loss_actor, entropy = helpers.pi_loss_fn(params, network, traj_batch, advantages, config)
    loss_actor, entropy = (0,0)
    total_loss = forward_loss + done_loss + reward_loss + loss_actor - (config["ENT_COEF"] * entropy)

    loss_dict = {   'forward_loss': forward_loss,
                    'done_loss': done_loss,
                    'reward_loss': reward_loss    
    }
    return total_loss, loss_dict

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    pi = distrax.Categorical(logits=jnp.zeros(n_actions))

    def train(rng):
        k = config.get('k', 32)
        rng, init_rng = jax.random.split(rng)
        is_continuous = isinstance(env.action_space(env_params), spaces.Box)
        action_dim = env.action_space(env_params).shape[0] if is_continuous else env.action_space(env_params).n
        config['IS_CONTINUOUS'] = is_continuous
        norm_type = 'layer_norm' if config['LAYER_NORM'] else 'None'

        network = IVActorCritic(
            action_dim=action_dim, is_continuous=False, z_dim=k, phi_dim=k, norm_type = norm_type
        )
        network_params = network.init(init_rng, jnp.zeros(obs_shape), jnp.zeros(action_dim))
        train_state = networks.initialize_flax_train_state(config, network, network_params,)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                value = network.apply(train_state.params, last_obs)
                pi = distrax.Categorical(logits=jnp.zeros((config['NUM_ENVS'], n_actions)))
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs)

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # --- ADVANTAGE CALCULATION ---
            advantages, target = helpers.calculate_gae(traj_batch, config["GAMMA"], config["GAE_LAMBDA"], )


            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(iv_loss_fn, has_aux=True)
                    
                    # 1. Unpack the auxiliary tuple here!
                    (total_loss, loss_dict), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    # 2. Return them all so they get stacked by the scan
                    return train_state, (total_loss, loss_dict)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                # loss_info is now a tuple of 4 arrays: (total_loss, value_loss, loss_actor, entropy)
                train_state, (total_loss, loss_dict) = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), (total_loss, loss_dict)

            initial_update_state = (train_state, traj_batch, advantages, target, rng)
            update_state, (total_loss, loss_dict) = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update({
                k: v.mean() 
                for k, v in loss_dict.items() 
            })
            
            # Shared Metrics
            metric.update(
                {
                    "total_loss": total_loss.mean(),
                    "mean_rew": traj_batch.reward.mean(),
                }
            )
            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=True, light=config.get("LIGHT_METRICS", True)
            )
            metric.update(value_metrics)

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

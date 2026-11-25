"""
PureJAXRL-style V-PPO trainer for MuJoCo Playground (MJX) with Brax PPO parity
and decoupled Actor/Critic updates for TD(lambda), E(0), E(lambda), and BSMC.
"""

import copy
import os
import sys
from typing import Mapping

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from gymnax.wrappers.purerl import LogWrapper

from mjx_rl.core.mjx_config import config as base_default_config
from mjx_rl.core.mjx_critics import (
    calculate_gae,
    calculate_targets,
    get_critic_loss_fn,
    get_critic_target_lambda,
    ppo_actor_loss_fn,
    prepare_critic_transitions,
)
from mjx_rl.core.mjx_network import (
    ActorCritic,
    ActorCriticDualCritic,
    initialize_split_train_state,
)
from mjx_rl.core.mjx_utils import Transition, shuffle_and_batch
from mjx_rl.core.mjx_wrappers import (
    ClipAction,
    NormalizeVecObservation,
    NormalizeVecReward,
    PlaygroundGymnaxWrapper,
    VecEnv,
)
from core.contraction_diagnostics import (
    compute_contraction_diagnostics,
    update_diagnostic_ema,
)


def _make_dummy_obs(obs_space):
    if hasattr(obs_space, "spaces") and isinstance(obs_space.spaces, Mapping):
        return {k: jnp.zeros(sp.shape) for k, sp in obs_space.spaces.items()}
    return jnp.zeros(obs_space.shape)


def _index_obs_last(obs):
    return jax.tree_util.tree_map(lambda x: x[-1], obs)


def extract_hparams(config: dict) -> dict[str, jnp.ndarray]:
    """
    Extracts scalar hyperparameters as JAX float32 arrays so sweeps can vary
    LRs, lambdas, clip_eps, ent_coef, etc. without triggering XLA recompilation.
    """
    return {
        "actor_lr": jnp.asarray(float(config["ACTOR_LR"]), dtype=jnp.float32),
        "critic_lr": jnp.asarray(float(config["CRITIC_LR"]), dtype=jnp.float32),
        "clip_eps": jnp.asarray(float(config["CLIP_EPS"]), dtype=jnp.float32),
        "ent_coef": jnp.asarray(float(config["ENT_COEF"]), dtype=jnp.float32),
        "vf_coef": jnp.asarray(float(config["VF_COEF"]), dtype=jnp.float32),
        "aux_vf_coef": jnp.asarray(float(config["AUX_VF_COEF"]), dtype=jnp.float32),
        "gae_lambda": jnp.asarray(float(config.get("ACTOR_GAE_LAMBDA", config["GAE_LAMBDA"])), dtype=jnp.float32),
        "target_lambda": jnp.asarray(float(get_critic_target_lambda(config)), dtype=jnp.float32),
        "e_lambda": jnp.asarray(float(config["E_LAMBDA"]), dtype=jnp.float32),
        "aux_lambda": jnp.asarray(float(config["AUX_LAMBDA"]), dtype=jnp.float32),
        "reward_scaling": jnp.asarray(float(config.get("REWARD_SCALING", 1.0)), dtype=jnp.float32),
    }


def static_compile_signature(config: dict) -> tuple:
    """
    Returns a hashable key of all compile-time static fields in `config`
    (shapes, scan lengths, epochs, critic type, network architecture, LR schedule).
    Configs with the same signature can share a single compiled XLA executable.
    """
    net_cfg = config["NETWORK_CONFIG"]
    return (
        config["ENV_NAME"],
        config.get("IMPL", "jax"),
        int(config["TOTAL_TIMESTEPS"]),
        int(config["NUM_ENVS"]),
        int(config["NUM_STEPS"]),
        int(config.get("ACTION_REPEAT", 1)),
        int(config.get("EPISODE_LENGTH", 1000)),
        int(config.get("ACTOR_EPOCHS", net_cfg["NUM_EPOCHS"])),
        int(config.get("CRITIC_EPOCHS", net_cfg["NUM_EPOCHS"])),
        int(config.get("ACTOR_MINIBATCH_SIZE", net_cfg["MINIBATCH_SIZE"])),
        int(config.get("CRITIC_MINIBATCH_SIZE", net_cfg["MINIBATCH_SIZE"])),
        config["CRITIC_TYPE"].lower(),
        config["E_VARIANT"].lower(),
        bool(config.get("SHARED_TORSO", False)),
        bool(config.get("AUX_STOP_GRAD_TORSO", False)),
        bool(config.get("SPLIT_AUX_OPTIMIZER", False)),
        net_cfg.get("LR_SCHEDULE", "constant").lower(),
        bool(net_cfg.get("NORMALIZE_ENV", True)),
        bool(net_cfg.get("NORMALIZE_REW", False)),
    )


def make_train(config, dynamic_hparams: bool = False):
    cfg = copy.deepcopy(base_default_config)
    net_cfg = copy.deepcopy(cfg["NETWORK_CONFIG"])
    if "NETWORK_CONFIG" in config:
        net_cfg.update(config["NETWORK_CONFIG"])
    cfg.update(config)
    cfg["NETWORK_CONFIG"] = net_cfg
    config = cfg

    batch_size = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    action_repeat = int(config.get("ACTION_REPEAT", 1))
    steps_per_update = batch_size * action_repeat
    config["NUM_UPDATES"] = max(1, int(config["TOTAL_TIMESTEPS"]) // steps_per_update)
    config["MAX_NUM_POLICIES"] = config["NUM_UPDATES"]

    actor_epochs = int(config.get("ACTOR_EPOCHS", net_cfg["NUM_EPOCHS"]))
    critic_epochs = int(config.get("CRITIC_EPOCHS", net_cfg["NUM_EPOCHS"]))
    config["ACTOR_EPOCHS"] = actor_epochs
    config["CRITIC_EPOCHS"] = critic_epochs

    if "ACTOR_MINIBATCH_SIZE" in config:
        actor_mb = min(batch_size, int(config["ACTOR_MINIBATCH_SIZE"]))
        actor_num_mb = max(1, batch_size // actor_mb)
    else:
        actor_num_mb = int(net_cfg.get("NUM_MINIBATCHES", max(1, batch_size // int(net_cfg["MINIBATCH_SIZE"]))))
        actor_mb = max(1, batch_size // actor_num_mb)
    config["ACTOR_MINIBATCH_SIZE"] = actor_mb
    config["ACTOR_NUM_MINIBATCHES"] = actor_num_mb

    if "CRITIC_MINIBATCH_SIZE" in config:
        critic_mb = min(batch_size, int(config["CRITIC_MINIBATCH_SIZE"]))
        critic_num_mb = max(1, batch_size // critic_mb)
    else:
        critic_num_mb = int(net_cfg.get("NUM_MINIBATCHES", max(1, batch_size // int(net_cfg["MINIBATCH_SIZE"]))))
        critic_mb = max(1, batch_size // critic_num_mb)
    config["CRITIC_MINIBATCH_SIZE"] = critic_mb
    config["CRITIC_NUM_MINIBATCHES"] = critic_num_mb

    env = PlaygroundGymnaxWrapper(
        env_name=config["ENV_NAME"],
        impl=config.get("IMPL", "jax"),
        episode_length=int(config.get("EPISODE_LENGTH", 1000)),
        action_repeat=action_repeat,
        full_reset=bool(config.get("FULL_RESET", False)),
        config_overrides=config.get("ENV_CONFIG_OVERRIDES", None),
    )
    env_params = None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if net_cfg.get("NORMALIZE_ENV", True):
        env = NormalizeVecObservation(env)
    if net_cfg.get("NORMALIZE_REW", False):
        env = NormalizeVecReward(env, config["GAMMA"])

    reward_scaling = float(config.get("REWARD_SCALING", 1.0))

    def _train_impl(rng, hparams):
        c_type = config["CRITIC_TYPE"].lower()
        e_variant = config["E_VARIANT"].lower()
        critic_loss_fn = get_critic_loss_fn(c_type, split=True)
        actor_loss_fn = ppo_actor_loss_fn

        gae_lambda = hparams["gae_lambda"]
        target_lambda = hparams["target_lambda"]
        r_scale = hparams["reward_scaling"]
        traced_cfg = {
            **config,
            "CLIP_EPS": hparams["clip_eps"],
            "ENT_COEF": hparams["ent_coef"],
            "VF_COEF": hparams["vf_coef"],
            "AUX_VF_COEF": hparams["aux_vf_coef"],
            "E_LAMBDA": hparams["e_lambda"],
            "AUX_LAMBDA": hparams["aux_lambda"],
        }
        uses_aux = (
            c_type in [
                "e_0", "e0", "e", "sampled_e", "surgical_v_aux", "surgical",
                "e_lambda", "elambda", "e_geometric",
                "bsmc", "bsmc_v_aux",
            ]
            and e_variant in ["aux", "e_aux"]
        )

        policy_layers = tuple(net_cfg.get("POLICY_HIDDEN_LAYER_SIZES", (32, 32, 32, 32)))
        value_layers = tuple(net_cfg.get("VALUE_HIDDEN_LAYER_SIZES", (256, 256, 256, 256, 256)))
        activation = net_cfg.get("ACTIVATION", "swish")
        dist_type = net_cfg.get("DISTRIBUTION_TYPE", "tanh_normal")
        kernel_init_type = net_cfg.get("KERNEL_INIT", "lecun_uniform")
        policy_obs_key = config.get("POLICY_OBS_KEY", "state")
        value_obs_key = config.get("VALUE_OBS_KEY", "state")

        if uses_aux:
            network = ActorCriticDualCritic(
                action_dim=env.action_space(env_params).shape[0],
                activation=activation,
                policy_hidden_layer_sizes=policy_layers,
                value_hidden_layer_sizes=value_layers,
                distribution_type=dist_type,
                kernel_init_type=kernel_init_type,
                layer_norm=config.get("LAYER_NORM", False),
                shared_torso=config.get("SHARED_TORSO", False),
                aux_stop_grad_torso=config.get("AUX_STOP_GRAD_TORSO", False),
                policy_obs_key=policy_obs_key,
                value_obs_key=value_obs_key,
            )
        else:
            network = ActorCritic(
                action_dim=env.action_space(env_params).shape[0],
                activation=activation,
                policy_hidden_layer_sizes=policy_layers,
                value_hidden_layer_sizes=value_layers,
                distribution_type=dist_type,
                kernel_init_type=kernel_init_type,
                n_value_heads=config.get("NUM_VALUE_HEADS", 1),
                layer_norm=config.get("LAYER_NORM", False),
                policy_obs_key=policy_obs_key,
                value_obs_key=value_obs_key,
            )

        rng, _rng = jax.random.split(rng)
        init_x = _make_dummy_obs(env.observation_space(env_params))
        network_params = network.init(_rng, init_x)
        train_state = initialize_split_train_state(config, network, network_params, hparams=hparams)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)
        log_contraction_diag = bool(config["LOG_CONTRACTION_DIAGNOSTICS"])
        diag_probe_envs = int(config["DIAG_PROBE_ENVS"])
        diag_ema = float(config["DIAG_EMA"])
        diag_alpha_ema_init = jnp.asarray(0.0, dtype=jnp.float32)
        diag_beta2_ema_init = jnp.asarray(0.0, dtype=jnp.float32)

        if c_type in ("td_0", "e_0", "e_0_basic", "e_0_v_aux", "msbe"):
            diag_lambda = jnp.asarray(0.0, dtype=jnp.float32)
        elif c_type in ("e_lambda", "e_lambda_v_aux"):
            diag_lambda = hparams["e_lambda"]
        else:
            diag_lambda = hparams["target_lambda"]

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx, diag_alpha_ema, diag_beta2_ema = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                if uses_aux:
                    value, v_aux = network.apply(train_state.params, last_obs, method=network.both)
                    pi = network.apply(train_state.params, last_obs, method=network.policy)
                else:
                    pi, value = network.apply(train_state.params, last_obs)
                    v_aux = jnp.zeros_like(value)

                raw_action = pi.sample_raw(seed=_rng)
                action = pi.postprocess(raw_action)
                log_prob = pi.log_prob(raw_action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                scaled_reward = reward * r_scale

                next_obs = info["real_next_obs"]
                is_timeout = info["is_timeout"]
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal
                info_with_raw = {**info, "raw_action": raw_action}

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=scaled_reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=next_obs,
                    info=info_with_raw,
                    is_start=is_start,
                    is_timeout=is_timeout,
                    v_aux=v_aux,
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. CALCULATE ADVANTAGES & TARGETS
            last_val_raw = network.apply(train_state.params, last_obs, method=network.value)
            regular_next_val = jnp.concatenate([traj_batch.value[1:], last_val_raw[None]], axis=0)
            timeout_next_val = network.apply(train_state.params, traj_batch.next_obs, method=network.value)
            if timeout_next_val.ndim > regular_next_val.ndim:
                timeout_next_val = timeout_next_val.squeeze(-1)
            next_value = jnp.where(traj_batch.is_timeout, timeout_next_val, regular_next_val)

            advantages, gae_targets = calculate_gae(
                traj_batch,
                next_value,
                gamma=config["GAMMA"],
                gae_lambda=gae_lambda,
            )
            targets = calculate_targets(
                traj_batch,
                next_value,
                gamma=config["GAMMA"],
                target_lambda=target_lambda,
            )

            if uses_aux and e_variant == "aux":
                final_next_aux_val = network.apply(
                    train_state.params,
                    _index_obs_last(traj_batch.next_obs),
                    method=network.v_aux,
                )
                if final_next_aux_val.ndim > 1:
                    final_next_aux_val = final_next_aux_val.squeeze(-1)
                regular_aux_next = jnp.concatenate([traj_batch.v_aux[1:], final_next_aux_val[None]], axis=0)
                timeout_aux_next = network.apply(train_state.params, traj_batch.next_obs, method=network.v_aux)
                if timeout_aux_next.ndim > regular_aux_next.ndim:
                    timeout_aux_next = timeout_aux_next.squeeze(-1)
                aux_next_value = jnp.where(traj_batch.is_timeout, timeout_aux_next, regular_aux_next)
                aux_lambda = hparams["aux_lambda"]
                traj_batch_aux = traj_batch._replace(value=traj_batch.v_aux)
                _, aux_targets = calculate_gae(
                    traj_batch_aux,
                    aux_next_value,
                    gamma=config["GAMMA"],
                    gae_lambda=aux_lambda,
                )
            elif uses_aux and e_variant == "e_aux":
                aux_lambda = hparams["aux_lambda"]
                aux_targets = calculate_targets(
                    traj_batch,
                    next_value,
                    gamma=config["GAMMA"],
                    target_lambda=aux_lambda,
                )
            else:
                aux_targets = targets

            true_terminal = traj_batch.done & ~traj_batch.is_timeout
            last_val = jnp.where(true_terminal[-1], 0.0, next_value[-1])
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(last_val)
            next_targets = jnp.where(traj_batch.is_timeout, next_value, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)

            traj_batch = traj_batch._replace(
                next_value=next_value,
                targets=targets,
                next_targets=next_targets,
                aux_targets=aux_targets,
            )

            # 3. PREPARE CRITIC TRANSITIONS
            rng, _rng = jax.random.split(rng)
            traj_batch = prepare_critic_transitions(
                critic_type=c_type,
                traj_batch=traj_batch,
                targets=targets,
                last_val=last_val,
                gamma=config["GAMMA"],
                e_lambda=hparams["e_lambda"],
                rng=_rng,
            )

            # 3b. ONLINE CONTRACTION & VARIANCE DIAGNOSTICS
            if log_contraction_diag:
                def _v_apply(c_params, x_flat):
                    full_p = {"params": {**train_state.actor.params, **c_params}}
                    v = network.apply(full_p, x_flat, method=network.value)
                    return v.squeeze(-1) if (v.ndim > 1 and v.shape[-1] == 1) else v

                diag_out = compute_contraction_diagnostics(
                    critic_params=train_state.critic.params,
                    v_apply_fn=_v_apply,
                    traj_batch=traj_batch,
                    gamma=config["GAMMA"],
                    lmbda=diag_lambda,
                    n_probe_envs=diag_probe_envs,
                    uses_aux=uses_aux,
                )
                diag_alpha_ema, diag_beta2_ema, diag_beta_eff, diag_r_rot_c = update_diagnostic_ema(
                    diag_alpha_ema,
                    diag_beta2_ema,
                    diag_out["alpha_eff"],
                    diag_out["beta2_eff"],
                    idx,
                    ema_decay=diag_ema,
                )
                diag_alpha_eff = diag_out["alpha_eff"]
                diag_rho_c = diag_out["rho_c"]
                diag_ret_var_ratio = diag_out["ret_var_ratio"]
                diag_aux_unexp_var = diag_out["aux_unexp_var"]
                diag_aux_vs_v = diag_out["aux_vs_v"]
            else:
                diag_alpha_eff = jnp.asarray(0.0, dtype=jnp.float32)
                diag_beta_eff = jnp.asarray(0.0, dtype=jnp.float32)
                diag_rho_c = jnp.asarray(0.0, dtype=jnp.float32)
                diag_r_rot_c = jnp.asarray(0.0, dtype=jnp.float32)
                diag_ret_var_ratio = jnp.asarray(0.0, dtype=jnp.float32)
                diag_aux_unexp_var = jnp.asarray(0.0, dtype=jnp.float32)
                diag_aux_vs_v = jnp.asarray(0.0, dtype=jnp.float32)

            # 4. CRITIC UPDATE PHASE
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    traj_mb = mb

                    def _loss_fn(critic_params):
                        return critic_loss_fn(
                            critic_params,
                            train_state.actor.params,
                            network,
                            traj_mb,
                            traj_mb.targets,
                            traced_cfg,
                        )

                    (_, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.critic.params
                    )
                    train_state = train_state.apply_critic_gradients(grads)
                    return train_state, aux

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(
                    _rng, critic_batch, config["CRITIC_NUM_MINIBATCHES"]
                )
                train_state, critic_metrics = jax.lax.scan(
                    _critic_minibatch, train_state, minibatches
                )
                return (train_state, critic_batch, rng), critic_metrics

            rng, _rng = jax.random.split(rng)
            critic_state = (train_state, traj_batch, _rng)
            (train_state, _, rng), critic_loss_info = jax.lax.scan(
                _critic_epoch, critic_state, None, config["CRITIC_EPOCHS"]
            )

            # 5. ACTOR UPDATE PHASE
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(carry, mb):
                    train_state, rng = carry
                    traj_mb, adv_mb = mb
                    rng, _rng = jax.random.split(rng)

                    def _loss_fn(actor_params):
                        return actor_loss_fn(
                            actor_params,
                            train_state.critic.params,
                            network,
                            traj_mb,
                            adv_mb,
                            traced_cfg,
                            rng=_rng,
                        )

                    (_, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.actor.params
                    )
                    train_state = train_state.apply_actor_gradients(grads)
                    return (train_state, rng), aux

                train_state, actor_batch, rng = actor_state
                rng, _rng_shuf, _rng_mb = jax.random.split(rng, 3)
                minibatches = shuffle_and_batch(
                    _rng_shuf, actor_batch, config["ACTOR_NUM_MINIBATCHES"]
                )
                (train_state, _), actor_metrics = jax.lax.scan(
                    _actor_minibatch, (train_state, _rng_mb), minibatches
                )
                return (train_state, actor_batch, rng), actor_metrics

            actor_batch = (traj_batch, advantages)
            rng, _rng = jax.random.split(rng)
            actor_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), actor_loss_info = jax.lax.scan(
                _actor_epoch, actor_state, None, config["ACTOR_EPOCHS"]
            )

            # 6. RUNTIME METRICS
            combined_loss_info = {**critic_loss_info, **actor_loss_info}
            metric = {
                "pi_idx": idx,
                "pi_num": idx,
                "advantage": advantages.mean(),
            }
            metric.update({k: v.mean() for k, v in combined_loss_info.items()})
            metric.update(
                {
                    k: v.mean()
                    for k, v in traj_batch.info.items()
                    if k
                    not in [
                        "real_next_obs",
                        "obs_jump",
                        "targets_jump",
                        "valid_jump",
                        "is_absorbing",
                        "raw_action",
                    ]
                    and hasattr(v, "mean")
                    and v.ndim <= 2
                }
            )

            if log_contraction_diag:
                metric["diag_rho_c"] = diag_rho_c
                metric["diag_r_rot_c"] = diag_r_rot_c
                metric["diag_ret_var_ratio"] = diag_ret_var_ratio
                metric["diag_aux_unexp_var"] = diag_aux_unexp_var
                metric["diag_aux_vs_v"] = diag_aux_vs_v
                metric["diag_alpha_eff"] = diag_alpha_eff
                metric["diag_beta_eff"] = diag_beta_eff

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_true_terminal,
                rng,
                idx + 1,
                diag_alpha_ema,
                diag_beta2_ema,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            last_true_terminal,
            _rng,
            1,
            diag_alpha_ema_init,
            diag_beta2_ema_init,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    if dynamic_hparams:
        return _train_impl

    def train(rng):
        return _train_impl(rng, extract_hparams(config))

    return train

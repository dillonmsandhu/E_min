#!/usr/bin/env python3
"""
benchmark_q_ppo.py
Benchmarks Q-PPO with Analytical Marginalized GAE against Standard PPO on CartPole-v1:
1. Standard PPO (TD λ=0.8)
2. Q-PPO (SARSA λ=0.8)
3. Q-PPO (E(0)-Q, no V_aux)
4. Q-PPO (E(λ=0.4)-Q, no V_aux)

Decoupled Actor-Critic (DualTrainState), 4 actor/critic epochs, minibatch size 512,
32 parallel envs x 256 rollout length = 8192 batch size, 32 updates (262,144 timesteps).
"""
import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp

from gymnax_rl.algos.ppo import make_train as make_ppo_train
from gymnax_rl.algos.q_ppo import make_train as make_q_ppo_train

OUTPUT_DIR = "/Users/dillonsandhu/.gemini/antigravity-ide/brain/dcbfdcae-de92-4f35-80ff-46e88e5a5f98/q_ppo_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANTS = [
    {
        "name": "Standard PPO (TD λ=0.8)",
        "algo": "ppo",
        "critic_type": "td",
        "lambda": 0.8,
        "color": "#1f77b4",
    },
    {
        "name": "Q-PPO (SARSA λ=0.8)",
        "algo": "q_ppo",
        "critic_type": "sarsa",
        "lambda": 0.8,
        "color": "#2ca02c",
    },
    {
        "name": "Q-PPO (E(0)-Q, no V_aux)",
        "algo": "q_ppo",
        "critic_type": "e_0_q",
        "lambda": 0.0,
        "color": "#ff7f0e",
    },
    {
        "name": "Q-PPO (E(λ=0.4)-Q, no V_aux)",
        "algo": "q_ppo",
        "critic_type": "e_lambda_q",
        "lambda": 0.4,
        "color": "#d62728",
    },
]

SEEDS = [42, 1337, 2026]


def run_benchmark():
    print(f"================================================================")
    print(f"Running Q-PPO Benchmark on CartPole-v1 across {len(SEEDS)} seeds")
    print(f"================================================================")

    common_config = {
        "ENV_NAME": "CartPole-v1",
        "NUM_ENVS": 32,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 262144,
        "ACTOR_EPOCHS": 4,
        "CRITIC_EPOCHS": 4,
        "NUM_EPOCHS": 4,
        "MINIBATCH_SIZE": 512,
        "ACTOR_MINIBATCH_SIZE": 512,
        "CRITIC_MINIBATCH_SIZE": 512,
        "LR": 3e-4,
        "CRITIC_LR": 1e-3,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.8,
    }

    results = {}

    for var in VARIANTS:
        v_name = var["name"]
        print(f"\nEvaluating variant: {v_name}")
        cfg = common_config.copy()
        cfg["CRITIC_TYPE"] = var["critic_type"]
        if "sarsa" in var["critic_type"] or "td" in var["critic_type"]:
            cfg["VALUE_LAMBDA"] = var["lambda"]
        cfg["E_LAMBDA"] = var["lambda"]

        if var["algo"] == "ppo":
            train_fn = make_ppo_train(cfg)
        else:
            train_fn = make_q_ppo_train(cfg)

        v_returns = []
        v_lengths = []
        v_v_preds = []
        v_terms = []
        v_truncs = []

        start_time = time.time()
        for seed in SEEDS:
            rng = jax.random.PRNGKey(seed)
            out = train_fn(rng)
            jax.block_until_ready(out)

            metrics = out["metrics"]
            v_returns.append(np.array(metrics["returned_episode_returns"]))
            v_lengths.append(np.array(metrics["returned_episode_lengths"]))
            v_v_preds.append(np.array(metrics["v_pred"]))
            v_terms.append(np.array(metrics["n_terminations"]))
            v_truncs.append(np.array(metrics["n_truncations"]))

        elapsed = time.time() - start_time
        v_returns = np.stack(v_returns, axis=0) # (seeds, updates)
        v_lengths = np.stack(v_lengths, axis=0)
        v_v_preds = np.stack(v_v_preds, axis=0)
        v_terms = np.stack(v_terms, axis=0)
        v_truncs = np.stack(v_truncs, axis=0)

        mean_final = np.mean(v_returns[:, -5:])
        auc = np.mean(v_returns)

        print(f"  -> Done in {elapsed:.2f}s | Final Return: {mean_final:.2f} | AUC: {auc:.2f}")

        results[v_name] = {
            "returns": v_returns,
            "lengths": v_lengths,
            "v_preds": v_v_preds,
            "terminations": v_terms,
            "truncations": v_truncs,
            "mean_final": mean_final,
            "auc": auc,
            "color": var["color"],
        }

    # Save raw results
    with open(os.path.join(OUTPUT_DIR, "q_ppo_benchmark_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Generate Learning Curves Figure
    plt.figure(figsize=(10, 6), dpi=300)
    updates = np.arange(1, common_config["TOTAL_TIMESTEPS"] // (common_config["NUM_ENVS"] * common_config["NUM_STEPS"]) + 1)
    timesteps = updates * (common_config["NUM_ENVS"] * common_config["NUM_STEPS"])

    for v_name, data in results.items():
        mean_ret = np.mean(data["returns"], axis=0)
        std_ret = np.std(data["returns"], axis=0)
        c = data["color"]
        plt.plot(timesteps, mean_ret, label=f"{v_name} (AUC: {data['auc']:.1f})", color=c, lw=2.2)
        plt.fill_between(timesteps, mean_ret - std_ret, mean_ret + std_ret, color=c, alpha=0.15)

    plt.axhline(500, color="gray", linestyle="--", alpha=0.6, label="Max Return (500)")
    plt.xlabel("Environment Timesteps", fontsize=12)
    plt.ylabel("Episodic Return", fontsize=12)
    plt.title("Q-PPO (Analytical Marginalized GAE) vs. Standard PPO on CartPole-v1", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "q_ppo_cartpole_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    # Generate Summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "q_ppo_cartpole_summary.csv")
    with open(csv_path, "w") as f:
        f.write("Algorithm,Final_Return_Mean,Final_Return_Std,AUC,Total_Terminations,Total_Truncations\n")
        for v_name, data in results.items():
            final_mean = np.mean(data["returns"][:, -5:])
            final_std = np.std(data["returns"][:, -5:])
            auc = data["auc"]
            tot_term = np.sum(data["terminations"])
            tot_trunc = np.sum(data["truncations"])
            f.write(f'"{v_name}",{final_mean:.2f},{final_std:.2f},{auc:.2f},{tot_term},{tot_trunc}\n')

    print(f"\nAll artifacts generated at: {OUTPUT_DIR}")
    print(f"Plot saved to: {plot_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    run_benchmark()

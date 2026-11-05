#!/usr/bin/env python3
"""
benchmark_v_aux_enhancements.py
Evaluates V_aux learning improvements on CartPole-v1 for E(0) and E(0.9):
1. Baseline: Pure MSE on V_aux
2. + Huber Loss: delta=10.0 on V_aux
3. + Trust Region: beta=1.0 on V_aux
4. + Both: Huber (delta=10.0) + Trust Region (beta=1.0) on V_aux

Evaluated across 3 seeds on CartPole-v1.
Tracks:
- Episodic Return learning curves & AUC
- MSE of V_aux vs return targets
- MSE of V_main vs return targets
- Cosine similarity between V_main and V_aux
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

from gymnax_rl.algos.ppo import make_train

OUTPUT_DIR = "/Users/dillonsandhu/.gemini/antigravity-ide/brain/dcbfdcae-de92-4f35-80ff-46e88e5a5f98/v_aux_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITIONS = [
    {"name": "Baseline (MSE)", "loss_type": "mse", "trust_coef": 0.0, "color": "#1f77b4"},
    {"name": "+ Huber Loss (δ=10)", "loss_type": "huber", "huber_delta": 10.0, "trust_coef": 0.0, "color": "#2ca02c"},
    {"name": "+ Trust Region (β=1)", "loss_type": "mse", "trust_coef": 1.0, "color": "#ff7f0e"},
    {"name": "+ Both (Huber + Trust)", "loss_type": "huber", "huber_delta": 10.0, "trust_coef": 1.0, "color": "#d62728"},
]

ALGORITHMS = [
    {"name": "E(0) with V_aux", "critic_type": "e_0", "e_lambda": 0.0},
    {"name": "E(0.9) with V_aux", "critic_type": "e_lambda", "e_lambda": 0.9},
]

SEEDS = [42, 1337, 2026]


def run_benchmark():
    print("=" * 70)
    print("V_aux Learning Enhancements Benchmark on CartPole-v1")
    print("=" * 70)

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
        "USES_AUX": True,
        "SURGICAL_V_AUX": True,
    }

    results = {}

    for algo in ALGORITHMS:
        a_name = algo["name"]
        results[a_name] = {}
        print(f"\n>>> Running Algorithm: {a_name}")

        for cond in CONDITIONS:
            c_name = cond["name"]
            full_name = f"{a_name} [{c_name}]"
            print(f"  Evaluating condition: {c_name} ...")

            cfg = common_config.copy()
            cfg["CRITIC_TYPE"] = algo["critic_type"]
            cfg["E_LAMBDA"] = algo["e_lambda"]
            cfg["VALUE_LAMBDA"] = algo["e_lambda"]
            cfg["AUX_LOSS_TYPE"] = cond["loss_type"]
            cfg["AUX_HUBER_DELTA"] = cond.get("huber_delta", 10.0)
            cfg["AUX_TRUST_REGION_COEF"] = cond["trust_coef"]

            train_fn = make_train(cfg)

            returns_all = []
            mse_aux_all = []
            mse_main_all = []
            sim_all = []

            start_t = time.time()
            for seed in SEEDS:
                rng = jax.random.PRNGKey(seed)
                out = train_fn(rng)
                jax.block_until_ready(out)

                metrics = out["metrics"]
                returns_all.append(np.array(metrics["returned_episode_returns"]))
                if "mse_v_aux" in metrics:
                    mse_aux_all.append(np.array(metrics["mse_v_aux"]))
                    mse_main_all.append(np.array(metrics["mse_v_main"]))
                    sim_all.append(np.array(metrics["v_similarity"]))

            elapsed = time.time() - start_t
            returns_arr = np.stack(returns_all, axis=0) # (seeds, updates)
            mean_final = np.mean(returns_arr[:, -5:])
            auc = np.mean(returns_arr)

            entry = {
                "returns": returns_arr,
                "mean_final": mean_final,
                "auc": auc,
                "color": cond["color"],
            }
            if len(mse_aux_all) > 0:
                entry["mse_v_aux"] = np.stack(mse_aux_all, axis=0)
                entry["mse_v_main"] = np.stack(mse_main_all, axis=0)
                entry["v_similarity"] = np.stack(sim_all, axis=0)

            print(f"    -> Done in {elapsed:.1f}s | Final: {mean_final:.2f} | AUC: {auc:.2f}")
            results[a_name][c_name] = entry

    # Save raw data
    with open(os.path.join(OUTPUT_DIR, "v_aux_benchmark_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Plot 1: Learning curves for E(0) and E(0.9) side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)
    updates = np.arange(1, common_config["TOTAL_TIMESTEPS"] // (common_config["NUM_ENVS"] * common_config["NUM_STEPS"]) + 1)
    timesteps = updates * (common_config["NUM_ENVS"] * common_config["NUM_STEPS"])

    for i, algo in enumerate(ALGORITHMS):
        ax = axes[i]
        a_name = algo["name"]
        for cond in CONDITIONS:
            c_name = cond["name"]
            data = results[a_name][c_name]
            mean_ret = np.mean(data["returns"], axis=0)
            std_ret = np.std(data["returns"], axis=0)
            c = data["color"]
            ax.plot(timesteps, mean_ret, label=f"{c_name} (AUC: {data['auc']:.1f})", color=c, lw=2.2)
            ax.fill_between(timesteps, mean_ret - std_ret, mean_ret + std_ret, color=c, alpha=0.12)

        ax.axhline(500, color="gray", linestyle="--", alpha=0.6, label="Max (500)")
        ax.set_title(f"{a_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Environment Timesteps", fontsize=11)
        if i == 0:
            ax.set_ylabel("Episodic Return", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=9.5, loc="lower right")

    plt.suptitle("Impact of V_aux Huber Loss & Trust-Region Regularization on CartPole-v1", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    curve_plot = os.path.join(OUTPUT_DIR, "v_aux_enhancement_learning_curves.png")
    plt.savefig(curve_plot)
    plt.close()

    # Plot 2: V_aux vs V_main MSE and Cosine Similarity
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    for i, algo in enumerate(ALGORITHMS):
        a_name = algo["name"]
        ax_mse = axes[0, i]
        ax_sim = axes[1, i]

        for cond in CONDITIONS:
            c_name = cond["name"]
            data = results[a_name][c_name]
            if "mse_v_aux" in data:
                mean_mse = np.mean(data["mse_v_aux"], axis=0)
                mean_sim = np.mean(data["v_similarity"], axis=0)
                c = data["color"]
                ax_mse.plot(timesteps, mean_mse, label=f"{c_name}", color=c, lw=2.0)
                ax_sim.plot(timesteps, mean_sim, label=f"{c_name}", color=c, lw=2.0)

        ax_mse.set_title(f"V_aux Target MSE: {a_name}", fontsize=12, fontweight="bold")
        ax_mse.set_xlabel("Timesteps")
        ax_mse.set_ylabel("MSE vs Return Target")
        ax_mse.grid(True, alpha=0.3)
        ax_mse.legend(fontsize=9)

        ax_sim.set_title(f"Cosine Similarity (V_main, V_aux): {a_name}", fontsize=12, fontweight="bold")
        ax_sim.set_xlabel("Timesteps")
        ax_sim.set_ylabel("Cosine Similarity")
        ax_sim.grid(True, alpha=0.3)
        ax_sim.legend(fontsize=9)

    plt.tight_layout()
    diag_plot = os.path.join(OUTPUT_DIR, "v_aux_mse_and_similarity.png")
    plt.savefig(diag_plot)
    plt.close()

    # Save Summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "v_aux_enhancements_summary.csv")
    with open(csv_path, "w") as f:
        f.write("Algorithm,Condition,Final_Return_Mean,Final_Return_Std,AUC\n")
        for algo in ALGORITHMS:
            a_name = algo["name"]
            for cond in CONDITIONS:
                c_name = cond["name"]
                data = results[a_name][c_name]
                final_mean = np.mean(data["returns"][:, -5:])
                final_std = np.std(data["returns"][:, -5:])
                auc = data["auc"]
                f.write(f'"{a_name}","{c_name}",{final_mean:.2f},{final_std:.2f},{auc:.2f}\n')

    print(f"\nAll artifacts generated at: {OUTPUT_DIR}")
    print(f"Curves saved to: {curve_plot}")
    print(f"Diagnostics saved to: {diag_plot}")
    print(f"Summary saved to: {csv_path}")


if __name__ == "__main__":
    run_benchmark()

"""
plot_lambda_spectrum_vs_E.py
CLI tool to plot the full spectrum of lambda curves vs E-minimization.

Usage:
    python notebooks/plot_lambda_spectrum_vs_E.py --sweep-dir results/ppo/sweeps/ppo_CartPole-v1_... --metric returned_episode_returns
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from notebooks.analyze_sweeps import (
    load_sweep_data,
    find_latest_run_dir,
    plot_lambda_spectrum_vs_E,
)


def main():
    parser = argparse.ArgumentParser(description="Plot lambda spectrum vs E minimization")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Sweep batch root directory")
    parser.add_argument("--td-algo", type=str, default="sampled_td_lambda", help="TD algorithm name folder")
    parser.add_argument("--e-algo", type=str, default="sampled_E", help="E algorithm name folder")
    parser.add_argument("--metric", type=str, default="returned_episode_returns", help="Metric to plot")
    parser.add_argument("--lambda-param", type=str, default="VALUE_LAMBDA", help="Hyperparameter name for lambda")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for saved plot")
    parser.add_argument("--log-scale", action="store_true", help="Plot on log scale")
    parser.add_argument("--email", type=str, default=None, help="Email recipient to send PDF results")

    args = parser.parse_args()

    td_path = os.path.join(args.sweep_dir, args.td_algo, "tuning")
    if not os.path.exists(td_path):
        td_path = os.path.join(args.sweep_dir, args.td_algo)

    e_path = os.path.join(args.sweep_dir, args.e_algo, "tuning")
    if not os.path.exists(e_path):
        e_path = os.path.join(args.sweep_dir, args.e_algo)

    td_sweep_data = load_sweep_data(td_path)
    e_sweep_data = load_sweep_data(e_path)

    out_dir = args.out_dir or os.path.join(args.sweep_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    png_save_path = os.path.join(out_dir, "comparison_lambda_spectrum_vs_E.png")
    pdf_save_path = os.path.join(out_dir, "comparison_lambda_spectrum_vs_E.pdf")

    env_name = td_sweep_data.get("env_name", "")
    fig = plot_lambda_spectrum_vs_E(
        td_sweep_data=td_sweep_data,
        e_sweep_data=e_sweep_data,
        metric_key=args.metric,
        lambda_param=args.lambda_param,
        title=f"E-Minimization vs. TD(λ) Spectrum ({env_name})",
        save_path=png_save_path,
        log_scale=args.log_scale,
    )
    if fig is not None:
        fig.savefig(pdf_save_path, bbox_inches="tight")
        print(f"Lambda spectrum comparison PDF saved to: {pdf_save_path}")

    recipient = args.email or os.environ.get("EMAIL_RECIPIENT")
    if recipient and os.path.exists(pdf_save_path):
        from core.mail import email_pdf
        email_pdf(
            pdf_save_path,
            recipient=recipient,
            subject=f"[{env_name}] E-Minimization vs. TD(λ) Spectrum Plot",
            body=f"Environment: {env_name}\nMetric: {args.metric}\nResults Directory: {args.sweep_dir}",
        )


if __name__ == "__main__":
    main()

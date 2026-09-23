import marimo

__generated_with = "0.10.0"
app = marimo.App(width="normal")


@app.cell
def __():
    import cloudpickle
    import glob
    import json
    import os
    import re
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return cloudpickle, glob, json, mo, np, os, pd, plt, re


@app.cell
def __(np, pd, re):
    TARGET_ALGS = [
        "E_lambda_diff",
        "E_lambda_fixed",
        "E_lambda_geometric",
        "E0",
        "ppo",
    ]

    ALGO_STYLE = {
        "E": {"label": "E(0) (1-step baseline)", "color": "#2ca02c", "linestyle": "-"},
        "E0": {"label": "E(0) (1-step baseline)", "color": "#2ca02c", "linestyle": "-"},
        "sampled_E": {"label": "E(0) (1-step baseline)", "color": "#2ca02c", "linestyle": "-"},
        "E_lambda_fixed": {"label": "E(λ) Fixed (FVI)", "color": "#1f77b4", "linestyle": "-"},
        "sampled_E_lambda": {"label": "E(λ) Fixed (FVI)", "color": "#1f77b4", "linestyle": "-"},
        "E_lambda_diff": {"label": "E(λ) Diff (Method 2)", "color": "#ff7f0e", "linestyle": "-"},
        "E_lambda_differentiable": {"label": "E(λ) Diff (Method 2)", "color": "#ff7f0e", "linestyle": "-"},
        "sampled_E_lambda_diff": {"label": "E(λ) Diff (Method 2)", "color": "#ff7f0e", "linestyle": "-"},
        "E_lambda_geom": {"label": "E(λ) Geom (Method 3)", "color": "#9467bd", "linestyle": "-"},
        "E_lambda_geometric": {"label": "E(λ) Geom (Method 3)", "color": "#9467bd", "linestyle": "-"},
        "sampled_E_lambda_geom": {"label": "E(λ) Geom (Method 3)", "color": "#9467bd", "linestyle": "-"},
        "ppo": {"label": "TD(λ) (PPO baseline)", "color": "#d62728", "linestyle": "--"},
        "sampled_td_lambda": {"label": "TD(λ) (PPO baseline)", "color": "#d62728", "linestyle": "--"},
        "td_lambda": {"label": "TD(λ) (PPO baseline)", "color": "#d62728", "linestyle": "--"},
        "mc": {"label": "Monte Carlo", "color": "#8c564b", "linestyle": ":"},
    }

    def parse_lambda(suffix_str, config_dict=None):
        """Extracts numerical lambda from folder suffix (e.g. lambda_0, lambda_09) or config."""
        match = re.search(r"lambda_([0-9.]+)", suffix_str)
        if match:
            raw = match.group(1)
            if "." in raw:
                return float(raw)
            if raw in ("0", "00"):
                return 0.0
            elif raw in ("1", "10"):
                return 1.0
            elif raw.startswith("0"):
                return float("0." + raw[1:])
            else:
                return float(raw)

        if config_dict:
            # Check E_LAMBDA first, then VALUE_LAMBDA
            e_lam = config_dict.get("E_LAMBDA")
            if e_lam is not None:
                return float(e_lam)
            val_lam = config_dict.get("VALUE_LAMBDA")
            if val_lam is not None:
                return float(val_lam)

        return None

    def smooth_series(series_1d, mode="Exponential Moving Average", span_or_window=20):
        """Applies smoothing suitable for single-seed noisy RL trajectories."""
        s = pd.Series(series_1d)
        if mode == "Exponential Moving Average":
            span = max(1, int(span_or_window))
            return s.ewm(span=span, adjust=False).mean().values
        elif mode == "Rolling Mean":
            window = max(1, int(span_or_window))
            return s.rolling(window=window, min_periods=1).mean().values
        else:
            return np.asarray(series_1d)

    def load_single_run(pkl_path, cloudpickle_mod, json_mod, os_mod):
        """Loads out.pkl and adjacent config.json, returning metrics and config."""
        try:
            with open(pkl_path, "rb") as f:
                data = cloudpickle_mod.load(f)
        except Exception:
            return None, None

        cfg = {}
        cfg_path = os_mod.path.join(os_mod.path.dirname(pkl_path), "config.json")
        if os_mod.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json_mod.load(f)
            except Exception:
                pass

        metrics = {}
        if isinstance(data, dict):
            if "metrics" in data and isinstance(data["metrics"], dict):
                metrics = data["metrics"]
            elif "returned_episode_returns" in data:
                metrics = data
            elif not cfg and "ENV_NAME" in data:
                cfg = data

        return metrics, cfg

    return ALGO_STYLE, TARGET_ALGS, load_single_run, parse_lambda, smooth_series


@app.cell
def __(TARGET_ALGS, mo):
    results_dir_input = mo.ui.text(
        value="results",
        label="Results Base Directory",
        placeholder="e.g. results or /path/to/cluster/results",
    )
    alg_selector = mo.ui.multiselect(
        options=TARGET_ALGS,
        value=TARGET_ALGS,
        label="Target Algorithms",
    )
    smoothing_type = mo.ui.dropdown(
        options=["Exponential Moving Average", "Rolling Mean", "None"],
        value="Exponential Moving Average",
        label="Smoothing Method",
    )
    smoothing_slider = mo.ui.slider(
        start=2,
        stop=100,
        step=1,
        value=20,
        label="Smoothing Span / Window",
    )
    show_raw_checkbox = mo.ui.checkbox(
        value=True,
        label="Show unsmoothed raw curves in background (semi-transparent)",
    )
    include_e0_baseline_checkbox = mo.ui.checkbox(
        value=True,
        label="Show E0 baseline on all λ panels",
    )
    return (
        alg_selector,
        include_e0_baseline_checkbox,
        results_dir_input,
        show_raw_checkbox,
        smoothing_slider,
        smoothing_type,
    )


@app.cell
def __(
    alg_selector,
    include_e0_baseline_checkbox,
    mo,
    results_dir_input,
    show_raw_checkbox,
    smoothing_slider,
    smoothing_type,
):
    control_panel = mo.md(
        f"""
        # Cluster Runs Comparison Dashboard
        Compare `returned_episode_returns` across target algorithms for each value of $\\lambda$.

        {mo.vstack([
            mo.hstack([results_dir_input, alg_selector], justify="start", gap=2),
            mo.hstack([smoothing_type, smoothing_slider, show_raw_checkbox, include_e0_baseline_checkbox], justify="start", gap=2),
        ])}
        """
    )
    return (control_panel,)


@app.cell
def __(control_panel):
    control_panel
    return


@app.cell
def __(
    alg_selector,
    cloudpickle,
    glob,
    json,
    load_single_run,
    os,
    parse_lambda,
    re,
    results_dir_input,
):
    _base_dir = results_dir_input.value
    _selected_algs = set(alg_selector.value or [])
    runs_catalog = []

    if os.path.exists(_base_dir) and _selected_algs:
        _candidate_pkls = []
        for _alg in _selected_algs:
            # 4-level pattern: results/{alg}/lambda_{lambda}/{env}/out.pkl
            _candidate_pkls.extend(glob.glob(os.path.join(_base_dir, _alg, "lambda_*", "*", "out.pkl")))
            # 3-level pattern: results/{alg}/lambda_{lambda}/out.pkl
            _candidate_pkls.extend(glob.glob(os.path.join(_base_dir, _alg, "lambda_*", "out.pkl")))

            # Check directory 'E' if 'E0' is selected
            if _alg == "E0":
                _candidate_pkls.extend(glob.glob(os.path.join(_base_dir, "E", "lambda_*", "*", "out.pkl")))
                _candidate_pkls.extend(glob.glob(os.path.join(_base_dir, "E", "lambda_*", "out.pkl")))
                # Also allow results/E0/{env}/out.pkl if saved without lambda suffix
                for _e_pkl in glob.glob(os.path.join(_base_dir, "E0", "*", "out.pkl")) + glob.glob(os.path.join(_base_dir, "E", "*", "out.pkl")):
                    _parent_name = os.path.basename(os.path.dirname(_e_pkl))
                    if _parent_name not in ("sweeps", "tuning", "checkpoints"):
                        _candidate_pkls.append(_e_pkl)

        _pkl_files = sorted(list(set(_candidate_pkls)))

        for _pkl in _pkl_files:
            _rel_path = os.path.relpath(_pkl, _base_dir)
            _parts = _rel_path.split(os.sep)

            # Find directory part matching lambda_
            _lambda_idx = -1
            for _i, _p in enumerate(_parts):
                if re.search(r"lambda_([0-9.]+)", _p):
                    _lambda_idx = _i
                    break

            if _lambda_idx != -1:
                _suffix = _parts[_lambda_idx]
                _algo_name = _parts[_lambda_idx - 1] if _lambda_idx > 0 else _parts[0]
                _env_candidate = _parts[_lambda_idx + 1] if _lambda_idx + 1 < len(_parts) - 1 else "Unknown"
            elif len(_parts) >= 2:
                _algo_name = _parts[0]
                _suffix = "lambda_0" if _algo_name in ("E", "E0") else _parts[1]
                _env_candidate = _parts[1] if len(_parts) == 3 else (_parts[2] if len(_parts) >= 4 else "Unknown")
            else:
                _algo_name = "Unknown"
                _suffix = ""
                _env_candidate = "Unknown"

            # Normalize E -> E0
            if _algo_name == "E":
                _algo_name = "E0"

            # Extra guard: ensure algo is in selected algs
            if _algo_name not in _selected_algs:
                continue

            # Extra guard: ensure suffix starts with lambda_
            if not _suffix.startswith("lambda_"):
                continue

            _metrics, _cfg = load_single_run(_pkl, cloudpickle, json, os)
            if not _metrics:
                continue

            _env_name = _cfg.get("ENV_NAME", _env_candidate) if _cfg else _env_candidate
            if _algo_name == "E0":
                _lam = 0.0
            else:
                _lam = parse_lambda(_suffix, _cfg)

            _ret_series = _metrics.get("returned_episode_returns")
            if _ret_series is None:
                _ret_series = _metrics.get("returned_discounted_episode_returns")

            if _ret_series is not None:
                _arr = _ret_series
                # Squeeze out potential seed or dummy axes
                while getattr(_arr, "ndim", 0) > 1:
                    _arr = _arr.mean(axis=0)

                runs_catalog.append({
                    "algo": _algo_name,
                    "suffix": _suffix,
                    "env": _env_name,
                    "lambda": _lam if _lam is not None else -1.0,
                    "lambda_str": f"λ = {_lam:.2f}" if _lam is not None else "Unknown λ",
                    "returns": _arr,
                    "config": _cfg,
                    "path": _pkl,
                })

    available_envs = sorted(list(set(r["env"] for r in runs_catalog))) if runs_catalog else []
    available_lambdas = sorted(list(set(r["lambda"] for r in runs_catalog if r["lambda"] >= 0))) if runs_catalog else []

    return available_envs, available_lambdas, runs_catalog


@app.cell
def __(available_envs, mo, runs_catalog):
    if not runs_catalog:
        env_selector = mo.ui.dropdown(options=[], value=None, label="Select Environment")
        _status_msg = mo.md(
            "⚠️ **No runs with `out.pkl` found.** Check the results path above."
        )
    else:
        env_selector = mo.ui.dropdown(
            options=available_envs,
            value=available_envs[0] if available_envs else None,
            label="Select Environment",
        )
        _status_msg = mo.md(
            f"✅ **Loaded {len(runs_catalog)} run(s)** across **{len(available_envs)} environment(s)**."
        )

    env_control = mo.vstack([_status_msg, env_selector])
    return env_control, env_selector


@app.cell
def __(env_control):
    env_control
    return


@app.cell
def __(
    ALGO_STYLE,
    available_lambdas,
    env_selector,
    include_e0_baseline_checkbox,
    np,
    plt,
    runs_catalog,
    show_raw_checkbox,
    smooth_series,
    smoothing_slider,
    smoothing_type,
):
    fig_grid = None
    _selected_env = env_selector.value

    if _selected_env and available_lambdas:
        _n_lambdas = len(available_lambdas)
        _n_cols = min(3, _n_lambdas)
        _n_rows = (_n_lambdas + _n_cols - 1) // _n_cols

        fig_grid, _axes = plt.subplots(
            _n_rows, _n_cols, figsize=(7.0 * _n_cols, 4.5 * _n_rows), squeeze=False
        )
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        for _idx, _lam in enumerate(available_lambdas):
            _row, _col = _idx // _n_cols, _idx % _n_cols
            _ax = _axes[_row, _col]

            # Filter runs for this environment and this lambda
            _lam_runs = [
                r for r in runs_catalog
                if r["env"] == _selected_env and np.isclose(r["lambda"], _lam, atol=1e-3)
            ]

            # Also allow 1-step E baseline (lambda=0 or constant) to appear on all panels if enabled
            if include_e0_baseline_checkbox.value:
                _e0_runs = [
                    r for r in runs_catalog
                    if r["env"] == _selected_env and r["algo"] in ("E", "E0", "sampled_E")
                ]
            else:
                _e0_runs = []
            _seen_algos = set()
            _plotted_runs = []

            for _r in _lam_runs + _e0_runs:
                if _r["algo"] not in _seen_algos:
                    _seen_algos.add(_r["algo"])
                    _plotted_runs.append(_r)

            # Sort runs consistently by algorithm name
            _plotted_runs.sort(key=lambda x: x["algo"])

            for _r in _plotted_runs:
                _style = ALGO_STYLE.get(
                    _r["algo"], {"label": _r["algo"], "color": "#333333", "linestyle": "-"}
                )
                _raw_y = np.asarray(_r["returns"])
                _steps = np.arange(len(_raw_y))

                _smooth_y = smooth_series(
                    _raw_y, mode=smoothing_type.value, span_or_window=smoothing_slider.value
                )

                if show_raw_checkbox.value:
                    _ax.plot(_steps, _raw_y, color=_style["color"], alpha=0.22, linewidth=1.0)

                _label = _style["label"]
                _ax.plot(
                    _steps,
                    _smooth_y,
                    color=_style["color"],
                    linestyle=_style["linestyle"],
                    linewidth=2.4,
                    label=_label,
                )

            _ax.set_title(f"$\lambda = {_lam}$ (Comparison)", fontsize=13, fontweight="bold")
            _ax.set_xlabel("Update Step", fontsize=11)
            _ax.set_ylabel("Returned Episode Returns", fontsize=11)
            _ax.grid(True, linestyle="--", alpha=0.5)
            if _plotted_runs:
                _ax.legend(loc="best", fontsize=9, frameon=True, framealpha=0.9)

        # Hide any unused subplots in the grid
        for _idx in range(_n_lambdas, _n_rows * _n_cols):
            _row, _col = _idx // _n_cols, _idx % _n_cols
            _axes[_row, _col].axis("off")

        fig_grid.suptitle(
            f"Multi-Algorithm Comparison by λ ({_selected_env})",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        fig_grid.tight_layout()

    return (fig_grid,)


@app.cell
def __(fig_grid, mo):
    if fig_grid is not None:
        grid_view = mo.vstack([
            mo.md("## All $\\lambda$ Values: Facet Grid Comparison"),
            mo.as_html(fig_grid),
        ])
    else:
        grid_view = mo.md("_Select an environment above to render the comparison grid._")
    return (grid_view,)


@app.cell
def __(grid_view):
    grid_view
    return


@app.cell
def __(available_lambdas, mo):
    lambda_focus_selector = mo.ui.dropdown(
        options=[f"{l:.2f}" for l in available_lambdas],
        value=f"{available_lambdas[0]:.2f}" if available_lambdas else None,
        label="Select Specific λ to Inspect",
    )
    return (lambda_focus_selector,)


@app.cell
def __(
    ALGO_STYLE,
    env_selector,
    include_e0_baseline_checkbox,
    lambda_focus_selector,
    np,
    plt,
    runs_catalog,
    show_raw_checkbox,
    smooth_series,
    smoothing_slider,
    smoothing_type,
):
    fig_focus = None
    _selected_env = env_selector.value
    _selected_lam_str = lambda_focus_selector.value

    if _selected_env and _selected_lam_str is not None:
        _target_lam = float(_selected_lam_str)
        fig_focus, _ax = plt.subplots(figsize=(10.5, 5.5))

        _lam_runs = [
            r for r in runs_catalog
            if r["env"] == _selected_env and np.isclose(r["lambda"], _target_lam, atol=1e-3)
        ]
        if include_e0_baseline_checkbox.value:
            _e0_runs = [
                r for r in runs_catalog
                if r["env"] == _selected_env and r["algo"] in ("E", "E0", "sampled_E")
            ]
        else:
            _e0_runs = []
        _seen = set()
        _runs = []
        for _r in _lam_runs + _e0_runs:
            if _r["algo"] not in _seen:
                _seen.add(_r["algo"])
                _runs.append(_r)

        _runs.sort(key=lambda x: x["algo"])

        for _r in _runs:
            _style = ALGO_STYLE.get(
                _r["algo"], {"label": _r["algo"], "color": "#333333", "linestyle": "-"}
            )
            _raw_y = np.asarray(_r["returns"])
            _steps = np.arange(len(_raw_y))
            _smooth_y = smooth_series(
                _raw_y, mode=smoothing_type.value, span_or_window=smoothing_slider.value
            )

            if show_raw_checkbox.value:
                _ax.plot(_steps, _raw_y, color=_style["color"], alpha=0.22, linewidth=1.0)

            _final_val = _smooth_y[-1] if len(_smooth_y) > 0 else 0.0
            _ax.plot(
                _steps,
                _smooth_y,
                color=_style["color"],
                linestyle=_style["linestyle"],
                linewidth=2.6,
                label=f"{_style['label']} (Final: {_final_val:.1f})",
            )

        _ax.set_title(
            f"Focused Comparison: λ = {_target_lam:.2f} ({_selected_env})",
            fontsize=14,
            fontweight="bold",
        )
        _ax.set_xlabel("Update Step", fontsize=12)
        _ax.set_ylabel("Returned Episode Returns", fontsize=12)
        _ax.grid(True, linestyle="--", alpha=0.5)
        _ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)
        fig_focus.tight_layout()

    return (fig_focus,)


@app.cell
def __(fig_focus, lambda_focus_selector, mo):
    if fig_focus is not None:
        focus_view = mo.vstack([
            mo.md("## Detailed Single-$\\lambda$ View"),
            lambda_focus_selector,
            mo.as_html(fig_focus),
        ])
    else:
        focus_view = mo.md("")
    return (focus_view,)


@app.cell
def __(focus_view):
    focus_view
    return


@app.cell
def __(
    ALGO_STYLE,
    available_lambdas,
    env_selector,
    np,
    pd,
    runs_catalog,
    smooth_series,
    smoothing_slider,
    smoothing_type,
):
    summary_dataframe = pd.DataFrame()
    _selected_env = env_selector.value

    if _selected_env and runs_catalog:
        _rows = []
        for _r in runs_catalog:
            if _r["env"] != _selected_env:
                continue

            _raw_y = np.asarray(_r["returns"])
            if len(_raw_y) == 0:
                continue

            _smooth_y = smooth_series(
                _raw_y, mode=smoothing_type.value, span_or_window=smoothing_slider.value
            )
            _style = ALGO_STYLE.get(_r["algo"], {"label": _r["algo"]})

            _w = min(20, len(_raw_y))
            _final_window_mean = float(_raw_y[-_w:].mean())
            _max_return = float(_raw_y.max())
            _auc = float(_raw_y.mean())

            _rows.append({
                "Algorithm": _style["label"],
                "Internal Algo": _r["algo"],
                "Lambda (λ)": _r["lambda"],
                "Suffix": _r["suffix"],
                "Final Window Return (Last 20)": round(_final_window_mean, 2),
                "Peak Return": round(_max_return, 2),
                "Mean Return (AUC)": round(_auc, 2),
                "Total Steps": len(_raw_y),
            })

        if _rows:
            summary_dataframe = pd.DataFrame(_rows)
            summary_dataframe.sort_values(
                by=["Lambda (λ)", "Final Window Return (Last 20)"],
                ascending=[True, False],
                inplace=True,
            )

    return (summary_dataframe,)


@app.cell
def __(mo, summary_dataframe):
    if not summary_dataframe.empty:
        summary_view = mo.vstack([
            mo.md("## Summary Metrics Table"),
            mo.ui.table(summary_dataframe),
        ])
    else:
        summary_view = mo.md("")
    return (summary_view,)


@app.cell
def __(summary_view):
    summary_view
    return


if __name__ == "__main__":
    app.run()

# Walkthrough: Modular Hyperparameter Sweep & Comparison Pipeline

We have implemented an end-to-end, modular, and interpretable hyperparameter sweep and cross-algorithm comparison pipeline for reinforcement learning / value error experiments.

---

## 🚀 Key Features Implemented

### 1. Enhanced Single-Algorithm Sweep Engine ([`core/sweep.py`](file:///Users/dillonsandhu/Documents/Research/bellman_error/core/sweep.py))
- **Comprehensive Statistics Across Seeds**: Calculates final mean, final std, min value along trajectory, and mean value along trajectory.
- **Explicit Winning Configuration**: Automatically identifies the best configuration and saves [`best_config.json`](file:///Users/dillonsandhu/Documents/Research/bellman_error/core/sweep.py#L182-L200) containing the full merged configuration, winning hyperparameters, rank #1 metadata, and metric values.
- **Detailed Summary Tables**: Saves both `tuning_summary.csv` and `tuning_summary.json` with ranking and stats.
- **Publication-Quality Visualizations**:
  - `hyperparameter_sweep_{metric_key}.png`: Visualizes all hyperparameter configurations averaged over seeds, labeled with parameter values and final performance, with the best configuration highlighted (`★ [BEST]`).
  - `best_config_seeds.png`: Visualizes all individual random seed learning curves for the winning configuration.
  - `all_configs_seeds_grid.png`: Small-multiples subplot grid showing seed trajectories for every configuration in the grid.

### 2. Modular Multi-Algorithm Sweep Pipeline ([`scripts/sweep_pipeline.py`](file:///Users/dillonsandhu/Documents/Research/bellman_error/scripts/sweep_pipeline.py))
- Evaluates a fixed task across your core suite: `exact_td`, `exact_mc`, `exact_E_gd`, `exact_td_lambda`, etc.
- Works across all policy regimes: `--policy fixed`, `--policy random`, or `--policy ppo`.
- Automatically tailors parameter grids (e.g. 2D grid over $\alpha$ and $\lambda$ for TD($\lambda$)).
- Automatically resolves valid pretrained policy checkpoints for fixed-policy evaluation.
- Generates a unified `comparison/` directory containing:
  - `comparison_best_configs.png`: Plots the best configuration of every algorithm on the same graph with mean curves and error bands (arithmetic ±1 std or geometric mean).
  - `comparison_summary.csv` & `comparison_summary.json`: Unified ranking table of all algorithms.

### 3. Modular Sweep Analysis Library ([`notebooks/analyze_sweeps.py`](file:///Users/dillonsandhu/Documents/Research/bellman_error/notebooks/analyze_sweeps.py))
- `discover_algorithm_sweeps(policy, env_name)`: Finds the latest sweep runs across batch sweeps and standalone tuning directories.
- `load_sweep_data(path)`: Loads configuration, metrics, summary DataFrame, and `best_config.json`.
- `extract_best_configuration(sweep_data)`: Extracts winning hyperparameters and 2D seed trajectories `(n_seeds, time_steps)`.
- `plot_algorithm_comparison(...)`: Generates cross-algorithm comparison graphs.
- `summarize_algorithm_comparison(...)`: Compares and ranks all algorithms.

### 4. Standalone Evaluation Script ([`notebooks/evaluate_sweeps.py`](file:///Users/dillonsandhu/Documents/Research/bellman_error/notebooks/evaluate_sweeps.py))
- Quick CLI / module that discovers the latest runs, generates seed plots for each algorithm's best configuration, and plots the cross-algorithm comparison.

---

## 📁 Output Folder Structure

When you run a sweep via `scripts/sweep_pipeline.py` or SLURM scripts (`scripts/run_slurm_sweep_exact.sh`), the outputs are organized in an interpretable hierarchy:

```
results/fixed/sweeps/fixed_FourRooms-misc_<timestamp>/
├── pipeline_config.json                 # Run specification
├── exact_td/
│   └── tuning/<timestamp>/FourRooms-misc/
│       ├── config.json                  # Base config
│       ├── best_config.json             # Winning configuration & metadata
│       ├── tuning_summary.csv           # Sorted hyperparameter ranking
│       ├── tuning_summary.json
│       ├── out.pkl                      # Raw metrics tensor across combos & seeds
│       ├── hyperparameter_sweep_nn_weighted_VE.png
│       ├── best_config_seeds.png        # Individual seed curves for #1 config
│       └── all_configs_seeds_grid.png   # Subplot grid of all configs
├── exact_mc/
│   └── tuning/...
├── exact_E_gd/
│   └── tuning/...
├── exact_td_lambda/
│   └── tuning/...
└── comparison/
    ├── comparison_summary.csv           # Multi-algorithm performance comparison
    ├── comparison_summary.json
    └── comparison_best_configs.png      # All best algorithms plotted together
```

---

## 💡 Usage Examples

### Running the Multi-Algorithm Sweep Pipeline

```bash
# Sweep all 4 main algorithms on FourRooms-misc (fixed policy evaluation, 5 seeds, 1000 timesteps)
python scripts/sweep_pipeline.py --policy fixed --env-name FourRooms-misc --n-seeds 5 --total-timesteps 1000

# Sweep specific algorithms with custom learning rates
python scripts/sweep_pipeline.py --policy fixed --algos exact_td exact_mc --lr-grid 0.05 0.01 0.005 0.001 0.0005

# Sweep random policy algorithms
python scripts/sweep_pipeline.py --policy random --env-name FourRooms-misc --n-seeds 5 --total-timesteps 2000

# Or submit via SLURM:
sbatch scripts/run_slurm_sweep_exact.sh
```

### Analyzing & Plotting Existing Runs

```bash
# Analyze and plot latest fixed policy sweeps
python notebooks/evaluate_sweeps.py --policy fixed --env-name FourRooms-misc

# Analyze using geometric mean error bands
python notebooks/analyze_sweeps.py --policy fixed --env-name FourRooms-misc --use-geom-mean
```

### Python / Notebook API

```python
from notebooks.analyze_sweeps import discover_algorithm_sweeps, plot_algorithm_comparison, summarize_algorithm_comparison

# 1. Discover latest sweeps
runs = discover_algorithm_sweeps(policy="fixed", env_name="FourRooms-misc")

# 2. View summary table
summary_df = summarize_algorithm_comparison(runs, metric_key="nn_weighted_VE")
print(summary_df)

# 3. Plot comparison
fig = plot_algorithm_comparison(runs, metric_key="nn_weighted_VE", save_path="comparison.png")
```

---

## ✅ Verification Results

We verified the pipeline by running a full multi-algorithm sweep across `exact_td`, `exact_mc`, `exact_E_gd`, and `exact_td_lambda` on `FourRooms-misc`:
- All 4 algorithms completed parallel vmapped sweeps over their parameter grids and seeds.
- `best_config.json`, `tuning_summary.csv`, `hyperparameter_sweep_nn_weighted_VE.png`, and `best_config_seeds.png` were generated for each algorithm.
- Cross-algorithm comparison summary and `comparison_best_configs.png` were generated in `results/fixed/sweeps/.../comparison/`.
- `over_engineered_plot_tune_script.py` executed successfully, automatically discovering runs and rendering comparisons.

# Environmental Properties Sweep (`env_properties_sweep`)

This sweep investigates the comparative stability, sample efficiency, and robustness of:
- **$E(0)$**: Symmetrized Dirichlet error minimization ([`algos/E.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/E.py))
- **$TD(0)$**: Classic 1-step TD learning with fitted value iteration updates ([`algos/ppo.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/ppo.py) with `VALUE_LAMBDA=0.0`)
- **$TD(\lambda)$**: Standard PPO multi-step baseline with $\lambda = 0.95$ ([`algos/ppo.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/ppo.py))

All runs hold **`NUM_EPOCHS = 16`** constant (along with learning rates and rollout sizes) to test how value learning dynamics respond when critic iterations are scaled up across environments varying in reward structure and control stochasticity.

---

## 1. Environments Evaluated

### Family 1: Mountain Car
1. **Sparse Continuous**: `MountainCarContinuous-v0`
2. **Dense Continuous**: `MountainCarDenseContinuous-v0` (Potential-Based Reward Shaping)
3. **Dense Discrete**: `MountainCarDenseDiscrete-v0` (3-action discrete bang-bang control)

### Family 2: Point Robot (Fully Observable MDP)
1. **Sparse Continuous**: `PointRobot-misc` (`DENSE_REWARD=False`)
2. **Dense Continuous**: `PointRobot-misc` (`DENSE_REWARD=True`)
3. **Dense Discrete**: `PointRobotDiscrete-misc` (`DENSE_REWARD=True`, 5 cardinal actions)

---

## 2. Experimental Conditions

Each environment is evaluated in two regimes:
1. **Clean**: Deterministic transition physics (`SLIP_PROB=0.0`, `TRANSITION_NOISE=0.0`).
2. **Noisy**:
   - `SLIP_PROB`: $0.05$ (5% chance of wheel/traction slip on each timestep)
   - `SLIP_FORCE_SCALE`: $0.5$ (delivers 50% commanded drive force / displacement during slip)
   - `TRANSITION_NOISE`: $0.001$ (additive Gaussian terrain jitter)

---

## 3. Visualizations

Each environment family produces a self-contained publication figure ($2 \times 3$ grid):
- **Row 1 (Clean)**: Sparse Continuous | Dense Continuous | Dense Discrete
- **Row 2 (Noisy)**: Sparse Continuous | Dense Continuous | Dense Discrete
- **Curves per subplot**: 3 curves ($E(0)$, $TD(0)$, $TD(\lambda)$) with shaded Mean $\pm$ 1 SEM over 8 independent seeds.

Outputs generated:
- `results/sweeps/<sweep_id>/mountain_car/mountain_car_env_properties.pdf` (and `.png`)
- `results/sweeps/<sweep_id>/point_robot/point_robot_env_properties.pdf` (and `.png`)
- `summary_<family>.csv` and `metrics_<family>.pkl`

---

## 4. Usage

### Local Smoke Test
```bash
python scripts/env_properties_sweep/sweep_env_properties.py \
    --env-family mountain_car \
    --total-timesteps 64 \
    --num-envs 4 \
    --num-steps 16 \
    --minibatch-size 64 \
    --n-seeds 2
```

### Full SLURM Cluster Execution
```bash
sbatch scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh
```
Or run individual tasks:
```bash
./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 0   # MountainCar
./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 1   # PointRobot
```

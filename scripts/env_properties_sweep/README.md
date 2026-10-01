# Environmental Properties Sweep (`env_properties_sweep`)

This sweep investigates the comparative stability, sample efficiency, and robustness of:
- **$E(0)$**: Symmetrized Dirichlet error minimization ([`algos/E.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/E.py))
- **$TD(0)$**: Classic 1-step TD learning with fitted value iteration updates ([`algos/ppo.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/ppo.py) with `VALUE_LAMBDA=0.0`)
- **$TD(\lambda)$**: Standard PPO multi-step baseline with $\lambda = 0.95$ ([`algos/ppo.py`](file:///Users/dillonsandhu/Documents/Research/E_min/algos/ppo.py))

All runs hold **`NUM_EPOCHS = 16`** constant (along with learning rates and rollout sizes) to test how value learning dynamics respond when critic iterations are scaled up across environments varying in reward structure and control stochasticity.

---

## 1. Environments Evaluated (2x2 Factorial Design)

Each environment family is evaluated across a complete $2 \times 2$ factorial matrix of **Reward Density** (Sparse vs. Dense) $\times$ **Action Space** (Continuous vs. Discrete):

### Family 1: Mountain Car
1. **Sparse Continuous**: `MountainCarDenseContinuous-v0` (`DENSE_REWARD=False`)
2. **Sparse Discrete**: `MountainCarDenseDiscrete-v0` (`DENSE_REWARD=False`, 3-action discrete bang-bang)
3. **Dense Continuous**: `MountainCarDenseContinuous-v0` (`DENSE_REWARD=True`, Potential-Based Reward Shaping)
4. **Dense Discrete**: `MountainCarDenseDiscrete-v0` (`DENSE_REWARD=True`, 3-action discrete bang-bang)

### Family 2: Point Robot (Fully Observable MDP)
1. **Sparse Continuous**: `PointRobot-misc` (`DENSE_REWARD=False`)
2. **Sparse Discrete**: `PointRobotDiscrete-misc` (`DENSE_REWARD=False`, 5 cardinal actions)
3. **Dense Continuous**: `PointRobot-misc` (`DENSE_REWARD=True`)
4. **Dense Discrete**: `PointRobotDiscrete-misc` (`DENSE_REWARD=True`, 5 cardinal actions)

---

## 2. Experimental Note: Dense Reward Failure in Point Robot

> [!WARNING]
> **Why Dense Reward Failed in Point Robot:**
> In `PointRobot`, the classical dense reward formulation is defined as $r_t = -\|\text{pos}_t - \text{goal}\|_2$ with zero positive goal bonus. Because reaching the target triggers a random re-spawn/teleportation far away from the goal ($r_{t+1} \approx -1.3$ to $-1.8$), entering the goal circle is severely penalized compared to simply hovering right outside the goal perimeter (where $r_t \approx -0.21$ continuously).
> 
> Without an episodic terminal state or a positive goal bonus, pure negative-distance reward shaping creates a perverse incentive to avoid the goal zone. In contrast, sparse reward provides $+1.0$ strictly upon goal arrival, successfully incentivizing the agent to visit the goal as many times as possible.

---

## 3. Experimental Conditions

Each environment is evaluated in two regimes:
1. **Clean**: Deterministic transition physics (`SLIP_PROB=0.0`, `TRANSITION_NOISE=0.0`).
2. **Noisy**:
   - `SLIP_PROB`: $0.05$ (5% chance of wheel/traction slip on each timestep)
   - `SLIP_FORCE_SCALE`: $0.5$ (delivers 50% commanded drive force / displacement during slip)
   - `TRANSITION_NOISE`: $0.001$ (additive Gaussian terrain jitter)

---

## 4. Visualizations

Each environment family produces a self-contained publication figure ($2 \times 4$ grid):
- **Row 1 (Clean)**: Sparse Cont. | Sparse Disc. | Dense Cont. | Dense Disc.
- **Row 2 (Noisy)**: Sparse Cont. | Sparse Disc. | Dense Cont. | Dense Disc.
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

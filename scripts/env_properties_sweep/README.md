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
3. **Dense Continuous**: `PointRobot-misc` (`DENSE_REWARD=True`, Potential-Based Reward Shaping)
4. **Dense Discrete**: `PointRobotDiscrete-misc` (`DENSE_REWARD=True`, Potential-Based Reward Shaping)

### Family 3: Space Invaders (MinAtar)
Tests credit assignment delay and horizon variance (all with **no noise** / 0% sticky actions):
1. **Standard SpaceInvaders**: Standard lethal bullet hits (episodic sudden death), projectile bullets with flight delay
2. **Fixed-Horizon SpaceInvaders**: Fixed 1,000 steps, -1 penalty per enemy bullet or alien hit rather than termination
3. **Instant-Bullets SpaceInvaders**: Standard lethal hits, hitscan bullets with 0 flight delay

---

## 2. Experimental Note: Dense Reward in Point Robot

> [!NOTE]
> **Potential-Based Reward Shaping (PBRS) in Point Robot:**
> Point Robot uses true Potential-Based Reward Shaping $\Phi(s) = -\|\text{pos} - \text{goal}\|_2$ (referenced to $0.0$ when goal is reached) with shaping reward $F(s, s') = \gamma \Phi(s') - \Phi(s)$. Approaching the goal yields $+0.1$/step progress rewards, reaching the goal awards the $+1.0$ base reward, and camping stationary earns zero progress reward ($s' = s \implies \Delta \Phi = 0$).

---

## 3. Experimental Conditions

1. **Mountain Car & Point Robot**:
   - **Clean**: Deterministic transition physics (`SLIP_PROB=0.0`, `TRANSITION_NOISE=0.0`).
   - **Noisy**: 5% chance of wheel/traction slip (`SLIP_PROB=0.05`, `SLIP_FORCE_SCALE=0.5`, `TRANSITION_NOISE=0.001`).
2. **Space Invaders**:
   - **No Noise**: 0% sticky actions (`STICKY_ACTION_PROB=0.0`).

---

## 4. Visualizations

Each environment family produces a self-contained publication figure:
- **Mountain Car & Point Robot** ($2 \times 4$ grid):
  - Row 0 (Clean): Sparse Cont. | Sparse Disc. | Dense Cont. | Dense Disc.
  - Row 1 (Noisy): Sparse Cont. | Sparse Disc. | Dense Cont. | Dense Disc.
- **Space Invaders** ($2 \times 3$ grid, No Noise):
  - Row 0 (Episode Return): Standard | Fixed-Horizon (-1/Hit) | Instant Bullets (Hitscan)
  - Row 1 (Episode Length / Survival Steps): Standard | Fixed-Horizon (-1/Hit) | Instant Bullets (Hitscan)

Outputs generated:
- `results/sweeps/<sweep_id>/mountain_car/mountain_car_env_properties.pdf` (and `.png`)
- `results/sweeps/<sweep_id>/point_robot/point_robot_env_properties.pdf` (and `.png`)
- `results/sweeps/<sweep_id>/space_invaders/space_invaders_env_properties.pdf` (and `.png`)
- `summary_<family>.csv` and `metrics_<family>.pkl`

---

## 5. Usage

### Local Smoke Test
```bash
python scripts/env_properties_sweep/sweep_env_properties.py \
    --env-family space_invaders \
    --total-timesteps 256 \
    --num-envs 4 \
    --num-steps 16 \
    --minibatch-size 64 \
    --num-epochs 1 \
    --n-seeds 1
```

### Full SLURM Cluster Execution
```bash
sbatch scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh
```
Or run individual tasks:
```bash
./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 0   # MountainCar
./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 1   # PointRobot
./scripts/env_properties_sweep/run_slurm_env_properties_sweep.sh 2   # SpaceInvaders
```

# Sampled $E(\lambda)$: Complete Code Architecture, Algorithmic Derivation, and Implementation Reference

This document serves as the complete, definitive technical reference for **Sampled $E(\lambda)$** (Bellman-Smoothed Value Regression with geometric jump sampling and auxiliary debiasing). It contains all relevant production code from the codebase alongside step-by-step mathematical and algorithmic explanations ("blurbs") detailing **what each code block does**, **why it is necessary**, and **what catastrophic failures occur if it is omitted**.

---

## Table of Contents
1. [Core Mathematical Framework & Theoretical Objectives](#1-core-mathematical-framework--theoretical-objectives)
2. [Target Computation & GAE Trajectory Processing (`calculate_gae`)](#2-target-computation--gae-trajectory-processing-calculate_gae)
3. [Error Construction & Step-$T$ Continuation Bootstrapping](#3-error-construction--step-t-continuation-bootstrapping)
4. [Geometric Jump Horizon Sampling ($m = K + 1$)](#4-geometric-jump-horizon-sampling-m--k--1)
5. [Lookahead Runway Windowing & Boundary Bootstrap Clamping](#5-lookahead-runway-windowing--boundary-bootstrap-clamping)
6. [Episode Boundaries: True Absorbing Terminals vs. Artificial Timeouts](#6-episode-boundaries-true-absorbing-terminals-vs-artificial-timeouts)
7. [Differentiable Stop-Gradient Surrogate with Auxiliary Critic $v_{\text{aux}}$](#7-differentiable-stop-gradient-surrogate-with-auxiliary-critic-v_textaux)
8. [Geometrically Decaying Start-State Reset Grounding ($\mathcal{L}_{\text{start}}$)](#8-geometrically-decaying-start-state-reset-grounding-mathcall_textstart)
9. [Exact Theoretical Prefactors and Loss Assembly](#9-exact-theoretical-prefactors-and-loss-assembly)
10. [Temporal State Threading & Minibatch Environment Batching](#10-temporal-state-threading--minibatch-environment-batching)
11. [Full Reference Implementation: Critic Loss (`core/helpers.py`)](#11-full-reference-implementation-critic-loss-corehelperspy)
12. [Full Reference Implementation: PPO Training Step (`ppo/E_lambda_geometric.py`)](#12-full-reference-implementation-ppo-training-step-ppoe_lambda_geometricpy)

---

## 1. Core Mathematical Framework & Theoretical Objectives

### The Objective
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ be a Markov reward process with stationary distribution $\mu$, diagonal weighting $D = \text{diag}(\mu)$, true value function $V$, parametric critic $v_\theta$, and error $e \doteq V - v_\theta$. 

Standard $\text{TD}(\lambda)$ follows the expected semi-gradient flow $\dot{\theta} = J_\theta^\top A_\lambda e$, where $A_\lambda \doteq D(I - \gamma P)(I - \gamma\lambda P)^{-1}$. Because $A_\lambda$ is non-symmetric, $\text{TD}(\lambda)$ is prone to divergence under function approximation.

We define the **Bellman-Smoothed Value Error $E(\lambda)$** as the quadratic form induced by the symmetric part $S_\lambda \doteq \frac{1}{2}(A_\lambda + A_\lambda^\top)$:
$$E(\lambda) \doteq \frac{1}{2} e^\top A_\lambda e = \frac{1}{2} e^\top D(I - \gamma P)(I - \gamma\lambda P)^{-1} e = \frac{1}{2} e^\top S_\lambda e$$

Using the compound resolvent operator $P_\lambda \doteq (1 - \gamma\lambda) P (I - \gamma\lambda P)^{-1} = (1 - \gamma\lambda) \sum_{k=0}^\infty (\gamma\lambda)^k P^{k+1}$ and effective discount $\tilde{\gamma} \doteq \frac{\gamma(1 - \lambda)}{1 - \gamma\lambda}$, this objective decomposes on stationary chains into:
$$E(\lambda) = \underbrace{\frac{1}{2}(1 - \tilde{\gamma}) \|e\|_D^2}_{\text{Magnitude Anchor}} + \underbrace{\frac{1}{2}\tilde{\gamma} \mathcal{E}_\lambda(e)}_{\text{Dirichlet Smoothness Energy}}$$
where:
$$\mathcal{E}_\lambda(e) \doteq \frac{1}{2} \sum_{i, j} \mu_i [P_\lambda]_{ij} (e_i - e_j)^2 = \frac{1}{2} \mathbb{E}_{\substack{s_i \sim \mu \\ K \sim \text{Geom}(1 - \gamma\lambda) \\ s_j \sim P^{K+1}(\cdot \mid s_i)}} \Big[ (e(s_i) - e(s_j))^2 \Big]$$

---

## 2. Target Computation & GAE Trajectory Processing (`calculate_gae`)

### The Code
```python
def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping Mask
        # If the episode ended due to a true absorbing termination, V(s_term) = 0.
        # If it ended due to an artificial timeout, continuation value must be bootstrapped.
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation Mask (Trajectory Boundary)
        # Sever the GAE recursive chain if the environment reset for ANY reason
        # (terminal OR timeout), since transition[t+1] belongs to an unrelated episode.
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = transition.reward + γ * transition.next_value * bootstrap_mask - transition.value
        
        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)
        
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    # Returns are target G_t = advantages + baseline value
    return (advantages, advantages + traj_batch.value)
```

### The Blurb: What it does & Why it is necessary
1. **Dual-Masking Architecture**:
   - `bootstrap_mask = 1.0 - (done & ~is_timeout)`: When an agent reaches a true goal or terminal failure, the absorbing state has true value zero ($V(s_{\text{term}}) = 0$). Bootstrapping from `next_value` would corrupt the target with arbitrary network predictions. Conversely, on an artificial timeout (`is_timeout = True`), the agent was arbitrarily stopped by a time limit; its true expected future return is non-zero, so we **must bootstrap** from `transition.next_value`.
   - `boundary_mask = 1.0 - done`: Regardless of whether the reset was caused by a terminal or a timeout, the simulator immediately places the agent into a newly reset state at step $t+1$. The recursive advantage accumulator `gae` flowing backward from $t+1$ belongs to the *next episode*. Accumulating across `done == True` would pollute the returns of Episode 1 with the rewards of Episode 2.
2. **Empirical Return Targets $G_t$**:
   - Setting `λ = config["RETURN_LAMBDA"] = 1.0` produces the exact Monte Carlo discounted return targets:
     $$G_t = \sum_{l=0}^{H-1} \gamma^l r_{t+l} + \gamma^H v(s_{t+H})$$
     These targets $G_t$ serve as the unbiased estimates of $V(s_t)$ for error construction and auxiliary critic training.

---

## 3. Error Construction & Step-$T$ Continuation Bootstrapping

### The Code
```python
    # 1. Construct rollout errors (for Term 2 & Magnitude) and aux errors (for Term 1)
    errors = sg(targets) - values           # \hat{e}_t = sg(G_t) - v_\theta(s_t), shape (T, B)
    e_aux = sg(aux_values) - values         # e_{aux, t} = sg(v_{aux}(s_t)) - v_\theta(s_t), shape (T, B)

    # Step T boundary extension (absorbing 0 on terminal, bootstrapped continuation otherwise)
    e_T = jnp.where(true_terminals[-1], 0.0, sg(next_target_T) - next_value_T)
    e_aux_T = jnp.where(true_terminals[-1], 0.0, sg(next_aux_value_T) - next_value_T)

    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)       # shape (T + 1, B)
    e_aux_ext = jnp.concatenate([e_aux, e_aux_T[None]], axis=0)     # shape (T + 1, B)
```

### The Blurb: What it does & Why it is necessary
1. **Definition of Error Vectors**:
   - `errors`: Empirical error $\hat{e}_t = G_t - v_\theta(s_t)$. Notice `sg(targets)`: the target returns $G_t$ act as fixed supervisory labels and must not backpropagate gradients into past time steps.
   - `e_aux`: Auxiliary error $e_{\text{aux}, t} = v_{\text{aux}}(s_t) - v_\theta(s_t)$. Notice `sg(aux_values)`: the auxiliary network parameters $\phi$ are held constant while evaluating the critic loss w.r.t. $\theta$, ensuring zero parameter gradient leakage.
2. **Boundary Extension at Step $T$**:
   - In a finite buffer $[0, T-1]$, geometric jumps originating near the end of the buffer (e.g. $t = T - 2$) can land at or beyond step $T$.
   - Rather than dropping these transitions, we append step $T$ continuation errors `e_T` and `e_aux_T` computed from `next_obs[-1]`.
   - If step $T-1$ was a `true_terminal`, the continuation state is absorbing, so `e_T = 0.0`. If it was not terminal, we bootstrap using `next_target_T - next_value_T`.
   - Concatenating creates arrays of length $T + 1$, allowing safe indexing up to index $T$ without out-of-bounds index errors.

---

## 4. Geometric Jump Horizon Sampling ($m = K + 1$)

### The Code
```python
    # 2. Geometric jump sampling: m = K + 1 where K ~ Geom(1 - gl)
    # Target distribution: P(K = k) = (1 - gl) * (gl)^k for k in {0, 1, 2, ...}
    u = jax.random.uniform(rng, shape=(T_start, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T_start, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1  # Total lookahead steps: m in {1, 2, 3, ...}
```

### The Blurb: What it does & Why it is necessary
1. **Simulating the Compound Resolvent $P_\lambda$**:
   - The matrix operator $P_\lambda = (1 - \gamma\lambda)\sum_{k=0}^\infty (\gamma\lambda)^k P^{k+1}$ is an infinite geometric mixture of transition powers $P^{k+1}$.
   - We cannot compute infinite matrix sums in deep RL. However, by the law of total expectation, sampling a random jump length $m = K + 1$ with $K \sim \text{Geometric}(1 - \gamma\lambda)$ and advancing $m$ steps along the trajectory evaluates $P_\lambda$ **unbiasedly in a single sample**.
2. **Numerically Stable Inverse-CDF Sampling**:
   - The CDF of $K \sim \text{Geom}(1 - p)$ on $\{0, 1, 2, \dots\}$ is $F(k) = 1 - p^{k+1}$.
   - Inverting the CDF for uniform $u \sim \text{Uniform}(0, 1)$ gives:
     $$k = \left\lfloor \frac{\ln(1 - u)}{\ln(\gamma\lambda)} \right\rfloor$$
   - `safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)` prevents division by zero or $\ln(1.0) = 0$.
   - When $\lambda = 0$ ($gl \to 0$), the branch `gl < 1e-6` deterministically forces `jumps = 0 \implies m = 1$, identically recovering the 1-step lookahead $t+1$ of $E(0)$.

---

## 5. Lookahead Runway Windowing & Boundary Bootstrap Clamping

### The Code
```python
    T, B = targets.shape[:2]
    # Runway: Only sample source states from t in [0, T_start)
    T_start = min(T, max(1, T - 16)) if num_start_states is None else min(num_start_states, T)

    t_arr = jnp.arange(T_start)[:, None]
    # Destination index: Clamped to the buffer boundary T
    j = jnp.minimum(t_arr + m, T)
```

### The Blurb: What it does & Why it is necessary
1. **The Failure of Naive Truncation**:
   - If one attempts to sample jumps from all $t \in [0, T-1]$, jumps near the end of the buffer ($t \approx T-1$) will land past $T$.
   - If truncated jumps are masked out to $0$, late states are only ever updated when $K = 0$, skewing the effective $\lambda \to 0$.
   - If surviving jumps are reweighted by an importance weight $w_{\text{trunc}}(t) = \frac{1}{1 - (\gamma\lambda)^{T-t}}$, the weight for $t = T-1$ reaches $\frac{1}{1 - \gamma\lambda}$ (up to $20\times$), introducing catastrophic gradient variance spikes.
2. **The Runway + Clamping Solution**:
   - **Prefix Runway Window ($T_{\text{start}} = T - 16$)**: By restricting source states to $t < T - 16$, the remaining 16 steps serve as a buffer runway. For $\gamma\lambda = 0.8$, $97.2\%$ of jumps land naturally inside the buffer without hitting $T$.
   - **Boundary Clamping ($j = \min(t + m, T)$)**: Any jump that does reach or exceed step $T$ is smoothly clamped to index $T$ and bootstrapped using the continuation error $e_T$. This eliminates all importance-sampling weights while retaining full gradient signals.

---

## 6. Episode Boundaries: True Absorbing Terminals vs. Artificial Timeouts

### The Code
```python
    # 3. Detect true absorbing terminals vs. artificial timeouts in [t, j - 1]
    is_timeout = dones & (~true_terminals)
    
    # Compute exclusive cumulative sums to query intervals [t, j - 1] in O(1) time
    term_cumsum_excl = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    timeout_cumsum_excl = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)],
        axis=0,
    )

    # A terminal/timeout was crossed if cumsum[j] - cumsum[t] > 0
    hit_true_terminal = (jnp.take_along_axis(term_cumsum_excl, j, axis=0) - term_cumsum_excl[:T_start]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum_excl, j, axis=0) - timeout_cumsum_excl[:T_start]) > 0

    e_start = errors[:T_start]
    e_aux_start = e_aux[:T_start]
    e_jump = jnp.take_along_axis(errors_ext, j, axis=0)
    e_aux_jump = jnp.take_along_axis(e_aux_ext, j, axis=0)

    # Per-transition Dirichlet energy
    diff_sq = jnp.where(
        hit_true_terminal,
        e_start ** 2,  # Absorbs at e_term = 0; backward branch at s_term vanishes
        jnp.where(
            hit_timeout,
            0.0,       # Artificial timeout: discard ungrounded cross-episode transition
            (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2,
        ),
    )
```

### The Blurb: What it does & Why it is necessary
1. **$O(1)$ Interval Event Detection**:
   - We must know if the trajectory encountered an episode boundary between $t$ and destination $j-1$.
   - Slicing and scanning variable-length windows $[t, j-1]$ per transition would break JAX vectorized compilation.
   - Instead, we construct an exclusive cumulative sum `term_cumsum_excl` of length $T+1$ with a prepended zero. The number of terminals in $[t, j-1]$ is computed in $O(1)$ parallel time via:
     $$\Delta \text{term}(t, j) = \text{term\_cumsum}[j] - \text{term\_cumsum}[t]$$
2. **True Absorbing Terminal Condition ($\mathcal{D} = \hat{e}_t^2$)**:
   - In an MDP, when an agent reaches a goal or dies at $t_{\text{term}} < j$, the state becomes absorbing and stays in $s_{\text{term}}$ through step $j$.
   - At $s_{\text{term}}$, true and estimated values are zero: $V(s_{\text{term}}) = v_\theta(s_{\text{term}}) = 0 \implies e_j = 0$ and $\nabla_\theta v_\theta(s_{\text{term}}) = \mathbf{0}$.
   - Differentiating the energy:
     - **Backward branch (Term 1) vanishes**: $(e_{\text{aux}, t} - 0) \nabla_\theta v_\theta(s_{\text{term}}) = \mathbf{0}$.
     - **Forward branch (Term 2) evaluates to**: $(\hat{e}_t - 0) \nabla_\theta v_\theta(s_t) = \hat{e}_t \nabla_\theta v_\theta(s_t)$.
   - The derivative of $\frac{1}{2}\hat{e}_t^2$ w.r.t. $\theta$ is $-\hat{e}_t \nabla_\theta v_\theta(s_t)$, which matches Term 2 identically!
   - Thus, **we do not mask out terminal transitions**; we set $\mathcal{D}(t, b) = \hat{e}_t^2$.
3. **Artificial Timeout Masking ($\mathcal{D} = 0.0$)**:
   - When an episode is truncated by an artificial time limit, the state at $t+1$ belongs to a new episode, but the agent was not in an absorbing zero-value state ($V(s) \neq 0$). Pairing $s_t$ with $s_j$ across a timeout would pair unrelated states from different episodes. Hence, timeouts are masked to `0.0`.

---

## 7. Differentiable Stop-Gradient Surrogate with Auxiliary Critic $v_{\text{aux}}$

### The Code
```python
    # Normal (non-terminal, non-timeout) transition surrogate:
    # Term 2 (Forward):    (e_start - sg(e_jump)) ** 2
    # Term 1 (Backward):   (sg(e_aux_start) - e_aux_jump) ** 2
    diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2
```

### The Blurb: What it does & Why it is necessary
1. **The Double-Sampling Dilemma in Multi-Step Jumps**:
   - Differentiating the Dirichlet term $\frac{1}{2}(e_i - e_j)^2$ produces two branches:
     $$\nabla_\theta \mathcal{E} = \underbrace{(e_i - e_j)\nabla_\theta v_\theta(s_j)}_{\text{Term 1: Backward / Destination}} - \underbrace{(e_i - e_j)\nabla_\theta v_\theta(s_i)}_{\text{Term 2: Forward / Source}}$$
   - **Term 2** is completely unbiased because $\nabla_\theta v_\theta(s_i)$ is fixed at step $i$, prior to realizing the jump or future returns.
   - **Term 1**, however, multiplies the empirical error at step $i$ ($\hat{e}_i = \sum \gamma^t r + \gamma^m G_j - v(s_i)$) by the destination gradient $\nabla_\theta v_\theta(s_j)$. Because both $\hat{e}_i$ and $s_j$ depend on the *same random trajectory*, single-trajectory sampling incurs a nonzero covariance bias:
     $$\mathbb{E}[\hat{e}_i \nabla_\theta v_\theta(s_j)] = e_i \mathbb{E}[\nabla_\theta v_\theta(s_j)] + \text{Cov}(\hat{e}_i, \nabla_\theta v_\theta(s_j)) \neq e_i \mathbb{E}[\nabla_\theta v_\theta(s_j)]$$
2. **Debiasing via $v_{\text{aux}}$**:
   - We replace the empirical error in Term 1 with the auxiliary proxy error $e_{\text{aux}}(s) \doteq \text{sg}(v_{\text{aux}}(s)) - v_\theta(s)$.
   - Because $v_{\text{aux}}(s_i)$ is a deterministic function of state $s_i$, it is conditionally independent of the sampled jump trajectory $s_i \to s_j$, eliminating the covariance bias completely.
   - Furthermore, because $e_{\text{aux}, i} - e_{\text{aux}, j} = (v_{\text{aux}, i} - v_{\text{aux}, j}) - (v_{\theta, i} - v_{\theta, j})$, any constant estimation bias in $v_{\text{aux}}$ cancels in the difference.
3. **Stop-Gradient Automatic Differentiation**:
   - By applying `sg` to destination $\hat{e}_j$ in the first term, differentiating $\frac{1}{2}(e_i - \text{sg}(e_j))^2$ only takes gradients w.r.t. $v_\theta(s_i)$, exactly generating $-\text{Term 2}$.
   - By applying `sg` to source $e_{\text{aux}, i}$ in the second term, differentiating $\frac{1}{2}(\text{sg}(e_{\text{aux}, i}) - e_{\text{aux}, j})^2$ only takes gradients w.r.t. $v_\theta(s_j)$, exactly generating $+\text{Term 1}$.

---

## 8. Geometrically Decaying Start-State Reset Grounding ($\mathcal{L}_{\text{start}}$)

### The Code
```python
    # 4. Geometrically decaying start-state reset grounding (scanning over true_terminals & timeouts)
    def scan_steps(k, step_flags):
        term, timeout = step_flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    # If unprovided, default to a large integer so t=0 is not falsely flagged as a reset
    init_k = jnp.full((B,), 10_000, dtype=jnp.int32) if init_steps_since_start is None else init_steps_since_start
    final_k, steps_since_start = jax.lax.scan(scan_steps, init_k, (true_terminals, is_timeout))

    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(steps_since_start[:T_start] == 0, 1.0, 0.0),
        safe_gl ** steps_since_start[:T_start],
    )

    # Loss coefficient: Exactly 0.25 * tilde_gamma
    start_weight = 0.25 * tilde_gamma
    start_loss = start_weight * jnp.mean(start_trace * (e_start ** 2))
```

### The Blurb: What it does & Why it is necessary
1. **The Probability Mass Leakage Problem**:
   - In an episodic environment, the empirical state distribution $\mu$ sampled in a continuing rollout buffer is stationary w.r.t. the simulator resets ($\mu^\top P_c = \mu^\top$), **not** the true transition matrix $P$:
     $$\mu^\top (I - P) = \nu_{\text{term}} (\mu_0 - \mathbf{1}_{\text{term}})^\top$$
   - When completing the square for $e^\top D(I - P_\lambda)e$, probability mass leaks out at terminal transitions. The exact boundary correction requires adding:
     $$\mathcal{L}_{\text{start}} = \frac{1}{2}\tilde{\gamma} \cdot \frac{1}{2} e^\top \big(D - \text{diag}[\mu^\top P_\lambda]\big)e = \frac{\tilde{\gamma}}{4} \nu_{\text{term}} \sum_{k=0}^\infty (\gamma\lambda)^k \mathbb{E}_{s_k \sim P^k(\cdot \mid \mu_0)} \big[ e(s_k)^2 \big]$$
2. **Geometric Decay Along Trajectories**:
   - In $E(0)$, grounding is applied only at step $k = 0$ of a new episode ($\mathbb{I}_{\text{start}}$).
   - In $E(\lambda)$, because of the compound resolvent $(I - \gamma\lambda P)^{-1} = \sum (\gamma\lambda)^k P^k$, grounding decays geometrically by $(\gamma\lambda)^k$ across all subsequent steps $k$ of the new episode.
3. **Why `is_timeout` Clears the Counter ($k \to 10,000$)**:
   - If an episode ends via a true terminal at $t_0$ (setting $k=0$), and then the subsequent episode ends via an artificial timeout (`is_timeout == True`) at $t_1 = t_0 + 5$, the newly reset episode starting at $t_1 + 1$ would inherit $k = 6, 7, \dots$ unless `is_timeout` clears the counter.
   - In episodic MDPs, probability mass leaks out of the system **only** when an episode reaches a true absorbing termination ($V(s_{\text{term}}) = 0$), which is what the start grounding term compensates for. Artificial timeouts do not represent probability mass leakage; the newly reset episode starting after a timeout is not compensating for an absorbing boundary deficit.
   - Resetting $k \to 10,000$ upon `is_timeout` (`jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))`) zeroes out the decaying reset trace ($(\gamma\lambda)^{10,000} \approx 0$) and guarantees zero cross-episode leakage of `start_trace`.
4. **Why the Weight is Strictly $0.25 \tilde{\gamma}$ (Not $0.5 \tilde{\gamma}$)**:
   - For Dirichlet loss $\mathcal{L}_{\text{dir}}$, the surrogate sums two squared differences $(d_{\text{fwd}}^2 + d_{\text{bwd}}^2)$, each differentiating w.r.t. one state, bringing down a factor of $2$ which cancels one $\frac{1}{2}$. Thus, `dirichlet_weight = 0.5 * tilde_gamma`.
   - For start loss $\mathcal{L}_{\text{start}}$, however, $e(s_t)^2$ is a **single squared term**:
     $$\nabla_\theta [c \cdot \hat{e}_t^2] = 2c \hat{e}_t (-\nabla_\theta v_\theta(s_t)) = -2c \hat{e}_t \nabla_\theta v_\theta(s_t)$$
   - The derivative factor of $2$ cancels only **one** of the two $\frac{1}{2}$ prefactors:
     $$\frac{1}{2} \tilde{\gamma} \cdot \frac{1}{2} \cdot 2 = \frac{1}{2} \tilde{\gamma}$$
   - Setting `start_weight = 0.25 * tilde_gamma` produces the exact analytical gradient $-\frac{1}{2}\tilde{\gamma}\nu_{\text{term}}(\gamma\lambda)^k e_t \nabla_\theta v_\theta(s_t)$.
5. **Defaulting `init_k = 10,000`**:
   - If `init_steps_since_start` is unknown (e.g. at the very start of training), initializing $k = 0$ would falsely assume every environment just reset at $t = 0$, falsely inflating reset grounding by $+\frac{1}{T}$. Initializing $k = 10,000$ ensures $(\gamma\lambda)^{10,000} \approx 0$, seamlessly treating the initial buffer as a continuing steady-state chain until the first true terminal occurs.

---

## 9. Exact Theoretical Prefactors and Loss Assembly

### The Code
```python
    # 5. Exact theoretical prefactors
    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma
    start_weight = 0.25 * tilde_gamma  # Double 1/2 factor

    magnitude_loss = magnitude_weight * jnp.mean(e_start ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    start_loss = start_weight * jnp.mean(start_trace * (e_start ** 2))
    value_loss = magnitude_loss + dirichlet_loss + start_loss

    # Auxiliary critic loss (mean squared error w.r.t. empirical Monte Carlo returns)
    aux_loss = 0.5 * jnp.mean((sg(targets) - aux_values) ** 2)
    total_loss = value_loss + aux_loss_coef * aux_loss
```

### The Blurb: What it does & Why it is necessary
1. **Algebraic Consistency Across All $\lambda \in [0, 1]$**:
   - At $\lambda = 0$: $\tilde{\gamma} = \gamma$, $1 - \tilde{\gamma} = 1 - \gamma$, $m = 1$, and $\text{start\_trace} = \mathbb{I}_{\text{start}}$. The objective identically recovers the exact 1-step $E(0)$ formulation.
   - At $\lambda = 1$: $\tilde{\gamma} = 0$, so `dirichlet_weight = 0.0` and `start_weight = 0.0`. The magnitude weight becomes:
     $$\text{magnitude\_weight} = 0.5 \frac{1 - \gamma}{1 - \gamma} = 0.5$$
     recovering standard Monte Carlo squared error regression $\frac{1}{2}\mathbb{E}[(G_t - v_\theta(s_t))^2]$.
2. **Auxiliary Critic Regression**:
   - `aux_loss` trains $v_{\text{aux}}$ directly on empirical returns $G_t$ via standard regression. Because $v_{\text{aux}}$ only appears inside `sg(aux_values)` in the main critic loss, $\nabla_\theta \mathcal{L}_{\text{total}}$ is completely unperturbed by $\mathcal{L}_{\text{aux}}$.

---

## 10. Rollout-Level Jump Pairing & Flat Minibatching (`prepare_e_lambda_jump_pairs`)

### The Architecture
Instead of resampling lookahead jumps and scanning episode boundaries repeatedly on every minibatch, we materialize geometric jump pairs $(s_t, s_{t+m})$ **once per rollout buffer** prior to the gradient descent loop. Each jump pair, along with its precomputed boundary masks and decaying start weights, is stored in a clean NamedTuple (`ELambdaJumpTransition`). Standard transition flattening and uniform random shuffling (`helpers.shuffle_and_batch`) then operate directly on these paired transitions, matching canonical PPO.

### The Code
```python
class ELambdaJumpTransition(NamedTuple):
    obs: jnp.ndarray
    dest_obs: jnp.ndarray
    target: jnp.ndarray
    dest_target: jnp.ndarray
    hit_true_terminal: jnp.ndarray
    hit_timeout: jnp.ndarray
    start_weight: jnp.ndarray
    action: Optional[jnp.ndarray] = None
    log_prob: Optional[jnp.ndarray] = None
    advantage: Optional[jnp.ndarray] = None


def prepare_e_lambda_jump_pairs(
    traj_batch,
    returns: jax.Array,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    steps_since_start: jax.Array,
    advantages: Optional[jax.Array] = None,
    runway_length: int = 16,
):
    T, B = returns.shape[:2]
    T_start = min(T, max(1, T - runway_length))
    gl = gamma * lmbda
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)

    # 1. Sample geometric jump lengths: m = K + 1 where K ~ Geom(1 - gl)
    u = jax.random.uniform(rng, shape=(T_start, B))
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T_start, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T_start)[:, None]
    j = jnp.minimum(t_arr + m, T)

    # 2. Episode boundary queries in [t, j - 1] via prefix sums
    is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
    true_terminals = traj_batch.done & ~is_timeout

    term_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    timeout_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)],
        axis=0,
    )

    hit_true_terminal = (jnp.take_along_axis(term_cumsum, j, axis=0) - term_cumsum[:T_start]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum, j, axis=0) - timeout_cumsum[:T_start]) > 0

    # 3. Track steps since start (clearing to 10,000 on timeout)
    def scan_steps(k, step_flags):
        term, timeout = step_flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    init_k = jnp.full((B,), 10_000, dtype=jnp.int32) if steps_since_start is None else steps_since_start
    final_steps_since_start, all_steps = jax.lax.scan(scan_steps, init_k, (true_terminals, is_timeout))

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(all_steps[:T_start] == 0, 1.0, 0.0),
        safe_gl ** all_steps[:T_start],
    )
    start_weight = 0.25 * tilde_gamma * start_trace

    # 4. Form extended arrays (length T + 1) for destination lookups at step T
    obs_ext = jnp.concatenate([traj_batch.obs, traj_batch.next_obs[-1][None]], axis=0)
    target_ext = jnp.concatenate([returns, traj_batch.next_value[-1][None]], axis=0)

    # Gather destination states and targets along time axis (axis 0)
    j_expanded = j.reshape(j.shape + (1,) * (obs_ext.ndim - 2))
    j_expanded = jnp.broadcast_to(j_expanded, (T_start, B) + obs_ext.shape[2:])
    dest_obs = jnp.take_along_axis(obs_ext, j_expanded, axis=0)
    dest_target = jnp.take_along_axis(target_ext, j, axis=0)

    action = traj_batch.action[:T_start] if hasattr(traj_batch, "action") else None
    log_prob = traj_batch.log_prob[:T_start] if hasattr(traj_batch, "log_prob") else None
    adv = advantages[:T_start] if advantages is not None else None

    jump_batch = ELambdaJumpTransition(
        obs=traj_batch.obs[:T_start],
        dest_obs=dest_obs,
        target=returns[:T_start],
        dest_target=dest_target,
        hit_true_terminal=hit_true_terminal,
        hit_timeout=hit_timeout,
        start_weight=start_weight,
        action=action,
        log_prob=log_prob,
        advantage=adv,
    )
    return jump_batch, final_steps_since_start
```

### The Blurb: What it does & Why it is necessary
1. **Single Pre-Materialization per Rollout**:
   - Rather than executing complex indexing, cumulative sums, and random geometric draws inside every epoch and every minibatch of SGD, `prepare_e_lambda_jump_pairs` runs **exactly once** after environment rollout collection.
   - Holding jump destinations fixed across epochs acts as standard empirical sample reuse during the policy/value update, stabilizing gradient updates across epochs.
2. **Native Flat Shuffling**:
   - `jump_batch` leaves `prepare_e_lambda_jump_pairs` with shape `(T_start, B, ...)`.
   - Calling canonical `helpers.shuffle_and_batch(rng, jump_batch, NUM_MINIBATCHES)` flattens `(T_start, B)` into $N = T_{\text{start}} \times B$, permutes all paired fields identically with 100% synchronization, and splits into flat 1D minibatches.
3. **Timeout Counter Reset ($k \to 10,000$)**:
   - Clearing $k \to 10,000$ on artificial timeouts ensures start-state grounding does not erroneously bleed across unrelated episodes if an episode terminates via truncation rather than a true absorbing boundary.

---

## 11. Full Reference Implementation: Pointwise Minibatch Critic Loss (`core/helpers.py`)

```python
def e_lambda_pointwise_critic_loss(
    v_start: jax.Array,
    v_jump: jax.Array,
    v_aux_start: jax.Array,
    v_aux_jump: jax.Array,
    mb: ELambdaJumpTransition,
    gamma: float,
    lmbda: float,
):
    """
    Evaluates the exact Dirichlet, magnitude, and start losses on a 1D minibatch of jump pairs.
    """
    sg = jax.lax.stop_gradient
    gl = gamma * lmbda

    e_start = sg(mb.target) - v_start
    e_jump = jnp.where(mb.hit_true_terminal, 0.0, sg(mb.dest_target) - v_jump)

    e_aux_start = sg(v_aux_start) - v_start
    e_aux_jump = jnp.where(mb.hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

    diff_sq = jnp.where(
        mb.hit_true_terminal,
        e_start ** 2,
        jnp.where(
            mb.hit_timeout,
            0.0,
            (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2,
        ),
    )

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    magnitude_loss = magnitude_weight * jnp.mean(e_start ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    start_loss = jnp.mean(mb.start_weight * (e_start ** 2))
    value_loss = magnitude_loss + dirichlet_loss + start_loss

    losses = {
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "start_loss": start_loss,
    }
    return value_loss, losses
```

---

## 12. Full Reference Implementation: PPO Training Step (`ppo/E_lambda_geometric.py`)

```python
    # 2. ADVANTAGES & RETURNS
    advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], config["GAE_LAMBDA"])
    _, returns = helpers.calculate_gae(traj_batch, config["GAMMA"], config["RETURN_LAMBDA"])

    # 3. PREPARE JUMP PAIRS ONCE PER ROLLOUT
    rng, _rng = jax.random.split(rng)
    jump_batch, final_steps_since_start = helpers.prepare_e_lambda_jump_pairs(
        traj_batch=traj_batch,
        returns=returns,
        advantages=advantages,
        gamma=config["GAMMA"],
        lmbda=config["VALUE_LAMBDA"],
        rng=_rng,
        steps_since_start=steps_since_start,
        runway_length=config.get("RUNWAY_LENGTH", 16),
    )

    # 4. UPDATE NETWORK OVER EPOCHS AND FLAT MINIBATCHES
    def _update_epoch(update_state, unused):
        def _update_minbatch(state, mb):
            train_state, aux_train_state = state

            grad_fn = jax.value_and_grad(helpers.e_lambda_ppo_minibatch_loss, has_aux=True)
            (total_loss, losses), grads = grad_fn(
                train_state.params,
                network,
                mb,
                config,
                aux_params=aux_train_state.params,
                aux_network=aux_network,
            )
            train_state = train_state.apply_gradients(grads=grads)

            # Update auxiliary critic on empirical returns
            def _aux_loss_fn(aux_p):
                aux_vals = aux_network.apply(aux_p, mb.obs)
                return 0.5 * jnp.mean((jax.lax.stop_gradient(mb.target) - aux_vals) ** 2)

            aux_loss, aux_grads = jax.value_and_grad(_aux_loss_fn)(aux_train_state.params)
            aux_train_state = aux_train_state.apply_gradients(grads=aux_grads)
            losses["aux_loss"] = aux_loss

            return (train_state, aux_train_state), losses

        train_state, aux_train_state, rng = update_state
        rng, _rng = jax.random.split(rng)
        minibatches = helpers.shuffle_and_batch(_rng, jump_batch, config["NUM_MINIBATCHES"])

        (train_state, aux_train_state), loss_info = jax.lax.scan(
            _update_minbatch, (train_state, aux_train_state), minibatches
        )
        return (train_state, aux_train_state, rng), loss_info

    initial_update_state = (train_state, aux_train_state, rng)
    update_state, loss_info = jax.lax.scan(
        _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
    )
    train_state, aux_train_state, rng = update_state
```

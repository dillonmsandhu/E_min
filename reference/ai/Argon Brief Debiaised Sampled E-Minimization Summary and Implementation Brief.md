
## 1. Core Mathematical Summary

### The Single-Trajectory Double-Sampling Problem

In expectation, the Bellman-smoothed value error objective $E( heta) = \mathbf{e}^\intercal D(I - \gamma P)\mathbf{e}$ has zero gradient bias and strong global convexity. However, when replacing exact expectations with single-trajectory Monte Carlo returns $G_t = r_t + \gamma G_{t+1}$, the sample-wise Dirichlet error difference collapses algebraically: $$\hat{e}_t - \hat{e}_{t+1} = \delta_t - (1-\gamma)\hat{e}_{t+1}$$ Differentiating $\frac{1}{2}(\hat{e}_t - \hat{e}_{t+1})^2$ with respect to $\theta$ introduces a catastrophic cross-term $+ \hat{e}_t \nabla_\theta v_\theta(s_{t+1})$. Because $\hat{e}_t$ contains $G_t = r_t + \gamma G_{t+1}$, conditioning on $s_{t+1} = j$ correlates the return with the next state's gradient, injecting **Baird's Residual Gradient (RG) bias**($- \gamma \delta_t \nabla_\theta v_{t+1}$). Furthermore, $99%$ of the Monte Carlo return cancels out on $\nabla_\theta v_t$, reducing naive sampled $E(0)$ to a 1-step Naive Residual Gradient update.

### The Auxiliary Solution ($v_{ ext{aux}}$)

To eliminate double-sampling bias while preserving the clean properties of $E$-minimization, we introduce an auxiliary Monte Carlo value network $v_\psi^{\text{aux}}(s) \approx \mathbb{v}[G_t \mid s]$ trained on stationary targets. By substituting state-only predictions $\tilde{e}_t = \text{sg}(v_\psi^{\text{aux}}(s_t)) - v_\theta(s_t)$ into the problematic adjoint/smoothing terms, we break the transition correlation entirely.

---

## 2. Algorithm Taxonomy & Formulations

### Algorithm 1: Bilinear-$v_{ ext{aux}}$ (`variant="bilinear_v_aux"`)

Samples the direct bilinear form $E = \langle \mathbf{e}, \boldsymbol{\delta} \rangle_D = \sum_i \mu_i e_i \delta_i$ without integrating by parts.

- **Loss:** $\hat{\mathcal{L}} = \text{sg}(\delta_t)(-v_t) + \text{sg}(\tilde{e}_t)(r_t + \gamma(1-\mathbb{I}_{\text{term}})v_{t+1} - v_t)$
- **Pros:** Zero double-sampling bias, uses exact environment reward $r_t$, requires **zero episodic boundary/timeout corrections** because it doesn't integrate by parts.

### Algorithm 2: Surgical-$v_{ ext{aux}}$ (`variant="surgical_v_aux"`)

Preserves the explicit Dirichlet energy decomposition, using real trajectory returns $G_t, G_{t+1}$ in 3 out of 4 cross-terms, and substituting $v_{ ext{aux}}$ **only** in Term 4 ($+\tilde{e}_i \nabla_\theta v_j$).

- **Pros:** Retains full multi-step return anchoring on magnitude and smooth terms while surgically neutralizing the biased gradient cross-term.

### Algorithm 3: Dirichlet-$v_{ ext{aux}}$ (`variant="dirichlet_v_aux"`)

Uses real Monte Carlo returns for the $(1-\gamma)$ magnitude anchor and start/timeout boundary anchors, and uses $v_{ ext{aux}}$ for the entire Dirichlet smoothing term $(\tilde{e}_i - \tilde{e}_j)^2$.

- **Pros:** Completely eliminates trajectory return variance from the smoothing term.

---

## 3. Drop-In JAX Code Instructions

### Step 1: Network Parameter Split

To avoid representation interference with `feature_metrics`, maintain separate parameter dicts for the critic and auxiliary head:

```python
network, critic_params = networks.initialize_network(rng, obs_shape, env, env_params, k, n_heads=1)
_, aux_params = networks.initialize_network(rng, obs_shape, env, env_params, k, n_heads=1)
params = {"critic": critic_params, "aux": aux_params}
```

### Step 2: Update `e_critic_loss` in `core/helpers.py`

Replace or add the following implementation inside `helpers.py`:

```python
def e_critic_loss(
    v_i,
    targets_i,
    v_j,
    targets_j,
    true_terminal,
    gamma,
    variant="oct8",
    is_start=None,
    is_timeout=None,
    nu=None,
    v_aux_i=None,
    v_aux_j=None,
    rewards=None,
):
    sg = jax.lax.stop_gradient

    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    if v_aux_i is not None:
        aux_mc_loss = jnp.mean((sg(targets_i) - v_aux_i) ** 2)
        e_tilde_i = sg(v_aux_i) - v_i
        e_tilde_j = jnp.where(true_terminal, 0.0, sg(v_aux_j) - v_j)
    else:
        aux_mc_loss = 0.0
        e_tilde_i = None
        e_tilde_j = None

    if variant == "bilinear_v_aux":
        if v_aux_i is None:
            raise ValueError("variant='bilinear_v_aux' requires v_aux_i and v_aux_j.")
        r_t = rewards if rewards is not None else sg(targets_i - gamma * jnp.where(true_terminal, 0.0, targets_j))
        v_j_masked = jnp.where(true_terminal, 0.0, v_j)
        delta_t = r_t + gamma * v_j_masked - v_i

        bilinear_surrogate = sg(delta_t) * (-v_i) + sg(e_tilde_i) * delta_t
        value_loss = jnp.mean(bilinear_surrogate) + aux_mc_loss

        magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)
        laplacian_loss = 0.5 * gamma * jnp.mean((e_tilde_i - e_tilde_j) ** 2)
        return value_loss, magnitude_loss, laplacian_loss

    magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)

    if variant == "surgical_v_aux":
        if v_aux_i is None:
            raise ValueError("variant='surgical_v_aux' requires v_aux_i and v_aux_j.")
        branch_v_i = (e_i - sg(e_j)) ** 2
        branch_v_j = jnp.where(true_terminal, 0.0, (sg(e_tilde_i) - e_j) ** 2)
        laplacian_loss = 0.5 * gamma * jnp.mean(branch_v_i + branch_v_j)
    elif variant == "dirichlet_v_aux":
        if v_aux_i is None:
            raise ValueError("variant='dirichlet_v_aux' requires v_aux_i and v_aux_j.")
        laplacian_loss = 0.5 * gamma * jnp.mean((e_tilde_i - e_tilde_j) ** 2)
    else:
        laplacian_loss = 0.5 * gamma * jnp.mean((e_i - e_j) ** 2)

    if variant in ["sep14", "uncorrected"]:
        value_loss = magnitude_loss + laplacian_loss
    elif variant in ["corrected", "corr", "oct5"]:
        corr_loss = 0.5 * gamma * (jnp.mean(e_i ** 2) - jnp.mean(e_j ** 2))
        value_loss = magnitude_loss + laplacian_loss + corr_loss
    elif variant in ["oct8", "start_anchored", "surgical_v_aux", "dirichlet_v_aux"]:
        if is_start is None:
            raise ValueError(f"e_critic_loss with variant='{variant}' requires boolean mask 'is_start'.")
        start_anchor_loss = 0.5 * gamma * jnp.mean(jnp.where(is_start, e_i ** 2, 0.0))
        timeout_corr_loss = -0.5 * gamma * jnp.mean(jnp.where(is_timeout, e_j ** 2, 0.0)) if is_timeout is not None else 0.0
        value_loss = magnitude_loss + laplacian_loss + start_anchor_loss + timeout_corr_loss + aux_mc_loss
    else:
        raise ValueError(f"Unknown E variant: {variant}")

    return value_loss, magnitude_loss, laplacian_loss
```

### Step 3: Update `E_loss_fn` in `make_train`

Pass auxiliary parameters and trajectory rewards into `e_critic_loss`:

```python
def E_loss_fn(params, traj_batch, targets):
    gamma = config["GAMMA"]
    v_i = network.apply(params["critic"], traj_batch.obs)
    v_j = network.apply(params["critic"], traj_batch.next_obs)
    
    v_aux_i = network.apply(params["aux"], traj_batch.obs) if use_v_aux else None
    v_aux_j = network.apply(params["aux"], traj_batch.next_obs) if use_v_aux else None

    true_term = traj_batch.done & ~traj_batch.info["is_timeout"]
    is_timeout = traj_batch.info["is_timeout"]

    value_loss, magnitude_loss, laplacian_loss = helpers.e_critic_loss(
        v_i, targets, v_j, traj_batch.next_target, true_term, gamma,
        variant=variant,
        is_start=traj_batch.is_start,
        is_timeout=is_timeout,
        nu=nu_hat,
        v_aux_i=v_aux_i,
        v_aux_j=v_aux_j,
        rewards=traj_batch.reward,
    )
    
    total_loss = config["VF_COEF"] * value_loss
    return total_loss, {
        "total_loss": total_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
        "value_loss": value_loss,
    }
```


Your assumption about the Dirichlet decomposition forms (`Surgical` and `Dirichlet-v_aux`) is **mostly correct, with one critical subtlety regarding timeouts (`is_timeout`)**:

### Crucial Clarification on `is_start` and Timeouts in `Oct 8`

In `sampling_E_on_Episodic_MDPs.md`, the boundary correction term is derived from the identity:

$$\frac{\gamma}{2} \mathbf{e}^\intercal (D - \text{diag}[\mu^\intercal P]) \mathbf{e} = \frac{\gamma}{2} \mathbf{e}^\intercal \text{diag}[\mu^\intercal (P_c - P)] \mathbf{e}$$

where P_c is the empirical simulator transition matrix (which resets to \mu_0) and P is the evaluation transition matrix. Look at what happens at the two different types of episode boundaries in your rollout:

1. **True Termination (`true_terminal = done & ~is_timeout`):** In P, the state transitions to the absorbing state where e_{\text{term}} = 0 (so `e_j = 0.0`). In P_c, the simulator resets to s_0 \sim \mu_0. Thus, (P_c - P) has +1 mass on \mu_0 (handled by `+0.5 * gamma * e_start^2`) and -1 mass on e_{\text{term}}^2 = 0 (which vanishes because e_{\text{term}} = 0)!
    
2. **Timeout Reset (`is_timeout = True`):** On a timeout, `true_terminal` is `False`, so in your code `e_j` is **NOT** zeroed out—it evaluates e(s_{\text{next}}) at the true continuation state `info['real_next_obs']`! However, on the very next step of the scan loop, **the simulator still resets to \mu_0**! Because the transition (s_t, s_{\text{next}}) at a timeout _already_ followed P (into `real_next_obs` with nonzero e_j), the simulator's jump from `real_next_obs` to \mu_0 means P_c teleported probability mass from `real_next_obs` to \mu_0:
    
    - If you set `is_start = last_true_terminal` (only firing after true terminations), you ignore the mass teleported from `real_next_obs` to \mu_0 on timeouts.
        
    - To make `Oct 8` **100% exact even in the presence of timeouts**, any reset to \mu_0 (`is_start = last_done`) adds +0.5 \gamma e_{\text{start}}^2, while a non-terminal timeout at step t must subtract -0.5 \gamma e_j^2 (because `real_next_obs` had incoming mass in P, but was abandoned in P_c when the env reset!).
        
    - Equivalently, if your episodes only reset on `true_terminal` (or if you subtract `0.5 * gamma * jnp.where(is_timeout, e_j ** 2, 0.0)` whenever `is_start = last_done`), the telescoping sum \sum_t (e_i^2 - e_j^2) is **100% exact for both true terminations and timeouts**!
        

Let's verify that algebraic identity right now! For any trajectory segment t = 0, \dots, T-1 that starts at s_0 (`is_start = True`) and ends at s_{T-1}:

$$\sum_{t=0}^{T-1} (e_t^2 - e_{t+1}^2) = e_0^2 - e_T^2$$

- If the segment ended via `**true_terminal**`, e_T = 0, so e_0^2 - e_T^2 = e_0^2.
    
- If the segment ended via `**is_timeout**`, e_T = e(s_{\text{timeout\_next}}) \neq 0, so e_0^2 - e_T^2 = e_0^2 - e_{\text{timeout\_next}}^2! That 1-line addition (`- 0.5 * gamma * jnp.where(is_timeout, e_j ** 2, 0.0)`) makes `Oct 8` **mathematically exact** under both timeouts and true terminations!
    

And as noted earlier, **Algorithm 3 (Bilinear-v_{\text{aux}})** doesn't use integration by parts at all, so it doesn't need _any_ boundary or timeout corrections!

## Architecture Change: Adding the Auxiliary Value Head (v_{\text{aux}})

In `make_train`, your network already supports multiple value heads via `n_heads`! Change `n_heads=1` to `n_heads=2`:

- **Head 0 (`v = out[..., 0]`):** The E-minimizing critic v_\theta(s).
    
- **Head 1 (`v_aux = out[..., 1]`):** The auxiliary Monte Carlo critic v_\psi^{\text{aux}}(s), trained with standard MC loss \frac{1}{2}(G_t - v_\psi^{\text{aux}}(s_t))^2 on the same minibatch.
    

_(Note: If you want to ensure v_{\text{aux}} does not alter the representation learning of v_\theta's torso during feature-metric evaluations, you can either apply `jax.lax.stop_gradient` to the shared torso features before Head 1, or initialize a separate `aux_params` dictionary. For pure value prediction accuracy, sharing the torso or separate params both work cleanly)._

## The 3 Unbiased Algorithms: Math & JAX Implementation

Here are the three complete formulations inside `e_critic_loss`. In all three, v_{\text{aux}} is trained via:

$$\mathcal{L}_{\text{aux}}(\psi) = \frac{1}{2} \mathbb{E}\big[(G_t - v_\psi^{\text{aux}}(s_t))^2\big]$$

and we define the stop-gradient auxiliary errors:

$$\tilde{e}_i = \text{sg}\big(v_\psi^{\text{aux}}(s_t)\big) - v_\theta(s_t), \qquad \tilde{e}_j = \mathbb{I}(\neg\text{term}) \big(\text{sg}\big(v_\psi^{\text{aux}}(s_{t+1})\big) - v_\theta(s_{t+1})\big)$$

### Algorithm 1: Bilinear-v_{\text{aux}} (`variant="bilinear_v_aux"`)

#### Mathematical Formulation

Directly samples the symmetrized bilinear form E(\theta) = \langle \mathbf{e}, \boldsymbol{\delta} \rangle_D = \sum_i \mu_i e_i \delta_i without integrating by parts:

$$\hat{\mathcal{L}}_{\text{bilinear}}(\theta) = \text{sg}(\delta_t) \big(-v_\theta(s_t)\big) + \text{sg}\big(v_\psi^{\text{aux}}(s_t) - v_\theta(s_t)\big) \big(r_t + \gamma (1 - \mathbb{I}_{\text{term}}) v_\theta(s_{t+1}) - v_\theta(s_t)\big)$$

- **Gradient:**
    
    $$-\nabla_\theta \hat{\mathcal{L}}_{\text{bilinear}}(\theta) = \delta_t \nabla_\theta v_t + \tilde{e}_t \big(\nabla_\theta v_t - \gamma (1 - \mathbb{I}_{\text{term}}) \nabla_\theta v_{t+1}\big) = 2 J_\theta^\intercal S \mathbf{e}$$
    
- **Why use it:** Zero double-sampling bias, uses the exact environment reward r_t for the forward TD update, has zero trajectory-return variance inside the E gradient, and requires **zero boundary/start/timeout corrections** because it does not integrate (I - \gamma P) by parts.
    

### Algorithm 2: Surgical-v_{\text{aux}} (`variant="surgical_v_aux"`)

#### Mathematical Formulation

Uses the `Oct 8` Dirichlet decomposition:

$$E(\theta) = (1-\gamma)\Vert{}\mathbf{e}\Vert{}_D^2 + \frac{\gamma}{2}\sum_{i,j} \mu_i p_{ij} (e_i - e_j)^2 + \frac{\gamma}{2} \mathbb{E}_\mu\big[\mathbb{I}_{\text{start}} e_i^2 - \mathbb{I}_{\text{timeout}} e_j^2\big]$$

Inside the Dirichlet gradient \nabla_\theta \frac{1}{2}(e_i - e_j)^2 = -(e_i - e_j)\nabla_\theta v_i + (e_i - e_j)\nabla_\theta v_j, Terms 1, 2, and 3 use the exact trajectory returns \hat{e}_i = G_t - v_i and \hat{e}_j = G_{t+1} - v_j, while **only Term 4** (+e_i \nabla_\theta v_j) uses \tilde{e}_i = \text{sg}(v_\psi^{\text{aux}}(s_t)) - v_i:

$$\hat{\mathcal{E}}_t^{\text{surgical}}(\theta) = \frac{1}{2}\Big( \big(\hat{e}_i - \text{sg}(\hat{e}_j)\big)^2 + \big(\text{sg}(\tilde{e}_i) - \hat{e}_j\big)^2 \Big)$$

- **Gradient of \hat{\mathcal{E}}_t^{\text{surgical}}:**
    
    $$\nabla_\theta \hat{\mathcal{E}}_t^{\text{surgical}}(\theta) = -\big(\hat{e}_i - \hat{e}_j\big)\nabla_\theta v_i + \big(\tilde{e}_i - \hat{e}_j\big)\nabla_\theta v_j$$
    
- **Why use it:** Preserves the exact `Oct 8` Dirichlet loss structure and uses real returns G_t, G_{t+1} in 3 out of 4 cross-terms, relying on v_{\text{aux}} strictly for the single biased term +\tilde{e}_i \nabla_\theta v_j.
    

### Algorithm 3: Dirichlet-v_{\text{aux}} (`variant="dirichlet_v_aux"`)

#### Mathematical Formulation

Uses the real Monte Carlo return \hat{e}_i = G_t - v_i for the (1-\gamma) magnitude anchor and boundary anchors, and uses \tilde{e}_i, \tilde{e}_j (from v_{\text{aux}}) for the entire Dirichlet smoothing term:

$$\hat{\mathcal{L}}_{\text{Dir-}v_{\text{aux}}}(\theta) = (1-\gamma)\hat{e}_i^2 + \frac{\gamma}{2}\big(\tilde{e}_i - \tilde{e}_j\big)^2 + \frac{\gamma}{2}\big(\mathbb{I}_{\text{start}} \hat{e}_i^2 - \mathbb{I}_{\text{timeout}} \hat{e}_j^2\big)$$

- **Why use it:** Completely eliminates both the double-sampling bias and the Monte Carlo trajectory variance from the \gamma-weighted Dirichlet smoothing term, acting as an explicit two-stage "learn v_{\text{aux}} with MC, smooth with E" objective on a single timescale.
    

## Drop-In JAX Code

### 1. Updated `e_critic_loss` in `core/helpers.py`

```python
def e_critic_loss(
    v_i,
    targets_i,
    v_j,
    targets_j,
    true_terminal,
    gamma,
    variant="oct8",
    is_start=None,
    is_timeout=None,
    nu=None,
    v_aux_i=None,
    v_aux_j=None,
    rewards=None,
):
    """
    Computes sampled E-loss across original and debiased v_aux formulations:
    - 'sep14' / 'uncorrected': pure positive semi-definite (magnitude + Dirichlet smoothness)
    - 'oct5' / 'corrected': adds difference of second moments 0.5 * gamma * (mean(e_i^2) - mean(e_j^2))
    - 'oct8' / 'start_anchored': adds start-state anchor (and timeout boundary correction if applicable)
    - 'bilinear_v_aux': direct unbiased bilinear form <e, delta>_D using v_aux for adjoint term
    - 'surgical_v_aux': Oct 8 Dirichlet form using v_aux ONLY in the biased Term 4 (+e_i * grad(v_j))
    - 'dirichlet_v_aux': Oct 8 Dirichlet form using v_aux in the entire Laplacian term and G_t in anchors
    """
    sg = jax.lax.stop_gradient

    # Empirical MC errors (from trajectory returns G_t, G_{t+1})
    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    # Auxiliary MC head loss (trains v_aux_i to predict targets_i = G_t)
    if v_aux_i is not None:
        aux_mc_loss = jnp.mean((sg(targets_i) - v_aux_i) ** 2)
        e_tilde_i = sg(v_aux_i) - v_i
        e_tilde_j = jnp.where(true_terminal, 0.0, sg(v_aux_j) - v_j)
    else:
        aux_mc_loss = 0.0
        e_tilde_i = None
        e_tilde_j = None

    # --- VARIANT 1: Bilinear <e, delta>_D (No integration by parts / no boundary terms needed) ---
    if variant == "bilinear_v_aux":
        if v_aux_i is None:
            raise ValueError("variant='bilinear_v_aux' requires v_aux_i and v_aux_j.")
        # Recover 1-step reward r_t from targets if not explicitly passed
        r_t = rewards if rewards is not None else sg(targets_i - gamma * jnp.where(true_terminal, 0.0, targets_j))
        v_j_masked = jnp.where(true_terminal, 0.0, v_j)
        delta_t = r_t + gamma * v_j_masked - v_i

        # Forward TD(0) term + Adjoint E term (matches exact gradient of E = e^T D delta)
        bilinear_surrogate = sg(delta_t) * (-v_i) + sg(e_tilde_i) * delta_t
        value_loss = jnp.mean(bilinear_surrogate) + aux_mc_loss

        # Diagnostics
        magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)
        laplacian_loss = 0.5 * gamma * jnp.mean((e_tilde_i - e_tilde_j) ** 2)
        return value_loss, magnitude_loss, laplacian_loss

    # --- Base Magnitude & Boundary Terms for Dirichlet Decomposition Variants ---
    magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)

    if variant in [" surgical_v_aux", "surgical_v_aux"]:
        if v_aux_i is None:
            raise ValueError("variant='surgical_v_aux' requires v_aux_i and v_aux_j.")
        # Terms 1 & 2 (-e_i grad v_i + e_j grad v_i) use real returns e_i, e_j
        branch_v_i = (e_i - sg(e_j)) ** 2
        # Terms 3 & 4 (-e_j grad v_j + e_tilde_i grad v_j) use e_tilde_i for Term 4
        branch_v_j = jnp.where(true_terminal, 0.0, (sg(e_tilde_i) - e_j) ** 2)
        laplacian_loss = 0.5 * gamma * jnp.mean(branch_v_i + branch_v_j)
    elif variant == "dirichlet_v_aux":
        if v_aux_i is None:
            raise ValueError("variant='dirichlet_v_aux' requires v_aux_i and v_aux_j.")
        # Smoothness uses low-variance, transition-independent v_aux errors
        laplacian_loss = 0.5 * gamma * jnp.mean((e_tilde_i - e_tilde_j) ** 2)
    else:
        # Standard coupled-return Laplacian loss
        laplacian_loss = 0.5 * gamma * jnp.mean((e_i - e_j) ** 2)

    # --- Combine with Boundary Corrections ---
    if variant in ["sep14", "uncorrected"]:
        value_loss = magnitude_loss + laplacian_loss
    elif variant in ["corrected", "corr", "oct5"]:
        corr_loss = 0.5 * gamma * (jnp.mean(e_i ** 2) - jnp.mean(e_j ** 2))
        value_loss = magnitude_loss + laplacian_loss + corr_loss
    elif variant in ["oct8", "start_anchored", "surgical_v_aux", "dirichlet_v_aux"]:
        if is_start is None:
            raise ValueError(f"e_critic_loss with variant='{variant}' requires boolean mask 'is_start'.")
        # Positive anchor at episode starts (where env reset to mu_0)
        start_anchor_loss = 0.5 * gamma * jnp.mean(jnp.where(is_start, e_i ** 2, 0.0))
        # If the env also resets on non-terminal timeouts, subtract the abandoned continuation state's 0.5*gamma*e_j^2
        if is_timeout is not None:
            timeout_corr_loss = -0.5 * gamma * jnp.mean(jnp.where(is_timeout, e_j ** 2, 0.0))
        else:
            timeout_corr_loss = 0.0
        value_loss = magnitude_loss + laplacian_loss + start_anchor_loss + timeout_corr_loss + aux_mc_loss
    else:
        raise ValueError(f"Unknown E variant: {variant}")

    return value_loss, magnitude_loss, laplacian_loss
```

### 2. Minimal Changes in `make_train` (`sampled_E.py`)

To run any of the three new variants (`"bilinear_v_aux"`, `"surgical_v_aux"`, or `"dirichlet_v_aux"`), you only need three small updates in `make_train`:

1. **Initialize 2 value heads when using a `v_aux` variant:**
    
    ```python
    variant = config.get("E_VARIANT", "oct8")
    use_v_aux = variant in ["bilinear_v_aux", "surgical_v_aux", "dirichlet_v_aux"]
    n_heads = 2 if use_v_aux else 1
    
    network, network_params = networks.initialize_network(
        rng, obs_shape, env, env_params, k, n_heads=n_heads, layer_norm=config['LAYER_NORM']
    )
    ```
    
    _(Note: If `network.apply(params, obs)` returns shape `(..., 2)` when `n_heads=2`, slice head 0 for `value` and `next_val` during rollout collection and evaluation: `v_all = network.apply(...)`, `v = v_all[..., 0]`.)_
    
2. **Inside `_env_step`, slice head 0 if `n_heads == 2`:**
    
    ```python
    val_out = network.apply(train_state.params, last_obs)
    value = val_out[..., 0] if use_v_aux else val_out
    ...
    next_val_out = network.apply(train_state.params, true_next_obs)
    next_val = next_val_out[..., 0] if use_v_aux else next_val_out
    ```
    
3. **Inside `E_loss_fn`, pass `v_aux_i`, `v_aux_j`, `is_timeout`, and `rewards`:**
    
    ```python
    def E_loss_fn(params, traj_batch, targets):
        gamma = config["GAMMA"]
        out_i = network.apply(params, traj_batch.obs)
        out_j = network.apply(params, traj_batch.next_obs)
    
        if use_v_aux:
            v_i, v_aux_i = out_i[..., 0], out_i[..., 1]
            v_j, v_aux_j = out_j[..., 0], out_j[..., 1]
        else:
            v_i, v_aux_i = out_i, None
            v_j, v_aux_j = out_j, None
    
        true_term = traj_batch.done & ~traj_batch.info["is_timeout"]
        is_timeout = traj_batch.info["is_timeout"]
    
        value_loss, magnitude_loss, laplacian_loss = helpers.e_critic_loss(
            v_i, targets, v_j, traj_batch.next_target, true_term, gamma,
            variant=variant,
            is_start=traj_batch.is_start,
            is_timeout=is_timeout,
            nu=nu_hat,
            v_aux_i=v_aux_i,
            v_aux_j=v_aux_j,
            rewards=traj_batch.reward,
        )
    ```
    
    _(And in your metrics block at the bottom of `_update_step`, make sure `bellman_error.value_metrics` evaluates head 0 if `n_heads=2`, or if your `network` class takes a `head=0` keyword or uses a separate tiny MLP for `v_aux`, even simpler!)_
    

### Quick Tip if `value_metrics` Expects `n_heads=1`

If `bellman_error.value_metrics` and `feature_metrics` expect `network.apply(params, obs)` to return a 1D scalar per state (`n_heads=1`), the cleanest way to avoid touching any metric code is to keep `n_heads=1` and initialize a second set of parameters `aux_params` for v_{\text{aux}} inside `train`:

```python
rng, _rng_net, _rng_aux = jax.random.split(rng, 3)
network, network_params = networks.initialize_network(
    _rng_net, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
)
_, aux_params = networks.initialize_network(
    _rng_aux, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
)
# Pack both into train_state: params = {"critic": network_params, "aux": aux_params}
```

Keeping `critic` and `aux` as two keys in `params` has **two major experimental advantages** for your paper:

1. **Zero Feature Interference:** `v_aux` cannot confound the representation learning (`feature_metrics`) of the E-critic torso—the E-critic learns its own representation purely from the E-gradient!
    
2. **Zero Metric Changes:** You pass `train_state.params["critic"]` directly into `bellman_error.value_metrics` and `feature_metrics` without changing a single line of your evaluation code.
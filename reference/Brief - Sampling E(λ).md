# Sampling the Symmetrized Value Objective $E(\lambda)$

The symmetrized $\lambda$-value objective balances the magnitude of value approximation errors with a multi-step Dirichlet graph smoothness penalty.

---

## 1. Expected Forms of $E(\lambda)$

Let $P$ be a row-stochastic transition probability matrix, $D = \text{diag}(\mu)$ be the diagonal stationary distribution matrix satisfying $\mu^\top P = \mu^\top$, $A = D(I - \gamma P)$, and $L = (I - \gamma \lambda P)^{-1}$. The error vector is defined as $e = V - v_\theta$.

### Matrix Definition and Symmetrized Gradient
The scalar objective $E(\lambda)$ is defined as:

$$\begin{align} E(\lambda) &= \frac{1}{2} e^\top A L e \\ &= \frac{1}{2} e^\top D(I - \gamma P)(I - \gamma \lambda P)^{-1} e \end{align}$$

Gradient descent on $E(\lambda)$ produces the exact parameter update:

$$\begin{align} \nabla_\theta E(\lambda) = -\frac{1}{2} J^\top (AL + L^\top A^\top) e \end{align}$$

where $J = \nabla_\theta v_\theta$ is the Jacobian matrix of value predictions with respect to parameters $\theta$.

### Multi-Step Dirichlet Expansion Form
Using the identity $(I - \gamma P)L = I - \gamma(1 - \lambda) P L$ and the graph Laplacian decomposition $\sum_{i,j} \mu_i [P^m]_{ij} e_i e_j = \Vert{}e\Vert{}_D^2 - \frac{1}{2} \sum_{i,j} \mu_i [P^m]_{ij} (e_i - e_j)^2$:

$$\begin{align} E(\lambda) &= \frac{1 - \gamma}{1 - \gamma \lambda} \Vert{}e\Vert{}_D^2 + \frac{\gamma(1 - \lambda)}{2} \sum_{k=0}^\infty (\gamma \lambda)^k \sum_{i,j} \mu_i [P^{k+1}]_{ij} (e_i - e_j)^2 \\ &= (1 - \tilde{\gamma}) \Vert{}e\Vert{}_D^2 + \frac{\tilde{\gamma}}{2} \sum_{i,j} \mu_i [P_\lambda]_{ij} (e_i - e_j)^2 \end{align}$$

where $P_\lambda = (1 - \gamma \lambda) P L = (1 - \gamma \lambda) \sum_{k=0}^\infty (\gamma \lambda)^k P^{k+1}$ is a valid row-stochastic transition matrix, and $\tilde{\gamma} = \frac{\gamma(1 - \lambda)}{1 - \gamma \lambda}$ is the effective discount factor.

---

## Method 1: FVI Stop-Gradient Form (Forward & Backward Error Traces)

### Description
This method evaluates sample errors $\hat{e}_t = G_t - v(s_t)$ along on-policy rollouts and accumulates downstream errors ($\bar{e}_{>t}$) and upstream errors ($\bar{e}_{<t}$) using discounted eligibility scans. Treating these traces as fixed constants via a stop-gradient operator yields a scalar regression target:

$$\begin{align} v_t \leftarrow G_t - \frac{\gamma(1 - \lambda)}{2} (\bar{e}_{>t} + \bar{e}_{<t}) \end{align}$$

### Unbiased Gradient Property
* Under stationary distribution $\mu$, the expectation of the product of Jacobian $\nabla_\theta v_t$ and the forward trace $\bar{e}_{>t} = \sum_{k=0}^\infty (\gamma \lambda)^k e_{t+k+1}$ satisfies $\mathbb{E}[\nabla_\theta v_t \bar{e}_{>t}] = J^\top D P L e$.
* Because reverse transitions in a stationary chain satisfy $\mathbb{P}(s_{t-k-1}=j \mid s_t=i) = \frac{\mu_j [P^{k+1}]_{ji}}{\mu_i}$, the backward trace expectation satisfies $\mathbb{E}[\nabla_\theta v_t \bar{e}_{<t}] = J^\top L^\top P^\top D e = J^\top (D P L)^\top e$.
* Summing forward and backward expectations yields the symmetrized operator $\mathbb{E}[\nabla_\theta v_t (\bar{e}_{>t} + \bar{e}_{<t})] = J^\top (D P L + L^\top P^\top D) e$.
* Combining this with the anchor gradient $\nabla_\theta (\frac{1-\gamma}{1-\gamma\lambda} \Vert{}e\Vert{}_D^2)$ reconstructs $-\frac{1}{2} J^\top (AL + L^\top A^\top) e \equiv \nabla_\theta E(\lambda)$.

### Pseudocode
```python
def fvi_two_sided_targets(traj_batch, gamma, lmbda):
    """
    Computes regression targets using forward and backward error traces.
    traj_batch: obs (T, B, ...), returns (T, B), values (T, B), dones (T, B)
    """
    errors = traj_batch.returns - traj_batch.values
    gl = gamma * lmbda

    # 1. Forward Trace: Accumulate future errors backwards in time
    def _backward_pass(trace, transition):
        done, e_next = transition
        mask = 1.0 - done
        trace = e_next + gl * mask * trace
        return trace, trace

    # Shift errors: step t receives e_{t+1} as immediate lookahead
    e_next = roll_forward(errors, shift=-1)
    _, forward_traces = scan(
        _backward_pass, zeros_like(errors[0]), (traj_batch.dones, e_next), reverse=True
    )

    # 2. Backward Trace: Accumulate past errors forwards in time
    def _forward_pass(trace, transition):
        done, e_prev = transition
        mask = 1.0 - done
        trace = e_prev + gl * mask * trace
        return trace, trace

    # Shift errors: step t receives e_{t-1} as immediate lookbehind
    e_prev = roll_backward(errors, shift=1)
    _, backward_traces = scan(
        _forward_pass, zeros_like(errors[0]), (traj_batch.dones, e_prev), reverse=False
    )

    # 3. Construct scalar regression target
    coeff = 0.5 * gamma * (1.0 - lmbda)
    smoothing_correction = coeff * (forward_traces + backward_traces)
    targets = traj_batch.returns - stop_gradient(smoothing_correction)

    return targets
```

---

## Method 2: Full Autodiff Moment Scan

### Description
Pairwise squared differences $(e_t - e_{t+k+1})^2$ expand into algebraic moments:

$$\begin{align} (e_t - e_{t+k+1})^2 = e_t^2 - 2 e_t e_{t+k+1} + e_{t+k+1}^2 \end{align}$$

A single backward scan computes the zero-th, first, and second geometric moments of downstream errors. This evaluates the multi-step Dirichlet energy in $O(T)$ operations while letting automatic differentiation backpropagate through both $v(s_t)$ and future states $v(s_{t+k+1})$.

### Unbiased Estimate Property
* Consecutive on-policy states satisfy $s_t \sim \mu$ and $s_{t+k+1} \sim P^{k+1}(\cdot \mid s_t)$.
* Because $\mathbb{E}[(G_t - v_t) - (G_{t+k+1} - v_{t+k+1}) \mid s_t, s_{t+k+1}] = e(s_t) - e(s_{t+k+1})$, the sample difference satisfies $\mathbb{E}[(\hat{e}_t - \hat{e}_{t+k+1})^2] = \sum_{i,j} \mu_i [P^{k+1}]_{ij} (e_i - e_j)^2$.
* Summing across horizons $k$ matches the multi-step Dirichlet expansion:

$$\begin{align} \mathbb{E}\left[ \sum_{k=0}^{T - t - 1} (\gamma \lambda)^k (\hat{e}_t - \hat{e}_{t+k+1})^2 \right] = \sum_{k=0}^\infty (\gamma \lambda)^k \sum_{i,j} \mu_i [P^{k+1}]_{ij} (e_i - e_j)^2 \end{align}$$

* Differentiating this sampled objective directly produces an unbiased estimate of $\nabla_\theta E(\lambda)$.

### Pseudocode
```python
def full_autodiff_dirichlet_loss(traj_batch, value_net, params, gamma, lmbda):
    """
    Computes differentiable E(lambda) loss using backward moment traces.
    """
    values = value_net.apply(params, traj_batch.obs)
    errors = traj_batch.returns - values
    gl = gamma * lmbda

    def _moment_step(traces, transition):
        w0, w1, w2 = traces
        done, e_curr, e_next = transition
        mask = 1.0 - done

        # Advance geometric traces backwards
        w0_t = 1.0 + gl * mask * w0
        w1_t = e_next + gl * mask * w1
        w2_t = (e_next ** 2) + gl * mask * w2

        # Quadratic expansion: w0 * e_t^2 - 2 * w1 * e_t + w2
        dirichlet_t = mask * (w0_t * (e_curr ** 2) - 2.0 * e_curr * w1_t + w2_t)
        return (w0_t, w1_t, w2_t), dirichlet_t

    e_next = roll_forward(errors, shift=-1)
    init_traces = (zeros_like(errors[0]), zeros_like(errors[0]), zeros_like(errors[0]))
    
    _, dirichlet_terms = scan(
        _moment_step, init_traces, (traj_batch.dones, errors, e_next), reverse=True
    )

    magnitude_loss = ((1.0 - gamma) / (1.0 - gl)) * mean(errors ** 2)
    laplacian_loss = (0.5 * gamma * (1.0 - lmbda)) * mean(dirichlet_terms)
    
    return magnitude_loss + laplacian_loss
```

---

## Method 3: Geometric Jump Sampling ($P_\lambda$)

### Description
This method samples transitions directly from the compound stochastic matrix $P_\lambda = (1 - \gamma \lambda) \sum_{k=0}^\infty (\gamma \lambda)^k P^{k+1}$. At step $t$, a lookahead skip $K \in \{0, 1, 2, \dots\}$ is drawn from a geometric distribution with termination parameter $(1 - \gamma \lambda)$. The loss evaluates only the sampled transition $(s_t, s_{t+K+1})$ scaled by $\tilde{\gamma} = \frac{\gamma(1 - \lambda)}{1 - \gamma \lambda}$.

### Unbiased Estimate Property
* The probability of drawing lookahead length $K = k$ is $\mathbb{P}(K = k) = (1 - \gamma \lambda)(\gamma \lambda)^k$.
* Marginalizing over jump lengths gives transition probabilities $\sum_{k=0}^\infty \mathbb{P}(K = k) [P^{k+1}]_{ij} = (1 - \gamma \lambda) \sum_{k=0}^\infty (\gamma \lambda)^k [P^{k+1}]_{ij} \equiv [P_\lambda]_{ij}$.
* Expectation over both the stationary trajectory and the random geometric skip satisfies:

$$\begin{align} \mathbb{E}_{K \sim \text{Geom}}\left[ (\hat{e}(s_t) - \hat{e}(s_{t+K+1}))^2 \right] = \sum_{i,j} \mu_i [P_\lambda]_{ij} (e_i - e_j)^2 \end{align}$$

* Weighting by $\frac{\tilde{\gamma}}{2}$ and adding the anchor term $(1 - \tilde{\gamma})\hat{e}(s_t)^2$ forms an unbiased sample estimate of $E(\lambda)$.

### Pseudocode
```python
def geometric_jump_sampling_loss(traj_batch, value_net, params, gamma, lmbda, rng_key):
    """
    Computes E(lambda) by sampling random geometric horizon jumps from P_lambda.
    """
    T, B = traj_batch.returns.shape[:2]
    gl = gamma * lmbda
    tilde_gamma = (gamma * (1.0 - lmbda)) / (1.0 - gl)

    # 1. Sample jump lengths K ~ Geometric(p = 1 - gl) with K >= 0
    u = random_uniform(rng_key, shape=(T, B))
    jumps = floor(log(1.0 - u) / log(gl)).astype(int)

    # 2. Compute destination indices and mask episode boundaries
    target_idx = minimum(arange(T)[:, None] + jumps + 1, T - 1)
    
    # Mask out jumps that cross an episode reset
    done_cumsum = cumsum(traj_batch.dones, axis=0)
    has_terminal = (take_along_axis(done_cumsum, target_idx, axis=0) - done_cumsum) > 0
    valid_mask = (1.0 - traj_batch.dones) * (1.0 - has_terminal.astype(float))

    # 3. Evaluate predictions and errors
    values = value_net.apply(params, traj_batch.obs)
    errors = traj_batch.returns - values
    errors_jump = take_along_axis(errors, target_idx, axis=0)

    # 4. Form sampled loss
    magnitude_loss = (1.0 - tilde_gamma) * mean(errors ** 2)
    laplacian_loss = 0.5 * tilde_gamma * mean(valid_mask * (errors - errors_jump) ** 2)

    return magnitude_loss + laplacian_loss
```

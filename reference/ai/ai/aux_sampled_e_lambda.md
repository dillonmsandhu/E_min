# Sampled $E(\lambda)$: Expected Formulation, Double-Sampling Resolution via $v_{\text{aux}}$, and Episodic Buffer Debiasing

This document presents a self-contained mathematical derivation and practical implementation guide for **Sampled $E(\lambda)$** (Bellman-Smoothed Value Regression with trace parameter $\lambda \in [0, 1]$). We begin from the stationary expected quadratic form, derive its exact gradient to pinpoint where multi-step double sampling arises, introduce the auxiliary critic $v_{\text{aux}}$ to debias the backward gradient branch, and finally address the practical subtleties of episodic resets, absorbing terminal states, and finite rollout buffer boundaries.

---

## 1. Expected Formulation of $E(\lambda)$ on a Compound Markov Chain

### 1.1 From $\text{TD}(\lambda)$ to the Symmetrized Energy $E(\lambda)$
Let $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ be a Markov reward process induced by policy $\pi$, with state transition matrix $P$, stationary distribution $\mu$, diagonal state-weighting matrix $D = \text{diag}(\mu)$, true value function $V \in \mathbb{R}^{|\mathcal{S}|}$, parametric approximation $v_\theta \in \mathbb{R}^{|\mathcal{S}|}$, and value prediction error $e \doteq V - v_\theta$.

In standard $\text{TD}(\lambda)$, let $L \doteq (I - \gamma\lambda P)^{-1} = \sum_{k=0}^\infty (\gamma\lambda)^k P^k$ denote the resolvent trace operator and $\delta \doteq R + \gamma P v_\theta - v_\theta = (I - \gamma P)e$ denote the one-step expected Bellman residual. The expected semi-gradient $\text{TD}(\lambda)$ update follows:
$$\dot{\theta}_{\text{TD}(\lambda)} = J_\theta^\top D L \delta = J_\theta^\top D (I - \gamma P) L e = J_\theta^\top A_\lambda e$$
where $J_\theta = \nabla_\theta v_\theta$ is the value Jacobian and $A_\lambda \doteq D(I - \gamma P)(I - \gamma\lambda P)^{-1}$. Because $A_\lambda$ is asymmetric under general Markov chains, $\text{TD}(\lambda)$ is not a true gradient flow and can converge to biased fixed points or diverge under nonlinear function approximation.

We instead define the **Bellman-Smoothed Objective $E(\lambda)$** as the quadratic form induced by $A_\lambda$:
$$E(\lambda) \doteq \frac{1}{2} e^\top A_\lambda e = \frac{1}{2} e^\top D (I - \gamma P)(I - \gamma\lambda P)^{-1} e = \frac{1}{2} e^\top S_\lambda e$$
where $S_\lambda \doteq \frac{1}{2}(A_\lambda + A_\lambda^\top)$ is symmetric and positive definite for all $\gamma \in [0, 1)$ and $\lambda \in [0, 1]$. Notice the boundary cases:
* As $\lambda \to 1$, $A_\lambda \to D$, so $E(1) = \frac{1}{2}\|e\|_D^2$ recovers standard Monte Carlo (MC) regression.
* At $\lambda = 0$, $A_0 = D(I - \gamma P)$, recovering the one-step Bellman-smoothed objective $E(0)$.

### 1.2 Reduction to $E(0)$ via the Compound Transition Operator $P_\lambda$
Using the algebraic identity $(I - \gamma P) = (I - \gamma\lambda P) - \gamma(1 - \lambda)P$, we can factor the resolvent product $(I - \gamma P)(I - \gamma\lambda P)^{-1}$ as:
$$(I - \gamma P)(I - \gamma\lambda P)^{-1} = I - \gamma(1 - \lambda)P(I - \gamma\lambda P)^{-1} = (1 - \tilde{\gamma})I + \tilde{\gamma}(I - P_\lambda)$$
where we define the **effective discount weight $\tilde{\gamma}$** and **compound transition matrix $P_\lambda$**:
$$\tilde{\gamma} \doteq \frac{\gamma(1 - \lambda)}{1 - \gamma\lambda}, \qquad 1 - \tilde{\gamma} = \frac{1 - \gamma}{1 - \gamma\lambda}, \qquad P_\lambda \doteq (1 - \gamma\lambda) P (I - \gamma\lambda P)^{-1} = (1 - \gamma\lambda) \sum_{k=0}^\infty (\gamma\lambda)^k P^{k+1}$$

Crucially, $P_\lambda$ has a direct probabilistic interpretation:
1. **Row-Stochasticity**: If $P\mathbf{1} = \mathbf{1}$, then $P_\lambda \mathbf{1} = (1 - \gamma\lambda)\sum_{k=0}^\infty (\gamma\lambda)^k \mathbf{1} = \mathbf{1}$.
2. **Stationarity Preservation**: If $\mu^\top P = \mu^\top$, then $\mu^\top P_\lambda = \mu^\top$.
3. **Geometric Horizon Sampling**: Sampling a destination state $s_j \sim P_\lambda(\cdot \mid s_i)$ is equivalent to drawing a lookahead skip $K \sim \text{Geometric}(1 - \gamma\lambda)$ on $\{0, 1, 2, \dots\}$ with $\mathbb{P}(K = k) = (1 - \gamma\lambda)(\gamma\lambda)^k$, setting the jump step size to $m = K + 1$, and advancing $m$ steps along the Markov chain: $s_j = s_{i+m} \sim P^m(\cdot \mid s_i)$.

Under a stationary distribution ($\mu^\top P_\lambda = \mu^\top$), for any $m \ge 1$ we have the identity:
$$e^\top D P^m e = \|e\|_D^2 - \frac{1}{2} \sum_{i,j} \mu_i [P^m]_{ij} (e_i - e_j)^2$$
Applying this identity across the geometric mixture $P_\lambda$ decomposes $E(\lambda)$ into a **magnitude anchor** plus a **multi-step Dirichlet smoothness energy $\mathcal{E}_\lambda(e)$**:
$$E(\lambda) = \frac{1}{2}(1 - \tilde{\gamma})\|e\|_D^2 + \frac{1}{2}\tilde{\gamma} \mathcal{E}_\lambda(e), \qquad \mathcal{E}_\lambda(e) \doteq \frac{1}{2} \sum_{i,j} \mu_i [P_\lambda]_{ij} (e_i - e_j)^2 = \frac{1}{2} \mathbb{E}_{\substack{s_i \sim \mu \\ K \sim \text{Geom}(1-\gamma\lambda) \\ s_j \sim P^{K+1}(\cdot \mid s_i)}} \Big[ (e_i - e_j)^2 \Big]$$

---

## 2. Gradient Decomposition, the Double-Sampling Issue, and $v_{\text{aux}}$

### 2.1 Forward and Backward Gradient Decomposition
Since $e_i = V(s_i) - v_\theta(s_i)$, its parameter gradient is $\nabla_\theta e_i = -\nabla_\theta v_\theta(s_i)$ (or $\nabla_\theta e = -J_\theta$). Differentiating $E(\lambda)$ with respect to $\theta$ yields:
$$\nabla_\theta E(\lambda) = -(1 - \tilde{\gamma}) J_\theta^\top D e + \frac{1}{2}\tilde{\gamma} \nabla_\theta \mathcal{E}_\lambda(e) = -J_\theta^\top S_\lambda e$$

Writing $\mathcal{E}_\lambda(e) = \sum_i \mu_i \mathcal{E}_{\lambda, i}(e)$, the gradient of the smoothness energy at start state $s_i$ splits into two distinct branches depending on whether the value gradient $\nabla_\theta v_\theta$ is evaluated at the **jumped destination state $s_j = s_{i+m}$ (Term 1 / Backward Branch)** or the **current start state $s_i$ (Term 2 / Forward Branch)**:
$$\begin{aligned}
\nabla_\theta \mathcal{E}_{\lambda, i}(e) &= \nabla_\theta \frac{1}{2} \sum_j [P_\lambda]_{ij} (e_i - e_j)^2 \\
&= \sum_j [P_\lambda]_{ij} (e_i - e_j) \big(\nabla_\theta v_\theta(s_j) - \nabla_\theta v_\theta(s_i)\big) \\
&= \underbrace{\sum_j [P_\lambda]_{ij} (e_i - e_j) \nabla_\theta v_\theta(s_j)}_{\text{Term (1): Backward / Destination-State Gradient}} - \underbrace{\sum_j [P_\lambda]_{ij} (e_i - e_j) \nabla_\theta v_\theta(s_i)}_{\text{Term (2): Forward / Start-State Gradient}}
\end{aligned}$$

In global matrix-vector form across $s_i \sim \mu$, this decomposition is:
$$\nabla_\theta \mathcal{E}_\lambda(e) = \underbrace{J_\theta^\top (P_\lambda^\top - I) D e}_{\text{Term (1)}} - \underbrace{J_\theta^\top D (I - P_\lambda) e}_{\text{Term (2)}}$$
Geometrically, unlike standard regression where $\nabla_\theta v_\theta(s_i)$ only responds to $e_i$, $\nabla_\theta \mathcal{E}_\lambda(e)$ contains cross-state inner products $J_\theta^\top D P_\lambda e$ and $J_\theta^\top P_\lambda^\top D e$. Term (2) adjusts $v_\theta(s_i)$ so its error $e_i$ matches the expected $m$-step successor error $[P_\lambda e]_i$, while Term (1) uses the time-reversed adjoint operator $P_\lambda^\top D$ to adjust $v_\theta(s_j)$ so its error $e_j$ matches its expected $m$-step predecessor error.

### 2.2 Pinpointing the Multi-Step Double-Sampling Issue
In practice, the true value function $V$ is unknown; along a single trajectory $(s_t, r_t, s_{t+1}, \dots)$, we only have empirical return estimates $\hat{e}_t = G_t - v_\theta(s_t)$ where $\mathbb{E}[G_t \mid s_t] = V(s_t)$. Suppose we sample a jump length $m = K + 1$ and transition along a single trajectory from $s_i$ to $s_j = s_{i+m}$. Let us examine each branch under single-trajectory sampling:

1. **Term (2) (Forward Branch at $s_i$) is 100% Unbiased**:
   $$\text{Term (2)} = \mathbb{E}_{s_j \sim P_\lambda(\cdot \mid s_i)}\big[e_i - e_j \mid s_i\big] \nabla_\theta v_\theta(s_i)$$
   Because $\nabla_\theta v_\theta(s_i)$ is a **deterministic function of $s_i$**, it is completely fixed *before* the jump length $K$ or the trajectory $s_i \to s_j$ is realized. Consequently, multiplying the single-trajectory error difference $(\hat{e}_i - \hat{e}_j)$ by $\nabla_\theta v_\theta(s_i)$ is completely unbiased:
   $$\mathbb{E}\Big[(\hat{e}_i - \hat{e}_j)\nabla_\theta v_\theta(s_i) \,\Big|\, s_i\Big] = \mathbb{E}\big[\hat{e}_i - \hat{e}_j \mid s_i\big] \nabla_\theta v_\theta(s_i) = \sum_j [P_\lambda]_{ij}(e_i - e_j)\nabla_\theta v_\theta(s_i)$$
   Moreover, because $\hat{e}_i - \hat{e}_j = \hat{\delta}_i^{(m)} - (1 - \gamma^m)\hat{e}_j$ on the same trajectory, the noisy future return tail $G_j$ beyond $s_j$ automatically cancels up to a factor of $(1 - \gamma^m)$.

2. **Term (1) (Backward Branch at $s_j$) Suffers from Multi-Step Double Sampling**:
   Expanding Term (1) into its two expectations over $s' = s_{i+m} \sim P_\lambda(\cdot \mid s_i)$ gives:
   $$\text{Term (1)} = e_i \, \mathbb{E}_{s' \sim P_\lambda(\cdot \mid s_i)}\!\big[\nabla_\theta v_\theta(s')\big] - \mathbb{E}_{s' \sim P_\lambda(\cdot \mid s_i)}\!\big[e(s') \nabla_\theta v_\theta(s')\big]$$
   * **Why the second sub-term is unbiased**: Conditioned on reaching destination state $s_j = s_{i+m}$, $\nabla_\theta v_\theta(s_j)$ is fixed, while the rollout randomness in $\hat{e}_j = G_j - v_\theta(s_j)$ occurs *strictly after* step $j$. By the tower property, $\mathbb{E}[\hat{e}_j \nabla_\theta v_\theta(s_j) \mid s_i] = \mathbb{E}_{s'}[e(s')\nabla_\theta v_\theta(s')]$.
   * **Why the first sub-term is biased**: Although $e_i = V(s_i) - v_\theta(s_i)$ is a fixed constant at $s_i$, estimating $e_i$ via the empirical return $\hat{e}_i = G_i - v_\theta(s_i) = \sum_{t=0}^{m-1} \gamma^t r_{i+t} + \gamma^m G_j - v_\theta(s_i)$ requires taking an expectation over the very same $m$-step trajectory $s_i \to s_j$ used to sample $s_j$: $e_i = \mathbb{E}_{s', G'}[\hat{e}_i \mid s_i]$. Thus, the first sub-term is a **product of two expectations** over $s' \sim P_\lambda(\cdot \mid s_i)$:
     $$e_i \, \mathbb{E}_{s'}\!\big[\nabla_\theta v_\theta(s')\big] = \mathbb{E}_{s', G'}\!\big[\hat{e}_i \mid s_i\big] \, \mathbb{E}_{s'}\!\big[\nabla_\theta v_\theta(s') \mid s_i\big]$$
     When estimated from a single trajectory, $\hat{e}_i$ and $\nabla_\theta v_\theta(s_j)$ share all $m = K + 1$ steps of environment transition stochasticity, policy action stochasticity, and jump-length randomness $K$. Since $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] + \text{Cov}(X, Y)$, a single-trajectory sample introduces an explicit **double-sampling covariance bias**:
     $$\mathbb{E}_{K, s_j, G_j}\!\big[\hat{e}_i \nabla_\theta v_\theta(s_j) \mid s_i\big] = e_i \mathbb{E}_{s'}\!\big[\nabla_\theta v_\theta(s') \mid s_i\big] + \text{Cov}_{K, s_j}\!\left(\sum_{t=0}^{K}\gamma^t r_{i+t} + \gamma^{K+1}V(s_j),\, \nabla_\theta v_\theta(s_j) \,\middle|\, s_i\right)$$
     Notice that this covariance bias does not vanish even when $v_\theta = V$ ($e = 0$).

### 2.3 Resolving Double Sampling with an Auxiliary Value Network $v_{\text{aux}}$
To eliminate the correlation between the error estimate at $s_i$ and the destination gradient $\nabla_\theta v_\theta(s_j)$ without resetting the simulator, we introduce an auxiliary critic $v_{\text{aux}, \phi}(s)$ trained via standard regression on the return targets $G_t$:
$$\mathcal{L}_{\text{aux}}(\phi) = \frac{1}{2} \mathbb{E}_{s_t \sim \mu}\Big[\big(\text{sg}(G_t) - v_{\text{aux}, \phi}(s_t)\big)^2\Big]$$
and define the auxiliary proxy error:
$$e_{\text{aux}}(s) \doteq \text{sg}\big(v_{\text{aux}, \phi}(s)\big) - v_\theta(s)$$

#### Why Use $e_{\text{aux}}$ for ALL of Term (1) (Rather Than Just the First Sub-Term)?
Because the second sub-term $\mathbb{E}_{s'}[e(s')\nabla_\theta v_\theta(s')]$ can be estimated unbiasedly by $\hat{e}_j \nabla_\theta v_\theta(s_j)$, one might be tempted to substitute $e_{\text{aux}}(s_i)$ only into the first sub-term $e_{\text{aux}}(s_i)\nabla_\theta v_\theta(s_j)$ while keeping the Monte Carlo return $\hat{e}_j$ in the second sub-term. However, doing so breaks the variance cancellation between $e_i$ and $e_j$: in $(e_i - e_j)$, shared trajectory noise cancels, whereas in $\big(e_{\text{aux}}(s_i) - \hat{e}_j\big)\nabla_\theta v_\theta(s_j)$, the deterministic $e_{\text{aux}}(s_i)$ cannot cancel the random return tail $G_j$ inside $\hat{e}_j$, exposing Term (1) to the full $\mathcal{O}((1-\gamma)^{-2})$ Monte Carlo variance.

Instead, we use $e_{\text{aux}}$ for the **entirety of Term (1)** while keeping the empirical return error $\hat{e} = G - v_\theta$ for **Term (2)** and the magnitude term:
$$\hat{g}_i(\lambda) = -(1 - \tilde{\gamma})\hat{e}_i \nabla_\theta v_\theta(s_i) + \frac{1}{2}\tilde{\gamma} \Big[ \underbrace{\big(e_{\text{aux}}(s_i) - e_{\text{aux}}(s_j)\big)\nabla_\theta v_\theta(s_j)}_{\widehat{\text{Term (1)}}\text{ (Backward via } v_{\text{aux}}\text{)}} - \underbrace{(\hat{e}_i - \hat{e}_j)\nabla_\theta v_\theta(s_i)}_{\widehat{\text{Term (2)}}\text{ (Forward via rollout }\hat{e}\text{)}} \Big]$$

#### Expected Gradient Dynamics Under $v_{\text{aux}}$
Let $\epsilon_{\text{aux}}(s) \doteq v_{\text{aux}, \phi}(s) - V(s)$ denote the approximation residual of the auxiliary critic, so that $e_{\text{aux}} = v_{\text{aux}} - v_\theta = (V - v_\theta) + (v_{\text{aux}} - V) = e + \epsilon_{\text{aux}}$. Because $e_{\text{aux}}(s_i)$ is a deterministic function of $s_i$ (marginalizing over all future action and transition stochasticity prior to sampling $K$ and $s_j$), taking the expectation of $\hat{g}_i(\lambda)$ yields:
$$\begin{aligned}
\mathbb{E}_{\mu, P_\lambda}\big[\hat{g}_i(\lambda)\big] &= -(1 - \tilde{\gamma}) J_\theta^\top D e + \frac{1}{2}\tilde{\gamma} J_\theta^\top (P_\lambda^\top - I) D (e + \epsilon_{\text{aux}}) - \frac{1}{2}\tilde{\gamma} J_\theta^\top D (I - P_\lambda) e \\
&= -J_\theta^\top \underbrace{\left[ D - \frac{\tilde{\gamma}}{2}(D P_\lambda + P_\lambda^\top D) \right]}_{S_\lambda} e + \frac{1}{2}\tilde{\gamma} J_\theta^\top (P_\lambda^\top - I) D \epsilon_{\text{aux}} \\
&= \nabla_\theta E(\lambda) + \frac{1}{2}\tilde{\gamma} J_\theta^\top (P_\lambda^\top - I) D \epsilon_{\text{aux}}
\end{aligned}$$
This decomposition guarantees three critical properties:
1. **Exact Symmetric Preconditioner $S_\lambda$**: Because $e_{\text{aux}}(s_i) - e_{\text{aux}}(s_j) = (v_{\text{aux}}(s_i) - v_{\text{aux}}(s_j)) - (v_\theta(s_i) - v_\theta(s_j))$ contains the exact difference of $v_\theta$, the matrix multiplying the true error $e$ is the exact symmetric positive-definite matrix $-J_\theta^\top S_\lambda$ regardless of any inaccuracy in $v_{\text{aux}}$.
2. **Zero Per-Transition Rollout Variance in Term (1)**: Both $e_{\text{aux}}(s_i)$ and $e_{\text{aux}}(s_j)$ are deterministic evaluations of the networks at $s_i$ and $s_j$.
3. **Difference-Filtering of Auxiliary Error**: Any constant offset or low-frequency error in $\epsilon_{\text{aux}}$ across $m$-step transitions ($\epsilon_{\text{aux}}(s_i) \approx \epsilon_{\text{aux}}(s_j)$) is annihilated by the difference operator $(P_\lambda^\top - I)D$.

### 2.4 Differentiable Stop-Gradient Surrogate Loss
To implement $\hat{g}_i(\lambda)$ inside an automatic differentiation framework (such as JAX or PyTorch) as a single scalar loss without writing custom vector-Jacobian products, we split the Dirichlet squared difference into forward and backward surrogate terms using `stop_gradient` ($\text{sg}$):
* **Forward surrogate (differentiates $v_\theta(s_i)$ only)**:
  $$d_{\text{fwd}} = \hat{e}_i - \text{sg}(\hat{e}_j) \implies \nabla_\theta \left[\frac{1}{2} d_{\text{fwd}}^2\right] = -(\hat{e}_i - \hat{e}_j)\nabla_\theta v_\theta(s_i)$$
* **Backward surrogate (differentiates $v_\theta(s_j)$ only)**:
  $$d_{\text{bwd}} = \text{sg}(e_{\text{aux}, i}) - e_{\text{aux}, j} \implies \nabla_\theta \left[\frac{1}{2} d_{\text{bwd}}^2\right] = +(e_{\text{aux}, i} - e_{\text{aux}, j})\nabla_\theta v_\theta(s_j)$$

Summing $\frac{1}{2}\tilde{\gamma} \cdot \frac{1}{2}\big(d_{\text{fwd}}^2 + d_{\text{bwd}}^2\big)$ reproduces the exact desired gradient $\frac{1}{2}\tilde{\gamma}\big(\widehat{\text{Term (1)}} - \widehat{\text{Term (2)}}\big)$.

---

## 3. Episodic Resets, Absorbing Boundaries, and Finite Rollout Buffers

In practical reinforcement learning, data is collected in finite rollout buffers of shape $(T, B)$ from episodic environments that reset upon termination or timeout. Naive application of the stationary formula above introduces three distinct sources of bias. Here we analyze each issue and derive its exact correction.

### 3.1 Issue 1: Non-Stationary Probability Mass at Episode Resets ($\mu^\top P_\lambda \neq \mu^\top$)

#### The Problem
In an episodic MDP, let $P$ denote the true episodic transition matrix (which transitions into a zero-value absorbing terminal state $s_{\text{term}}$ with probability $p_{i, T} = P(s_{\text{term}} \mid s_i)$), and let $P_c$ denote the continuing simulator transition matrix (which immediately resets to an initial state distribution $\mu_0$ upon termination):
$$P_c = P + \mathbf{p}_T (\mu_0 - \mathbf{1}_{\text{term}})^\top$$
The empirical state distribution $\mu$ sampled in a rollout buffer is stationary with respect to the continuing simulator ($\mu^\top P_c = \mu^\top$), which means **$\mu$ is NOT stationary with respect to the true MDP transition matrix $P$**:
$$\mu^\top (I - P) = \mu^\top (P_c - P) = (\mu^\top \mathbf{p}_T)(\mu_0 - \mathbf{1}_{\text{term}})^\top = \nu_{\text{term}} (\mu_0 - \mathbf{1}_{\text{term}})^\top$$
where $\nu_{\text{term}} \doteq \sum_i \mu_i p_{i, T}$ is the per-step episode termination rate.

Because $\sum_i \mu_i [P_\lambda]_{ij} = [\mu^\top P_\lambda]_j \neq \mu_j$, completing the square for $e^\top D(I - P_\lambda)e$ leaves an extra diagonal boundary term:
$$\begin{aligned}
e^\top D (I - P_\lambda) e &= \sum_{i,j} \mu_i [P_\lambda]_{ij} \left(\frac{1}{2} e_i^2 - e_i e_j + \frac{1}{2} e_j^2\right) + \frac{1}{2} \sum_j \big(\mu_j - [\mu^\top P_\lambda]_j\big) e_j^2 \\
&= \mathcal{E}_\lambda(e) + \frac{1}{2} e^\top \big(D - \text{diag}[\mu^\top P_\lambda]\big) e
\end{aligned}$$
If this boundary term is omitted, probability mass leaks out of the Dirichlet energy at terminal states without being grounded at the start states $\mu_0$.

#### The Exact Solution: Geometrically Decaying Reset Trace ($\mathcal{L}_{\text{start}}$)
Using the identity $I - P_\lambda = (I - P)(I - \gamma\lambda P)^{-1}$, we can evaluate the row vector $\mu^\top (I - P_\lambda)$ in closed form:
$$\mu^\top (I - P_\lambda) = \mu^\top (I - P)(I - \gamma\lambda P)^{-1} = \nu_{\text{term}} (\mu_0 - \mathbf{1}_{\text{term}})^\top \sum_{k=0}^\infty (\gamma\lambda)^k P^k$$
Since $e(s_{\text{term}}) \equiv 0$, multiplying by $\frac{1}{2}\tilde{\gamma}$ (from the outer $\frac{1}{2}\tilde{\gamma} e^\top D(I - P_\lambda)e$ objective) gives the exact start-state grounding loss:
$$\mathcal{L}_{\text{start}}(\theta) = \frac{\tilde{\gamma}}{4} e^\top \big(D - \text{diag}[\mu^\top P_\lambda]\big) e = \frac{\tilde{\gamma}}{4} \nu_{\text{term}} \sum_{k=0}^\infty (\gamma\lambda)^k \mathbb{E}_{s_0 \sim \mu_0,\, s_k \sim P^k(\cdot \mid s_0)}\!\Big[ e(s_k)^2 \, \mathbb{I}(s_k \neq s_{\text{term}}) \Big]$$

**Vectorized Implementation**:
Along any simulator rollout, a new episode starts at $\mu_0$ immediately after a true terminal transition, which occurs at the empirical rate $\hat{\nu}_{\text{term}}$. Therefore, if we track the integer step counter $k_{t,b} \in \{0, 1, 2, \dots\}$ counting the number of steps elapsed since the most recent true episode termination, taking the simple average of $(\gamma\lambda)^{k_{t,b}} \hat{e}_{t,b}^2$ across the rollout buffer *automatically* includes the factor $\hat{\nu}_{\text{term}}$:
$$\mathcal{L}_{\text{start}}(\theta) = \frac{1}{4}\tilde{\gamma} \cdot \frac{1}{T_{\text{start}} B} \sum_{t=0}^{T_{\text{start}}-1} \sum_{b=0}^{B-1} (\gamma\lambda)^{k_{t,b}} \hat{e}_{t,b}^2$$
* When $\lambda = 0$, $(\gamma\lambda)^{k_{t,b}} = \mathbb{I}(k_{t,b} = 0) = \mathbb{I}_{\text{start}}(t, b)$, recovering the exact $E(0)$ start-state grounding term.
* Notice the coefficient $\frac{1}{4}\tilde{\gamma} = 0.25\tilde{\gamma}$: because $\hat{e}_{t,b}^2$ is a single squared term (whereas $d_{\text{fwd}}^2 + d_{\text{bwd}}^2$ sums two squared terms), differentiating $\frac{1}{4}\tilde{\gamma}\hat{e}_{t,b}^2$ brings down a factor of $2$, yielding the exact gradient $-\frac{1}{2}\tilde{\gamma}\nu_{\text{term}}(\gamma\lambda)^k e_t \nabla_\theta v_\theta(s_t)$.

---

### 3.2 Issue 2: Multi-Step Jumps Crossing Episode Boundaries (Absorbing Terminals vs. Timeouts)

#### The Problem
When sampling a multi-step jump $m = K + 1$ from state $s_t$, the lookahead window $[t, t + m]$ may cross an episode boundary (`done == True`) at some intermediate step $t_{\text{done}} \in [t, t + m - 1]$. Because vectorized environments immediately reset to a new episode at $t_{\text{done}} + 1$, naively indexing $s_{t+m}$ from the rollout array would pair $s_t$ from Episode 1 with an unrelated state $s_{t+m}$ from Episode 2 (cross-episode pollution). Conversely, simply masking out (`diff_sq = 0.0`) any jump that hits a terminal state silently deletes the terminal boundary condition and discards up to $50\%$–$80\%$ of the gradient signal for states near the end of an episode.

#### The Exact Solution: Distinguishing True Absorbing Terminals from Artificial Timeouts
We must distinguish between two fundamentally different types of episode endings in RL:

1. **True Absorbing Terminal States (`true_terminal == True`)**:
   In the underlying MDP, when the agent reaches a terminal transition at $t_{\text{term}} \in [t, t + m - 1]$, the Markov chain enters the absorbing state $s_{\text{term}}$ and remains in $s_{\text{term}}$ for all subsequent steps of the jump, including step $t + m$.
   * At $s_{\text{term}}$, the true value and predicted value are identically zero: $V(s_{\text{term}}) = v_\theta(s_{\text{term}}) = v_{\text{aux}}(s_{\text{term}}) \equiv 0$.
   * Therefore, the destination errors and destination gradient are identically zero:
     $$e_j = 0, \qquad e_{\text{aux}, j} = 0, \qquad \nabla_\theta v_\theta(s_j) = \mathbf{0}$$
   * Substituting these into Term (1) and Term (2):
     * **Term (1) (Backward branch) vanishes**: $(e_{\text{aux}, t} - 0)\nabla_\theta v_\theta(s_{\text{term}}) = \mathbf{0}$.
     * **Term (2) (Forward branch) reduces to the squared source error**: $d_{\text{fwd}}^2 = (\hat{e}_t - 0)^2 = \hat{e}_t^2$.
   * Thus, whenever a jump crosses at least one `true_terminal` in $[t, \min(t+m, T) - 1]$, the transition is **valid** and its Dirichlet loss contribution is simply $\mathcal{D}(t, b) = \hat{e}_{t,b}^2$.

2. **Artificial Timeouts / Truncations (`is_timeout = done & (~true_terminal)`)**:
   When an episode is cut short by an arbitrary time limit (`TimeLimit.truncated`), the state is *not* an absorbing zero-value state ($V(s) \neq 0$), and the next state in the buffer belongs to a newly reset episode. Therefore, any jump that crosses an artificial timeout in $[t, \min(t+m, T) - 1]$ cannot be evaluated across the reset and is masked out ($\mathcal{D}(t, b) = 0$).

---

### 3.3 Issue 3: Finite Rollout Buffer Length $T$ and Lookahead Runway

#### The Problem
In an on-policy rollout buffer of length $T$ (e.g., $T = 128$), the expected jump length is $\mathbb{E}[m] = \frac{1}{1 - \gamma\lambda}$ (which is $5$ steps for $\gamma\lambda = 0.8$ and $20$ steps for $\gamma\lambda = 0.95$). For start states $s_t$ near the end of the buffer ($t \approx T - 1$), almost any jump $K > 0$ will land beyond the buffer horizon ($t + m > T$).
* If jumps with $t + m > T$ are masked to $0.0$, states near the end of the buffer only ever receive updates when $K = 0$, biasing their effective trace parameter $\lambda$ toward $0$ and starving late states of gradient updates.
* If surviving jumps $K \le T - 1 - t$ are instead multiplied by an importance weight $w_{\text{trunc}}(t) = \frac{1}{1 - (\gamma\lambda)^{T - t}}$, two new problems arise: (a) for $t = T - 1$, $w_{\text{trunc}}(T-1) = \frac{1}{1 - \gamma\lambda}$ inflates variance by up to $20\times$, and (b) on trajectories that terminate naturally at $t_{\text{term}} < T$, no jumps are ever truncated at $T$, so multiplying non-terminal jumps by $w_{\text{trunc}}(t) > 1$ makes the probabilities sum to greater than $1$.

#### The Solution: Prefix Runway Window $T_{\text{start}}$ + Boundary Clamping
We resolve finite-buffer truncation cleanly without importance weights using two mechanisms:
1. **Prefix Runway Window ($T_{\text{start}} = T - L_{\text{runway}}$)**:
   We only evaluate start states $t \in [0, T_{\text{start}})$, reserving the final $L_{\text{runway}}$ steps of the rollout (e.g., $L_{\text{runway}} = 16$) as a lookahead runway so that jumps originating near $T_{\text{start}} - 1$ have room to land inside the buffer. For $\gamma\lambda = 0.8$ and $L_{\text{runway}} = 16$, over $97.2\%$ of jumps from $t = T_{\text{start}} - 1$ (and $>99.5\%$ of jumps overall) land naturally inside $[0, T]$.
2. **Boundary Bootstrap Clamping ($j = \min(t + m, T)$)**:
   Standard rollout buffers already compute continuation bootstrap values at step $T$ (`next_value_T`, `next_aux_value_T`, `next_target_T`). By appending the step-$T$ continuation error $e_T$ to form an extended array of length $T + 1$ and clamping the destination index to $j = \min(t + m, T)$, the tiny fraction of non-terminal jumps that exceed $T$ smoothly bootstrap at the buffer boundary $T$ rather than being discarded or requiring high-variance importance weights.

---

## 4. Complete Loss Summary and Reference JAX Implementation

### 4.1 Unified Loss Equations
For a rollout buffer of length $T$, batch size $B$, and start window $T_{\text{start}} = T - L_{\text{runway}}$, let $m_{t,b} = K_{t,b} + 1$ with $K_{t,b} \sim \text{Geom}(1 - \gamma\lambda)$ and $j_{t,b} = \min(t + m_{t,b}, T)$. The complete differentiable critic objective is:

$$\boxed{\mathcal{L}_{\text{total}}(\theta, \phi) = \mathcal{L}_{\text{mag}}(\theta) + \mathcal{L}_{\text{dir}}(\theta; \text{sg}(\phi)) + \mathcal{L}_{\text{start}}(\theta) + c_{\text{aux}} \mathcal{L}_{\text{aux}}(\phi)}$$

where:
$$\begin{aligned}
\mathcal{L}_{\text{mag}}(\theta) &= \frac{1}{2}(1 - \tilde{\gamma}) \frac{1}{T_{\text{start}} B} \sum_{t=0}^{T_{\text{start}}-1} \sum_{b=0}^{B-1} \hat{e}_{t,b}^2 \\
\mathcal{L}_{\text{dir}}(\theta; \text{sg}(\phi)) &= \frac{1}{2}\tilde{\gamma} \frac{1}{T_{\text{start}} B} \sum_{t=0}^{T_{\text{start}}-1} \sum_{b=0}^{B-1} \mathcal{D}(t, b) \\
\mathcal{L}_{\text{start}}(\theta) &= \frac{1}{4}\tilde{\gamma} \frac{1}{T_{\text{start}} B} \sum_{t=0}^{T_{\text{start}}-1} \sum_{b=0}^{B-1} (\gamma\lambda)^{k_{t,b}} \hat{e}_{t,b}^2 \\
\mathcal{L}_{\text{aux}}(\phi) &= \frac{1}{2} \frac{1}{T B} \sum_{t=0}^{T-1} \sum_{b=0}^{B-1} \big(\text{sg}(G_{t,b}) - v_{\text{aux}, \phi}(s_{t,b})\big)^2
\end{aligned}$$

with the per-transition Dirichlet term defined as:
$$\mathcal{D}(t, b) = \begin{cases}
\hat{e}_{t,b}^2, & \text{if } \exists \tau \in [t, j_{t,b}-1] \text{ with } \text{true\_terminal}_{\tau, b} = \text{True} \\
0, & \text{if } \exists \tau \in [t, j_{t,b}-1] \text{ with } \text{is\_timeout}_{\tau, b} = \text{True} \\
\big(\hat{e}_{t,b} - \text{sg}(\hat{e}_{j_{t,b}, b})\big)^2 + \big(\text{sg}(e_{\text{aux}, t, b}) - e_{\text{aux}, j_{t,b}, b}\big)^2, & \text{otherwise}
\end{cases}$$

---

### 4.2 Reference JAX Implementation

```python
import jax
import jax.numpy as jnp


def e_lambda_geometric_critic_loss(
    values: jax.Array,                  # v_theta(s_t), shape (T, B)
    aux_values: jax.Array,              # v_aux(s_t), shape (T, B)
    targets: jax.Array,                 # G_t empirical returns, shape (T, B)
    dones: jax.Array,                   # done flags (terminals | timeouts), shape (T, B)
    gamma: float,                       # discount factor, e.g. 0.99
    lmbda: float,                       # trace parameter, e.g. 0.8
    rng: jax.Array,                     # PRNGKey for geometric jump sampling
    true_terminals: jax.Array = None,   # absorbing terminals (excl. timeouts), shape (T, B)
    next_value_T: jax.Array = None,     # bootstrap v_theta at step T, shape (B,)
    next_aux_value_T: jax.Array = None, # bootstrap v_aux at step T, shape (B,)
    next_target_T: jax.Array = None,    # bootstrap target G_T at step T, shape (B,)
    num_start_states: int = None,       # T_start window (default: T - 16)
    init_steps_since_start: jax.Array = None,  # step count k at t=0, shape (B,)
    aux_loss_coef: float = 1.0,
):
    """
    Sampled E(lambda) critic loss with auxiliary critic v_aux,
    geometrically decaying start-state reset grounding, and absorbing boundary conditions.
    """
    if true_terminals is None:
        true_terminals = dones
    if next_value_T is None:
        next_value_T = jnp.zeros_like(values[0])
    if next_aux_value_T is None:
        next_aux_value_T = jnp.zeros_like(aux_values[0])
    if next_target_T is None:
        next_target_T = jnp.zeros_like(targets[0])

    T, B = targets.shape[:2]
    T_start = min(T, max(1, T - 16)) if num_start_states is None else min(num_start_states, T)

    gl = gamma * lmbda
    sg = jax.lax.stop_gradient

    # 1. Construct rollout errors (for Term 2 & Magnitude) and aux errors (for Term 1)
    errors = sg(targets) - values
    e_aux = sg(aux_values) - values

    e_T = jnp.where(true_terminals[-1], 0.0, sg(next_target_T) - next_value_T)
    e_aux_T = jnp.where(true_terminals[-1], 0.0, sg(next_aux_value_T) - next_value_T)

    errors_ext = jnp.concatenate([errors, e_T[None]], axis=0)       # (T + 1, B)
    e_aux_ext = jnp.concatenate([e_aux, e_aux_T[None]], axis=0)     # (T + 1, B)

    # 2. Sample geometric jump lengths: m = K + 1 where K ~ Geom(1 - gl)
    u = jax.random.uniform(rng, shape=(T_start, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T_start, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T_start)[:, None]
    j = jnp.minimum(t_arr + m, T)

    # 3. Detect true absorbing terminals vs. artificial timeouts in [t, j - 1]
    is_timeout = dones & (~true_terminals)
    term_cumsum_excl = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    timeout_cumsum_excl = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)],
        axis=0,
    )

    hit_true_terminal = (jnp.take_along_axis(term_cumsum_excl, j, axis=0) - term_cumsum_excl[:T_start]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum_excl, j, axis=0) - timeout_cumsum_excl[:T_start]) > 0

    e_start = errors[:T_start]
    e_aux_start = e_aux[:T_start]
    e_jump = jnp.take_along_axis(errors_ext, j, axis=0)
    e_aux_jump = jnp.take_along_axis(e_aux_ext, j, axis=0)

    diff_sq = jnp.where(
        hit_true_terminal,
        e_start ** 2,  # Absorbs at e_term = 0; backward branch at s_term vanishes
        jnp.where(
            hit_timeout,
            0.0,
            (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2,
        ),
    )

    # 4. Geometrically decaying start-state reset grounding (scanning over true_terminals & timeouts)
    def scan_steps(k, step_flags):
        term, timeout = step_flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    init_k = jnp.full((B,), 10_000, dtype=jnp.int32) if init_steps_since_start is None else init_steps_since_start
    final_k, steps_since_start = jax.lax.scan(scan_steps, init_k, (true_terminals, is_timeout))

    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(steps_since_start[:T_start] == 0, 1.0, 0.0),
        safe_gl ** steps_since_start[:T_start],
    )

    # 5. Exact theoretical prefactors
    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma
    start_weight = 0.25 * tilde_gamma  # Double 1/2 factor: outer 1/2 * inner 1/2

    magnitude_loss = magnitude_weight * jnp.mean(e_start ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    start_loss = start_weight * jnp.mean(start_trace * (e_start ** 2))
    value_loss = magnitude_loss + dirichlet_loss + start_loss

    aux_loss = 0.5 * jnp.mean((sg(targets) - aux_values) ** 2)
    total_loss = value_loss + aux_loss_coef * aux_loss

    losses = {
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "start_loss": start_loss,
        "aux_loss": aux_loss,
        "total_loss": total_loss,
    }
    return total_loss, losses, final_k
```
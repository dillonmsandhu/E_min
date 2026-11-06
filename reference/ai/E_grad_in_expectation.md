# Explicit Derivation of the Double Sampling Issue in Dirichlet Energy ($\mathcal{E}$)

This document provides a rigorous, step-by-step derivation of the gradient of the Dirichlet energy $\mathcal{E}$ in expectation, revealing exactly how and where the **product of two expectations** arises (the classic **Double Sampling Problem**), its structural equivalence to Baird's Bellman Error Minimization (BEM), and why an auxiliary value function $v_{\text{aux}}$ resolves it.

---

## 1. Matrix Formulation and Setup

Let:
- $\mathcal{S}$ denote the finite state space with $|\mathcal{S}|$ states.
- $\theta \in \mathbb{R}^d$ be the parameter vector of the value function $v_\theta$.
- $\mathbf{v}_\theta \in \mathbb{R}^{|\mathcal{S}|}$ be the value prediction vector, where entry $i$ is $v_\theta(s_i)$.
- $V \in \mathbb{R}^{|\mathcal{S}|}$ be the true value vector under policy $\pi$.
- $\mathbf{e} \triangleq V - \mathbf{v}_\theta \in \mathbb{R}^{|\mathcal{S}|}$ be the value error vector.
- $J \in \mathbb{R}^{|\mathcal{S}| \times d}$ be the Jacobian of the value function with respect to parameters $\theta$:
  $$J = \begin{bmatrix} \nabla_\theta v_\theta(s_1)^\top \\ \vdots \\ \nabla_\theta v_\theta(s_{|\mathcal{S}|})^\top \end{bmatrix} \implies J^\top = \begin{bmatrix} \nabla_\theta v_\theta(s_1) & \cdots & \nabla_\theta v_\theta(s_{|\mathcal{S}|}) \end{bmatrix}$$
  Since $V$ does not depend on $\theta$, the Jacobian of the error is:
  $$\nabla_\theta \mathbf{e} = -J$$
- $P \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ be the transition probability matrix under $\pi$.
- $D = \text{diag}(\mu)$ be the diagonal stationary distribution matrix satisfying $\mu^\top P = \mu^\top$ (or $P^\top D \mathbf{1} = D \mathbf{1}$).

The Dirichlet energy $\mathcal{E}$ is defined as the quadratic form:
$$\mathcal{E} = \mathbf{e}^\top D (I - P) \mathbf{e}$$

---

## 2. Symmetrization via the Graph Laplacian $L$

For any real square matrix $M$, the quadratic form $\mathbf{x}^\top M \mathbf{x}$ depends only on the symmetric part $\frac{1}{2}(M + M^\top)$:
$$\mathbf{e}^\top D(I-P) \mathbf{e} = \mathbf{e}^\top \left[ \frac{D(I - P) + (D(I - P))^\top}{2} \right] \mathbf{e}$$

Define the symmetric operator $L$ (the directed graph Laplacian):
$$L \triangleq \frac{1}{2} \Big( D(I - P) + (I - P)^\top D \Big) = D - \frac{1}{2} \big( D P + P^\top D \big)$$

Thus:
$$\mathcal{E} = \mathbf{e}^\top L \mathbf{e}$$

---

## 3. Parameter Gradient of $\mathcal{E}$ in Pure Matrix Form

Applying the matrix chain rule:
$$\nabla_\theta \mathcal{E} = (\nabla_\theta \mathbf{e})^\top \nabla_{\mathbf{e}} \mathcal{E} = 2 (\nabla_\theta \mathbf{e})^\top L \mathbf{e} = -2 J^\top L \mathbf{e}$$

Plugging in $L = \frac{1}{2} \Big( D(I - P) + (I - P)^\top D \Big)$:
$$\nabla_\theta \mathcal{E} = -2 J^\top \left( \frac{D(I - P) + (I - P)^\top D}{2} \right) \mathbf{e}$$
$$\nabla_\theta \mathcal{E} = - J^\top D (I - P) \mathbf{e} - J^\top (I - P^\top) D \mathbf{e}$$

Now, distribute both terms in matrix form:
$$\nabla_\theta \mathcal{E} = \underbrace{- J^\top D \mathbf{e} + J^\top D (P \mathbf{e})}_{\text{Term A: Forward Term}} \quad \underbrace{- J^\top D \mathbf{e} + J^\top P^\top D \mathbf{e}}_{\text{Term B: Backward Term}}$$

---

## 4. The Matrix Operators and Their Expectation Meaning

Before expanding into indices, we can stay strictly in **matrix operator form**. 

### 4.1 The $D$-Inner Product as Stationary Expectation
For any vector $\mathbf{b} \in \mathbb{R}^{|\mathcal{S}|}$ and any matrix $A \in \mathbb{R}^{|\mathcal{S}| \times d}$:
$$A^\top D \mathbf{b} = \sum_{s \in \mathcal{S}} \mu(s) A(s)^\top b(s) = \mathbb{E}_{s \sim \mu} \Big[ A(s)^\top b(s) \Big]$$
Here, $D$ is simply the expectation operator over the stationary distribution $\mu$.

### 4.2 The Markov Operator $P$ on Vectors vs. Jacobians
The transition matrix $P$ is the **conditional expectation operator** $\mathbb{E}_{s' \sim P(\cdot \mid s)}[\cdot \mid s]$:

1. **Acting on the Error Vector $\mathbf{e} \in \mathbb{R}^{|\mathcal{S}|}$:**
   $$(P \mathbf{e})(s) = \sum_{s'} P(s' \mid s) e(s') = \mathbb{E}_{s' \sim P(\cdot \mid s)} \big[ e(s') \;\big|\; s \big]$$
   $P\mathbf{e}$ is the **expected next-state error**.

2. **Acting on the Jacobian Matrix $J \in \mathbb{R}^{|\mathcal{S}| \times d}$:**
   Recall that row $s$ of $J$ is the gradient at state $s$: $J(s, :) = \nabla_\theta v_\theta(s)^\top$.
   Applying $P$ to $J$:
   $$(P J)(s, :) = \sum_{s'} P(s' \mid s) \nabla_\theta v_\theta(s')^\top = \mathbb{E}_{s' \sim P(\cdot \mid s)} \big[ \nabla_\theta v_\theta(s')^\top \;\big|\; s \big]$$
   $P J$ is the **expected next-state Jacobian matrix**!

3. **The Transpose Identity $J^\top P^\top = (P J)^\top$:**
   Taking the transpose of the matrix product:
   $$J^\top P^\top = (P J)^\top \in \mathbb{R}^{d \times |\mathcal{S}|}$$
   Column $s$ of $(P J)^\top$ is the **expected gradient at the next state $s'$ conditioned on current state $s$**:
   $$\big[ (P J)^\top \big]_{:, s} = \mathbb{E}_{s' \sim P(\cdot \mid s)} \big[ \nabla_\theta v_\theta(s') \;\big|\; s \big]$$

---

## 5. Matrix Form of the Gradient as Expectations

Now rewrite all four matrix terms directly using these expectation operators:

$$\nabla_\theta \mathcal{E} = \underbrace{- J^\top D \mathbf{e} + J^\top D (P \mathbf{e})}_{\text{Term A: Forward}} \quad \underbrace{- J^\top D \mathbf{e} + (P J)^\top D \mathbf{e}}_{\text{Term B: Backward}}$$

Let us inspect the cross-terms in each:

| Term | Matrix Expression | Who does $P$ act on? | Meaning in Expectation |
| :--- | :--- | :--- | :--- |
| **Current Gradient $\times$ Current Error** | $- J^\top D \mathbf{e}$ | Neither | $- \mathbb{E}_{s \sim \mu} \big[ \nabla_\theta v_\theta(s) \, e(s) \big]$ |
| **Forward Cross-Term** | $+ J^\top D (P \mathbf{e})$ | **Error** ($P \mathbf{e}$) | $+ \mathbb{E}_{s \sim \mu} \Big[ \nabla_\theta v_\theta(s) \cdot \mathbb{E}_{s' \mid s}[e(s')] \Big]$ |
| **Backward Cross-Term** | $+ (P J)^\top D \mathbf{e}$ | **Jacobian** ($P J$) | $+ \mathbb{E}_{s \sim \mu} \Big[ \mathbb{E}_{s' \mid s}[\nabla_\theta v_\theta(s')] \cdot e(s) \Big]$ |

### Why $(PJ)^\top D \mathbf{e}$ is a Product of Two Expectations
In the backward cross-term $(PJ)^\top D \mathbf{e}$:
- The first factor is the expected next-state gradient: $\mathbb{E}_{s' \sim P(\cdot \mid s)}[\nabla_\theta v_\theta(s') \mid s]$.
- The second factor is the value error at state $s$: $e(s) = \mathbb{E}[\hat{e}(s) \mid s]$.

Both are conditional expectations given the **same state $s$**:
$$\mathbf{(P J)^\top D \mathbf{e} = \mathbb{E}_{s \sim \mu} \left[ \Big( \mathbb{E}_{s' \sim P(\cdot \mid s)} \big[ \nabla_\theta v_\theta(s') \big] \Big) \cdot \Big( \mathbb{E}\big[\hat{e}(s) \mid s\big] \Big) \right]}$$

In contrast, in the forward cross-term $J^\top D (P\mathbf{e})$:
$$\mathbf{J^\top D (P\mathbf{e}) = \mathbb{E}_{s \sim \mu} \left[ \nabla_\theta v_\theta(s) \cdot \mathbb{E}_{s' \sim P(\cdot \mid s)} [e(s') \mid s] \right]}$$
Because $\nabla_\theta v_\theta(s)$ is already fixed at state $s$, it is not an expectation over transitions from $s$, so no double sampling occurs.

---

## 6. Detailed Entry-by-Entry Coordinate Expansion

### 6.1 Forward Term: Term (2)
Consider the vector $\mathbf{u} = D(I - P)\mathbf{e}$:
$$u_i = [D(I - P)\mathbf{e}]_i = \mu_i \left( e(s_i) - \sum_j P_{ij} e(s_j) \right) = \mu_i \sum_j P_{ij} \big( e(s_i) - e(s_j) \big)$$
where we used $\sum_j P_{ij} = 1$.

Multiplying by $-J^\top$:
$$-J^\top D (I - P) \mathbf{e} = - \sum_i \mu_i \sum_j P_{ij} \big( e(s_i) - e(s_j) \big) \nabla_\theta v_\theta(s_i)$$

In expectation form:
$$\mathbf{\text{Term (2)}} = - \mathbb{E}_{s_i \sim \mu, \, s_j \sim P(\cdot \mid s_i)} \Big[ \big( e(s_i) - e(s_j) \big) \nabla_\theta v_\theta(s_i) \Big]$$

> **Key Observation for Term (2):**
> The value gradient $\nabla_\theta v_\theta(s_i)$ is evaluated at the **current** state $s_i$.
> Conditioning on $s_i$, $\nabla_\theta v_\theta(s_i)$ is deterministic:
> $$\mathbb{E}_{s_j \sim P(\cdot \mid s_i)} \Big[ \big( e(s_i) - e(s_j) \big) \nabla_\theta v_\theta(s_i) \;\Big|\; s_i \Big] = \left( e(s_i) - \mathbb{E}_{s_j \mid s_i}[e(s_j)] \right) \nabla_\theta v_\theta(s_i)$$
> The transition noise from $s_j$ only appears inside $e(s_j)$ and is **uncoupled** from the gradient $\nabla_\theta v_\theta(s_i)$.
> Therefore, **Term (2) can be sampled without double sampling bias** along standard trajectories!

---

### 6.2 Backward Term: Term (1)
Now consider Term (1):
$$J^\top (P^\top - I) D \mathbf{e} = J^\top P^\top D \mathbf{e} - J^\top D \mathbf{e}$$

Let us inspect each piece:
1. First piece: $J^\top P^\top D \mathbf{e}$
   The $j$-th entry of $P^\top D \mathbf{e}$ is:
   $$[P^\top D \mathbf{e}]_j = \sum_i P_{ij} \mu_i e(s_i)$$
   Multiplying by $J^\top$:
   $$J^\top P^\top D \mathbf{e} = \sum_j \nabla_\theta v_\theta(s_j) [P^\top D \mathbf{e}]_j = \sum_j \nabla_\theta v_\theta(s_j) \sum_i \mu_i P_{ij} e(s_i) = \sum_i \mu_i \sum_j P_{ij} e(s_i) \nabla_\theta v_\theta(s_j)$$

2. Second piece: $- J^\top D \mathbf{e}$
   $$- J^\top D \mathbf{e} = - \sum_j \mu_j e(s_j) \nabla_\theta v_\theta(s_j)$$
   Using stationarity $\mu_j = \sum_i \mu_i P_{ij}$:
   $$- J^\top D \mathbf{e} = - \sum_i \mu_i \sum_j P_{ij} e(s_j) \nabla_\theta v_\theta(s_j)$$

Combining both pieces:
$$J^\top (P^\top - I) D \mathbf{e} = \sum_i \mu_i \sum_j P_{ij} \big( e(s_i) - e(s_j) \big) \nabla_\theta v_\theta(s_j)$$

In expectation form:
$$\mathbf{\text{Term (1)}} = \mathbb{E}_{s_i \sim \mu, \, s_j \sim P(\cdot \mid s_i)} \Big[ \big( e(s_i) - e(s_j) \big) \nabla_\theta v_\theta(s_j) \Big]$$

---

## 7. Emergence of the Product of Two Expectations

Let us decompose Term (1) into its two constituent parts:
$$\text{Term (1)} = \underbrace{\sum_i \mu_i \sum_j P_{ij} e(s_i) \nabla_\theta v_\theta(s_j)}_{\text{Term (1A)}} - \underbrace{\sum_i \mu_i \sum_j P_{ij} e(s_j) \nabla_\theta v_\theta(s_j)}_{\text{Term (1B)}}$$

### Examining Term (1B):
$$\text{Term (1B)} = \sum_j \mu_j e(s_j) \nabla_\theta v_\theta(s_j) = \mathbb{E}_{s_j \sim \mu} \big[ e(s_j) \nabla_\theta v_\theta(s_j) \big]$$
Here, both $e$ and $\nabla_\theta v_\theta$ are evaluated at the exact same state $s_j$. A single sample return $\hat{e}(s_j) = G(s_j) - v_\theta(s_j)$ provides an unbiased estimate:
$$\mathbb{E}_{G_j \mid s_j}[\hat{e}(s_j) \nabla_\theta v_\theta(s_j) \mid s_j] = \mathbb{E}[G_j - v_\theta(s_j) \mid s_j] \nabla_\theta v_\theta(s_j) = e(s_j) \nabla_\theta v_\theta(s_j)$$
Term (1B) is completely unbiased.

---

### Examining Term (1A) — The Double Sampling Issue:
Now examine Term (1A) by factoring out $e(s_i)$ from the inner summation over next states $s_j$:
$$\text{Term (1A)} = \sum_i \mu_i \, e(s_i) \left( \sum_j P_{ij} \nabla_\theta v_\theta(s_j) \right) = \mathbb{E}_{s_i \sim \mu} \left[ e(s_i) \cdot \mathbb{E}_{s_j \sim P(\cdot \mid s_i)} \big[ \nabla_\theta v_\theta(s_j) \big] \right]$$

Now substitute what $e(s_i)$ actually is in practice.
The true error $e(s_i) = V(s_i) - v_\theta(s_i)$ is not known in closed form; it must be estimated from environment samples.
By the Bellman equation, $V(s_i) = \mathbb{E}_{s_j}[r(s_i, s_j) + \gamma V(s_j) \mid s_i]$.
Thus, using the sample return $\hat{e}_i = r(s_i, s_j) + \gamma G_j - v_\theta(s_i)$:
$$e(s_i) = \mathbb{E}_{s_j \sim P(\cdot \mid s_i), \, G_j} \big[ \hat{e}(s_i; s_j) \;\big|\; s_i \big]$$

Substituting this expectation into Term (1A) yields:
$$\boxed{\mathbf{\text{Term (1A)} = \mathbb{E}_{s_i \sim \mu} \left[ \Big( \mathbb{E}_{s_j \sim P(\cdot \mid s_i)} \big[ \hat{e}(s_i; s_j) \big] \Big) \cdot \Big( \mathbb{E}_{s_j' \sim P(\cdot \mid s_i)} \big[ \nabla_\theta v_\theta(s_j') \big] \Big) \right]}}$$

**This is the exact product of two conditional expectations!**

$$\mathbb{E}_{s_j \sim P(\cdot \mid s_i)}[\hat{e}(s_i; s_j)] \times \mathbb{E}_{s_j' \sim P(\cdot \mid s_i)}[\nabla_\theta v_\theta(s_j')]$$

---

## 8. Why Naive Trajectory Sampling Incurs Bias

In standard reinforcement learning without a generative model (simulator), an agent observes a single transition:
$$(s_i, r_i, s_j) \quad \text{where } s_j \sim P(\cdot \mid s_i)$$

A naive stochastic gradient estimator attempts to evaluate the product using that **single sample** $s_j$:
$$\widehat{\text{Term (1A)}}_{\text{naive}} = \hat{e}(s_i; s_j) \nabla_\theta v_\theta(s_j)$$

Taking the expectation conditioned on $s_i$:
$$\mathbb{E}_{s_j \mid s_i} \big[ \hat{e}(s_i; s_j) \nabla_\theta v_\theta(s_j) \big] = \mathbb{E}_{s_j \mid s_i}[\hat{e}(s_i; s_j)] \cdot \mathbb{E}_{s_j \mid s_i}[\nabla_\theta v_\theta(s_j)] + \text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( \hat{e}(s_i; s_j), \, \nabla_\theta v_\theta(s_j) \big)$$

Because the return $\hat{e}(s_i; s_j) = r(s_i, s_j) + \gamma G(s_j) - v_\theta(s_i)$ contains the downstream return $G(s_j)$ (whose expectation is $V(s_j)$):
$$\text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( \hat{e}(s_i; s_j), \, \nabla_\theta v_\theta(s_j) \big) = \gamma \, \text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( V(s_j), \, \nabla_\theta v_\theta(s_j) \big) + \text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( r(s_i, s_j), \, \nabla_\theta v_\theta(s_j) \big)$$

This covariance is **strictly non-zero** in stochastic MDPs:
$$\mathbb{E}_{s_j \mid s_i} \big[ \hat{e}(s_i; s_j) \nabla_\theta v_\theta(s_j) \big] \neq e(s_i) \, \mathbb{E}_{s_j \mid s_i}[\nabla_\theta v_\theta(s_j)]$$

### The Double Sampling Requirement
To evaluate this product without bias, the agent would need **two independent successor states from the same state $s_i$**:
1. Transition 1: $s_j^{(1)} \sim P(\cdot \mid s_i)$ to compute the error $\hat{e}(s_i; s_j^{(1)})$,
2. Transition 2: $s_j^{(2)} \sim P(\cdot \mid s_i)$ to compute the next-state Jacobian $\nabla_\theta v_\theta(s_j^{(2)})$.

Because $s_j^{(1)}$ and $s_j^{(2)}$ are conditionally independent given $s_i$:
$$\mathbb{E}_{s_j^{(1)}, s_j^{(2)} \mid s_i} \big[ \hat{e}(s_i; s_j^{(1)}) \nabla_\theta v_\theta(s_j^{(2)}) \big] = \mathbb{E}_{s_j^{(1)} \mid s_i}[\hat{e}(s_i; s_j^{(1)})] \cdot \mathbb{E}_{s_j^{(2)} \mid s_i}[\nabla_\theta v_\theta(s_j^{(2)})] = e(s_i) \sum_j P_{ij} \nabla_\theta v_\theta(s_j)$$

In episodic/online RL, branching into two independent futures from $s_i$ is impossible without resetting a simulator. This is the **Double Sampling Problem**.

---

## 9. Comparison: Dirichlet Energy vs. Baird's Bellman Error Minimization (BEM)

It is instructive to contrast this with the classic double sampling problem in Bellman Error Minimization (Baird's Residual Gradient).

### 7.1 Bellman Error Minimization
The Mean Squared Bellman Error objective is:
$$\text{MSBE}(\theta) = \frac{1}{2} \|\mathbf{\bar{\delta}}\|_D^2 = \frac{1}{2} \mathbf{\bar{\delta}}^\top D \mathbf{\bar{\delta}}, \quad \text{where } \mathbf{\bar{\delta}} \triangleq T v_\theta - v_\theta = R + \gamma P \mathbf{v}_\theta - \mathbf{v}_\theta$$

Its parameter gradient is:
$$\nabla_\theta \text{MSBE} = - J^\top (I - \gamma P)^\top D \mathbf{\bar{\delta}} = - J^\top D \mathbf{\bar{\delta}} + \gamma J^\top P^\top D \mathbf{\bar{\delta}}$$

The problematic term is:
$$\gamma J^\top P^\top D \mathbf{\bar{\delta}} = \gamma \sum_i \mu_i \bar{\delta}(s_i) \sum_j P_{ij} \nabla_\theta v_\theta(s_j) = \gamma \, \mathbb{E}_{s_i} \left[ \Big( \mathbb{E}_{s_j}[\delta(s_i, s_j)] \Big) \cdot \Big( \mathbb{E}_{s_j'}[\nabla_\theta v_\theta(s_j')] \Big) \right]$$

The resulting sampling bias on a single transition $(s_i, s_j)$ is:
$$\text{Bias}_{\text{BEM}}(s_i) = \gamma^2 \, \text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( v_\theta(s_j), \, \nabla_\theta v_\theta(s_j) \big) + \gamma \, \text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( r(s_i, s_j), \, \nabla_\theta v_\theta(s_j) \big)$$

### 7.2 Comparison Summary Table

| Property | Bellman Error Minimization (BEM) | Dirichlet Energy ($\mathcal{E}$) |
| :--- | :--- | :--- |
| **Objective** | $\frac{1}{2} \|\bar{\delta}\|_D^2 = \frac{1}{2} \bar{\delta}^\top D \bar{\delta}$ | $\mathcal{E} = \mathbf{e}^\top D (I - P) \mathbf{e}$ |
| **Gradient Form** | $-J^\top D \bar{\delta} + \gamma J^\top P^\top D \bar{\delta}$ | $-J^\top D (I - P)\mathbf{e} + J^\top (P^\top - I)D \mathbf{e}$ |
| **Forward Term** | $-J^\top D \bar{\delta}$ (Unbiased) | $-J^\top D (I - P)\mathbf{e}$ (Unbiased) |
| **Backward Term** | $\gamma J^\top P^\top D \bar{\delta}$ (**Biased**) | $J^\top (P^\top - I)D \mathbf{e}$ (**Biased**) |
| **Product of Expectations** | $\mathbb{E}_{s_j}[\delta_i] \times \mathbb{E}_{s_j'}[\nabla_\theta v(s_j')]$ | $\mathbb{E}_{s_j}[\hat{e}_i] \times \mathbb{E}_{s_j'}[\nabla_\theta v(s_j')]$ |
| **Single-Sample Bias** | $\gamma^2 \text{Cov}(v_\theta(s_j), \nabla_\theta v(s_j) \mid s_i)$ | $\gamma \text{Cov}(V(s_j), \nabla_\theta v(s_j) \mid s_i)$ |
| **Target Function in Covariance** | $v_\theta(s_j)$ (**Non-stationary**, moving weights) | $V(s_j)$ (**Stationary**, true value function) |
| **Scope of Double Sampling** | Affects the **entire** target gradient | Localized to **only 1 of 4 terms** in the total energy $(1-\gamma)\|\mathbf{e}\|_D^2 + \gamma \mathcal{E}$ |

---

## 10. Resolution via the Auxiliary Value Network ($v_{\text{aux}}$)

To eliminate the double-sampling bias without requiring branching transitions, we introduce an auxiliary network $v_{\text{aux}}$ trained via Monte Carlo regression to estimate $V(s)$.

Define the auxiliary error proxy:
$$e_{\text{aux}}(s) \triangleq v_{\text{aux}}(s) - v_\theta(s)$$

We substitute $e_{\text{aux}}$ **only into the problematic Term (1)**:
$$\widehat{\text{Term (1)}} = \big( e_{\text{aux}}(s_i) - e_{\text{aux}}(s_j) \big) \nabla_\theta v_\theta(s_j)$$

### Why this eliminates the double-sampling covariance:
1. $v_{\text{aux}}(s_i)$ is evaluated purely as a function of the current state $s_i$.
2. It does **not** depend on the realization of the successor state $s_j \sim P(\cdot \mid s_i)$ on this transition.
3. Therefore:
   $$\text{Cov}_{s_j \sim P(\cdot \mid s_i)} \big( e_{\text{aux}}(s_i), \, \nabla_\theta v_\theta(s_j) \big) = 0$$
4. Conditioning on $s_i$:
   $$\mathbb{E}_{s_j \mid s_i} \big[ e_{\text{aux}}(s_i) \nabla_\theta v_\theta(s_j) \big] = e_{\text{aux}}(s_i) \sum_j P_{ij} \nabla_\theta v_\theta(s_j)$$
   which is the exact product of expectations, evaluated without requiring a second transition sample.

This yields the complete, unbiased stochastic gradient estimator of Equation~(348) (`eq:full_sample_grad`):
$$\hat{g}_i = \underbrace{-2(1-\gamma)\hat{e}_i \nabla_\theta v_i}_{\text{Magnitude MC Term}} + \underbrace{\gamma \big(e_{\text{aux},i} - e_{\text{aux},j}\big)\nabla_\theta v_j}_{\text{Term (1): Debiaised with } v_{\text{aux}}} - \underbrace{\gamma \big(\hat{e}_i - \hat{e}_j\big)\nabla_\theta v_i}_{\text{Term (2): Unbiased Forward Term}}$$

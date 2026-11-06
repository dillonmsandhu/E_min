# Energy Gradient Sampling on DAGs
Looking directly at the decomposition from the paper:

$$\nabla_\theta \mathcal{E}_i = \sum_j p_{ij} (e_i - e_j) \big[ \underbrace{\nabla_\theta v_\theta(s_j)}_{\text{Term (1)}} - \underbrace{\nabla_\theta v_\theta(s_i)}_{\text{Term (2)}} \big]$$

When we take the expectation over the state occupancy $\mu$, the total Dirichlet gradient is:

$$\nabla_\theta \mathcal{E} = \underbrace{\sum_i \mu_i \sum_j p_{ij} (e_i - e_j) \nabla_\theta v_\theta(s_j)}_{\text{Total Term (1): Backward}} - \underbrace{\sum_i \mu_i \sum_j p_{ij} (e_i - e_j) \nabla_\theta v_\theta(s_i)}_{\text{Total Term (2): Forward}}$$

And **Forward Only** simply drops Term (1), keeping only $-\text{Term (2)}$.

Mathematically, why does dropping Term (1) preserve **98.2% alignment** in Space Invaders, but **catastrophically destroy the gradient ($\rho = -0.12$) in FourRooms**?

---

### 1. The Core Mechanism: Near-Cancellation vs. Independent Features

Look at the difference vector inside the brackets of the exact Dirichlet gradient:

$$\big[ \nabla_\theta v_\theta(s_j) - \nabla_\theta v_\theta(s_i) \big]$$

The entire behavior of Forward Only hinges on **how similar $\nabla_\theta v(s_j)$ is to $\nabla_\theta v(s_i)$**:

#### A. In Spatial Gridworlds (FourRooms & EightRooms): High Gradient Cancellation
In FourRooms, $s_i$ and $s_j$ are adjacent spatial cells in a maze (e.g., $(x, y)$ and $(x+1, y)$). 
Because neural networks generalize smoothly across adjacent spatial inputs:
$$\nabla_\theta v_\theta(s_j) \approx \nabla_\theta v_\theta(s_i)$$

Now look at what happens in the exact gradient:
$$\nabla_\theta \mathcal{E}_i = (e_i - e_j) \big[ \nabla_\theta v(s_j) - \nabla_\theta v(s_i) \big]$$

Because $\nabla v(s_j)$ and $\nabla v(s_i)$ are almost the same vector, **Term (1) and Term (2) almost perfectly cancel each other out!** 
What survives to form the true Dirichlet gradient is only their **tiny difference vector** $\Delta \nabla v = \nabla v(s_j) - \nabla v(s_i)$ (the spatial curvature of the neural representation).

When you run **Forward Only** in FourRooms:
- You drop $\nabla v(s_j)$.
- You completely unbalance the near-perfect cancellation!
- Instead of taking the tiny curvature vector $\Delta \nabla v$, you are left with the massive raw vector $-\nabla v(s_i)$.
- This raw vector points in an entirely different direction from the curvature vector, causing the alignment to collapse to **$\rho = -0.12$**.

---

#### B. In a Progression DAG (Space Invaders): No Spatial Cancellation
In Space Invaders, transitions $s_i \rightarrow s_j$ (especially with geometric jumps $m \ge 1$) do **not** represent neighboring tiles on a static map. They represent **temporally distinct stages of the game**:
- State $s_i$: Full alien grid, high shields, early wave.
- State $s_j$: Fragmented alien fleet, damaged shields, advanced wave.

The feature representations $\nabla v(s_i)$ and $\nabla v(s_j)$ are **not nearly identical vectors**:
$$\nabla_\theta v(s_j) \not\approx \nabla_\theta v(s_i)$$

Because there is **no near-cancellation between Term (1) and Term (2)**:
- Term (2) ($-\nabla v(s_i)$) already carries the primary descent direction of the loss.
- Term (1) does not define the direction through a delicate subtraction; it acts as a secondary perturbation.

---

### 2. What Term (1) and Term (2) Physically Do to the Parameters

To see why Term (1) is directionally redundant in a DAG, use the Bellman identity $e_i - \gamma e_j = -\delta_i$. As $\gamma \to 1$:
$$e_i - e_j \approx -\delta_i$$

Now rewrite both terms in terms of the Bellman error $\delta_i = r_i + \gamma v(s_j) - v(s_i)$:

#### Forward Term ($-\text{Term (2)}$):
$$-\text{Term (2)} \approx \delta_i \nabla_\theta v(s_i)$$
This is **standard semi-gradient TD(0)**. It updates the weights of state $s_i$ to push $v(s_i)$ toward its future return.

#### Backward Term ($\text{Term (1)}$):
$$\text{Term (1)} \approx - \delta_i \nabla_\theta v(s_j)$$
This is **Baird's residual gradient**. It updates the weights of the *destination* state $s_j$ along $-\delta_i$.

---

### 3. Why Baird's Term (Term 1) is Superfluous in a DAG

In general RL, why did Baird introduce $\text{Term (1)} = -\delta_i \nabla v(s_j)$ in the first place?
To prevent **semi-gradient TD from diverging on cyclic graphs** (the Deadly Triad).
In a cyclic graph with function approximation, semi-gradient TD ($-\text{Term 2}$) is not a true gradient and can circulate endlessly. Baird added Term (1) to turn the update into the true gradient of an objective.

#### But in a DAG:
1. **$P$ is Strictly Upper-Triangular (Nilpotent)**:
   All eigenvalues of $P$ are strictly zero ($\rho(P) = 0$). There are no feedback loops.
2. **Semi-Gradient TD Cannot Diverge**:
   On a DAG, dynamic programming sweeps strictly backward from the terminal state. Semi-gradient TD is a strict topological contraction.
   Therefore, **the stabilizing role of Baird's backward term (Term 1) is completely unnecessary in a DAG.**

---

### 4. Why Space Invaders Chokes on Sampling Term (1)

If Term (1) is directionally aligned with the loss surface in expectation, why does keeping it in Naive Oct 8 cause such high error (MSE 0.862 vs. 0.098 for Forward Only)?

Because along a single rollout trajectory:
$$\hat{e}_i - \hat{e}_j = (r_i + \gamma G_j - v_i) - (G_j - v_j) = \hat{\delta}_i - (1-\gamma)e_j$$

The future return $G_j$ cancels out. The sample backward term collapses into:
$$\hat{g}_{\text{backward}} = \hat{\delta}_i \nabla_\theta v(s_j)$$

In Space Invaders:
- The return scale is large ($V^* \approx 18.5$, so variance $\sim 18.5^2 \approx 340$).
- The product $\hat{\delta}_i \nabla v(s_j)$ carries massive single-transition variance and double-sampling covariance.
- **The Empirical Reality**:
  - In theory, Term (1) adds a mild correction to the forward descent.
  - In practice, on a single trajectory, Term (1) injects return-scale noise that swamps the update.

---

### Summary

| MDP Type | Relation of $\nabla v(s_i)$ to $\nabla v(s_j)$ | Role of Term (1) vs. Term (2) | What Happens if You Drop Term (1)? |
| :--- | :--- | :--- | :--- |
| **Spatial / Maze (FourRooms)** | $\nabla v(s_i) \approx \nabla v(s_j)$ (Neighbors) | **Near-cancellation**: The true gradient is defined by their small difference $\Delta \nabla v$. | **Catastrophic ($\rho = -0.12$)**: Unbalances the cancellation, replacing curvature with a massive raw vector. |
| **Acyclic / Progression (Space Invaders)** | $\nabla v(s_i) \not\approx \nabla v(s_j)$ (Separate game phases) | **Dominant forward driver**: Term (2) drives the descent; Term (1) is a secondary Baird correction. | **Optimal ($\rho = 0.982$, 9x lower MSE)**: Loses almost no directional signal while eliminating Baird's return-scale sampling variance. |

# Measuring Forward vs. Backward Divergence: DAGs vs. Cyclic MDPs

## 1. The Core Decomposition

The exact parameter gradient of the Dirichlet Energy $\mathcal{E} = \mathbf{e}^\top D(I - P)\mathbf{e}$ decomposes into Forward and Backward components:
$$\nabla_\theta \mathcal{E} = g_{\text{fwd}} + g_{\text{bwd}}$$

where:
$$\begin{aligned}
g_{\text{fwd}} &= -J^\top D(I - P)\mathbf{e} = \underbrace{J^\top D P \mathbf{e}}_{\text{Forward Cross-Term}} - \underbrace{J^\top D \mathbf{e}}_{\text{Self Term}} \\
g_{\text{bwd}} &= -J^\top (I - P^\top)D\mathbf{e} = \underbrace{(PJ)^\top D \mathbf{e}}_{\text{Backward Cross-Term}} - \underbrace{J^\top D \mathbf{e}}_{\text{Self Term}}
\end{aligned}$$

Here:
- $J^\top D P \mathbf{e} = \mathbb{E}_{(s, s') \sim (\mu, P)} [\nabla_\theta v(s) \, e(s')]$ (feature at source, error at destination).
- $(PJ)^\top D \mathbf{e} = J^\top P^\top D \mathbf{e} = \mathbb{E}_{(s, s') \sim (\mu, P)} [\nabla_\theta v(s') \, e(s)]$ (feature at destination, error at source).
- $J^\top D \mathbf{e} = \mathbb{E}_{s \sim \mu} [\nabla_\theta v(s) \, e(s)]$ is the identical self-term.

---

## 2. The Identity: Gradient Difference IS Cross-Operator Asymmetry

Because the self-term $-J^\top D \mathbf{e}$ is identical in both, subtracting them eliminates the self-term completely:

$$g_{\text{fwd}} - g_{\text{bwd}} = J^\top D P \mathbf{e} - (PJ)^\top D \mathbf{e} = J^\top (DP - P^\top D) \mathbf{e} = -2 J^\top K \mathbf{e}$$

where $K = -\frac{1}{2}(DP - P^\top D)$ is the skew-symmetric circulation matrix.

Meanwhile, their sum is the true gradient along the symmetric Laplacian $S = D - \frac{1}{2}(DP + P^\top D)$:
$$g_{\text{fwd}} + g_{\text{bwd}} = -2 J^\top S \mathbf{e} = \nabla_\theta \mathcal{E}$$

Therefore, **the cross-operator asymmetry IS the gradient difference vector**.

---

## 3. The 3 Essential Metrics

Instead of a sprawling catalog of metrics, the entire difference between DAGs and cyclic MDPs reduces to **three precise quantities**:

### Metric 1: Normalized Gradient Difference (Relative Circulation Force)
$$\mathcal{M}_1 \triangleq \frac{\|g_{\text{fwd}} - g_{\text{bwd}}\|_2}{\|\nabla_\theta \mathcal{E}\|_2} = \frac{\|J^\top D P \mathbf{e} - (PJ)^\top D \mathbf{e}\|_2}{2 \|J^\top S \mathbf{e}\|_2} = \frac{\|J^\top K \mathbf{e}\|_2}{\|J^\top S \mathbf{e}\|_2}$$

- **Physical Meaning**: Measures how large the non-reversible circulation force is relative to the true Dirichlet descent force.
- **In Samples**: Along a batch of transitions:
  $$\widehat{\mathcal{M}}_1 = \frac{\|\hat{g}_{\text{fwd}} - \hat{g}_{\text{bwd}}\|_2}{\|\hat{g}_{\text{fwd}} + \hat{g}_{\text{bwd}}\|_2}$$

---

### Metric 2: Feature Transition Collinearity (Pure Representation Continuity)
How similar are the critic representations between transition neighbors $s \to s'$?

$$\mathcal{M}_2 \triangleq \frac{\mathbb{E}_{(s, s') \sim (\mu, P)} \big[ \langle \nabla_\theta v(s), \nabla_\theta v(s') \rangle \big]}{\mathbb{E}_{s \sim \mu} \big[ \|\nabla_\theta v(s)\|_2^2 \big]}$$

In matrix form:
$$\mathbb{E}_{(s, s')} [\langle \nabla v(s), \nabla v(s') \rangle] = \sum_{s, s'} \mu(s) P(s, s') [J J^\top]_{s, s'} = \text{Tr}(J^\top D P J)$$
$$\mathbb{E}_{s} [\|\nabla v(s)\|^2] = \sum_s \mu(s) [J J^\top]_{s, s} = \text{Tr}(J^\top D J)$$

Therefore:
$$\mathcal{M}_2 = \frac{\text{Tr}(J^\top D P J)}{\text{Tr}(J^\top D J)}$$

- **Physical Meaning**: This is independent of the value error $\mathbf{e}$. It measures whether the function approximator treats consecutive states as **near-identical neighbors** ($\mathcal{M}_2 \approx 1$) or **distinct orthogonal stages** ($\mathcal{M}_2 \approx 0$).
  - **Gridworlds / Mazes**: $\mathcal{M}_2 \approx 0.90 - 0.98$ (smooth spatial generalization). High cancellation regime: $\nabla v(s') - \nabla v(s) \approx 0$.
  - **Progression DAGs**: $\mathcal{M}_2 \approx 0.00 - 0.10$ (temporally separated stages). No cancellation: forward term drives descent directly.
- **In Samples**:
  $$\widehat{\mathcal{M}}_2 = \frac{1}{B} \sum_{b=1}^B \frac{\langle \nabla_\theta v(s_t^{(b)}), \nabla_\theta v(s_{t+1}^{(b)}) \rangle}{\|\nabla_\theta v(s_t^{(b)})\|_2 \|\nabla_\theta v(s_{t+1}^{(b)})\|_2}$$

---

### Metric 3: Cross-Term Parameter Alignment (Error-Weighted Cross Cosine)
What is the cosine similarity between the Forward Cross-Term and the Backward Cross-Term in parameter space $\mathbb{R}^d$?

$$\mathcal{M}_3 \triangleq \frac{\langle J^\top D P \mathbf{e}, \, (PJ)^\top D \mathbf{e} \rangle}{\|J^\top D P \mathbf{e}\|_2 \, \|(PJ)^\top D \mathbf{e}\|_2}$$

In matrix form:
$$\langle J^\top D P \mathbf{e}, \, (PJ)^\top D \mathbf{e} \rangle = \mathbf{e}^\top P^\top D \, (J J^\top) \, P^\top D \mathbf{e}$$

- **Physical Meaning**: This is the error-weighted alignment between pulling the source state toward future return error ($J^\top D P \mathbf{e}$) vs. pushing the destination state away from current error ($(PJ)^\top D \mathbf{e}$).
  - In reversible gridworlds, this alignment is large and positive, driving the cancellation.
  - In DAGs, this alignment drops or decouples because state errors and representations at stage $t$ are decoupled from stage $t+1$.
- **In Samples**:
  $$\widehat{\mathcal{M}}_3 = \frac{\langle \hat{g}_{\text{cross, fwd}}, \, \hat{g}_{\text{cross, bwd}} \rangle}{\|\hat{g}_{\text{cross, fwd}}\|_2 \|\hat{g}_{\text{cross, bwd}}\|_2}$$
  where $\hat{g}_{\text{cross, fwd}} = \frac{1}{B}\sum_b e(s_{t+1}^{(b)}) \nabla v(s_t^{(b)})$ and $\hat{g}_{\text{cross, bwd}} = \frac{1}{B}\sum_b e(s_t^{(b)}) \nabla v(s_{t+1}^{(b)})$.

---

## 4. Summary: How to Interpret Results

| Regime | Metric 1: Relative Gradient Difference $\frac{\|g_{\text{fwd}} - g_{\text{bwd}}\|}{\|\nabla \mathcal{E}\|}$ | Metric 2: Feature Transition Collinearity $\frac{\text{Tr}(J^\top D P J)}{\text{Tr}(J^\top D J)}$ | Metric 3: Cross-Term Alignment $\cos(J^\top D P \mathbf{e}, (PJ)^\top D \mathbf{e})$ | Forward-Only Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Spatial / Cyclic (FourRooms, MountainCar)** | Small relative to raw norms; huge near-cancellation | **High ($\approx 0.90 - 0.95$)** | **High ($\approx +0.90$)** | **Fails catastrophically ($\rho \le 0$)** because dropping backward destroys delicate curvature subtraction |
| **Progression DAG (Space Invaders)** | Governed by strictly nilpotent $P$; forward term dominates | **Low ($\approx 0.00 - 0.15$)** | **Low ($\approx 0.00 - 0.20$)** | **Optimal ($\rho \approx 0.98$)** because forward term directly carries descent and avoids Baird variance |

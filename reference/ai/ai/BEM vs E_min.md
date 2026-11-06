# Bellman Error Minimization (BEM) vs. Dirichlet Energy Minimization ($E$-Min)
### Theoretical Analysis of Double-Sampling Bias, Auxiliary Predictors, and Comparison to GTD/TDC

---

## Executive Summary

| Dimension | Naive BEM / Residual Gradient | TDC / GTD2 (MSPBE) | $E$-Min (Naive Trajectory) | $E$-Min with Auxiliary Net ($V_{\text{aux}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Objective** | $\text{MSBE} = \frac{1}{2} \|\bar{\delta}\|_D^2$ | $\text{MSPBE} = \frac{1}{2} \|\Pi_J \bar{\delta}\|_D^2$ | $E = \langle \mathbf{e}, \mathbf{\bar{\delta}} \rangle_D = (1-\gamma)\|\mathbf{e}\|_D^2 + \gamma \mathcal{E}(\mathbf{e})$ | $E = (1-\gamma)\|\mathbf{e}\|_D^2 + \gamma \mathcal{E}(\mathbf{e})$ |
| **Global Optimum** | Over-smooths / biased value function | TD fixed point ($\Pi_J \bar{\delta} = 0$) | True value $V$ (when realizable) | True value $V$ (when realizable) |
| **Double-Sampling Bias** | $\gamma \, \text{Cov}(r + \gamma v', J' \mid s)$ (100% of update) | $0$ (for MSPBE); non-zero projection defect to MSBE | $\text{Cov}(r + \gamma V', J' \mid s)$ (localized to 1 of 4 terms) | **Identically Zero (0)** |
| **Auxiliary Target** | None (or $\bar{\delta}(s)$) | $\delta_t(\theta)$ (**Non-stationary**, shifts every step) | None | $G_t$ (**Stationary** mean $V(s)$) |
| **Fraction of Gradient Dependent on Aux Net** | N/A | **100%** (Controls entire correction & projection) | 0% | **$\le 25\% \times \gamma$** ($> 75\%$ pure MC trajectory data) |
| **Hessian / Curvature Issues** | None | Drops $\nabla_\theta^2 v_\theta$; inverts stiff NTK ($> 10^5$) | None | None (Pure first-order Euclidean SGD) |
| **Empirical Speed** | Fast, but converges to wrong solution | **Notoriously slow & unstable** | Fast, but biased on stochastic transitions | **Fast, stable, and unbiased** |

---

## 1. Mathematical Foundations and Notation

Let the MDP state space have size $|\mathcal{S}|$. The value function parameter vector is $\theta \in \mathbb{R}^d$.
- **Value vector:** $\mathbf{v}_\theta \in \mathbb{R}^{|\mathcal{S}|}$, where entry $i$ is $v_\theta(s_i)$.
- **Jacobian matrix:** $J \in \mathbb{R}^{|\mathcal{S}| \times d}$, where row $i$ is the parameter gradient at state $s_i$:
  $$J = \begin{bmatrix} \nabla_\theta v_\theta(s_1)^\top \\ \vdots \\ \nabla_\theta v_\theta(s_{|\mathcal{S}|})^\top \end{bmatrix}, \qquad J(s_i) \triangleq \nabla_\theta v_\theta(s_i) \in \mathbb{R}^d$$
- **State distribution:** $D = \text{diag}(\mu)$, with stationary condition $\mu^\top P = \mu^\top$.
- **Weighted inner product & norm:** $\langle \mathbf{x}, \mathbf{y} \rangle_D = \mathbf{x}^\top D \mathbf{y} = \mathbb{E}_{s \sim \mu}[x(s) y(s)]$, and $\|\mathbf{x}\|_D^2 = \langle \mathbf{x}, \mathbf{x} \rangle_D$.
- **Tangent projection operator $\Pi_J$:** Orthogonal projector onto the subspace spanned by the Jacobian columns:
  $$\Pi_J \triangleq J (J^\top D J)^{-1} J^\top D$$
  where $C \triangleq J^\top D J = \mathbb{E}_{s \sim \mu}[J(s) J(s)^\top] \in \mathbb{R}^{d \times d}$ is the empirical Fisher / Neural Tangent Kernel (NTK) matrix.
- **Value error vector:** $\mathbf{e} \triangleq V - \mathbf{v}_\theta \implies \nabla_\theta \mathbf{e} = -J$.
- **Expected Bellman error vector:** $\mathbf{\bar{\delta}} \triangleq Tv_\theta - \mathbf{v}_\theta = R - (I - \gamma P)\mathbf{v}_\theta = (I - \gamma P)\mathbf{e} \implies \nabla_\theta \mathbf{\bar{\delta}} = -(I - \gamma P) J$.
- **Sample TD error:** $\delta_t \triangleq r_t + \gamma v_\theta(s_{t+1}) - v_\theta(s_t)$.
- **Sample empirical error:** $\hat{e}_t \triangleq G_t - v_\theta(s_t)$, where $G_t = r_t + \gamma G_{t+1}$ is the single-trajectory Monte Carlo return.

---

## 2. Question 1: How Biased is $E$-Min Compared to Naive BEM and MSPBE?

### 2.1 Naively Sampled BEM (Baird's Residual Gradient)
The exact BEM objective is the Mean Squared Bellman Error:
$$\text{MSBE}(\theta) = \frac{1}{2} \|\mathbf{\bar{\delta}}\|_D^2 = \frac{1}{2} \sum_i \mu_i \bar{\delta}_i^2$$
Its exact gradient is:
$$\nabla_\theta \text{MSBE} = - J^\top (I - \gamma P)^\top D \mathbf{\bar{\delta}} = -\sum_i \mu_i \bar{\delta}_i \sum_j p_{ij} \big(J(s_i) - \gamma J(s_j)\big)$$

Naively evaluating this along a single observed transition $(s_t = i, r_t = r_{ij}, s_{t+1} = j)$ yields:
$$\widehat{\nabla_\theta \text{MSBE}}_t = -\delta_t \big(J(s_t) - \gamma J(s_{t+1})\big) = -\delta_t J(s_t) + \gamma \delta_t J(s_{t+1})$$

Conditioned on $s_t = i$:
- The first term is unbiased: $\mathbb{E}[-\delta_t J(s_t) \mid s_t = i] = -\bar{\delta}_i J(s_i)$.
- The second term couples the sample TD error $\delta_{ij}$ with the next-state Jacobian $J(s_j)$:
  $$\mathbb{E}\big[\gamma \delta_t J(s_{t+1}) \;\big|\; s_t = i\big] = \gamma \sum_j p_{ij} \delta_{ij} J(s_j)$$
Whereas the exact target requires:
$$\gamma \bar{\delta}_i \sum_j p_{ij} J(s_j) = \gamma \left(\sum_k p_{ik} \delta_{ik}\right)\left(\sum_j p_{ij} J(s_j)\right)$$

Subtracting the true target yields the **exact double-sampling bias of BEM**:
$$\mathbf{\text{Bias}_{\text{BEM}}(s_i) = \gamma \, \text{Cov}_{s_{t+1} \sim P(\cdot|s_t)}\Big( r_t + \gamma v_\theta(s_{t+1}), \, J(s_{t+1}) \;\Big|\; s_t = i \Big)}$$

---

### 2.2 Naively Sampled MSPBE
The exact MSPBE objective projects $\mathbf{\bar{\delta}}$ onto the tangent space:
$$\text{MSPBE}(\theta) = \frac{1}{2} \|\Pi_J \mathbf{\bar{\delta}}\|_D^2 = \frac{1}{2} \mathbf{\bar{\delta}}^\top D J (J^\top D J)^{-1} J^\top D \mathbf{\bar{\delta}}$$
Its exact gradient (holding the tangent projection plane locally fixed) is:
$$\nabla_\theta \text{MSPBE} = - \underbrace{\mathbb{E}\big[ (J(s_t) - \gamma J(s_{t+1})) J(s_t)^\top \big]}_{A^\top} \cdot \underbrace{\Big(\mathbb{E}\big[J(s_t) J(s_t)^\top\big]\Big)^{-1}}_{C^{-1}} \cdot \underbrace{\mathbb{E}\big[\delta_t J(s_t)\big]}_{b}$$

If sampled naively without an auxiliary network or a second timescale (reusing $(s_t, s_{t+1})$ for both $A^\top$ and $b$):
- **Single-sample matrix inversion:** If $C$ is estimated from the single sample $J(s_t) J(s_t)^\top$, the projection collapses:
  $$J(s_t)^\top \big(J(s_t) J(s_t)^\top\big)^{-1} J(s_t) = 1$$
  and naively sampled MSPBE **collapses identically into naively sampled BEM**.
- **Known covariance $C = \mathbb{E}[JJ^\top]$:** If $C$ is known, the update has bias:
  $$\mathbf{\text{Bias}_{\text{naive-MSPBE}}(s_i) = \gamma \big(J(s_i)^\top C^{-1} J(s_i)\big) \, \text{Cov}_{s_{t+1} \sim P(\cdot|s_t)}\Big( r_t + \gamma v_\theta(s_{t+1}), \, J(s_{t+1}) \;\Big|\; s_t = i \Big)}$$
  which is the exact BEM covariance scaled by the state's feature leverage metric.

---

### 2.3 Naively Sampled $E$-Min
The continuing Dirichlet energy loss is:
$$E(\theta) = (1-\gamma) \sum_i \mu_i e_i^2 + \frac{\gamma}{2} \sum_{ij} \mu_i p_{ij} (e_i - e_j)^2$$
The exact gradient of the Dirichlet component $\mathcal{E}(\mathbf{e}) = \frac{1}{2} \sum_{ij} \mu_i p_{ij} (e_i - e_j)^2$ is:
$$\nabla_\theta \mathcal{E}(\mathbf{e}) = \sum_i \mu_i \left[ -e_i J(s_i) + \left(\sum_j p_{ij} e_j\right) J(s_i) - \sum_j p_{ij} e_j J(s_j) + \sum_j p_{ij} e_i J(s_j) \right]$$

Sampling the loss $\hat{\mathcal{E}}_t = \frac{1}{2}(\hat{e}_t - \hat{e}_{t+1})^2$ along a single trajectory produces four sample terms:
$$\nabla_\theta \hat{\mathcal{E}}_t = \underbrace{-\hat{e}_t J(s_t)}_{\text{Term 1}} + \underbrace{\hat{e}_{t+1} J(s_t)}_{\text{Term 2}} - \underbrace{\hat{e}_{t+1} J(s_{t+1})}_{\text{Term 3}} + \underbrace{\hat{e}_t J(s_{t+1})}_{\text{Term 4}}$$

Conditioning on $s_t = i$:
- **Term 1 is Unbiased:** $\mathbb{E}[-\hat{e}_t J(s_t) \mid s_t = i] = -e_i J(s_i)$.
- **Term 2 is Unbiased:** $\mathbb{E}[\hat{e}_{t+1} J(s_t) \mid s_t = i] = (\sum_j p_{ij} e_j) J(s_i)$.
- **Term 3 is Unbiased:** Conditioned on $s_{t+1} = j$, the Markov return $G_{t+1}$ averages to $V(s_j)$, so $\mathbb{E}[-\hat{e}_{t+1} J(s_{t+1}) \mid s_t = i] = -\sum_j p_{ij} e_j J(s_j)$.
- **Term 4 is BIASED:** Because $\hat{e}_t = r_t + \gamma G_{t+1} - v(s_t)$, conditioning on the realized transition $s_{t+1} = j$ gives:
  $$\mathbb{E}[\hat{e}_t J(s_{t+1}) \mid s_t = i] = \sum_j p_{ij} \big(r_{ij} + \gamma V(s_j) - v(s_i)\big) J(s_j)$$
Subtracting the exact target $e_i \sum_j p_{ij} J(s_j) = \sum_j p_{ij} (V(s_i) - v(s_i)) J(s_j)$ yields the **exact bias of $E$-min**:
$$\mathbf{\text{Bias}_{E\text{-min}}(s_i) = \text{Cov}_{s_{t+1} \sim P(\cdot|s_t)}\Big( r_t + \gamma V(s_{t+1}), \, J(s_{t+1}) \;\Big|\; s_t = i \Big)}$$

### Comparison of the Biases:
1. **Scope:** In BEM, the entire target gradient is coupled to transition noise. In $E$-min, **3 of the 4 Dirichlet terms (and the entire $(1-\gamma)\|\mathbf{e}\|_D^2$ magnitude term) are completely unbiased**. The bias is localized entirely to Term 4.
2. **Evaluated Function:** BEM's covariance depends on the parameter-dependent value estimate $v_\theta(s_{t+1})$, which changes every update. $E$-min's covariance depends on the **ground-truth value function** $V(s_{t+1})$, which is fixed.

---

## 3. Question 2: What is the Best Way to Use an Auxiliary Network for $E$-Min?

Along any trajectory, the empirical error satisfies the exact recursion:
$$\hat{e}_t = r_t + \gamma G_{t+1} - v(s_t) = \big(r_t + \gamma v(s_{t+1}) - v(s_t)\big) + \gamma \big(G_{t+1} - v(s_{t+1})\big) = \mathbf{\delta_t + \gamma \hat{e}_{t+1}}$$

Substituting this into the biased Term 4 reveals its internal structure:
$$\text{Term 4} = \hat{e}_t J(s_{t+1}) = \mathbf{\delta_t J(s_{t+1}) + \gamma \hat{e}_{t+1} J(s_{t+1})}$$
Taking the expectation conditioned on $s_t = i, s_{t+1} = j$:
$$\mathbb{E}[\gamma \hat{e}_{t+1} J(s_{t+1}) \mid s_t = i] = \gamma \sum_j p_{ij} e_j J(s_j)$$
The second piece $\gamma \hat{e}_{t+1} J(s_{t+1})$ is **already unbiased** for $\gamma \sum_j p_{ij} e_j J(s_j)$! The bias comes from the first piece: the correlation between 1-step transition noise in $\delta_t$ and $J(s_{t+1})$.

---

### Option A: Using $\delta_{\text{aux}}$ to Predict Bellman Error
Train an auxiliary network $\delta_\psi(s) \approx \bar{\delta}(s) = \mathbb{E}[\delta_t \mid s_t = s]$.
- **The Control Variate Update:** Since $\mathbb{E}[\delta_t - \delta_\psi(s_t) \mid s_t] = 0$, we can subtract the sample covariance from Term 4:
  $$\widehat{\text{Term 4}}^{\text{cov-sub}} = \hat{e}_t J(s_{t+1}) - \big(\delta_t - \delta_\psi(s_t)\big) J(s_{t+1}) = \delta_\psi(s_t) J(s_{t+1}) + \gamma \hat{e}_{t+1} J(s_{t+1})$$
- **Remaining Bias:** Combined with Term 3 ($-\hat{e}_{t+1} J(s_{t+1})$), the expectation is:
  $$\mathbb{E}[\text{Term 3} + \widehat{\text{Term 4}}^{\text{cov-sub}} \mid s_t = i] = \bar{\delta}_i \sum_j p_{ij} J(s_j) - (1-\gamma)\sum_j p_{ij} e_j J(s_j)$$
  The 1-step reward/transition noise covariance is eliminated! However, because $e_i = (I-\gamma P)^{-1} \bar{\delta}_i$, a residual downstream covariance $-\gamma \, \text{Cov}(e_j, J_j)$ remains unless $\delta_\psi$ is accumulated across future steps.

---

### Option B: Using $V_{\text{aux}}$ to Predict the Value Function (The Recommended Best Way)
Instead of predicting the 1-step TD error $\delta$, train an auxiliary network $V_\psi(s)$ on Monte Carlo returns:
$$\min_\psi \mathbb{E}\big[ (G_t - V_\psi(s_t))^2 \big] \implies V_\psi^*(s) = V(s)$$
Define the auxiliary error:
$$e_{\text{aux}}(s_t) \triangleq V_\psi(s_t) - v_\theta(s_t)$$

Now substitute $e_{\text{aux}}(s_t)$ **strictly into Term 4**:
$$\nabla_\theta \hat{\mathcal{E}}_t^{\text{hybrid}} = \underbrace{-\hat{e}_t J(s_t) + \hat{e}_{t+1} J(s_t) - \hat{e}_{t+1} J(s_{t+1})}_{\text{Terms 1, 2, 3: Pure Trajectory Return } G_t, G_{t+1}} + \underbrace{e_{\text{aux}}(s_t) J(s_{t+1})}_{\text{Term 4: Decoupled via Auxiliary Net}}$$

#### Why Option B is Superior:
1. **Zero Double-Sampling Bias in a Single Step:**
   Because $e_{\text{aux}}(s_t)$ depends strictly on $s_t$, conditioning on $s_t = i$ yields:
   $$\mathbb{E}\big[ e_{\text{aux}}(s_t) J(s_{t+1}) \;\big|\; s_t = i \big] = e_{\text{aux}}(i) \sum_j p_{ij} J(s_j) = e_i \sum_j p_{ij} J(s_j)$$
   **100% of the bias is eliminated immediately.**
2. **Stationary Supervised Learning Target:**
   $V_\psi$ is trained by regression on $G_t$. For fixed policy evaluation, $G_t$ has a **stationary conditional mean $V(s)$**. In contrast, $\delta_{\text{aux}}$ is trained on $\delta_t(\theta)$, which changes constantly as $\theta$ updates.

---

## 4. Question 3: Auxiliary Network Dependency Analysis ($E$-Min vs. TDC)

A central vulnerability of auxiliary-network algorithms is how much the primary update relies on the auxiliary predictions. If an algorithm depends heavily on the auxiliary network, approximation errors in that network corrupt the entire optimization trajectory.

### 4.1 Dependency in TDC / GTD
The TDC update rule is:
$$\Delta \theta = \alpha \Big( \delta_t J(s_t) - \gamma J(s_{t+1}) \big(J(s_t)^\top w\big) \Big)$$
where $w$ is the auxiliary linear weight vector estimating the projected Bellman error:
$$\Pi_J \bar{\delta}(s_t) = J(s_t)^\top w \approx J(s_t)^\top (J^\top D J)^{-1} J^\top D \mathbf{\bar{\delta}}$$

- **100% Objective Dependency:** The objective being minimized is $\text{MSPBE} = \frac{1}{2} \|\Pi_J \mathbf{\bar{\delta}}\|_D^2$. If $w$ has error, the gradient vector points in the wrong direction entirely.
- **100% Correction Dependency:** The second term (which prevents divergence of TD under off-policy / nonlinear approximation) is entirely scaled by the auxiliary prediction $J(s_t)^\top w$.
- **Curvature Inversion Dependency:** The auxiliary weights $w$ must track the inverse feature covariance $(J^\top D J)^{-1}$. In deep neural networks, $J^\top D J$ is the Neural Tangent Kernel, whose condition number routinely exceeds $10^5$. Any tracking lag in $w$ destabilizes the update.
- **Error Resilience (Severe Vulnerability to Lag & Inversion Errors):**
  Suppose the auxiliary network has tracking/estimation error $\epsilon_w \triangleq w - w^* \in \mathbb{R}^d$.
  The error injected into the expected parameter update is:
  $$\text{Gradient Error}_{\text{TDC}} = \gamma \, \mathbb{E}\big[ J(s_{t+1}) J(s_t)^\top \big] \epsilon_w = \gamma J^\top P^\top D J \epsilon_w$$
  This causes three cascading failure modes in TDC:
  1. **Loss of Negative Definiteness (Divergence & Limit Cycles):**
     TDC's first term $\mathbb{E}[\delta_t J(s_t)]$ is the standard TD semi-gradient governed by the matrix $-J^\top D (I - \gamma P) J$, which is *not* negative definite under off-policy sampling or function approximation (the Deadly Triad). The entire mathematical purpose of the auxiliary correction $-\gamma \mathbb{E}[J(s_{t+1}) J(s_t)^\top] w^*$ is to cancel the skew-symmetric/positive eigenvalues and produce the strictly negative semi-definite system matrix $-A^\top C^{-1} A$. If $\epsilon_w$ is non-zero, this cancellation fails; positive eigenvalues re-emerge, leading to divergence, explosive updates, or oscillatory limit cycles.
  2. **Curvature Amplification through NTK Inversion ($C^{-1}$):**
     In continuous optimization, $w$ chases $w^*(\theta) = C^{-1} b(\theta)$ with lag $\epsilon_w \approx C^{-1} \big(\frac{dw^*}{d\theta} \dot{\theta}\big)$, where $C = J^\top D J$. Because the NTK matrix $C$ has near-zero eigenvalues ($\lambda_{\min} \sim 10^{-4}$ to $10^{-6}$), any movement in $\theta$ creates **massive, ill-conditioned tracking errors** in $\epsilon_w$ along the flat directions of the feature space.
  3. **Zero Anchor to True Returns:**
     Unlike Monte Carlo methods, TDC has no ground-truth return anchor. If the auxiliary network $w$ stalls or tracks poorly, the algorithm has no independent signal to fall back on; it drifts or collapses into degenerate local solutions.

---

### 4.2 Dependency in $E$-Min with $V_{\text{aux}}$
Recall the full continuing objective:
$$E(\theta) = (1-\gamma) \|\mathbf{e}\|_D^2 + \gamma \mathcal{E}(\mathbf{e})$$
The gradient update is:
$$\nabla_\theta \hat{E}_t = 2(1-\gamma) \underbrace{\big(-\hat{e}_t J(s_t)\big)}_{\text{Magnitude Term}} + \gamma \Big[ \underbrace{-\hat{e}_t J(s_t) + \hat{e}_{t+1} J(s_t) - \hat{e}_{t+1} J(s_{t+1})}_{\text{Dirichlet Terms 1, 2, 3}} + \underbrace{e_{\text{aux}}(s_t) J(s_{t+1})}_{\text{Dirichlet Term 4 (Auxiliary)}} \Big]$$

Let us compute the exact fraction of the gradient that depends on the auxiliary network:
1. **Magnitude Component ($(1-\gamma)$ weight):** **0%** depends on the auxiliary net. It uses pure Monte Carlo return $G_t$.
2. **Dirichlet Component ($\gamma$ weight):** Exactly **1 out of the 4 terms (25%)** uses the auxiliary net. The other three terms use pure trajectory returns $G_t, G_{t+1}$.
3. **Total Gradient Dependency:**
   $$\text{Auxiliary Weight Fraction} = \frac{\frac{\gamma}{4}}{2(1-\gamma) + \gamma} = \frac{\gamma}{8 - 4\gamma}$$
   - For $\gamma = 0.9$: Auxiliary dependency is $\frac{0.9}{8 - 3.6} = \mathbf{20.4\%}$.
   - For $\gamma = 0.99$: Auxiliary dependency is $\frac{0.99}{8 - 3.96} = \mathbf{24.5\%}$.
   - **Over 75% to 80% of the gradient is driven by pure, ground-truth Monte Carlo trajectory data.**

4. **Error Resilience:**
   Suppose the auxiliary network has prediction error $\epsilon(s) = V_{\text{aux}}(s) - V(s)$. The error introduced into the parameter update is:
   $$\text{Gradient Error} = \gamma \, \mathbb{E}\big[ \epsilon(s_t) J(s_{t+1}) \big]$$
   Because $\epsilon(s_t)$ does not invert any curvature matrix and only perturbs a single cross-term, the optimization remains strongly anchored by the remaining 75%+ Monte Carlo gradient flow. It cannot diverge or enter the limit cycles typical of GTD.

---

## 5. Why TDC-Type Methods are Slow in Practice & Why $E$-Min Avoids Their Pitfalls

### Three Reasons TDC is Slow in Practice:

1. **The Dropped Network Hessian:**
   The exact gradient of $\text{MSPBE} = \frac{1}{2} \mathbf{\bar{\delta}}^\top D J (J^\top D J)^{-1} J^\top D \mathbf{\bar{\delta}}$ requires differentiating through $J(\theta)$, which introduces the network Hessian tensor $\nabla_\theta^2 v_\theta(s)$. Nonlinear TDC drops this Hessian to remain $O(d)$. Dropping the Hessian turns the update into a **non-conservative vector field** (it is no longer the gradient of any scalar objective), causing rotational dynamics, oscillations, and slow convergence.
2. **Ill-Conditioned Auxiliary Inversion ($J^\top D J$):**
   The auxiliary network $w$ is implicitly solving a linear system governed by $J^\top D J$. In deep networks, the NTK spectrum decays exponentially. The condition number is extremely large ($\kappa > 10^5$), making SGD on $w$ extraordinarily stiff and slow.
3. **Severe Non-Stationarity of the Auxiliary Target:**
   TDC's auxiliary net trains on $\delta_t(\theta) = r_t + \gamma v_\theta(s_{t+1}) - v_\theta(s_t)$. Every time $\theta$ steps, the target shifts. The fast timescale is perpetually chasing a moving target.

### Why $E$-Min with $V_{\text{aux}}$ Solves These Issues:

1. **Pure First-Order Euclidean Gradient:**
   The objective $E(\theta) = \mathbf{e}^\top S \mathbf{e}$ is a quadratic form in $\mathbf{e}$. Its gradient $\nabla_\theta E = -2 J^\top S \mathbf{e}$ relies solely on the Jacobian $J(s) = \nabla_\theta v_\theta(s)$. **No Hessian terms are dropped**, and no non-conservative vector fields are created.
2. **No Matrix Inversion or Stiff Conditioning:**
   $V_{\text{aux}}$ is trained with standard MSE loss on returns: $\min_\psi \mathbb{E}[(G_t - V_\psi(s_t))^2]$. This is standard regression without inverting the feature covariance or NTK.
3. **Stationary Training Target:**
   For any evaluation policy, the expected return $\mathbb{E}[G_t \mid s_t = s] = V(s)$ is a fixed, stationary ground truth. $V_{\text{aux}}$ converges cleanly under standard supervised learning schedules.
4. **Majority Monte Carlo Signal:**
   By keeping Terms 1, 2, and 3 on the actual trajectory return, the algorithm retains the rapid convergence of Monte Carlo methods while using the Dirichlet structure to enforce smooth spatial interpolation across states.

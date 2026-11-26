# Contraction-Loss & Return-Variance Diagnostics Key

This document is the interpretation guide for the four online diagnostic metrics logged across **Gymnax**, **Brax**, **MJX** (via `core/contraction_diagnostics.py`), and **Tabular** (`bellman_error`) sweeps, as well as the plots and 2×2 scatter grids generated automatically by `scripts/sweep_diagnostic_plotting.py`.

---

## 1. The Four Primary Measures (Always About the Main Critic $v$)

All four primary metrics are defined **identically for every algorithm** (`TD`, `MC`, `E`, `BSMC`, `Rot-Gated`) on the **main critic $v$** — never overloaded with `v_aux`.

| Metric Key | Symbol | Measures (on main critic $v$) | Expected Correlation with $E / \mathrm{TD}$ Gain |
| :--- | :---: | :--- | :---: |
| `diag_rho_c` | $\rho_c(v)$ | **1. Aliasing** — centered alignment between $\nabla v(s_t)$ and its $\lambda$-trace successor gradient | **Positive ($+$)** |
| `diag_r_rot_c` | $r_{\text{rot},c}(v)$ | **2. Rotation** — centered split-half non-normality ($\|K_{J_c} u_c\| / u_c^\top S_{J_c} u_c$) | **Positive ($+$)** |
| `diag_ret_var_ratio` | $\mathrm{VR}_{\text{ret}}(v)$ | **3. Variance** — return-to-TD error ratio $\mathbb{E}[(G_t^{(1)} - v(s_t))^2] / \mathbb{E}[\delta_t^2]$ | Context-dependent |
| `diag_v_r2` | $R^2(v)$ | **4. Main Critic Accuracy** — $1 - \mathrm{MSE}(v, G_t^{(1)}) / \mathrm{Var}(G_t^{(1)})$ (**higher is better**, `1.0` = perfect) | **Negative ($-$)** (when TD's $v$ is inaccurate, $E$ has room to win) |

### Separate `v_aux`-Only Diagnostics (Logged as `NaN` on non-`v_aux` Baselines)

| Metric Key | Symbol | Measures (strictly on `v_aux`) |
| :--- | :---: | :--- |
| `diag_aux_r2` | $R^2(v_{\text{aux}})$ | **`v_aux` Accuracy** — $1 - \mathrm{MSE}(v_{\text{aux}}, G^{\text{aux}}) / \mathrm{Var}(G^{\text{aux}})$ (**higher is better**) |
| `diag_aux_vs_v` | $S_{\text{aux}}$ | **`v_aux` vs. $v$ Skill** — $1 - \mathrm{MSE}(v_{\text{aux}}, G^{\text{aux}}) / \mathrm{MSE}(v, G^{\text{aux}})$ (**$> 0$: `v_aux` better than $v$**) |

---

## 2. Exact Definitions & Online Estimation

All four quantities are computed at each PPO update before the critic gradient step, using the rollout batch of shape `(T, B)`.

### 2.1 Centered Gradient Aliasing (`diag_rho_c`, $\rho_c$)
Let $J_c = J - \mathbb{E}_\mu[J]$ be the state-mean-centered Jacobian of the critic $v_\theta$, and let $u_c$ be the unit centered $\mathrm{TD}(\lambda)$ semi-gradient direction computed on a **disjoint** slice of rollout environments (`[E_eval : min(2*E_eval, B)]`). Let $j_t = (J_c u_c)_t$ be the centered JVP and let $j_{\text{fwd},t}$ be its discounted future $\lambda$-trace:
$$j_{\text{fwd},t} = (1 - d_t)\Big(\gamma (1 - \lambda) j_{t+1} + \gamma \lambda\, j_{\text{fwd},t+1}\Big), \qquad \gamma_{\text{eff}} = \frac{\gamma (1 - \lambda)}{1 - \gamma \lambda}.$$
Then:
$$\rho_c = \frac{\mathbb{E}_\mu[j_t \, j_{\text{fwd},t}]}{\gamma_{\text{eff}}\,\mathbb{E}_\mu[j_t^2]}.$$

- **Why centering matters:** Without subtracting $\mathbb{E}_\mu[J]$, episodic boundary flux causes a constant DC shift that makes raw cosine similarity look artificially high even when centered features are orthogonal.
- **Interpretation thresholds:**
  - $\rho_c < 0.50$: **Low aliasing** — consecutive states have distinct critic gradients; TD contracts cleanly.
  - $0.50 \le \rho_c < 0.85$: **Moderate aliasing** — some feature sharing across transitions.
  - $\rho_c \ge 0.85$: **High aliasing** — $\nabla v(s_t) \approx \nabla v(s_{t+1})$; TD contraction weakens ($\alpha_{\text{eff},c} = (1 - \gamma_{\text{eff}}\rho_c)\mathbb{E}[j_t^2] \to 0$).
  - $\rho_c > 1.0$: **Super-unit / amplifying aliasing** — possible when $\|j_{t+1}\| > \|j_t\|$ along trajectories (off-policy / non-stationary / boundary growth); TD can transiently expand ($\alpha_{\text{eff},c} < 0$).

---

### 2.2 Centered Split-Half Rotation Ratio (`diag_r_rot_c`, $r_{\text{rot},c}$)
Decompose the projected $\mathrm{TD}(\lambda)$ operator $A_{J_c,\lambda} = S_{J_c,\lambda} + K_{J_c,\lambda}$ into its symmetric (contraction) part $S_{J_c,\lambda}$ and antisymmetric (rotational) part $K_{J_c,\lambda}$.
Using $E_{\text{eval}}$ probe environments (disjoint from $u_c$), we compute:
1. **Unbiased contraction rate** (across all $E_{\text{eval}}$ probe environments):
   $$\hat{\alpha}_{\text{eff},c} = \mathbb{E}[j_t^2] - \mathbb{E}[j_t\, j_{\text{fwd},t}] = u_c^\top \hat{S}_{J_c,\lambda} u_c.$$
2. **Unbiased split-half squared rotation norm** (splitting $E_{\text{eval}}$ into two independent halves $(1)$ and $(2)$):
   $$\hat{\beta}_{\text{eff},c}^2 = \left\langle \hat{K}_{J_c,\lambda}^{(1)} u_c,\; \hat{K}_{J_c,\lambda}^{(2)} u_c \right\rangle.$$
   Because $\mathbb{E}[\|\hat{K} u\|^2] = \|Ku\|^2 + \mathrm{Var}(\hat{K}u)$, a naive single-batch norm suffers a positive $O(1/\sqrt{B})$ noise floor on reversible or stochastic MDPs. The split-half inner product is **strictly unbiased** ($\mathbb{E}[\hat{\beta}^2] = \|K_{J_c,\lambda} u_c\|^2$, with zero noise floor).
3. **EMA before square root:** Raw signed $\hat{\alpha}_{\text{eff},c}$ and $\hat{\beta}_{\text{eff},c}^2$ are smoothed with an exponential moving average (`DIAG_EMA = 0.95`) *before* taking the square root and ratio:
   $$r_{\text{rot},c} = \frac{\sqrt{\max\big(0,\; \mathrm{EMA}(\hat{\beta}_{\text{eff},c}^2)\big)}}{\max\big(\mathrm{EMA}(\hat{\alpha}_{\text{eff},c}),\; 10^{-6}\big)}.$$

- **Relationship to $\rho_c$:** $\rho_c$ measures *local* gradient aliasing (how small the denominator $\alpha_{\text{eff},c}$ gets), while $r_{\text{rot},c}$ measures *global* non-normality (whether the antisymmetric backward-propagation term $K_{J_c,\lambda}$ actually dominates $S_{J_c,\lambda}$). High $\rho_c$ on a **reversible** MDP has small $r_{\text{rot},c}$ (TD is slow, not rotational); high $\rho_c$ on a **directed** MDP has large $r_{\text{rot},c}$ (TD rotates and overshoots, and $E$'s backward term directly cancels the rotation).
- **Interpretation thresholds:**
  - $r_{\text{rot},c} < 0.25$: **Contraction-dominated** (below the default rotation-gate threshold $\tau = 0.25$); TD is well-conditioned and $E$'s backward term offers little stability benefit.
  - $0.25 \le r_{\text{rot},c} < 1.0$: **Moderate rotational coupling**; TD experiences mild contraction cancellation; rotation-gated $E$ begins opening $\omega \in (0, 0.6)$.
  - $r_{\text{rot},c} \ge 1.0$: **Rotation-dominated** ($\|K u\| \ge u^\top S u$); TD loses strong contraction / suffers rotational overshoot, and $E$'s backward term strongly improves conditioning.

---

### 2.3 Return-to-TD Variance Ratio (`diag_ret_var_ratio`, $\mathrm{VR}_{\text{ret}}$)
Over the full rollout `(T, B)`, let $\delta_t = r_t + \gamma(1 - d_t^{\text{true}}) v(s_{t+1}) - v(s_t)$ be the 1-step TD error, and let $e_t^{\text{MC}} = G_t^{(1)} - v(s_t)$ be the rollout-truncated Monte Carlo return error computed via the reverse scan $e_t^{\text{MC}} = \delta_t + \gamma(1 - d_t) e_{t+1}^{\text{MC}}$. Then:
$$\mathrm{VR}_{\text{ret}} = \frac{\mathbb{E}\!\left[(e_t^{\text{MC}})^2\right]}{\max\!\left(\mathbb{E}\!\left[\delta_t^2\right],\; 10^{-8}\right)}.$$
*(Using the second-moment ratio avoids $0/0$ blowups on constant-reward environments like CartPole or MountainCar at initialization when $\mathrm{Var}(\delta_t) \approx 0$, and equals $\mathrm{Var}(G_t^{(1)} - v(s_t)) / \mathrm{Var}(\delta_t)$ as soon as $\mathbb{E}[\delta_t] \approx 0$.)*

- **What it measures:** How much variance increases when moving from 1-step bootstrapping ($\delta_t$) to full multi-step returns ($G_t^{(1)}$). This quantity is **environment + policy intrinsic** (defined identically for all algorithms).
- **Interpretation thresholds:**
  - $\mathrm{VR}_{\text{ret}} \approx 1\text{--}5$: **Low variance inflation** — dense/smooth rewards or short effective horizon; $E$'s multi-step / `v_aux` targets are cheap and clean.
  - $\mathrm{VR}_{\text{ret}} \approx 5\text{--}25$: **Moderate variance inflation** — typical for medium-horizon continuous control or MinAtar.
  - $\mathrm{VR}_{\text{ret}} > 25$: **High variance inflation** — long horizon, noisy dynamics, or sparse/delayed rewards; multi-step targets and `v_aux` regression face high target variance.

---

### 2.4 Auxiliary / Critic Unexplained Variance (`diag_aux_unexp_var`, $1 - R^2_{\text{aux}}$)
- For **`v_aux` critics** (`e_0_v_aux`, `e_lambda_v_aux`, `bsmc_v_aux`, `e_0_rot_gated`, `e_lambda_rot_gated`):
  $$1 - R^2_{\text{aux}} = \frac{\mathrm{mean}\!\left((v_{\text{aux}}(s_t) - G_t^{\text{aux}})^2\right)}{\max\big(\mathrm{Var}(G_t^{\text{aux}}),\; 10^{-8}\big)}.$$
- For **non-`v_aux` critics** (`td_0`, `td_lambda`, `bsmc_fwd`):
  $$1 - R^2_v = \frac{\mathrm{mean}\!\left((G_t^{(1)} - v(s_t))^2\right)}{\max\big(\mathrm{Var}(G_t^{(1)}),\; 10^{-8}\big)}.$$

- **What it measures:** Debiased sampled $E$-minimization relies on $v_{\text{aux}}(s_t) \approx \mathbb{E}[G_t^{\text{aux}} \mid s_t]$ to keep the backward gradient term unbiased. If $v_{\text{aux}}$ lags behind a rapidly changing policy or cannot fit high-variance returns, $1 - R^2_{\text{aux}}$ stays high and injects bias/noise into $E$'s critic update.
- **Interpretation thresholds:**
  - $1 - R^2_{\text{aux}} < 0.25$: **Well-fit `v_aux`** ($R^2 > 0.75$); debiased $E$ gradient is accurate.
  - $0.25 \le 1 - R^2_{\text{aux}} < 0.60$: **Moderately noisy `v_aux`** ($R^2 \in (0.40, 0.75)$).
  - $1 - R^2_{\text{aux}} \ge 0.60$: **Poorly fit `v_aux`** ($R^2 \le 0.40$); auxiliary critic is lagging or overwhelmed by return variance, hurting $E$ relative to TD.
  - $1 - R^2_{\text{aux}} > 1.0$: Worse than predicting the batch mean (early training or critic divergence).

---

### 2.5 `v_aux` vs. `v` Skill (`diag_aux_vs_v`, $S_{\text{aux}}$)
For **`v_aux` critics**:
$$S_{\text{aux}} = 1 - \frac{\mathbb{E}\!\left[(v_{\text{aux}}(s_t) - G_t^{\text{aux}})^2\right]}{\max\!\left(\mathbb{E}\!\left[(v(s_t) - G_t^{\text{aux}})^2\right],\; 10^{-8}\right)}$$
(logged as `0.0` for non-`v_aux` baselines).
- **Why it is noise-cancelled:** Irreducible conditional return variance $\mathbb{E}[\mathrm{Var}(G_t^{\text{aux}} \mid s_t)]$ adds identically to numerator and denominator, so **$S_{\text{aux}} > 0 \iff \|v_{\text{aux}} - V^\pi\|_\mu^2 < \|v - V^\pi\|_\mu^2$**.
- **Interpretation:**
  - **$S_{\text{aux}} > 0$ (positive):** `v_aux` is closer to $V^\pi$ than $v$ is — the error estimate $\hat{e}_t = v_{\text{aux}}(s_t) - v(s_t)$ is genuinely helping $E$'s backward term.
  - **$S_{\text{aux}} \le 0$ (zero or negative):** `v_aux` is no better than $v$ itself — the backward term is adding noise/bias rather than debiasing.

---

## 3. Diagnostic Regime Matrix: Predicting $E(\lambda)$ vs. $\mathrm{TD}(\lambda)$

| | **Low Variance & Accurate `v_aux`** (low $\mathrm{VR}_{\text{ret}}$, low $1 - R^2_{\text{aux}}$) | **High Variance or Lagging `v_aux`** (high $\mathrm{VR}_{\text{ret}}$ or high $1 - R^2_{\text{aux}}$) |
| :--- | :--- | :--- |
| **High Contraction Loss** (high $\rho_c$, high $r_{\text{rot},c} \ge 0.25$) | **$E$ wins decisively.** TD suffers rotational overshoot/cancellation while $E$ cancels rotation cleanly with low noise. | **Contested / Rotation-Gated $E$ shines.** TD loses contraction, but full $E$ pays high variance; gating $\omega(r_{\text{rot},c})$ balances both. |
| **Low Contraction Loss** (low $\rho_c$, low $r_{\text{rot},c} < 0.25$) | **Tie / slight edge to $\mathrm{TD}(\lambda)$.** TD already contracts cleanly; $E$'s backward term is unnecessary overhead. | **$\mathrm{TD}(\lambda)$ wins.** TD contracts cleanly and avoids $E$'s return-variance / `v_aux` fitting penalty. |

---

## 4. Reading the Automated Sweep Plots

At the completion of any standardized sweep (`Gymnax`, `Brax`, `MJX`), `scripts/sweep_diagnostic_plotting.py` writes:

1. **`plots/diagnostics/learning_curves_diag_*.{png,pdf}`** — Time-series of each diagnostic over environment steps for every environment and algorithm (shows *when* in training contraction loss or `v_aux` lag occurs).
2. **`plots/diagnostics/diagnostic_auc_per_env.{png,pdf}`** & **`tables/diagnostic_auc_*.{csv,md}`** — Per-environment time-averaged AUC for all 4 diagnostics.
3. **`plots/diagnostics/diagnostic_benchmark_avg_auc.{png,pdf}`** & **`tables/diagnostic_benchmark_summary.{csv,md}`** — Benchmark-average AUC (mean ± SE across environments) for comparing entire suites (e.g., Gymnax vs. Brax vs. MJX).
4. **`plots/diagnostics/scatter_2x2_vs_rel_perf_lam*.{png,pdf}`** — **2×2 scatter grid** plotting each diagnostic's per-environment AUC ($x$-axis) against the relative return gain $\frac{\mathrm{AUC}(E(\lambda)) - \mathrm{AUC}(\mathrm{TD}(\lambda))}{|\mathrm{AUC}(\mathrm{TD}(\lambda))|} \times 100\%$ ($y$-axis) at matched $\lambda$ (default $\lambda = 0.8$, plus $\lambda = 0$ when available):
   - Points above the dashed `0%` line are **green** ($E$ wins); below are **red** ($\mathrm{TD}$ wins).
   - **Top row ($\rho_c$, $r_{\text{rot},c}$):** A strong **positive** slope / Pearson $r$ / Spearman $\rho$ confirms that TD's contraction loss explains where $E$ wins.
   - **Bottom row ($\mathrm{VR}_{\text{ret}}$, $1 - R^2_{\text{aux}}$):** A strong **negative** slope confirms that return variance / `v_aux` fitting error explains where $E$ underperforms TD.

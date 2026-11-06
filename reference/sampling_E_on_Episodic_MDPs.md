**Main Takeaway 1: Previous Sampling form for E is Biased for Episodic MDPs**
I have been sampling Equation (0), which led to bias for tasks with frequent termination. In the experiments, we show this as the variant `Sept 14`, which is also what I have been doing in all experiments thus far.

**Main Takeaway 2: Unbiased Sampling form of E has High Variance**
Equation (1) allows for a fully general sampling distribution. However, the final term in Equation (1) is a sum of squared errors $e_i ^2- e_j^2$. When sampling, its variance will be quite large. In the experiments below, we call sampling Equation (1) directly `Oct 5`.

**Main Takeaway 3: Sampling E (unbiased form) is the same as sampling the value loss at terminal states**

**Main Takeaway 4: Most High Variance Terms in Equation (1) are Mean Zero**

---

Recall the following value error from Tang and Munos 2023:
$$E = \mathbf{e}^\intercal D(I-\gamma P) \mathbf{e}$$
where $\mathbf{e} = \mathbf{V}-\mathbf{v}_\theta$ is a length $N$ vector. We previously derived an error-smoothing form for the continuing case, which has low sampling variance. Now, we consider episodic MDPs.

We analyze $E$ for episodic MDPs where the simulator resets after terminating. Denote the transition matrix for the continuing version of the MDP by $P_c$, and the true transition matrix, which transitions to an absorbing state with zero reward, by $P$. We assume that the sampling distribution $\mu$ is the stationary distribution associated with $P_c$:
$$\mu^\intercal P \neq \mu^\intercal P_c = \mu.$$
In the continuing case, where $\mu^\intercal P = \mu$, we previously derived that the following was identical to $E$.
$$\begin{align}
E_c &= (1-\gamma) \|e\|_D^2 + \gamma \mathcal{E}(\mathbf{e}) \qquad (0)\\
\end{align}
$$
Where $\mathcal{E}(\mathbf{e}) \doteq \frac{1}{2}\sum_{ij} \mu_i p_{ij} (e_i - e_j)^2$ is a Dirichlet energy, which measures how "rough" the state-function $\mathbf{e}$ is.

**Equation (1)** is equivalent to $E$. The proof is at the end of this document. Equation (1) contains an additional term that goes to zero when $\mu$ is the stationary distribution, and is valid for both infinite horizon and episodic MDPs.
$$\begin{align}

E &= E_c + \frac{\gamma}{2} \sum_{j} (\mu_j -  (\mathbf{\mu}^\intercal P)_j) e_j^2 \\
 &= E_c +\frac{\gamma}{2} \mathbf{e}^\intercal (D - \text{diag}[\mathbf{\mu}^\intercal P])\mathbf{e}  \qquad \text{(1)}\\

\end{align}

$$
Note that $\text{diag}[\mathbf{x}]$ is a function that transforms a vector into a diagonal matrix, e.g. $D = \text{diag}[\mu]$.

**Equation (1) In Code**
```python
D = np.diag(mu)
D_in = np.diag(P.T @ mu)

# 1. Individual term matrices
mc_mat   = (1.0 - gamma) * D
energy_mat  = 0.5 * gamma * (D + D_in - (D @ P + P.T @ D))
correction_mat = 0.5 * gamma * (D - D_in)

# 2. Total matrix
E = mc_mat + energy_mat + correction_mat

# 3. Matches exact_E matrix identically
A = D @ (np.eye(len(P)) - gamma * P)
S = 0.5 * (A + A.T)

assert np.allclose(M_total, S)  # True!
```

**Main Takeaway 1: Sampling Equation (0) is Biased for Episodic MDPs**
Since I previously sampled equation (0), the updates were biased -- especially for tasks with frequent termination. We consider sampling Equation (1) instead. In the experiments, we show sampling Equation (0) as the variant `Sept 14`. It corresponds to all experiments thus far.

**Main Takeaway 2: Sampling Equation (1) has High Variance**
Equation (1) allows for a fully general sampling distribution. However, the final term in Equation (1) is a sum of squared errors $e_i ^2- e_j^2$. When sampling, its variance will be quite large. In the experiments below, we call sampling Equation (1) directly `Oct 5`.

**Termination Masking**
The boundary value at terminal states anchors $E$. We apply terminal masking, setting $e_k=0$ for all terminal state indices $k$ . When sampling, if a transition $(s,r,s')$ is observed, and $s'$ is the absorbing terminal state (or a done flag is `True`), we set $e(i)=r-v(s)$ and $e(j) = 0$.  Examining each term of $(1)$:

- MC Loss: $(1-\gamma)  e(s)^2$
- Dirichlet energy: $\frac{\gamma}{2} (e(s) -0)^2 = \frac{\gamma}{2} e(s)^2$
- Boundary condition (last term of (1)): $$\frac{\gamma}{2} (e(s)^2 - 0^2) = \frac{\gamma}{2} e(s)^2$$
Summing them up gives $e(s)^2 = (r-v(s))^2$ (Main Takeaway 3)
 
**Lowering the Variance of the Correction Term**
We'd like to evaluate $(D - \text{diag}[\mathbf{\mu}^\intercal P])$. Using the fact that $\mu^\intercal P_c = \mu^\intercal$, we see that correction term depends solely on states where $P$ and $P_C$ differ (states that can transition into terminal).
$$\begin{align}
D - \text{diag}(\mu^\intercal P) &= \text{diag}(\mu - \mu^\intercal P) \\
&=\text{diag}(\mu^\intercal P_c - \mu^\intercal P) \\
&=\text{diag} [\mu^\intercal (P_c -P)] \\
\end{align}
$$
Let $p_{i,T}$ be the probability that state $i$ terminates. For any state $i$ where $p_{i,T} >0$, row $i$ of $(P_c -P)$ will have $p_{i,T}$ mass on the start distribution and $-p_{i,T}$ on the terminal state. Otherwise, it is zero. To formalize that logic, let the start state distribution be $\mu_0$, and let $\mathbf{1}_{\text{term}}$ be a one-hot basis vector for the terminal state. Then, row $i$ of $(P_c - P)$ is:
$$(P_c - P)[i,:] = p_{i,T}(\mu_0^\intercal - \mathbf{1}_{\text{term}}^\intercal)$$
This can be expressed as the following outer product, where $\mathbf{p}_T$ is the final column of $P_c$.
$$P_c-P =\mathbf{p}_T (\mathbf{\mu} - \mathbf{1}_{term})^\intercal$$
Multiplying by row vector $\mu^\intercal$ obtains:
$$\begin{align}
\mathbf{w}^\intercal &= \mu^\intercal (P_c - P) = (\mathbf{\mu}^\intercal\mathbf{p}_T) (\mathbf{\mu}_0 - \mathbf{1}_{term})^\intercal \\
 &= \left(\sum_i \mu_i p_{i,T}\right) [d_0 - \mathbf{1}_{\text{term}}]^\intercal \\
&= \mathbb{E}_\mu[\text{terminates}] [\mu_0 - \mathbf{1}_{\text{term}}]^\intercal
\end{align}$$
Finally, we evaluate the last term of Equation (1):
$$
\begin{align}
\frac{\gamma}{2} \mathbf{e}^\intercal (D - \text{diag}[\mathbf{\mu}^\intercal P])\mathbf{e}   &=
\frac{\gamma}{2}\mathbf{e}^\intercal  \text{diag}(\mathbf{w}^\intercal)\mathbf{e}\\ 
&= \frac{\gamma}{2} \sum_j w_j e_j^2  \\
&= \frac{\gamma}{2}\mathbb{E}_\mu[\text{terminates}] \left[ \sum_{j} \mu_0(j) e_j^2 - \, e_{\text{term}}^2\right] \\
&= \frac{\gamma}{2}\mathbb{E}_\mu[\text{terminates}]  \sum_{j} \mu_0(j) e_j^2
\end{align}
$$
The final line is because the value at the terminal state is assumed to be zero.

Interestingly, this term reflects a grounding learning signal (based on Monte Carlo), for the start states only! We can sample the sum with $\mathbb{I}(start) e(s)^2$. The weighting based on the termination probability can be estimated with $\sum_{\text{trajectories}} \sum_{states} \mathbb{I}(\text{terminal})$.

**Main Takeaway 4: Most High Variance Terms in Equation (1) are Mean Zero**
The win is that, if $\mu_0$ is sparse (there are only a few starting states), then the variance is significantly lower than directly sampling based on Equation (1). Sampling according to Equation (2) below is called the `October 8` variant. It's the same as the September 14 version, except it adds a weighted Monte-Carlo loss term to the start states. This reflects that the start states have more incoming probability mass than simply following $P$ would imply.
$$\begin{align}
E 
 &= E_c +\frac{\gamma}{2}\mathbb{E}_\mu[\text{terminates}]  \sum_{j} \mu_0(j) e_j^2  \qquad \text{(2)}\\
\end{align}
$$
Empirical Results
----
I run *expected* PPO (no sample variance) for all three variants:
- `Sept 14`
- `Oct 5`
- `Oct 8`

The environments are environments where I have the tabular model and a matrix-based simulator thereof. I have dense and sparse versions of most envs. Also, I have a version of fourrooms with mines added (not in the doorways). There are 9 environments overall.

Running expected PPO, at each policy update step, I collect rollouts of size $128 \times 128$ from the environment (the start state comes from $\mu$). Using those samples, I estimate the gradient due to $E$-minimization for each variant. I compare these with the *ground truth expected gradient* due to gradient descent on $E$. Figure 1 shows the MSE during learning. Observe that the biased variant (Sept 14) has higher MSE on environments with terminations. 

Figure 2 shows aggregates during learning. It has cosine similarity of the gradient vectors, average MSE during learning, and average trajectory variance (e.g. empirical variance of the gradient estimate across trajectories).

The newest variant always has the highest cosine similarity and lowest MSE. It also always has lower variance than the other unbiased variant `Oct 5`.

**Figure 1: Gradient MSE on Envs**
![[gradient_mse_across_envs.pdf]]




**Figure 2: Learning Average Cosine Similarity, MSE, and variance**

![[aggregate_gradient_metrics_across_envs.pdf]]

----
**Proof of Equation (1)**

 Define $D = \text{diag}[\mu]$. We can rewrite $E= \mathbf{e}^\intercal D(I-\gamma P)\mathbf{e}$ as:
$$\begin{align}
E &= (1-\gamma) \mathbf{e}^\intercal D \mathbf{e} + \gamma \mathbf{e}^\intercal D(I-P)\mathbf{e}  \\
\end{align}
$$
The first term is $(1-\gamma) \|\mathbf{e}\|_D^2$. 

Now we begin a somewhat involved rewriting of $\mathbf{e}^\intercal D(I-P)\mathbf{e}$:
$$\begin{align}
\mathbf{e}^\intercal D(I-P)\mathbf{e} =\sum_{i} \mu_i e_i^2 -  \sum_{ij} \mu_i p_{ij} e_i e_j
\end{align}
$$
Using the expansion $2 e_i e_j = e_i^2 + e_j^2 - (e_i-e_j)^2$, we get:
$$\begin{align}
\mathbf{e}^\intercal D(I-P)\mathbf{e} =\sum_{i} \mu_i e_i^2 -  \frac{1}{2} \sum_{ij} \mu_i p_{ij} (e_i^2 + e_j^2 - (e_i-e_j)^2)
\end{align}
$$
Since $P$ is row stochastic, $\sum_j p_{ij} = 1$. Plugging this in simplifies the expression to:
$$\begin{align}
\mathbf{e}^\intercal D(I-P)\mathbf{e} &= \frac{1}{2}\sum_{i} \mu_i e_i^2 -  \frac{1}{2} \sum_{ij} \mu_i p_{ij} (e_j^2 - (e_i-e_j)^2) \\
&= \frac{1}{2} \left( \sum_{i} \mu_i e_i^2+ \sum_{ij} \mu_i p_{ij}(e_i-e_j)^2 - \sum_{ij} \mu_i p_{ij} e_j^2  \right) \\
&= \mathcal{E}(\mathbf{e}) + \frac{1}{2} \left(\sum_{i} \mu_i e_i^2 -  \sum_{ij} \mu_i p_{ij} e_j^2\right)\\
\end{align}
$$
When $\mu$ is the stationary distribution, the terms in parentheses cancel. Otherwise, we rewrite the last term:
$$\begin{align}
&\sum_{ij} \mu_i p_{ij} e_j^2 \\ =&\sum_j \left(\sum_i \mu_i p_{ij}  \right)_j e_j^2 \\=& \sum_j \left(\mathbf{\mu}^\intercal P\right)_j e_j^2
\end{align}$$
Plugging this in:
$$\begin{align}
\mathbf{e}^\intercal D(I-P)\mathbf{e} &= \mathcal{E}(\mathbf{e}) + \frac{1}{2} \sum_{j} (\mu_j -  (\mathbf{\mu}^\intercal P)_j) e_j^2 \\
 &= \mathcal{E}(\mathbf{e}) + \frac{1}{2} \mathbf{e}^\intercal (D - \text{diag}[\mathbf{\mu}^\intercal P]) \mathbf{e}\\
\end{align}
$$
----
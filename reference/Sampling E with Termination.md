In the continuing case, we have the error-smoothing form, which can be sampled.
$$\begin{align}
E &= (1-\gamma) \|e\|_D^2 + \frac{\gamma}{2} \sum_{ij}\mu_i p_{ij} (e_i - e_j)^2
\end{align}
$$
Where the smoothing term is the proportional to the *[Dirichlet energy](https://medium.com/@anupamapb/1-4-the-laplacian-quadratic-form-is-dirichlet-energy-61c783a529d4)* associated with the symmetrized MDP:
$$\mathcal{E} \doteq \sum_{ij} \mu_i p_{ij} (e_i - e_j)^2$$
Thus, for a continuing MDP, we have:
$$\begin{align}
E &= (1-\gamma) \|e\|_D^2 + \gamma \frac{\mathcal{E}}{2} 
\end{align}
$$
When the MDP has terminal transitions, we can't sample $E$ in that form. This is because the stationary distribution $\mu_i$ is based on the continuing version of $P$. Letting $P_{cont}$ be the version of $P$ where terminal states reset to the start state. The sampling distribution is due to $P_{cont}$, not $P$:
$$D = \text{diag}[\mu] \quad \text{with} \quad \begin{align}\mu^\intercal P_{cont} = \mu
\end{align}$$
The derivation of $E = (1-\gamma) \|e\|_D^2 + \gamma \frac{\mathcal{E}}{2}$ assumed $\mu$ was the stationary distribution *of* $P$, and therefore fails at that step. I have been using that form to sample, and therefore am sampling $E$ incorrectly.

So what can we sample instead?
Direct Expansion of E
------
We have that
$$\begin{equation}
E = \mathbf{e}^\intercal D(I-\gamma P)\mathbf{e} = \|e\|_D^2 - \gamma\sum_{ij} \mu_i p_{ij} e_i e_j \quad (1)
\end{equation}
$$The last term is $\gamma \mathbb{E}_{s, s' \sim \mu}e(s) e(s')$
$$\begin{equation}
E =  \mathbb{E}_{s \sim \mu} \left[ e(s)^2 - \gamma e(s) \mathbb{E}_{s' \sim P} e(s') \right]
\end{equation}
$$
This can be sampled.

Smoothing Term with a Correction
------
The cross term from equation $(1)$ can be written as follows 
$$\sum_{ij} \mu_i p_{ij} e_i e_j = \frac{1}{2} \left[ \mathcal{E} + \|e\|_D^2  - \sum_j (P^\intercal \mu)_j e_j^2 \right] \quad (2)$$
Plugging giving:
$$\begin{equation}
E = (1-\gamma)\|e\|_D^2 + \frac{\gamma}{2}\left[\mathcal{E} + \|e\|_D^2  - \sum_j (P^\intercal \mu)_j e_j^2 \right] \quad (3)
\end{equation}
$$
*IF* $\mu$ is the stationary distribution of $P$, then $P^\intercal \mu = \mu^\intercal$, and $(3)$ reverts to the continuing form:
$$\begin{equation}
E = (1-\gamma)\|e\|_D^2 + \frac{\gamma \mathcal{E}}{2} \quad (4)
\end{equation}
$$
This is the expression I have been using for sampling. However, if $\mu$ is not the stationary distribution for $P$ (e.g. terminal states reset), then we have to use $(3)$, which differs from $(4)$ by:
$$\begin{align}
(3)-(4) = \frac{\gamma}{2} \left[ \|e\|_D^2 - \sum_j (P^\intercal \mu)_j e_j^2\right]
\end{align}$$
Since I have been sampling Equation $(4)$, but should have sampled Equation $(3)$, what I've sampled under-estimates the true loss $E$ by the above, and it must be added back to correct it. 

Looking at $\sum_j (P^\intercal \mu)_j e_j^2 = \sum_j (\mu^\intercal P^\intercal)_j e_j^2$

*Proof of Equation $(2)$*
----
Re-label the cross term as $C \doteq \sum_{ij} \mu_i p_{ij} e_i e_j$. 

Starting from $\mathcal{E}$ and expanding $(e_j -e_j)^2$:
$$\begin{align}
\mathcal{E} &= \sum_{ij} \mu_i p_{ij} (e_i^2 - 2 e_i e_j + e_j^2) \\
 &=  \sum_{ij} \mu_i p_{ij} e_i^2 - 2C + \sum_{ij} \mu_i p_{ij}e_j^2 \\
&=  \|e\|_D^2 - 2 C + \sum_{ij} \mu_i p_{ij}e_j^2 \qquad (\text{since} \sum_j p_{ij} = 1) \\
&=  \|e\|_D^2 - 2C + \sum_{j} \sum_i [\mu_i p_{ij} ]e_j^2 \\
\mathcal{E}&=  \|e\|_D^2 - 2C + \sum_{j} (P^\intercal \mu)_j e_j^2
\end{align}$$
Solving for the cross term:
$$C = \frac{1}{2} \left[ \|e\|_D^2 + \sum_j (P^\top \mu)_j e_j^2 - \mathcal{E} \right] \quad (1)$$
To relate it to the earlier loss, we can write:
$$
\begin{align}
C  = \frac{1}{2}\left[\|e\|_D^2 - 2C + \|e\|_D^2 \right] \quad (2)
\end{align}
$$

Plugging $(1)$ into $(2)$ gives:
$$C = \frac{1}{2}\left[\mathcal{E} + \|e\|_D^2 -\sum_{j} (P^\intercal \mu)_j e_j^2 \right]$$
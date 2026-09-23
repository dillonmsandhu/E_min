### Summary:
Two interpretations of the $E$ loss's smoothing term:
1. Ensure the difference in consecutive predicted state values is accurate, i.e. that $v(s')-v(s) \approx V(s')-V(s)$.
2. The smoothing term can be broken into a "forward" and "backward" loss for a given state. They are of the form $v(s_i) \approx V(s_i) + [v(s_j) - V(s_j)]$, where $s_j$ is a neighboring state that either goes *into* $s_i$ (backward loss) or that $s_i$ transitions to (forward loss). In effect, if a neighboring state $s_j$ over-estimates its value, $v(s_i)$ is increased.

Additionally, I show that there are two different ways to sample the gradient of $E$, which might be useful for alternative learning updates, e.g. using classification, or help stabilize learning, e.g. by ensuring the gradient is always for a single input, which may help the optimizer. 
## Preliminaries
Recall the value error from Tang and Munos 2023:
$$E = e^\intercal D(I-\gamma P) e$$
where $e = V-v_\theta$ is a length $N$ vector. Similar to Proposition 3.1 from [A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189) we can rewrite it as follows:
$$\begin{align}
E &= (1-\gamma) \|e\|_D^2 + \frac{\gamma}{2} \sum_{ij}\mu_i p_{ij} (e_i - e_j)^2 \\
\end{align}
$$
There are two ways to sample the gradient of the second term, a pairwise way using transition $s_t \rightarrow s_{t+1}$ and a second way that incorporates stop gradients using data from the triple  $s_{t-1} \rightarrow s_t \rightarrow s_{t+1}$.
#### Gradient 1: Pairwise Sampling of the Smoothing Loss
To estimate the second term, we can sample $s_{t} \sim \mu_t$ and $s_{t+1}$ according to the $t^{th}$ row of $P$, as well as returns $G_t$. Letting $\hat{e}_t = G_t - v_t$, this gives the following pairwise loss:
$$
L_{pw} = \frac{\gamma}{2} \sum_t  (\hat{e}_t - \hat{e}_{t+1})^2
$$
Taking the gradient with respect to $\theta$:
$$
\nabla L_{pw} = \gamma \sum_t  ( \hat{e}_{t+1} - \hat{e}_t) \nabla v_{t} -\gamma \sum_t  (\hat{e}_{t+1} - \hat{e}_t)\nabla v_{t+1}
$$
This can be rewritten by grouping terms based on the index of $\nabla v_{t}$. This rewritten form will ultimately match the stop gradient formulation in the next section.
$$
\nabla L_{pw} = -\gamma \sum_t [(\hat{e}_t - \hat{e}_{t+1})+ (\hat{e}_t - \hat{e}_{t-1})]\nabla v_{t} 
$$
**Interpretation: Accurate Difference in Consecutive State Values**
The smoothing term can be written as $e-e'= (V-V') - (v-v')$
This makes it clear that the smoothing term is designed to ensure the difference in predicted state values is accurate, i.e. that $v'-v \approx V'-V$.

### Gradient 2: Constructing Targets with a Stop Gradient
Alternatively, we can apply a stop gradient to the value estimates of $s_{t+1}$ and $s_{t-1}$ to construct the following smoothing loss for each time step:
$$l_{triple}(t) = \frac{\gamma}{2} (\hat{e}_t - \text{sg} (\hat{e}_{t+1}))^2 + \frac{\gamma}{2}(\hat{e}_t - \text{sg} (\hat{e}_{t-1}))^2$$
The gradient for each time step ends up being one component of the summation for $\nabla L_{pw}$.
$$\nabla_\theta l_{triple}(t) = -\gamma (\hat{e}_t -  \hat{e}_{t+1}) \nabla v_t -\gamma (\hat{e}_t -  \hat{e}_{t-1}) \nabla v_t$$
That is, $\nabla L_{triple} = \sum_t \nabla_\theta l_{triple}(t) = \nabla L_{pw}$

**Interpretation: Forward and Backward smoothing targets**
One can view the above form as fitting $v_t$ to a forward and backward target:
- **Forward Target:** $v_t \approx (G_t - \hat{e}_{t+1})$
- **Backward Target:** $v_t \approx (G_t - \hat{e}_{t-1})$

If the value is *overestimated* for the next state $(\hat{e}_{t+1} <0)$, it adjusts the $G$ up compared to Monte Carlo. If the value is underestimated for the next state, it adjusts the $G$ down from its monte carlo estimate. 
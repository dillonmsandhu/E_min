We can do a similar analysis as Tang and Munos 2023 for $TD(\lambda)$.

### TD($\lambda$)
Letting $L \doteq (I-\gamma \lambda P)^{-1}$, $TD(\lambda)$ makes the following update in expectation:
$$\begin{align}
\dot\theta  &=  \alpha J^\intercal D L\delta \\
(k\times 1) &=(k \times N)(N\times N)(N\times N)(N\times 1)
\end{align}
$$
where $J = \nabla_\theta v_\theta$ is the Jacobian of the current value estimate. It's straightforward to derive this using $T^\lambda v  = v + LD \delta$ and the $TD$ update $\dot{\theta} = J^\intercal (T^\lambda v-v)$.

Since $D L \delta = ALe$ (as I show below), the update can be rewritten as:
$$
\begin{align}
&\text{TD Update} \\
\dot\theta  &= \alpha J^\intercal ALe
\end{align}
$$
*Proof that $D L \delta = ALe$:
$$\begin{align}
D L \delta 
&=DL(R - (I-\gamma P)v)\\
&=DL(I-\gamma P)(V - v)\\
&=DL(I-\gamma P)e\\
&=D(I-\gamma P)Le \\
&=ALe
\end{align}
$$
The second-to-last line follows since $L$ and $(I-\gamma P)$ commute.*
### Gradient Descent on $E_\lambda$
Consider the following symmetrized $\lambda$ loss $E_\lambda \doteq \frac{1}{2}e^\intercal AL e$. For the classic TD($\lambda$) sanity check, note that as $\lambda \rightarrow 1$, $E_\lambda \rightarrow e^\intercal D e$, and when $\lambda=0$, $E_\lambda = e^\intercal A e$. 

Gradient descent on $E_\lambda$ makes the update:
$$\begin{align}
&\text{GD on } E(\lambda) \\
\dot \theta &= \alpha \frac{1}{2} J^\intercal (AL + L^\intercal A^\intercal)e \\
\end{align}$$
Or, when $AL$ is symmetric, 
$$\begin{align}
\dot \theta = \alpha J^\intercal ALe \\
\end{align}$$
This shows that $TD(\lambda)$ is the same as gradient descent on $E_\lambda$ when $AL$ is symmetric. 


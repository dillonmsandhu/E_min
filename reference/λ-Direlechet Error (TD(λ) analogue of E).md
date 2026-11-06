We can do a similar analysis as Tang and Munos 2023 for $TD(\lambda)$.

Letting $L = (I-\gamma \lambda P)^{-1}$, $TD(\lambda)$ makes the following update in expectation:
$$\begin{align}
\dot\theta  &=  \alpha J^\intercal D L\delta \\
(k\times 1) &=(k \times N)(N\times N)(N\times N)(N\times 1)
\end{align}
$$
where $J = \nabla_\theta v_\theta$ is the Jacobian of the current value estimate. 

Since $D L \delta = ALe$ (as I show below), the update can be rewritten as:
$$
\begin{align}
\dot\theta  &= \alpha J^\intercal ALe  \\
\dot\theta  &=  \langle J, ALe \rangle \qquad \\
\end{align}
$$
The final expression is a shorthand for the length $k$ vector of inner products of the columns of $J$ and rows of $LAe$, which is a  $k \times 1 \text{ vector with entry } i = \langle\frac{dv}{d\theta_i} ,ALe \rangle$.

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

Now consider gradient descent directly on $E_\lambda \doteq \frac{1}{2}e^\intercal AL e$.
$$
\begin{align}
\nabla_\theta E_\lambda = -\frac{1}{2} J^\intercal (AL + L^\intercal A^\intercal)e \\
\nabla_\theta E_\lambda = -\frac{1}{2} \langle J, (AL + L^\intercal A^\intercal)e \rangle
\end{align}
$$
Thus, gradient descent on $E_\lambda$ produces the update:
$$\begin{align}
\dot \theta = \alpha \frac{1}{2} J^\intercal (AL + L^\intercal A^\intercal)e \\
\end{align}$$
Or, when $AL$ is symmetric, 
$$\begin{align}
\dot \theta = \alpha J^\intercal ALe \\
\end{align}$$
This shows that $TD(\lambda)$ is the same as gradient descent on $E_\lambda$ when $AL$ is symmetric.

For the classic TD($\lambda$) sanity check, note that $E_\lambda = e^\intercal D (I-\gamma P)(I-\lambda \gamma P)^{-1} e$, becomes $e^\intercal D e$ (Monte Carlo) as $\lambda \rightarrow 1$, and $e^\intercal A e$ as $\lambda \rightarrow 0$.

.... I had another question about this $E_\lambda$, but I can't remember it...
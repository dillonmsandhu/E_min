### Positive Definiteness of AL
The contraction property of $T^\lambda$ means that $AL$ is positive definite, in a strong sense.  

**Proof**: 
The definition of $AL$ is positive definite is $x^\intercal AL x >0$ for all $x$. I will start with the fact that $T^\lambda$ is a contraction and then derive this.

We have that $|T^\lambda u - T^\lambda v| \leq c_\lambda |u-v|$, where $|\cdot|$ is the $D$-norm and $c_\lambda$ is the contraction rate. Rewriting the LHS, first rewrite $T^\lambda u$ as:
$$T^\lambda u = u + L(R - (I-\gamma P)u)$$
This implies
$$\begin{align}
T^\lambda u-T^\lambda v &= (u-v) - L(I-\gamma P)(u-v) \\
&= (I - L(I-\gamma P))(u-v) \\
\end{align}$$
Now define $x \doteq u-v$ and $M = L(I-\gamma P)$ 

Plugging this in and squaring both sides gives the result:
$$T^\lambda \text{ is a contraction} \iff |(I - M)x|^2_D \leq c_\lambda^2|x|^2_D$$
Expanding $|(I - M)x|^2_D$
$$\begin{align}
|(I - M)x|^2_D &= x^\intercal(I-M^\intercal) D(I-M)x\\
&= (x^\intercal-x^\intercal M^\intercal) D(x-Mx)\\
&= x^\intercal D x -x^\intercal DMx - x^\intercal M^\intercal D x  +x^\intercal M^\intercal D M x\\
\end{align}$$
Note that third term is a scalar, and therefore equal to its own transpose (the second term). Also observe that $DM = AL$ since $L$ commutes with $(I-\gamma P)$. Combine and rewrite the middle two terms as $2x^\intercal AL x$.
$$\begin{align}
|(I - M)x|^2_D
&= x^\intercal D x -2 x^\intercal ALx+x^\intercal M^\intercal D M x\\
\end{align}$$
Plugging into the contraction condition, and simplifying shows that the contraction property is equivalent to:
$$2 x^\intercal ALx \geq (1-c_\lambda^2 ) |x|_D^2+|M x|^2_D$$
Since the RHS is positive, $AL$ is positive definite. 
**Sanity Check for $\lambda=0$ and $\lambda \rightarrow 1$:**
When $\lambda=0$, we have that $L = I$ and $c_\lambda = \gamma$. Contraction then is equivalent to:
$$2 x^\intercal Ax \geq (1-\gamma^2 ) |x|_D^2+|(I-\gamma P) x|^2_D$$
Letting $\lambda \rightarrow 1$, we have that $c_\lambda = \frac{\gamma(1-\lambda)}{1-\gamma \lambda} \rightarrow 0$, and $L = (I-\gamma P)^{-1}$, implying $M=I$and $AL \rightarrow D$, an equality, meaning that contraction is necessarily true. 
$$2 x^\intercal Dx = 2|x|_D^2$$

# Numerical building blocks

## AR(1) shocks and their discretisation

Exogenous processes follow $z_{t+1} = (1-\rho)\mu + \rho z_t + \varepsilon_{t+1}$ with
$\varepsilon \sim N(0,\sigma_\varepsilon^2)$, $|\rho|<1$, and stationary standard deviation
$\sigma_z = \sigma_\varepsilon/\sqrt{1-\rho^2}$. With `log=True` the state is $e^{z}$.

**Tauchen.** On an evenly spaced grid $\{z_j\}$ with step $\Delta$ spanning $\pm m\sigma_z$,
$$\Pi_{ij} = \Phi\!\Big(\tfrac{z_j-\rho z_i+\Delta/2}{\sigma_\varepsilon}\Big)-\Phi\!\Big(\tfrac{z_j-\rho z_i-\Delta/2}{\sigma_\varepsilon}\Big),$$
with the end bins absorbing the tails.

**Rouwenhorst.** Start from $\Pi_2=\begin{pmatrix}p&1-p\\1-p&p\end{pmatrix}$, $p=(1+\rho)/2$ and
recurse
$$\Pi_N = p\begin{pmatrix}\Pi_{N-1}&0\\0&0\end{pmatrix}+(1-p)\begin{pmatrix}0&\Pi_{N-1}\\0&0\end{pmatrix}
+(1-p)\begin{pmatrix}0&0\\\Pi_{N-1}&0\end{pmatrix}+p\begin{pmatrix}0&0\\0&\Pi_{N-1}\end{pmatrix},$$
halving the interior rows. The grid is $\pm\sigma_z\sqrt{N-1}$ evenly spaced. It reproduces the
AR(1) variance and autocorrelation **exactly** (tests: relative error below $10^{-10}$), and
$E[z'\mid z]=\rho z$ on every node.

## Quadrature

Gauss-Hermite for $X\sim N(0,1)$: $E[f(X)]\approx\sum_k w_k f(x_k)$ with transformed nodes
$x_k=\sqrt2 t_k$ and weights $w_k/\sqrt\pi$; exact for polynomials of degree $\le 2n-1$.
Gauss-Legendre rescaled to $[a,b]$ with weights summing to one gives $E[f(U)]$ for $U\sim U(a,b)$.

## Grids

Uniform grids, and the curved grid
$$x(u) = \ell + (h-\ell)\,\frac{e^{ku}-1}{e^{k}-1},\quad u\in[0,1],$$
which puts more nodes near the lower bound (the borrowing limit), where policy functions have
their kink. $k\to0$ recovers the uniform grid.

## Linear interpolation

$V(x)\approx(1-t)V(x_i)+tV(x_{i+1})$, $t=(x-x_i)/(x_{i+1}-x_i)$. Outside the grid `clamp` holds the
end values; `linear` continues the end segments (needed by EGM so high-wealth consumption is not
biased).

## Markov chains: the GTH algorithm

The stationary distribution $\pi=\pi P$ is computed by Grassmann-Taksar-Heyman state reduction:
for $k=n,\dots,2$ set $s=\sum_{j<k}A_{kj}$, scale $A_{jk}\leftarrow A_{jk}/s$ for $j<k$ and
update $A_{ij}\leftarrow A_{ij}+A_{ik}A_{kj}$ for $i,j<k$; then $\pi_1=1$,
$\pi_j=\sum_{i<j}\pi_iA_{ij}$ and normalise. It uses only additions of non-negative numbers, hence is
accurate to machine precision with no eigen-solver.

## The fixed-point driver

Every iterative solver minimises $\|T(x)-x\|$ with the damped update
$x\leftarrow\omega T(x)+(1-\omega)x$. Convergence is declared when the **undamped** residual
$\|T(x)-x\|_\infty$ (or $\ell_1$ for distributions) falls below the tolerance.

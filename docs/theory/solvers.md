# Dynamic programming and perturbation

## Value function iteration

For a `saving` problem the household solves
$$V(a,z)=\max_{a'}\ \Big\{u\big(m(a,z)-a'\big)+\beta\sum_{z'}\Pi(z,z')V(a',z')\Big\},\qquad m(a,z)-a'>0,$$
with $m$ the `resources` function. The payoff tensor $U[j,i,k]=u(m(a_i,z_j)-a_k)$ ($-\infty$ if
infeasible) is built once; each iteration is $EV[k,j]=\beta\sum_{j'}\Pi_{jj'}V[k,j']$ followed by
$V_{new}[i,j]=\max_k(U[j,i,k]+EV[j,k])$. $T$ is a contraction with modulus $\beta$ in sup norm.

**Howard acceleration.** After each maximisation holding the policy $k^*(i,j)$ fixed,
$V\leftarrow U[j,i,k^*]+\beta\sum_{j'}\Pi_{jj'}V[k^*,j']$ is applied $H$ times (no maximisation).

**Monotone search.** The savings policy is increasing in assets for concave utility, so the search
for state $i+1$ starts at $k^*_i$.

VFI policies are exact on the grid, so their accuracy is $O(\text{grid spacing})$.

## Endogenous grid method (Carroll 2006)

With $c(a,z)$ the consumption function on the asset grid, the Euler equation is
$$u'(c(a,z)) = \beta\sum_{z'}\Pi(z,z')\,R(a',z')\,u'(c(a',z')),\qquad a'=m(a,z)-c.$$
For every *end-of-period* asset $a'_k$ and current shock $z_j$ the right-hand side is known, so
$$c^{end}_{kj}=(u')^{-1}(\text{rhs}_{kj}),\qquad m^{end}_{kj}=c^{end}_{kj}+a'_k$$
give consumption at the endogenous cash-on-hand $m^{end}$. Below $m^{end}_{0j}$ the constraint binds,
$c=m-a_{min}$. The new policy interpolates the knots $\{(m^{end}_{kj},c^{end}_{kj})\}$ (plus the
constrained segment) at $m(a_i,z_j)$ with *linear extrapolation*; $m^{end}$ must be strictly
increasing in $k$ (concavity). Euler errors are $\log_{10}|1-c_{implied}/c|$.

## Stationary equilibrium

Given a price $p$ (an interest rate) the household problem and its stationary distribution $\mu_p$
determine aggregate supply $A(p)=\sum\mu_p a$. The equilibrium price solves the excess-demand
equation $A(p)-D(p)=0$ by Brent's method on a bracket.

## First-order perturbation (Klein 2000)

Linearising $E_tf(y_{t+1},y_t)=0$ around the steady state gives $A\,E_t\hat y_{t+1}=B\hat y_t$
with $A=\partial f/\partial y_{t+1}$, $B=-\partial f/\partial y_t$. Take the generalised Schur form
$A=QSZ^H$, $B=QTZ^H$ with stable generalised eigenvalues $|t_{ii}/s_{ii}|<1$ ordered first
($s_{ii}=0$ counts as unstable). If the number of stable roots equals the number $n_x$ of
predetermined variables (Blanchard-Kahn), partition $Z=\begin{pmatrix}Z_{11}&Z_{12}\\Z_{21}&Z_{22}\end{pmatrix}$:
$$u_t=Z_{21}Z_{11}^{-1}x_t,\qquad x_{t+1}=Z_{11}S_{11}^{-1}T_{11}Z_{11}^{-1}x_t+\Sigma\varepsilon_{t+1}.$$
Unconditional covariance of the states solves $\Gamma_0=P\Gamma_0P'+\Sigma\Sigma'$ (discrete Lyapunov),
autocovariances are $\Gamma_k=P^k\Gamma_0$.

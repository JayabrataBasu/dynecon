# Estimation theory

## SMM

Let $\hat m$ be data moments (sample size $N$) and $m(\theta)$ model moments simulated on a sample
$S$ times as long (common random numbers). The estimator minimises
$g(\theta)'Wg(\theta)$, $g=m(\theta)-\hat m$. With $G=\partial m/\partial\theta'$ and
$\Omega=\lim N\,\mathrm{Var}(\hat m)$,
$$\mathrm{Var}(\hat\theta)=\Big(1+\tfrac1S\Big)\frac{(G'WG)^{-1}G'W\Omega WG(G'WG)^{-1}}{N}.$$
With $W=\Omega^{-1}$ this is $(1+1/S)(G'\Omega^{-1}G)^{-1}/N$. For $q>p$ moments the Hansen statistic
$$J=\frac{N}{1+1/S}\,g'\Omega^{-1}g\ \xrightarrow{d}\ \chi^2(q-p).$$
Bounded parameters are estimated on the unconstrained scale $u=\operatorname{logit}\frac{\theta-\ell}{h-\ell}$
(or $\log(\theta-\ell)$), which respects bounds without clipping.

## Maximum likelihood

Standard errors are $\sqrt{\operatorname{diag}[(-H)^{-1}]}$ with $H$ the Hessian of the log-likelihood.

## Kalman filter

For $y_t=Za_t+e_t$, $a_{t+1}=Ta_t+\eta_t$ ($e\sim N(0,H)$, $\eta\sim N(0,Q)$) with predicted state
$a_{t|t-1}$ and covariance $P_{t|t-1}$:
$v_t=y_t-Za_{t|t-1}$, $F_t=ZP_{t|t-1}Z'+H$, $K_t=P_{t|t-1}Z'F_t^{-1}$,
$a_{t+1|t}=T(a_{t|t-1}+K_tv_t)$, $P_{t+1|t}=T(P_{t|t-1}-K_tZP_{t|t-1})T'+Q$, and
$\ell=-\tfrac12\sum_t\big(k\log2\pi+\log|F_t|+v_t'F_t^{-1}v_t\big)$. Missing entries are dropped
from $y_t$, $Z$ and $H$ for that period. For the local-level model with $q=\sigma_\eta^2/\sigma_\varepsilon^2$
the steady-state predicted variance solves $\bar P^2-q\bar P-q=0$.

## Metropolis-Hastings

Random-walk proposals $N(x,\,2.38^2\,\hat\Sigma/d)$ with $\hat\Sigma$ the running chain covariance
(Haario et al.). Split-$\hat R$ compares within- and between-sequence variances of half chains;
the effective sample size uses Geyer's initial positive sequence of autocorrelations.

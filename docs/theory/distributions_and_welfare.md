# Distributions, inequality and welfare

## Young (2010) histogram

For a policy value $a'(i,j)$ let $a_k\le a'<a_{k+1}$ and $w=(a_{k+1}-a')/(a_{k+1}-a_k)$. Mass at
$(a_i,z_j)$ moves to $a_k$ with weight $w$ and to $a_{k+1}$ with weight $1-w$, then mixes over next
shocks with $\Pi$:
$$\mu'(k,j')=\sum_{j}\Pi_{jj'}\sum_{i}\mu(i,j)\,\big[w_{ij}\mathbf 1_{k=k_{ij}}+(1-w_{ij})\mathbf 1_{k=k_{ij}+1}\big].$$
The lottery preserves the policy mean exactly, $wa_k+(1-w)a_{k+1}=a'$, and total mass. The
stationary $\mu$ is the fixed point; the equivalent sparse transition matrix is the NumPy/SciPy
reference.

## Lorenz curve and Gini

For values $v_i\ge0$ with weights $w_i$ sorted ascending, the Lorenz curve is
$(\sum_{k\le i}w_k,\ \sum_{k\le i}w_kv_k/\sum w_kv_k)$ and
$$G = 1-\sum_i w_i\,(L_i+L_{i-1}).$$
Uniform on $[0,1]$ gives $1/3$; a lognormal with log-std $\sigma$ gives $2\Phi(\sigma/\sqrt2)-1$.

## Consumption-equivalent variation

$\lambda$ solves $V_{base}\big((1+\lambda)c\big)=V_{reform}$. For CRRA utility with $\sigma\ne1$,
$V$ scales by $(1+\lambda)^{1-\sigma}$, so
$$\lambda=\Big(\frac{V_{reform}}{V_{base}}\Big)^{1/(1-\sigma)}-1;$$
for log utility $V$ shifts by $\ln(1+\lambda)/(1-\beta)$, so $\lambda=\exp\{(1-\beta)(V_{reform}-V_{base})\}-1$.
The policy value is $V=(I-\beta T)^{-1}u$ with $T$ the policy-induced transition matrix. The aggregate
$\lambda$ uses expected values over the baseline stationary distribution.

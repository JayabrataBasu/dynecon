"""JAX backend: kernel parity, autodiff, gradients, batched solves, dispatch semantics."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import dynecon as ek  # noqa: E402
from dynecon import backends, models  # noqa: E402
from dynecon.core.errors import BackendError, BackendUnavailableError, ValidationError  # noqa: E402
from dynecon.estimation import mle  # noqa: E402
from dynecon.numerics import _dispatch  # noqa: E402
from dynecon.numerics.interpolation import linear_interp_1d  # noqa: E402
from dynecon.numerics.markov import markov_paths  # noqa: E402
from dynecon.simulation import iterate_distribution  # noqa: E402
from dynecon.solvers.batch import solve_many  # noqa: E402

pytestmark = pytest.mark.jax


def test_setup_enables_x64_and_describe_lists_jax() -> None:
    text = backends.describe()
    assert "x64 enabled: True" in text and "jax devices" in text
    assert backends.is_available("jax") and "jax" in backends.available_backends()


@pytest.mark.parametrize("mode", ["clamp", "linear"])
def test_interpolation_parity(mode: str) -> None:
    rng = np.random.default_rng(0)
    xp = np.sort(rng.uniform(-3, 3, 40)) + np.arange(40) * 1e-3
    fp = rng.standard_normal(40)
    x = rng.uniform(-4, 4, 300)
    ref = linear_interp_1d(x, xp, fp, extrapolate=mode, backend="numpy")
    out = linear_interp_1d(x, xp, fp, extrapolate=mode, backend="jax")
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10)
    # ties at nodes and the end nodes
    np.testing.assert_allclose(linear_interp_1d(xp, xp, fp, backend="jax"), fp, atol=1e-12)


def test_markov_paths_exact_parity() -> None:
    rng = np.random.default_rng(1)
    P = rng.random((4, 4))
    P /= P.sum(axis=1, keepdims=True)
    u = rng.random((80, 25))
    init = rng.integers(0, 4, 25)
    np.testing.assert_array_equal(
        markov_paths(P, 81, init, u, backend="numpy"), markov_paths(P, 81, init, u, backend="jax")
    )


def test_vfi_and_distribution_parity() -> None:
    model = models.load("consumption_savings", n_a=50, n_z=3, a_max=40.0).compile()
    a = ek.solve(model, method="vfi", backend="numpy", tol=1e-10)
    b = ek.solve(model, method="vfi", backend="jax", tol=1e-10)
    assert b.backend == "jax"
    np.testing.assert_array_equal(a.policy["assets_next_index"], b.policy["assets_next_index"])
    np.testing.assert_allclose(a.value, b.value, atol=1e-9)
    mu = np.random.default_rng(2).random(a.policy["assets_next"].shape)
    mu /= mu.sum()
    np.testing.assert_allclose(
        iterate_distribution(mu, a, 4, backend="numpy"),
        iterate_distribution(mu, a, 4, backend="jax"),
        atol=1e-13,
    )
    e1 = ek.solve(model, method="egm", backend="numpy")
    e2 = ek.solve(model, method="egm", backend="jax")
    np.testing.assert_allclose(e1.policy["consumption"], e2.policy["consumption"], atol=1e-10)


def test_autodiff_jacobians_match_finite_differences_and_analytic() -> None:
    rbc = models.load("rbc").compile()
    fd, ad = ek.solve(rbc, backend="numpy"), ek.solve(rbc, backend="jax")
    np.testing.assert_allclose(ad.P, fd.P, atol=1e-7)
    np.testing.assert_allclose(ad.F, fd.F, atol=1e-7)
    bm = ek.solve(models.load("brock_mirman", problem="dsge").compile(), backend="jax")
    np.testing.assert_allclose(bm.P, [[0.36, 1.0], [0.0, 0.9]], atol=1e-10)  # exact derivative
    np.testing.assert_allclose(bm.F, [[0.36, 1.0]], atol=1e-10)
    assert bm.backend == "jax"


def test_non_traceable_equations_raise_clear_error() -> None:
    from dataclasses import replace

    from dynecon.model import Model

    base = models.load("brock_mirman", problem="dsge").compile()
    m = Model("bad")
    m.params(**{k: base.parameters[k] for k in base.parameters})
    m.set_problem(
        "dsge",
        variables=["k", "z", "c"],
        states=["k", "z"],
        shocks=["e"],
        log_variables=["k", "z", "c"],
        equations=lambda yp, y, p: [
            yp.k - y.z * y.k**p.alpha + y.c,
            np.log(yp.z) - p.rho * np.log(y.z),  # numpy log: not traceable
            1.0 / y.c - p.beta * p.alpha * yp.z * yp.k ** (p.alpha - 1.0) / yp.c,
        ],
        shock_loadings=base.problem.shock_loadings,
        steady_state=base.problem.steady_state,
    )
    del replace
    with pytest.raises(ValidationError, match="traceable"):
        ek.solve(m.compile(), backend="jax")


def test_kalman_gradient_and_mle_with_jax_gradient() -> None:
    import jax.numpy as jnp

    from dynecon.estimation.kalman_jax import kalman_loglik_jax

    rng = np.random.default_rng(3)
    mu = np.cumsum(0.3 * rng.standard_normal(300))
    y = mu + 0.7 * rng.standard_normal(300)
    Z, T = jnp.ones((1, 1)), jnp.ones((1, 1))

    def ll(th):
        H = jnp.array([[jnp.exp(th[0])]])
        Q = jnp.array([[jnp.exp(th[1])]])
        return kalman_loglik_jax(y, Z, H, T, Q, jnp.zeros(1), 10.0 * jnp.ones((1, 1)))

    th0 = np.array([0.0, -1.0])
    g = np.asarray(jax.grad(ll)(jnp.asarray(th0)))
    h = 1e-6
    fd = np.array([(float(ll(th0 + h * e)) - float(ll(th0 - h * e))) / (2 * h) for e in np.eye(2)])
    np.testing.assert_allclose(g, fd, atol=1e-5)
    with_grad = mle(ll, th0, gradient="jax", method="BFGS")
    without = mle(lambda t: float(ll(jnp.asarray(t))), th0, method="Nelder-Mead")
    np.testing.assert_allclose(with_grad.estimates, without.estimates, atol=2e-3)
    assert np.all(with_grad.std_errors > 0)
    with pytest.raises(ValidationError):
        mle(ll, th0, gradient="torch")


def test_solve_many_matches_sequential_solves() -> None:
    model = models.load("consumption_savings", n_a=60, n_z=3, a_max=40.0).compile()
    betas = np.linspace(0.92, 0.97, 8)
    batch = solve_many(model, {"beta": betas})
    assert len(batch) == 8 and all(s.converged and s.backend == "jax" for s in batch)
    for b, sol in zip(betas, batch, strict=True):
        ref = ek.solve(model.with_params(beta=float(b)), method="egm", backend="numpy")
        np.testing.assert_allclose(sol.policy["consumption"], ref.policy["consumption"], atol=1e-8)
        assert sol.params["beta"] == pytest.approx(b)
    two = solve_many(model, [{"beta": 0.95, "r": 0.02}, {"beta": 0.95, "r": 0.03}])
    assert two[0].policy["consumption"].mean() != two[1].policy["consumption"].mean()
    with pytest.raises(ValidationError):
        solve_many(model, {"beta": [0.9, 0.95], "r": [0.02]})
    with pytest.raises(ValidationError):
        solve_many(model, {"rho": [0.5, 0.6]})  # changes the shock process
    with pytest.raises(ValidationError):
        solve_many(model, [])
    with pytest.raises(ValidationError):
        solve_many(models.load("nk3").compile(), {"beta": [0.9]})


def test_unavailable_jax_strict_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_dispatch._jax_state, "loaded", False)
    monkeypatch.setitem(_dispatch._jax_state, "warned", False)
    monkeypatch.setattr(backends, "is_available", lambda name: name == "numpy")
    ek.config.strict_backend = True
    try:
        with pytest.raises(BackendUnavailableError):
            _dispatch.resolve_backend("simulate_saving", backend="jax")
    finally:
        ek.config.strict_backend = False
    with pytest.warns(UserWarning, match="not installed"):
        name, _ = _dispatch.resolve_backend("simulate_saving", backend="jax")
    assert name == "numpy"
    ek.config.strict_backend = True
    try:
        monkeypatch.setattr(backends, "is_available", lambda name: True)
        monkeypatch.setitem(_dispatch._jax_state, "loaded", True)
        with pytest.raises(BackendError):  # simulate_saving has no JAX kernel
            _dispatch.resolve_backend("simulate_saving", backend="jax")
    finally:
        ek.config.strict_backend = False

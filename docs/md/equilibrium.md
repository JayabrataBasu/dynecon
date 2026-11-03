# Stationary equilibrium

`solve_stationary_equilibrium(model_factory, price_name, bracket, excess_demand)` finds the price
that clears a market in a heterogeneous-agent economy.

```text
        price p ──► model_factory(p) ──► solve (warm-started) ──► Young distribution mu
           ▲                                                            │
           │                                                            ▼
      brentq on f(p) = excess_demand(p, solution, mu)  ◄───────── aggregates
```

- **Aiyagari** (`econkit.models.aiyagari.equilibrium`): households save in capital; excess demand
  is `A(r) - K_d(r)` with `K_d = L ((r + delta) / alpha)^(1/(alpha-1))`. The equilibrium rate is
  below the complete-markets rate `1/beta - 1` (precautionary saving) and falls with income risk.
- **Huggett** (`econkit.models.huggett.equilibrium`): bonds are in zero net supply; excess
  demand is aggregate bond holdings.

## What the result holds

`EquilibriumSolution` is a `Solution` of the household problem at the clearing price plus
`price`, `excess_demand`, `aggregates` (market-specific dict), `household_solution`,
`distribution` and `history` (evaluated prices, excess demands, inner iterations; the total is
`history["inner_iterations_total"]`).

## Extending to a new market

Provide (1) `model_factory(price) -> CompiledModel` and (2) `excess_demand(price, sol, mu) ->
(excess, aggregates)`. If excess demand has the same sign at both ends of `bracket` a
`ConvergenceError` reports the endpoint values. The asset grid's upper bound must stay well
above the stationary support at every candidate price (the built-in models scale it with
`1 / (1/beta - 1 - r)` and report `mass_at_top`).

Transition dynamics between equilibria are out of scope for v1: comparisons are stationary.

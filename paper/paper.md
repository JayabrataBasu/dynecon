---
title: 'EconKit: a Python library for computational economics'
tags:
  - Python
  - economics
  - dynamic programming
  - heterogeneous agents
  - structural estimation
authors:
  - name: Jayabrata Basu
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 3 November 2026
bibliography: paper.bib
---

# Summary

EconKit is a Python library for specifying, solving, simulating and estimating dynamic
economic models. A model is declared once with a small domain-specific language (parameters,
states, shocks, a problem template), compiled to an immutable and fingerprinted object, and then
solved by value function iteration, the endogenous grid method [@carroll2006], first-order
perturbation [@klein2000] or a fixed point on prices; heterogeneous-agent stationary equilibria
are computed with Young's histogram method [@young2010]. Simulated models can be estimated by the
simulated method of moments, maximum likelihood with a Kalman filter, or adaptive
Metropolis-Hastings, and compared across policy regimes with consumption-equivalent welfare.

# Statement of need

Computational economics is served by excellent but fragmented tools: QuantEcon [@quantecon]
for teaching and algorithms, Dolo for perturbation and time iteration, HARK [@hark] for
lifecycle consumption models, Dynare for DSGE estimation. Researchers who move between
dynamic programming, general-equilibrium heterogeneous-agent models and structural estimation
repeat the same plumbing. EconKit provides one coherent function-and-result API across these
tasks, with a NumPy reference implementation that every compiled backend (Numba, JAX) is tested
against, analytic benchmarks for every built-in model (Brock-Mirman, permanent income,
the McCall reservation wage, the three-equation New Keynesian closed form, the Diamond OLG
steady state, the log-utility Lucas tree), and reproducible experiments that record the
software and model fingerprint behind every result. Performance claims in the repository come
only from a recorded benchmark harness.

# Acknowledgements

The design follows the methods literature cited above. No funding was received.

# References

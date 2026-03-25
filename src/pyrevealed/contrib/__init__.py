"""Contributed algorithms — not part of the core graph-based engine.

These modules implement specialized econometric methods (MLE, regression,
bootstrap, nonlinear optimization) that don't use the core graph primitives
(Floyd-Warshall, SCC, transitive closure). They remain fully functional but
are not accelerated by the Rust backend.

Modules:
    stochastic: Random utility models, IIA tests
    risk: CRRA estimation, expected utility axioms
    integrability: Slutsky matrix tests
    gross_substitutes: Slutsky decomposition, Hicksian demand
    ranking: Bradley-Terry model, preference aggregation
    context_effects: Decoy/compromise effect detection
    inference: Bootstrap confidence intervals
    power_analysis: Bronars power, Selten measure
    bronars: Bronars random demand generation
    acyclical_p: Strict preference acyclicity (redundant with GARP)
    differentiable: Smooth preferences / SARP
    gapp: Generalized axiom of price preferences
    welfare: Compensating/equivalent variation (CV/EV)
    additive: Additive separability tests
    separability: Weak separability tests
    spatial: Ideal point / metric preferences
    intertemporal: Exponential discounting tests
"""

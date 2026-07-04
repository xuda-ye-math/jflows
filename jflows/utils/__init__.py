"""Sampling utilities for jflows — the flat `jflows.utils` namespace.

Split across four modules (PLAN.md §3.4), re-exported here so
`jflows.utils.<fn>` call sites read the same as in zflows:

    metrics      : importance_weights_{F,G} (+log, + aliases),
                   compute_ESS, compute_ESS_log, coverage, resample
    optimization : lbfgs                                       (pending)
    rejuvenation : langevin, stochastic_heun,
                   hamiltonian_monte_carlo                     (pending)
    annealing    : sequential_monte_carlo,
                   annealed_importance_sampling_{F,G}          (pending)
"""

from .metrics import (
    compute_ESS,
    compute_ESS_log,
    coverage,
    importance_weights,
    importance_weights_F,
    importance_weights_G,
    importance_weights_log,
    importance_weights_log_F,
    importance_weights_log_G,
    resample,
)

__all__ = [
    "compute_ESS",
    "compute_ESS_log",
    "coverage",
    "importance_weights",
    "importance_weights_F",
    "importance_weights_G",
    "importance_weights_log",
    "importance_weights_log_F",
    "importance_weights_log_G",
    "resample",
]

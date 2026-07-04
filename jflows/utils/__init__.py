"""Sampling utilities for jflows — the flat `jflows.utils` namespace.

Split across four modules (PLAN.md §3.4), re-exported here so
`jflows.utils.<fn>` call sites read the same as in zflows:

    metrics      : importance_weights_{F,G} (+log, + aliases),
                   compute_ESS, compute_ESS_log, coverage, resample
    optimization : lbfgs (alias optimization), adamw + the low-level
                   LBFGS_State / lbfgs_init / lbfgs_step and
                   AdamW_State / adamw_init / adamw_step kernels
    rejuvenation : langevin (alias rejuvenation), stochastic_heun,
                   hamiltonian_monte_carlo (alias hmc) + the low-level
                   *_step kernels and the leapfrog integrator
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
from .optimization import (
    AdamW_State,
    LBFGS_State,
    adamw,
    adamw_init,
    adamw_step,
    lbfgs,
    lbfgs_init,
    lbfgs_step,
    optimization,
)
from .rejuvenation import (
    hamiltonian_monte_carlo,
    hmc,
    hmc_step,
    langevin,
    langevin_step,
    leapfrog,
    rejuvenation,
    stochastic_heun,
    stochastic_heun_step,
)

__all__ = [
    "AdamW_State",
    "LBFGS_State",
    "adamw",
    "adamw_init",
    "adamw_step",
    "compute_ESS",
    "compute_ESS_log",
    "coverage",
    "hamiltonian_monte_carlo",
    "hmc",
    "hmc_step",
    "importance_weights",
    "importance_weights_F",
    "importance_weights_G",
    "importance_weights_log",
    "importance_weights_log_F",
    "importance_weights_log_G",
    "langevin",
    "langevin_step",
    "lbfgs",
    "lbfgs_init",
    "lbfgs_step",
    "leapfrog",
    "optimization",
    "rejuvenation",
    "resample",
    "stochastic_heun",
    "stochastic_heun_step",
]

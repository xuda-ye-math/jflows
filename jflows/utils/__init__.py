"""Sampling utilities for jflows — the flat `jflows.utils` namespace.

Split across five modules, re-exported here in the flat
`jflows.utils` namespace:

    metrics      : importance_weights (+log; the type argument names the
                   transform type 'F'/'G'), compute_ESS, compute_ESS_log,
                   coverage, resample
    optimization : lbfgs (alias optimization), adamw + the low-level
                   LBFGS_State / lbfgs_init / lbfgs_step and
                   AdamW_State / adamw_init / adamw_step kernels
    rejuvenation : langevin (alias rejuvenation), stochastic_heun,
                   hamiltonian_monte_carlo (alias hmc) + the low-level
                   *_step kernels and the leapfrog integrator
    anneal       : sequential_monte_carlo (alias smc),
                   annealed_importance_sampling (alias ais; same 'F'/'G'
                   type argument)
    quench       : quench_and_temper (alias qt)
"""

from .anneal import (
    ais,
    annealed_importance_sampling,
    sequential_monte_carlo,
    smc,
)
from .metrics import (
    compute_ESS,
    compute_ESS_log,
    coverage,
    importance_weights,
    importance_weights_log,
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
from .quench import (
    qt,
    quench_and_temper,
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
    "ais",
    "annealed_importance_sampling",
    "compute_ESS",
    "compute_ESS_log",
    "coverage",
    "hamiltonian_monte_carlo",
    "hmc",
    "hmc_step",
    "importance_weights",
    "importance_weights_log",
    "langevin",
    "langevin_step",
    "lbfgs",
    "lbfgs_init",
    "lbfgs_step",
    "leapfrog",
    "optimization",
    "qt",
    "quench_and_temper",
    "rejuvenation",
    "resample",
    "sequential_monte_carlo",
    "smc",
    "stochastic_heun",
    "stochastic_heun_step",
]

"""jflows — JAX normalizing flows for unconditional energy-based sampling.

The JAX + equinox twin of `zflows` (same public surface, same conventions).

Public surface (built up phase by phase, see PLAN.md):

    jflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    jflows.potential : Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
                       Nlog_Gaussian_Mixture, linear_combination (+ the operator
                       algebra c*U, U+V, U-V, -U, U/c, sum([...]))
    jflows.loss      : reverse_KL_{F,G}, forward_KL_{F,G}, OT_loss
    jflows.utils     : metrics / optimization / rejuvenation / annealing

Internals (`jflows.core.*`) follow zflows' layout, with the deliberate
divergences listed in PLAN.md §3 — most notably: explicit PRNG `key`
arguments everywhere randomness occurs, immutable modules (`zeros()`
returns a new instance), no `beta` temperature arguments, and no
`torch.compile`-style machinery (`jax.jit` covers compilation).
"""

from . import flow, loss, potential, utils

__all__ = ["flow", "loss", "potential", "utils"]

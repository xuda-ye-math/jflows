"""jflows — JAX normalizing flows for unconditional energy-based sampling.

The JAX + equinox twin of `zflows` (same public surface, same conventions).

Public surface (built up phase by phase, see PLAN.md):

    jflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    jflows.potential : Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
                       Nlog_Gaussian_Mixture, linear_combination (+ the operator
                       algebra c*U, U+V, U-V, -U, U/c, sum([...]))
    jflows.loss      : reverse_KL, forward_KL (type='F'/'G'), OT_loss
    jflows.train     : train_reverse_KL, train_forward_KL — packed
                       single-stage training
    jflows.utils     : metrics / optimization / rejuvenation / annealing

Internals (`jflows.core.*`) follow zflows' layout, with the deliberate
divergences listed in PLAN.md §3 — most notably: explicit PRNG `key`
arguments everywhere randomness occurs, immutable modules (`zeros()`
returns a new instance), no `beta` temperature arguments, no
`torch.compile`-style machinery (`jax.jit` covers compilation), and a
flow-level high-level API: losses, importance weights, AIS, and
training all take the `Flow` itself (`flow.t()` is the advanced
composition layer).
"""

from . import flow, loss, potential, train, utils

__all__ = ["flow", "loss", "potential", "train", "utils"]

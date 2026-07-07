"""jflows — JAX normalizing flows for unconditional energy-based sampling.

Built on JAX and equinox.

Public surface:

    jflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    jflows.potential : Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
                       Nlog_Gaussian_Mixture, linear_combination (+ the operator
                       algebra c*U, U+V, U-V, -U, U/c, sum([...]))
    jflows.loss      : reverse_KL, forward_KL (type='F'/'G'), forward_KLX_G
    jflows.train     : train_reverse_KL, train_forward_KL,
                       train_forward_KLX_G (packed single-stage training),
                       boltzmann_reverse_KL, boltzmann_forward_KL,
                       boltzmann_forward_KLX_G (adaptive-ladder Boltzmann
                       generators), Monitor (live training-status reporter)
    jflows.utils     : metrics / optimization / rejuvenation / annealing

Internals (`jflows.core.*`) are a stripped-down port of zuko's flow/transform
machinery, with deliberate divergences — most notably: explicit PRNG `key`
arguments everywhere randomness occurs, immutable modules (`zeros()`
returns a new instance), no `beta` temperature arguments, no
`torch.compile`-style machinery (`jax.jit` covers compilation), and a
flow-level high-level API: losses, importance weights, AIS, and
training all take the `Flow` itself (`flow.t()` is the advanced
composition layer).
"""

from . import flow, loss, potential, train, utils

__all__ = ["flow", "loss", "potential", "train", "utils"]

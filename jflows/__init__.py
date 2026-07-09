"""jflows — JAX normalizing flows for unconditional energy-based sampling.

Built on JAX and equinox.

Public surface:

    jflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    jflows.potential : Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
                       Nlog_Gaussian_Mixture, linear_combination (+ the operator
                       algebra c*U, U+V, U-V, -U, U/c, sum([...]))
    jflows.loss      : reverse_KL_F, forward_KL_G, forward_KLX_G,
                       forward_X_G
    jflows.train     : train_reverse_KL_F, train_forward_KL_G,
                       train_forward_KLX_G, train_forward_KLXX_G (packed
                       single-stage training), Monitor (live training-status
                       reporter)
    jflows.boltzmann : boltzmann_reverse_KL_F, boltzmann_forward_KL_G,
                       boltzmann_forward_KLX_G, boltzmann_forward_KLXX_G
                       (adaptive-ladder Boltzmann generators built on the
                       stage trainers of jflows.train), plus the `_fixed`
                       twins that train along a caller-supplied fixed t_list
    jflows.utils     : metrics / optimization / rejuvenation / anneal / quench

Internals (`jflows.core.*`) are a stripped-down port of zuko's flow/transform
machinery, with deliberate divergences — most notably: explicit PRNG `key`
arguments everywhere randomness occurs, immutable modules (`zeros()`
returns a new instance), no `beta` temperature arguments, no
`torch.compile`-style machinery (`jax.jit` covers compilation), and a
flow-level high-level API: losses, importance weights, AIS, and
training all take the `Flow` itself (`flow.t()` is the advanced
composition layer).
"""

from . import boltzmann, flow, loss, potential, train, utils

__all__ = ["boltzmann", "flow", "loss", "potential", "train", "utils"]

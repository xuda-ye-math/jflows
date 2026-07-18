"""jflows — JAX normalizing flows for unconditional energy-based sampling.

Built on JAX and equinox.

Public surface:

    jflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    jflows.potential : Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
                       Nlog_Gaussian_Mixture, linear_combination (+ the operator
                       algebra c*U, U+V, U-V, -U, U/c, sum([...]))
    jflows.loss      : reverse_KL_F, forward_KL_G, forward_KLX_G,
                       forward_X_G
    jflows.train     : Monitor and all train_* stage drivers
    jflows.boltzmann : all adaptive and fixed-schedule boltzmann_* generators
    jflows.artifacts : save and load medium-level training outputs
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

from . import artifacts, boltzmann, flow, loss, potential, train, utils
from .version import __version__

__all__ = [
    "__version__",
    "artifacts",
    "boltzmann",
    "flow",
    "loss",
    "potential",
    "train",
    "utils",
]

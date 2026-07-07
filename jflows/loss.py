"""Training objectives for jflows flows.

Ported from `zflows/loss.py`. Every loss returns the PER-SAMPLE loss
vector, shape [N] aligned with the batch — no internal reduction. Take
`.mean()` for the scalar objective, or reweight / clip / trim the vector
first for post-hoc regularisation. There are no temperature arguments
(every potential is the energy of `exp(-U)`), and no compile wrappers
(`jax.jit` the training step at the call site).

Public API (the `type` argument names the transform type, 'F' or 'G'):
    reverse_KL  — source samples; transform is F (source -> target,
                  type='F') or G (target -> source, type='G')
    forward_KL  — target samples; same `type` convention
"""

from jax import Array

from .flow import Flow
from .potential import Potential


__all__ = [
    "forward_KL",
    "reverse_KL",
]


# ──────────────────────────────────────────────────────────────────────
# KL loss estimators — Monte Carlo KL divergences on source / target
# ──────────────────────────────────────────────────────────────────────

def reverse_KL(x: Array, target: Potential, flow: Flow, type: str) -> Array:
    """
    KL loss using source samples. The flow acts either as the forward
    map F (type='F', source -> target) or as the inverse map G = F^{-1}
    (type='G', target -> source); either way it is differentiated in its
    native direction only. Returns the per-sample contributions
        target(F(x)) - log|det J_F(x)|,      F(x) = G^{-1}(x),
    whose mean is a Monte Carlo estimate of KL(F_# source || exp(-target))
    up to an additive const.
    Input:
        x:      Array [N, d]   samples drawn from the source distribution
        target: Potential      negative log-density of the target (up to const)
        flow:   Flow           the normalizing flow
        type:   str            'F' if the flow maps source -> target;
                               'G' if it maps target -> source
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    if type == "F":
        y, ladj = flow.call_and_ladj(x)  # y = F(x), log|det J_F(x)|
    elif type == "G":
        y, ladj = flow.inv_and_ladj(x)   # y = G^-1(x), log|det J_{G^-1}(x)|
    else:
        raise ValueError(f"reverse_KL: type must be 'F' or 'G', got {type!r}")
    return target(y) - ladj


def forward_KL(y: Array, source: Potential, flow: Flow, type: str) -> Array:
    """
    KL loss using target samples. The flow acts either as the forward
    map F (type='F', source -> target) or as the inverse map G = F^{-1}
    (type='G', target -> source); either way it is differentiated in its
    native direction only. Returns the per-sample contributions
        source(G(y)) - log|det J_G(y)|,      G(y) = F^{-1}(y),
    whose mean is a Monte Carlo estimate of KL(target || (G^{-1})_# exp(-source))
    up to an additive const.
    Input:
        y:      Array [N, d]   samples drawn from the target distribution
        source: Potential      negative log-density of the source (up to const)
        flow:   Flow           the normalizing flow
        type:   str            'F' if the flow maps source -> target;
                               'G' if it maps target -> source
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    if type == "F":
        x, ladj = flow.inv_and_ladj(y)   # x = F^-1(y), log|det J_{F^-1}(y)|
    elif type == "G":
        x, ladj = flow.call_and_ladj(y)  # x = G(y), log|det J_G(y)|
    else:
        raise ValueError(f"forward_KL: type must be 'F' or 'G', got {type!r}")
    return source(x) - ladj

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
    OT_loss     — reverse KL + optimal-transport regularizers
"""

from jax import Array

from .flow import ComposedTransform, OTFlow
from .potential import Potential


__all__ = [
    "OT_loss",
    "forward_KL",
    "reverse_KL",
]


# ──────────────────────────────────────────────────────────────────────
# KL loss estimators — Monte Carlo KL divergences on source / target
# ──────────────────────────────────────────────────────────────────────

def reverse_KL(x: Array, target: Potential, transform: ComposedTransform, type: str) -> Array:
    """
    KL loss using source samples. The flow is supplied either as the
    forward map F (type='F', source -> target) or as the inverse map
    G = F^{-1} (type='G', target -> source); either way the transform is
    differentiated in its native direction only. Returns the per-sample
    contributions
        target(F(x)) - log|det J_F(x)|,      F(x) = G^{-1}(x),
    whose mean is a Monte Carlo estimate of KL(F_# source || exp(-target))
    up to an additive const.
    Input:
        x:         Array [N, d]       samples drawn from the source distribution
        target:    Potential          negative log-density of the target (up to const)
        transform: ComposedTransform  the flow map (typically obtained as flow.t())
        type:      str                'F' if `transform` is the forward map
                                      source -> target; 'G' if it is the
                                      inverse map target -> source
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    if type == "F":
        y, ladj = transform.call_and_ladj(x)      # y = F(x), log|det J_F(x)|
    elif type == "G":
        y, ladj = transform.inv.call_and_ladj(x)  # y = G^-1(x), log|det J_{G^-1}(x)|
    else:
        raise ValueError(f"reverse_KL: type must be 'F' or 'G', got {type!r}")
    return target(y) - ladj


def forward_KL(y: Array, source: Potential, transform: ComposedTransform, type: str) -> Array:
    """
    KL loss using target samples. The flow is supplied either as the
    forward map F (type='F', source -> target) or as the inverse map
    G = F^{-1} (type='G', target -> source); either way the transform is
    differentiated in its native direction only. Returns the per-sample
    contributions
        source(G(y)) - log|det J_G(y)|,      G(y) = F^{-1}(y),
    whose mean is a Monte Carlo estimate of KL(target || (G^{-1})_# exp(-source))
    up to an additive const.
    Input:
        y:         Array [N, d]       samples drawn from the target distribution
        source:    Potential          negative log-density of the source (up to const)
        transform: ComposedTransform  the flow map (typically obtained as flow.t())
        type:      str                'F' if `transform` is the forward map
                                      source -> target; 'G' if it is the
                                      inverse map target -> source
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    if type == "F":
        x, ladj = transform.inv.call_and_ladj(y)  # x = F^-1(y), log|det J_{F^-1}(y)|
    elif type == "G":
        x, ladj = transform.call_and_ladj(y)      # x = G(y), log|det J_G(y)|
    else:
        raise ValueError(f"forward_KL: type must be 'F' or 'G', got {type!r}")
    return source(x) - ladj


# ──────────────────────────────────────────────────────────────────────
# OT_loss — reverse KL + optimal-transport regularizers (OTFlow only)
# ──────────────────────────────────────────────────────────────────────

def OT_loss(
    x: Array,
    target: Potential,
    otflow: OTFlow,
    alpha_C: float = 1.0,
    alpha_R: float = 1.0,
) -> Array:
    """
    Full OT-Flow training objective: reverse KL plus the two optimal-transport
    regularizers. Specific to `jflows.flow.OTFlow` — it integrates the
    4-channel augmented ODE (position, log-det, transport cost, HJB residual)
    in a single pass via `OTFlowTransform.call_full`. Returns the per-sample
    contributions
        target(F(x)) - log|det J_F(x)| + alpha_C * C(x) + alpha_R * R(x),
    where the first two terms are exactly `reverse_KL` (the energy-based
    objective), C(x) = integral_0^1 (1/2)|grad Phi|^2 dt is the transport cost,
    and R(x) = integral_0^1 |(1/2)|grad Phi|^2 - d_t Phi| dt is the HJB residual.
    Setting alpha_C = alpha_R = 0 recovers `reverse_KL(x, target, otflow.t(), type='F')`.
    Input:
        x:       Array [N, d]   samples drawn from the source distribution
        target:  Potential      negative log-density of the target (up to const)
        otflow:  OTFlow         the optimal-transport flow (passed as the flow
                                object, not its transform, so the augmented ODE
                                is reachable)
        alpha_C: float          weight on the transport-cost regularizer (default 1.0)
        alpha_R: float          weight on the HJB-residual regularizer (default 1.0)
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    F = otflow.t().transforms[0]            # the underlying OTFlowTransform
    y, ladj, C, R = F.call_full(x)
    return target(y) - ladj + alpha_C * C + alpha_R * R

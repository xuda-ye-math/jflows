"""Training objectives for jflows flows.

Ported from `zflows/loss.py`. Every loss returns the PER-SAMPLE loss
vector, shape [N] aligned with the batch — no internal reduction. Take
`.mean()` for the scalar objective, or reweight / clip / trim the vector
first for post-hoc regularisation. There are no temperature arguments
(every potential is the energy of `exp(-U)`), and no compile wrappers
(`jax.jit` the training step at the call site).

Public API (F / G name the transform type explicitly — no bare aliases):
    reverse_KL_F  — source samples train F (source -> target)
    reverse_KL_G  — source samples train G (target -> source)
    forward_KL_F  — target samples train F
    forward_KL_G  — target samples train G
    OT_loss                    — reverse KL + optimal-transport regularizers
"""

from jax import Array

from .flow import ComposedTransform, OTFlow
from .potential import Potential


__all__ = [
    "OT_loss",
    "forward_KL_F",
    "forward_KL_G",
    "reverse_KL_F",
    "reverse_KL_G",
]


# ──────────────────────────────────────────────────────────────────────
# KL loss estimators — Monte Carlo KL divergences on source / target
# ──────────────────────────────────────────────────────────────────────

def reverse_KL_F(x: Array, target: Potential, F: ComposedTransform) -> Array:
    """
    KL loss using source samples to train F (the source -> target map).
    Returns the per-sample contributions  target(F(x)) - log|det J_F(x)|,
    whose mean is a Monte Carlo estimate of KL(F_# source || exp(-target))
    up to an additive const.
    Input:
        x:      Array [N, d]       samples drawn from the source distribution
        target: Potential          negative log-density of the target (up to const)
        F:      ComposedTransform  forward flow map (typically obtained as flow.t())
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    y, ladj = F.call_and_ladj(x)  # get y = F(x) and log_abs_det_jacobian
    return target(y) - ladj



def reverse_KL_G(x: Array, target: Potential, G: ComposedTransform) -> Array:
    """
    KL loss using source samples to train G (the target -> source map).
    Returns the per-sample contributions  target(G^-1(x)) - log|det J_{G^-1}(x)|,
    whose mean is a Monte Carlo estimate of KL(source || G_# exp(-target))
    up to an additive const.
    Input:
        x:      Array [N, d]       samples drawn from the source distribution
        target: Potential          negative log-density of the target (up to const)
        G:      ComposedTransform  flow map target -> source (typically flow.t())
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    y, ladj = G.inv.call_and_ladj(x)  # y = G^-1(x), ladj = log|det J_{G^-1}(x)|
    return target(y) - ladj


def forward_KL_F(y: Array, source: Potential, F: ComposedTransform) -> Array:
    """
    KL loss using target samples to train F (the source -> target map).
    Returns the per-sample contributions  source(F^-1(y)) - log|det J_{F^-1}(y)|,
    whose mean is a Monte Carlo estimate of KL(target || F_# exp(-source))
    up to an additive const.
    Input:
        y:      Array [N, d]       samples drawn from the target distribution
        source: Potential          negative log-density of the source (up to const)
        F:      ComposedTransform  forward flow map (typically obtained as flow.t())
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    x, ladj = F.inv.call_and_ladj(y)  # x = F^-1(y), ladj = log|det J_{F^-1}(y)|
    return source(x) - ladj



def forward_KL_G(y: Array, source: Potential, G: ComposedTransform) -> Array:
    """
    KL loss using target samples to train G (the target -> source map).
    Returns the per-sample contributions  source(G(y)) - log|det J_G(y)|,
    whose mean is a Monte Carlo estimate of KL(G_# target || exp(-source))
    up to an additive const.
    Input:
        y:      Array [N, d]       samples drawn from the target distribution
        source: Potential          negative log-density of the source (up to const)
        G:      ComposedTransform  flow map target -> source (typically flow.t())
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    x, ladj = G.call_and_ladj(y)  # x = G(y), ladj = log|det J_G(y)|
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
    where the first two terms are exactly `reverse_KL_F` (the energy-based
    objective), C(x) = integral_0^1 (1/2)|grad Phi|^2 dt is the transport cost,
    and R(x) = integral_0^1 |(1/2)|grad Phi|^2 - d_t Phi| dt is the HJB residual.
    Setting alpha_C = alpha_R = 0 recovers `reverse_KL_F(x, target, otflow.t())`.
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

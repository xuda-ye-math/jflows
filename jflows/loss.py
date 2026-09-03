"""Training objectives for jflows flows.

Every loss returns the PER-SAMPLE loss vector, shape [N] aligned with
the batch — no internal reduction. Take `.mean()` for the scalar
objective, or reweight / clip / trim the vector first for post-hoc
regularisation. There are no temperature arguments
(every potential is the energy of `exp(-U)`), and no compile wrappers
(`jax.jit` the training step at the call site).

Public API (each loss fixes the flow direction its suffix names —
F: source -> target; G: target -> source):
    reverse_KL_F — source samples; flow fixed as F (source -> target)
    forward_KL_G — target samples; flow fixed as G (target -> source)
    forward_KLX_G — target samples; forward KL + X functional, flow fixed
                  as G (target -> source), so no `type` argument
    forward_X_G — the standalone X functional on samples of a weight
                  measure omega (the target for X_pi; the mixture
                  alpha*hat_pi + (1 - alpha)*bar_nu for the mixture term),
                  flow fixed as G
    pairwise_variation — the exact batch estimate of the X functional from
                  the per-sample log-ratios, by one sort (no random pairing)
    forward_KLL1_G — target samples; forward KL + the centered L1 log-dispersion
                  of LDR-L1, |z - mean z|, flow fixed as G
"""

import jax.numpy as jnp
from jax import Array

from .flow import Flow
from .potential import Potential


__all__ = [
    "forward_KL_G",
    "forward_KLL1_G",
    "forward_KLX_G",
    "forward_X_G",
    "pairwise_variation",
    "reverse_KL_F",
]


# ──────────────────────────────────────────────────────────────────────
# KL loss estimators — Monte Carlo KL divergences on source / target
# ──────────────────────────────────────────────────────────────────────

def _trace_flow(flow: Flow, trace_key: Array | None) -> Flow:
    if trace_key is None:
        return flow
    return flow.with_trace_key(trace_key)


def reverse_KL_F(
    x: Array,
    target: Potential,
    flow: Flow,
    trace_key: Array | None = None,
) -> Array:
    """
    KL loss using source samples, with the flow fixed as the forward map F
    (source -> target), differentiated in its native direction. Returns
    the per-sample contributions
        target(F(x)) - log|det J_F(x)|,
    whose mean is a Monte Carlo estimate of KL(F_# source || exp(-target))
    up to an additive const.
    Input:
        x:      Array [N, d]   samples drawn from the source distribution
        target: Potential      negative log-density of the target (up to const)
        flow:   Flow           the normalizing flow, applied as F (source -> target)
        trace_key: Array | None optional Hutchinson key for `CNF(exact=False)`;
                              packed trainers refresh it every step. None uses
                              the constructor key for reproducible evaluation
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    flow = _trace_flow(flow, trace_key)
    y, ladj = flow.call_and_ladj(x)  # y = F(x), log|det J_F(x)|
    return target(y) - ladj


def forward_KL_G(
    y: Array,
    source: Potential,
    flow: Flow,
    trace_key: Array | None = None,
) -> Array:
    """
    KL loss using target samples, with the flow fixed as the inverse map G
    (target -> source), differentiated in its native direction. Returns
    the per-sample contributions
        source(G(y)) - log|det J_G(y)|,
    whose mean is a Monte Carlo estimate of KL(target || (G^{-1})_# exp(-source))
    up to an additive const.
    Input:
        y:      Array [N, d]   samples drawn from the target distribution
        source: Potential      negative log-density of the source (up to const)
        flow:   Flow           the normalizing flow, applied as G (target -> source)
        trace_key: Array | None optional Hutchinson key for `CNF(exact=False)`;
                              packed trainers refresh it every step
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    flow = _trace_flow(flow, trace_key)
    x, ladj = flow.call_and_ladj(y)  # x = G(y), log|det J_G(y)|
    return source(x) - ladj


def pairwise_variation(z: Array) -> Array:
    """
    Per-sample contributions whose mean is the exact mean absolute difference
    of `z` over all pairs i != j of the batch (the Gini mean difference):
        mean_{i != j} |z_i - z_j| = 2 / (N (N - 1)) * sum_i (2 r_i - N - 1) z_i,
    where r_i is the position of z_i in the ascending order of the batch. The
    per-sample contribution is
        2 (2 r_i - N - 1) / (N - 1) * z_i,
    computed from one O(N log N) sort with no random pairing and no extra
    energy evaluation; its mean is unbiased for E|z - z'| under the law the
    batch was drawn from, and its gradient in z_i is the batch sign
    correlation (1/(N - 1)) sum_{j != i} sgn(z_i - z_j).
    Input:
        z: Array [N]   per-sample log-ratios
    Output:
        v: Array [N]   contributions (reduce with .mean() for the variation)
    """
    n = z.shape[0]
    position = jnp.argsort(jnp.argsort(z)).astype(z.dtype) + 1.0
    return 2.0 * (2.0 * position - n - 1.0) / (n - 1.0) * z


def forward_KLX_G(y: Array, source: Potential, target: Potential, flow: Flow,
                  coeff_lambda: float = 1.0,
                  trace_key: Array | None = None) -> Array:
    """
    Forward KL regularised by the X functional, using target samples, with the
    flow fixed as the inverse map G (target -> source). Writing the per-sample
    log-ratio z = log(pi/nu) as
        z = source(G(y)) - target(y) - log|det J_G(y)|,
    this returns the per-sample contributions
        z + coeff_lambda * pairwise_variation(z),
    whose mean is the forward KL estimate mean(z) plus coeff_lambda times the
    X functional mean_{i != j} |z_i - z_j|, the mean absolute pairwise
    variation of z under the target, which penalises the spread of the
    log-ratio that the plain forward KL leaves free.
    Input:
        y:            Array [N, d]   samples drawn from the target distribution
        source:       Potential      negative log-density of the source (up to const)
        target:       Potential      negative log-density of the target (up to const)
        flow:         Flow           the normalizing flow, applied as G (target -> source)
        coeff_lambda: float          weight of the X functional term
        trace_key:    Array | None   optional Hutchinson key for an approximate CNF
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    flow = _trace_flow(flow, trace_key)
    x, ladj = flow.call_and_ladj(y)      # x = G(y), log|det J_G(y)|
    z = source(x) - target(y) - ladj     # per-sample log-ratio  log(pi/nu)
    return z + coeff_lambda * pairwise_variation(z)


def forward_X_G(y: Array, source: Potential, target: Potential, flow: Flow,
                trace_key: Array | None = None) -> Array:
    """
    The standalone X functional X_omega, using samples of a weight measure
    omega, with the flow fixed as the inverse map G (target -> source).
    Writing the per-sample log-ratio z = log(pi/nu) as
        z = source(G(y)) - target(y) - log|det J_G(y)|,
    this returns the per-sample contributions
        pairwise_variation(z),
    whose mean is the X functional X_omega = E_{y, y' ~ omega} |z(y) - z(y')|,
    the mean absolute pairwise variation of z under omega — the X term of
    `forward_KLX_G` on its own, so the weight is free to differ from the
    target: y ~ target gives the shape term X_pi, and y drawn from the mixture
    alpha*hat_pi + (1 - alpha)*bar_nu — hat_pi the quench and temper
    wide-coverage measure (mode discovery), bar_nu the detached pushforward
    of source samples through G^{-1} (leakage suppression) — gives the
    mixture term X_{alpha hat_pi + (1 - alpha) bar_nu} of the total
    training loss.
    Input:
        y:      Array [N, d]   samples drawn from the weight measure omega
        source: Potential      negative log-density of the source (up to const)
        target: Potential      negative log-density of the target (up to const)
        flow:   Flow           the normalizing flow, applied as G (target -> source)
        trace_key: Array | None optional Hutchinson key for an approximate CNF
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    flow = _trace_flow(flow, trace_key)
    x, ladj = flow.call_and_ladj(y)      # x = G(y), log|det J_G(y)|
    z = source(x) - target(y) - ladj     # per-sample log-ratio  log(pi/nu)
    return pairwise_variation(z)


def forward_KLL1_G(y: Array, source: Potential, target: Potential, flow: Flow,
                   coeff_lambda: float = 1.0,
                   trace_key: Array | None = None) -> Array:
    """
    Forward KL regularised by the centered L1 log-dispersion of LDR-L1, using
    target samples, with the flow fixed as the inverse map G (target -> source).
    With the per-sample log-ratio z = source(G(y)) - target(y) - log|det J_G(y)|
    this returns the per-sample contributions
        z + coeff_lambda * |z - mean(z)|,
    whose mean is the forward KL estimate plus coeff_lambda times the mean
    absolute deviation of z from its batch mean, the L1 log-dispersion
    regularization of Schopmans et al. (LDR-L1); the batch mean is the center
    and is differentiated through.
    Input:
        y:            Array [N, d]   samples drawn from the target distribution
        source:       Potential      negative log-density of the source (up to const)
        target:       Potential      negative log-density of the target (up to const)
        flow:         Flow           the normalizing flow, applied as G (target -> source)
        coeff_lambda: float          weight of the dispersion term
        trace_key:    Array | None   optional Hutchinson key for an approximate CNF
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    flow = _trace_flow(flow, trace_key)
    x, ladj = flow.call_and_ladj(y)      # x = G(y), log|det J_G(y)|
    z = source(x) - target(y) - ladj     # per-sample log-ratio  log(pi/nu)
    return z + coeff_lambda * jnp.abs(z - z.mean())

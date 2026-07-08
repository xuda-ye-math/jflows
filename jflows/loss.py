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
                  measure omega (the target for X_mu; the mixture
                  alpha*hat_mu + beta*bar_nu for the mixture term),
                  flow fixed as G
"""

import jax
import jax.numpy as jnp
from jax import Array

from .flow import Flow
from .potential import Potential


__all__ = [
    "forward_KL_G",
    "forward_KLX_G",
    "forward_X_G",
    "reverse_KL_F",
]


# ──────────────────────────────────────────────────────────────────────
# KL loss estimators — Monte Carlo KL divergences on source / target
# ──────────────────────────────────────────────────────────────────────

def reverse_KL_F(x: Array, target: Potential, flow: Flow) -> Array:
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
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    y, ladj = flow.call_and_ladj(x)  # y = F(x), log|det J_F(x)|
    return target(y) - ladj


def forward_KL_G(y: Array, source: Potential, flow: Flow) -> Array:
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
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    x, ladj = flow.call_and_ladj(y)  # x = G(y), log|det J_G(y)|
    return source(x) - ladj


def forward_KLX_G(y: Array, source: Potential, target: Potential, flow: Flow,
                  key: Array, coeff_lambda: float = 1.0) -> Array:
    """
    Forward KL regularised by the X functional, using target samples, with the
    flow fixed as the inverse map G (target -> source). Writing the per-sample
    log-ratio z = log(mu/nu) as
        z = source(G(y)) - target(y) - log|det J_G(y)|,
    this returns the per-sample contributions
        z + coeff_lambda * |z - z[perm]|
    for a random permutation `perm` of the batch. Its mean is the forward KL
    estimate mean(z) plus coeff_lambda times the X functional mean(|z - z[perm]|),
    the mean absolute pairwise variation of z under the target, which penalises
    the spread of the log-ratio that the plain forward KL leaves free.
    Input:
        y:            Array [N, d]   samples drawn from the target distribution
        source:       Potential      negative log-density of the source (up to const)
        target:       Potential      negative log-density of the target (up to const)
        flow:         Flow           the normalizing flow, applied as G (target -> source)
        key:          Array          PRNG key seeding the batch permutation
        coeff_lambda: float          weight of the X functional term
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    x, ladj = flow.call_and_ladj(y)      # x = G(y), log|det J_G(y)|
    z = source(x) - target(y) - ladj     # per-sample log-ratio  log(mu/nu)
    perm = jax.random.permutation(key, y.shape[0])
    return z + coeff_lambda * jnp.abs(z - z[perm])


def forward_X_G(y: Array, source: Potential, target: Potential, flow: Flow,
                key: Array) -> Array:
    """
    The standalone X functional X_omega, using samples of a weight measure
    omega, with the flow fixed as the inverse map G (target -> source).
    Writing the per-sample log-ratio z = log(mu/nu) as
        z = source(G(y)) - target(y) - log|det J_G(y)|,
    this returns the per-sample contributions
        |z - z[perm]|
    for a random permutation `perm` of the batch. Its mean is the X functional
    X_omega = E_{y, y' ~ omega} |z(y) - z(y')|, the mean absolute pairwise
    variation of z under omega — the X term of `forward_KLX_G` on its own, so
    the weight is free to differ from the target: y ~ target gives the shape
    term X_mu, and y drawn from the mixture alpha*hat_mu + beta*bar_nu —
    hat_mu the quench-and-temper wide-coverage measure (mode discovery),
    bar_nu the detached pushforward of source samples through G^{-1}
    (leakage suppression) — gives the mixture term
    X_{alpha hat_mu + beta bar_nu} of the total training loss.
    Input:
        y:      Array [N, d]   samples drawn from the weight measure omega
        source: Potential      negative log-density of the source (up to const)
        target: Potential      negative log-density of the target (up to const)
        flow:   Flow           the normalizing flow, applied as G (target -> source)
        key:    Array          PRNG key seeding the batch permutation
    Output:
        loss: Array [N]   per-sample losses (reduce with .mean() for the objective)
    """
    x, ladj = flow.call_and_ladj(y)      # x = G(y), log|det J_G(y)|
    z = source(x) - target(y) - ladj     # per-sample log-ratio  log(mu/nu)
    perm = jax.random.permutation(key, y.shape[0])
    return jnp.abs(z - z[perm])
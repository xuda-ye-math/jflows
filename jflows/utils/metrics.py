"""Sampling diagnostics, importance weights, and resampling for jflows.

Ported from `zflows/utils.py` (the ESS diagnostics, the flow
importance-sampling weights, and multinomial resampling; the CESS
variants are dropped — PLAN.md §3.4).

Public API (in pipeline order):
    importance_weights_log_{F,G} / importance_weights_log
                               — unnormalized IS log-weights through a flow
    importance_weights_{F,G} / importance_weights
                               — linear-space weights (max-shifted exp)
    compute_ESS                — effective sample size from linear weights
    compute_ESS_log            — effective sample size from log-weights (stable)
    coverage                   — k-NN mode-collapse diagnostic (Naeem et al., 2020)
    resample                   — multinomial resampling with replacement (key-first)

The canonical pipeline: importance_weights_log_* produce the per-sample
log-ratio log(mu/nu); compute_ESS_log summarizes it into the (0, 1]
diagnostic; resample bootstraps the particle set from the linear weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..flow import ComposedTransform
from ..potential import Potential


__all__ = [
    "compute_ESS",
    "compute_ESS_log",
    "coverage",
    "importance_weights",
    "importance_weights_F",
    "importance_weights_G",
    "importance_weights_log",
    "importance_weights_log_F",
    "importance_weights_log_G",
    "resample",
]


# ──────────────────────────────────────────────────────────────────────
# Importance weights — log/linear-space flow IS reweighting
# ──────────────────────────────────────────────────────────────────────

def importance_weights_log_F(
    samples: Array,
    source: Potential,
    target: Potential,
    F: ComposedTransform,
    chunk: int = 1,
) -> Array:
    """
    Self-normalized importance-sampling log-weights for the proposal
    `nu = F_# mu_0` against the target `mu_1`, where
        mu_0(x) ~ exp(-source(x)),   mu_1(y) ~ exp(-target(y)),
    and `F` is the trained FORWARD bijection (source -> target) that pushes
    source samples toward the target.

    For x drawn from the source, y = F(x), the proposal density is
        log nu(y) = -source(x) - log|det J_F(x)|.
    The unnormalized log-importance-weight is therefore
        log w(y) = log mu_1(y) - log nu(y)
                 = -target(y) + source(x) + log|det J_F(x)|,
    using `Potential` energies U = -log mu (up to additive constants
    that cancel after self-normalization).

    `importance_weights_log` is an alias for this forward-map variant; the
    twin `importance_weights_log_G` is identical but takes the INVERSE map
    `G = F^{-1}` (target -> source).

    Input:
        samples: Array [N, d]      particles drawn from `source`
        source:  Potential         source (proposal-base) potential U_0
        target:  Potential         target potential U_1
        F:       ComposedTransform forward flow map (e.g. flow.t())
        chunk:   int               split `samples` along dim 0 into this many
                                   chunks and concatenate the per-chunk
                                   log-weights. Reduces peak memory at the cost
                                   of wall time; statistically and numerically
                                   equivalent to chunk=1 (each sample's
                                   log-weight depends only on its own (x, F(x))).
    Output:
        log_w: Array [N]   unnormalized log importance weights, ready to
                           feed into compute_ESS_log or to exponentiate
                           (after subtracting max).
    """
    out = []
    for x in jnp.array_split(samples, chunk, axis=0):
        y, ladj = F.call_and_ladj(x)  # y = F(x), ladj = log|det J_F(x)|
        out.append(-target(y) + source(x) + ladj)
    return jnp.concatenate(out, axis=0)


def importance_weights_log_G(
    samples: Array,
    source: Potential,
    target: Potential,
    G: ComposedTransform,
    chunk: int = 1,
) -> Array:
    """
    Inverse-map twin of `importance_weights_log_F`: the same self-normalized
    IS log-weights for `nu = F_# mu_0` against `mu_1`, but the flow is supplied
    as the INVERSE map `G = F^{-1}` (target -> source) rather than the forward
    `F` (same convention as `reverse_KL_F` / `reverse_KL_G`).

    For x ~ source, the forward image is `y = F(x) = G^-1(x)`, recovered via
    `G.inv.call_and_ladj(x) -> (y, log|det J_{G^-1}(x)|)` with
        log|det J_{G^-1}(x)| = log|det J_F(x)|,
    so the weight is identical to the `_F` variant:
        log w(y) = -target(y) + source(x) + log|det J_F(x)|.

    Input: as `importance_weights_log_F`, but `G: ComposedTransform` is the
    inverse flow map target -> source (G = F^{-1}, e.g. flow.t()).
    Output:
        log_w: Array [N]   unnormalized log importance weights.
    """
    out = []
    for x in jnp.array_split(samples, chunk, axis=0):
        y, ladj = G.inv.call_and_ladj(x)  # y = G^-1(x) = F(x)
        out.append(-target(y) + source(x) + ladj)
    return jnp.concatenate(out, axis=0)


# alias: the forward-map variant is the default importance-weight log-routine
importance_weights_log = importance_weights_log_F


def importance_weights_F(
    samples: Array,
    source: Potential,
    target: Potential,
    F: ComposedTransform,
    chunk: int = 1,
) -> Array:
    """
    Linear-space self-normalized importance weights for the proposal
    `nu = F_# mu_0` against the target `mu_1` (FORWARD-map variant). Thin
    convenience wrapper around `importance_weights_log_F`: subtract the max
    log-weight for numerical stability, then exponentiate.

        w_i = exp(log_w_i - max_j log_w_j),   w in [0, 1].

    The omitted factor `exp(max log_w)` is a sample-dependent scalar that
    cancels in every *self-normalized* downstream use (ratios in compute_ESS,
    resample draws, MC averages of bounded test functions). Use this when the
    consumer expects plain non-negative weights (e.g. `resample`); use
    `importance_weights_log_F` + `compute_ESS_log` when log-space stability
    is required.

    `importance_weights` is an alias for this; `importance_weights_G` is the
    inverse-map (`G = F^{-1}`) twin.

    Input: as `importance_weights_log_F`.
    Output:
        w: Array [N]   unnormalized importance weights in [0, 1].
    """
    log_w = importance_weights_log_F(samples, source, target, F, chunk=chunk)
    return jnp.exp(log_w - log_w.max())


def importance_weights_G(
    samples: Array,
    source: Potential,
    target: Potential,
    G: ComposedTransform,
    chunk: int = 1,
) -> Array:
    """
    Inverse-map twin of `importance_weights_F`: linear-space self-normalized
    importance weights with the flow supplied as the INVERSE map `G = F^{-1}`
    (target -> source). Thin wrapper around `importance_weights_log_G`
    (subtract max log-weight, exponentiate).

    Input: as `importance_weights_F`, but `G: ComposedTransform` is the inverse map.
    Output:
        w: Array [N]   unnormalized importance weights in [0, 1].
    """
    log_w = importance_weights_log_G(samples, source, target, G, chunk=chunk)
    return jnp.exp(log_w - log_w.max())


# alias: the forward-map variant is the default importance-weight routine
importance_weights = importance_weights_F


# ──────────────────────────────────────────────────────────────────────
# ESS — effective sample size diagnostics
# ──────────────────────────────────────────────────────────────────────

def compute_ESS(weights: Array) -> Array:
    """
    Compute the Effective Sample Size (ESS) of samples with given
    weights. The ESS lies in [0, 1] by Cauchy's inequality.
    Input:
        weights: Array [N]   (non-negative, not required to be normalized)
    Output:
        ESS: Array (scalar in [0, 1])
    """
    N = weights.shape[0]
    return weights.sum() ** 2 / (N * (weights**2).sum())


def compute_ESS_log(log_weights: Array) -> Array:
    """
    Compute the Effective Sample Size (ESS) from log-weights, using
    logsumexp for numerical stability. The ESS lies in [0, 1] by
    Cauchy's inequality.

        log(ESS) = 2 * logsumexp(log_w) - log(N) - logsumexp(2 * log_w)

    Input:
        log_weights: Array [N]   unnormalized log-weights
    Output:
        ESS: Array (scalar in [0, 1])
    """
    N = log_weights.shape[0]
    log_num = 2 * jax.scipy.special.logsumexp(log_weights, axis=0)
    log_den = jax.scipy.special.logsumexp(2 * log_weights, axis=0) + jnp.log(
        jnp.asarray(N, dtype=log_weights.dtype)
    )
    return jnp.exp(log_num - log_den)


# ──────────────────────────────────────────────────────────────────────
# coverage — k-NN mode-collapse diagnostic
# ──────────────────────────────────────────────────────────────────────

def coverage(y: Array, x: Array, k: int = 5) -> Array:
    """
    Coverage metric (Naeem et al., 2020): the fraction of reference
    points x_i whose k-NN ball (radius = distance to the k-th nearest
    neighbor WITHIN x) contains at least one candidate y_j. Lies in
    [0, 1]; values near 1 mean the candidates reach every neighborhood
    of the reference set, while a missed mode lowers the value by that
    mode's share of the reference points.

    Complementary to the ESS as a mode-collapse diagnostic: the ESS can
    sit near its ceiling while a mode is absent from the proposal's
    support (the importance weights are only weighed on the support
    reached), whereas coverage against a wide-coverage reference set
    exposes the missing mode directly.

    Memory: builds the [P, P] and [P, N] distance matrices — O(P(P+N)).
    Input:
        y: Array [N, d]   candidate samples (e.g. the flow's pushforward)
        x: Array [P, d]   reference samples (e.g. a wide-coverage measure)
        k: int            neighborhood order (default 5)
    Output:
        coverage: Array (scalar in [0, 1])
    """
    P = x.shape[0]
    dxx = jnp.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    dxx = dxx.at[jnp.arange(P), jnp.arange(P)].set(jnp.inf)
    nnd_k = jnp.sort(dxx, axis=1)[:, k - 1]                 # [P]: distance to k-th nearest neighbor in x
    dxy = jnp.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)  # [P, N]
    return (dxy < nnd_k[:, None]).any(axis=1).mean()


# ──────────────────────────────────────────────────────────────────────
# resample — multinomial resampling with replacement
# ──────────────────────────────────────────────────────────────────────

def resample(key: Array, samples: Array, weights: Array, N: int | None = None) -> Array:
    """
    Multinomial resampling from weighted distribution with replacement.
    Input:
        key:     PRNG key
        samples: Array [M, d]
        weights: Array [M]   (non-negative, not required to be normalized)
        N: number of independent samples to return; defaults to samples.shape[0]
    Output:
        resampled: Array [N, d]
    """
    if N is None:
        N = samples.shape[0]
    idx = jax.random.categorical(key, jnp.log(weights), shape=(N,))
    return samples[idx]

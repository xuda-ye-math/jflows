"""Sampling diagnostics, importance weights, and resampling for jflows.

Public API (in pipeline order; the `type` argument names the transform
type: 'F' if the flow maps source -> target, 'G' if target -> source):
    importance_weights_log — unnormalized IS log-weights through a flow
    importance_weights     — linear-space weights (max-shifted exp)
    compute_ESS            — effective sample size from linear weights
    compute_ESS_log        — effective sample size from log-weights (stable)
    coverage               — k-NN mode-collapse diagnostic (Naeem et al., 2020)
    resample               — multinomial resampling with replacement (key-first)

The canonical pipeline: importance_weights_log produces the per-sample
log-ratio log(mu/nu); compute_ESS_log summarizes it into the (0, 1]
diagnostic; resample bootstraps the particle set from the linear weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..flow import Flow
from ..potential import Potential


__all__ = [
    "compute_ESS",
    "compute_ESS_log",
    "coverage",
    "importance_weights",
    "importance_weights_log",
    "resample",
]


# ──────────────────────────────────────────────────────────────────────
# Importance weights — log/linear-space flow IS reweighting
# ──────────────────────────────────────────────────────────────────────

def importance_weights_log(
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    chunk: int = 1,
) -> Array:
    """
    Self-normalized importance-sampling log-weights for the proposal
    `nu = F_# mu_0` against the target `mu_1`, where
        mu_0(x) ~ exp(-source(x)),   mu_1(y) ~ exp(-target(y)).
    The flow acts either as the forward map F (type='F',
    source -> target) or as the inverse map G = F^{-1} (type='G',
    target -> source).

    For x drawn from the source, y = F(x), the proposal density is
        log nu(y) = -source(x) - log|det J_F(x)|.
    The unnormalized log-importance-weight is therefore
        log w(y) = log mu_1(y) - log nu(y)
                 = -target(y) + source(x) + log|det J_F(x)|,
    using `Potential` energies U = -log mu (up to additive constants
    that cancel after self-normalization). With type='G' the same
    quantity is computed through G, using y = G^{-1}(x) and
    log|det J_{G^-1}(x)| = log|det J_F(x)|.

    Input:
        samples: Array [N, d]   particles drawn from `source`
        source:  Potential      source (proposal-base) potential U_0
        target:  Potential      target potential U_1
        flow:    Flow           the normalizing flow
        type:    str            'F' if the flow maps source -> target;
                                'G' if it maps target -> source
        chunk:   int            split `samples` along dim 0 into this many
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
    if type == "F":
        push = flow.call_and_ladj   # y = F(x), log|det J_F(x)|
    elif type == "G":
        push = flow.inv_and_ladj    # y = G^-1(x), log|det J_{G^-1}(x)|
    else:
        raise ValueError(f"importance_weights_log: type must be 'F' or 'G', got {type!r}")
    out = []
    for x in jnp.array_split(samples, chunk, axis=0):
        y, ladj = push(x)
        out.append(-target(y) + source(x) + ladj)
    return jnp.concatenate(out, axis=0)


def importance_weights(
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    chunk: int = 1,
) -> Array:
    """
    Linear-space self-normalized importance weights for the proposal
    `nu = F_# mu_0` against the target `mu_1`. Thin convenience wrapper
    around `importance_weights_log`: subtract the max log-weight for
    numerical stability, then exponentiate.

        w_i = exp(log_w_i - max_j log_w_j),   w in [0, 1].

    The omitted factor `exp(max log_w)` is a sample-dependent scalar that
    cancels in every *self-normalized* downstream use (ratios in compute_ESS,
    resample draws, MC averages of bounded test functions). Use this when the
    consumer expects plain non-negative weights (e.g. `resample`); use
    `importance_weights_log` + `compute_ESS_log` when log-space stability
    is required.

    Input: as `importance_weights_log` (same `type` convention).
    Output:
        w: Array [N]   unnormalized importance weights in [0, 1].
    """
    log_w = importance_weights_log(samples, source, target, flow, type, chunk=chunk)
    return jnp.exp(log_w - log_w.max())


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

def coverage(y: Array, x: Array, k: int = 5, chunk: int = 1) -> Array:
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

    Memory: squared distances are computed in [P/chunk, P] and
    [P/chunk, N] blocks from dot products — the [P, P, d] broadcast is
    never formed. Peak memory is O(P (P + N) / chunk); raise `chunk`
    for large sample sets (e.g. chunk >= P N / 1e8 at float32).
    Input:
        y: Array [N, d]   candidate samples (e.g. the flow's pushforward)
        x: Array [P, d]   reference samples (e.g. a wide-coverage measure)
        k: int            neighborhood order (default 5)
        chunk: int        split the reference set into this many row blocks
    Output:
        coverage: Array (scalar in [0, 1])
    """
    x2 = (x * x).sum(axis=-1)                                   # [P]
    y2 = (y * y).sum(axis=-1)                                   # [N]
    covered = []
    offset = 0
    for xc in jnp.array_split(x, chunk, axis=0):
        c = xc.shape[0]
        xc2 = (xc * xc).sum(axis=-1)
        dxx2 = xc2[:, None] - 2.0 * (xc @ x.T) + x2[None, :]    # [c, P] squared distances
        rows = jnp.arange(c)
        dxx2 = dxx2.at[rows, offset + rows].set(jnp.inf)        # mask self-distance
        nnd2 = -jax.lax.top_k(-dxx2, k)[0][:, k - 1]            # squared k-NN radius within x
        dxy2 = xc2[:, None] - 2.0 * (xc @ y.T) + y2[None, :]    # [c, N]
        covered.append((dxy2 < nnd2[:, None]).any(axis=1))
        offset += c
    return jnp.concatenate(covered).mean()


# ──────────────────────────────────────────────────────────────────────
# resample — multinomial resampling with replacement
# ──────────────────────────────────────────────────────────────────────

def resample(key: Array, samples: Array, weights: Array, N: int | None = None) -> Array:
    """
    Multinomial resampling from weighted distribution with replacement,
    drawn by inverse-CDF search: cumulative sum of the weights, N
    uniform draws, `searchsorted`. Memory is O(M + N) — safe at large
    particle counts (a categorical draw would materialize an [N, M]
    Gumbel matrix, which explodes at N = M ~ 1e5).
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
    cdf = jnp.cumsum(weights)
    u = jax.random.uniform(key, (N,), dtype=cdf.dtype) * cdf[-1]
    idx = jnp.searchsorted(cdf, u, side="right")
    idx = jnp.minimum(idx, weights.shape[0] - 1)  # u can round up to cdf[-1]
    return samples[idx]

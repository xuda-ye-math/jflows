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

import operator

import jax
import jax.numpy as jnp
from jax import Array

from ..flow import Flow
from ..potential import Potential
from ._compat import legacy_keywords


__all__ = [
    "compute_ESS",
    "compute_ESS_log",
    "coverage",
    "importance_weights",
    "importance_weights_log",
    "linear_weights_from_log",
    "resample",
]


def _linear_weights_from_log(log_weights: Array) -> Array:
    """Convert log-weights to a safe max-shifted linear representation.

    Positive infinities share all mass, finite values use the usual shifted
    exponential, and an undefined vector (any NaN, or all -inf) becomes all
    zeros. `resample` gives that zero vector a deliberate uniform fallback.
    """
    log_weights = jnp.asarray(log_weights)
    if not jnp.issubdtype(log_weights.dtype, jnp.inexact):
        log_weights = log_weights.astype(jnp.result_type(float))
    has_nan = jnp.any(jnp.isnan(log_weights))
    posinf = jnp.isposinf(log_weights)
    has_posinf = jnp.any(posinf)
    finite = jnp.isfinite(log_weights)
    has_finite = jnp.any(finite)
    shift = jnp.max(jnp.where(finite, log_weights, -jnp.inf))
    regular = jnp.where(finite, jnp.exp(log_weights - shift), 0.0)
    weights = jnp.where(has_posinf, posinf.astype(log_weights.dtype), regular)
    valid = ~has_nan & (has_posinf | has_finite)
    return jnp.where(valid, weights, jnp.zeros_like(weights))


def linear_weights_from_log(log_weights: Array) -> Array:
    """Return safe max-shifted linear weights from unnormalized log weights.

    Finite values are shifted before exponentiation, positive infinities share
    the mass, and an undefined vector (any NaN or all negative infinity)
    returns zeros. Passing that zero vector to :func:`resample` invokes its
    documented uniform fallback.
    """

    return _linear_weights_from_log(log_weights)


# ──────────────────────────────────────────────────────────────────────
# Importance weights — log/linear-space flow IS reweighting
# ──────────────────────────────────────────────────────────────────────

@legacy_keywords(chunk="chunks")
def importance_weights_log(
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    chunks: int = 1,
    trace_key: Array | None = None,
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
        chunks:  int            split `samples` along dim 0 into this many
                                     chunks and concatenate the per-chunk
                                     log-weights. Reduces peak memory at the cost
                                     of wall time; statistically and numerically
                                     equivalent to chunks=1 (each sample's
                                     log-weight depends only on its own (x, F(x))).
        trace_key: Array | None optional base key for stochastic CNF
                                log-Jacobian probes
    Output:
        log_w: Array [N]   unnormalized log importance weights, ready to
                           feed into compute_ESS_log or to exponentiate
                           (after subtracting max).
    """
    if type not in ("F", "G"):
        raise ValueError(f"importance_weights_log: type must be 'F' or 'G', got {type!r}")
    flow_trace = flow if trace_key is None else flow.with_trace_key(trace_key)
    out = []
    for x in jnp.array_split(samples, chunks, axis=0):
        push = flow_trace.call_and_ladj if type == "F" else flow_trace.inv_and_ladj
        y, ladj = push(x)
        out.append(-target(y) + source(x) + ladj)
    return jnp.concatenate(out, axis=0)


@legacy_keywords(chunk="chunks")
def importance_weights(
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    chunks: int = 1,
    trace_key: Array | None = None,
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
    log_w = importance_weights_log(
        samples, source, target, flow, type, chunks=chunks, trace_key=trace_key
    )
    return _linear_weights_from_log(log_w)


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
    weights = jnp.asarray(weights)
    if weights.ndim != 1 or weights.shape[0] == 0:
        raise ValueError("compute_ESS: weights must be a non-empty vector")
    if not jnp.issubdtype(weights.dtype, jnp.inexact):
        weights = weights.astype(jnp.result_type(float))
    N = weights.shape[0]
    total = weights.sum()
    square = jnp.square(weights).sum()
    raw = total**2 / (N * square)

    # Preserve the ordinary calculation whenever it is finite. If its
    # intermediate sum/square overflowed despite finite inputs, recompute on
    # a scale-normalized vector. Invalid/negative/all-zero weights have no
    # meaningful ESS and deliberately return 0 rather than NaN.
    scale = jnp.max(jnp.abs(weights))
    scaled = jnp.where(
        weights != 0,
        jnp.sign(weights) * jnp.exp(jnp.log(jnp.abs(weights)) - jnp.log(scale)),
        0.0,
    )
    fallback_den = N * jnp.square(scaled).sum()
    fallback = jnp.where(
        fallback_den > 0, scaled.sum() ** 2 / fallback_den, 0.0
    )
    ess = jnp.where(jnp.isfinite(raw), raw, fallback)
    valid = jnp.all(jnp.isfinite(weights) & (weights >= 0)) & (scale > 0)
    return jnp.where(valid, jnp.clip(ess, 0.0, 1.0), 0.0)


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
    log_weights = jnp.asarray(log_weights)
    if log_weights.ndim != 1 or log_weights.shape[0] == 0:
        raise ValueError("compute_ESS_log: log_weights must be a non-empty vector")
    if not jnp.issubdtype(log_weights.dtype, jnp.inexact):
        log_weights = log_weights.astype(jnp.result_type(float))
    N = log_weights.shape[0]
    # Keep the established logsumexp computation bit-for-bit on ordinary
    # finite inputs; use the safe linear representation only for degenerate
    # vectors where the original expression is non-finite.
    log_num = 2 * jax.scipy.special.logsumexp(log_weights, axis=0)
    log_den = jax.scipy.special.logsumexp(2 * log_weights, axis=0) + jnp.log(
        jnp.asarray(N, dtype=log_weights.dtype)
    )
    raw = jnp.exp(log_num - log_den)
    weights = _linear_weights_from_log(log_weights)
    fallback = compute_ESS(weights)
    ess = jnp.where(jnp.isfinite(raw), raw, fallback)
    return jnp.where(jnp.any(jnp.isnan(log_weights)), 0.0, ess)


# ──────────────────────────────────────────────────────────────────────
# coverage — k-NN mode-collapse diagnostic
# ──────────────────────────────────────────────────────────────────────

@legacy_keywords(chunk="chunks")
def coverage(y: Array, x: Array, k: int = 5, chunks: int = 1) -> Array:
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

    Memory: squared distances are computed in [P/chunks, P] and
    [P/chunks, N] blocks from dot products — the [P, P, d] broadcast is
    never formed. Peak memory is O(P (P + N) / chunks); raise `chunks`
    for large sample sets (e.g. chunks >= P N / 1e8 at float32).
    Input:
        y: Array [N, d]   candidate samples (e.g. the flow's pushforward)
        x: Array [P, d]   reference samples (e.g. a wide-coverage measure)
        k: int            neighborhood order (default 5), 1 <= k < P
        chunks: int       split the reference set into this many row blocks
    Output:
        coverage: Array (scalar in [0, 1])
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] \
            or x.shape[0] < 2 or y.shape[0] < 1:
        raise ValueError(
            "coverage: x/y must be non-empty rank-2 arrays with matching feature "
            f"dimensions and at least two reference rows; got x={x.shape}, y={y.shape}"
        )
    if isinstance(k, bool) or isinstance(chunks, bool):
        raise ValueError("coverage: k and chunks must be integers, not booleans")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError(f"coverage: k must be an integer, got {k!r}") from exc
    try:
        chunks = operator.index(chunks)
    except TypeError as exc:
        raise ValueError(f"coverage: chunks must be an integer, got {chunks!r}") from exc
    if not (1 <= k < x.shape[0]):
        raise ValueError(f"coverage: k must satisfy 1 <= k < P={x.shape[0]}, got {k!r}")
    if not (1 <= chunks <= x.shape[0]):
        raise ValueError(
            f"coverage: chunks must satisfy 1 <= chunks <= P={x.shape[0]}, got {chunks!r}"
        )
    x2 = (x * x).sum(axis=-1)                                   # [P]
    y2 = (y * y).sum(axis=-1)                                   # [N]
    covered = []
    offset = 0
    for xc in jnp.array_split(x, chunks, axis=0):
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
        weights: Array [M]   non-negative, not required to be normalized.
                 Positive infinities share all mass. A zero-total or invalid
                 vector falls back to uniform resampling rather than silently
                 selecting the final particle.
        N: number of independent samples to return; defaults to samples.shape[0]
    Output:
        resampled: Array [N, d]
    """
    weights = jnp.asarray(weights)
    M = samples.shape[0]
    if M == 0 or weights.ndim != 1 or weights.shape[0] != M:
        raise ValueError(
            f"resample: samples and weights need matching non-empty leading axes; "
            f"got samples={samples.shape}, weights={weights.shape}"
        )
    if N is None:
        N = M
    if N < 1:
        raise ValueError(f"resample: N must be positive, got {N!r}")
    if not jnp.issubdtype(weights.dtype, jnp.inexact):
        weights = weights.astype(jnp.result_type(float))

    posinf = jnp.isposinf(weights)
    has_posinf = jnp.any(posinf)
    valid = jnp.all((jnp.isfinite(weights) | posinf) & (weights >= 0))
    finite_weights = jnp.where(jnp.isfinite(weights), weights, 0.0)
    total = finite_weights.sum()
    scale = jnp.max(finite_weights)
    # On some float32 accelerator kernels, reciprocal(scale) underflows for
    # scale ~ 1e38, making the seemingly safe `weights / scale` all zeros.
    # The log difference avoids that reciprocal path and is used only when
    # the ordinary total has overflowed.
    scaled = jnp.where(
        finite_weights > 0,
        jnp.exp(jnp.log(finite_weights) - jnp.log(scale)),
        0.0,
    )
    regular = jnp.where(jnp.isfinite(total), finite_weights, scaled)
    regular_ok = valid & (scale > 0)
    safe = jnp.where(
        valid & has_posinf,
        posinf.astype(weights.dtype),
        jnp.where(regular_ok, regular, jnp.ones_like(weights)),
    )
    cdf = jnp.cumsum(safe)
    u = jax.random.uniform(key, (N,), dtype=cdf.dtype) * cdf[-1]
    idx = jnp.searchsorted(cdf, u, side="right")
    idx = jnp.minimum(idx, weights.shape[0] - 1)  # u can round up to cdf[-1]
    return samples[idx]

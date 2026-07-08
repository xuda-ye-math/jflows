"""Annealed transport samplers for jflows — SMC and flow-proposal AIS.

Built on the other utils modules:
importance reweighting + `resample` (metrics) alternating with Langevin
rejuvenation (rejuvenation), whose `taming` stabilizer is exposed here
as well.

Key convention: rung k (1-indexed) uses `fold_in(key, k)`, split into
one resampling key and one rejuvenation key — so a manual composition of
`resample` + `langevin` with the same derivation reproduces the loops.

There are no temperature arguments: to anneal between tempered
distributions, pass the scaled potentials (the potential algebra covers
tempering, e.g. `beta * U`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..flow import Flow
from ..potential import Potential, linear_combination
from .metrics import compute_ESS_log, resample
from .rejuvenation import langevin


__all__ = [
    "ais",
    "annealed_importance_sampling",
    "sequential_monte_carlo",
    "smc",
]


# ──────────────────────────────────────────────────────────────────────
# SMC — annealed Langevin on a linear bridge of potentials (no flow)
# ──────────────────────────────────────────────────────────────────────

def sequential_monte_carlo(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    ladder: int = 1,
    step: float = 1e-3,
    iters: int = 100,
    adjust: bool = True,
    taming: float = 0,
    chunk: int = 1,
) -> tuple[Array, Array]:
    """
    Sequential Monte Carlo (annealed Langevin) that transports the input
    particles from the source `mu_0 ~ exp(-source)` to the target
    `mu_1 ~ exp(-target)` (both on the SAME space, no flow) through a
    ladder of M = `ladder` linearly-interpolated bridge potentials,
    alternating importance reweighting (with multinomial resampling) and
    Langevin rejuvenation **on each bridge** at every rung.

    The bridge at rung k is the linear combination
        u_k(x) = (1 - k/M) * source(x) + (k/M) * target(x),   k = 0, ..., M,
    so u_0 = source (the distribution `samples` follow) and u_M = target.
    Built via `linear_combination([target, source], [k/M, 1 - k/M])`,
    rebuilt each rung (same pytree structure — no recompile).

    For each k = 1, ..., M, starting from particles x ~ exp(-u_{k-1}):
      1. Incremental self-normalised importance weights from u_{k-1} to u_k:
             log w(x) = u_{k-1}(x) - u_k(x) = (1/M) * (source(x) - target(x)),
         exponentiated after subtracting the max for numerical stability.
      2. Multinomial resampling of x by w.
      3. Langevin rejuvenation targeting exp(-u_k) — i.e. ON THE
         BRIDGE POTENTIAL u_k itself — for `iters` steps, with the tamed
         drift when `taming > 0`.
    After the final rung the particles approximate exp(-target).

    Contrast with `annealed_importance_sampling`, which uses a
    trained flow as the proposal and rejuvenates only in the final target
    `mu_1`: here there is no flow, the bridge is built directly in
    potential space, and Langevin runs on each intermediate `u_k`.

    Input:
        key:     PRNG key (rung k uses fold_in(key, k), split into the
                 resampling and rejuvenation keys)
        samples: Array [N, d]   particles drawn from exp(-source)
        source:  Potential      source potential
        target:  Potential      target potential
        ladder:  int            number of annealing rungs M (>= 1). M=1 is a
                                single reweight + resample + Langevin hop
                                straight from source to target; larger M
                                bridges low-overlap source/target pairs.
        step:    float          Langevin step size, shared across rungs
        iters:   int            Langevin steps per rung
        adjust:  bool           if True, MALA rejuvenation on each bridge
                                (unbiased); if False, ULA (see `langevin`)
        taming:  float          if > 0, tamed Langevin drift
                                grad u_k / (1 + taming * ||grad u_k||) —
                                stabilizes the rejuvenation on potentials
                                whose gradients grow super-linearly
        chunk:   int            split along dim 0 into this many chunks
                                inside each Langevin call to bound peak
                                memory (statistically equivalent to chunk=1)
    Output:
        samples: Array [N, d]   particles approximating exp(-target)
        ess:     Array [M]      per-rung effective sample size (in [0, 1]) of
                                the incremental importance weights, computed
                                *before* resampling — a diagnostic of how well
                                consecutive bridges overlap (close to 1 =
                                well-spaced ladder)
    """
    M = ladder
    x = samples
    ess = []  # per-rung effective sample size of the incremental weights
    for k in range(1, M + 1):
        c = k / M
        u_k = linear_combination([target, source], [c, 1.0 - c])
        # (1) incremental IS weights from u_{k-1} to u_k on x ~ exp(-u_{k-1}).
        #     The bridge difference u_{k-1}(x) - u_k(x) telescopes (for every
        #     k) to (1/M) * (source(x) - target(x)).
        log_w = (source(x) - target(x)) / M
        ess.append(compute_ESS_log(log_w))
        w = jnp.exp(log_w - log_w.max())  # self-normalised, in [0, 1]
        # (2) resample onto high-weight particles, then (3) Langevin-rejuvenate
        #     ON the bridge u_k to obtain fresh samples ~ exp(-u_k).
        key_r, key_l = jax.random.split(jax.random.fold_in(key, k))
        x = resample(key_r, x, w)
        x = langevin(key_l, x, u_k, step=step, iters=iters, adjust=adjust, taming=taming, chunk=chunk)
    return x, jnp.stack(ess)


# ──────────────────────────────────────────────────────────────────────
# AIS — flow-proposal SMC along the geometric path to the target
# ──────────────────────────────────────────────────────────────────────

def annealed_importance_sampling(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    ladder: int = 1,
    step: float = 1e-3,
    iters: int = 100,
    adjust: bool = True,
    taming: float = 0,
    chunk: int = 1,
) -> Array:
    """
    Annealed importance sampling (an SMC sampler) that uses a trained flow
    as the proposal. The source `mu_0 ~ exp(-source)` and target
    `mu_1 ~ exp(-target)` live in the same space, and the flow has been
    trained so that the pushforward `F_# mu_0 ~~ mu_1`. The input `samples`
    are drawn from `mu_0`; the routine returns samples from `mu_1`.

    The flow acts either as the forward map F (type='F',
    source -> target) or as the inverse map G = F^{-1} (type='G',
    target -> source), the same `type` convention as `importance_weights`.

    Annealing follows the geometric path between the flow proposal and the
    target,
        pi_k(y) = mu_1(y) ** (k/M) * (F_# mu_0)(y) ** (1 - k/M),
        k = 0, ..., M   (M = `ladder`),
    so pi_0 = F_# mu_0 and pi_M = mu_1. With the change-of-variables
    pushforward density, the full proposal -> target importance weight is,
    for y with pre-image x = F^{-1}(y),
        log w(y) = -target(y) + source(x) + log|det J_F(x)|
    (cf. `importance_weights_log`), and the incremental weight along the
    geometric path is exactly its 1/M-th. With type='G' the same
    quantities are computed through G: x = G(y) and
    log|det J_F(x)| = -log|det J_G(y)|.

    Steps:
      (0) y <- F(samples)                # push source samples to pi_0 = F_# mu_0
      (1) for k = 1, ..., M:
            x     <- F^{-1}(y)           # refresh the latent pre-images
            log w <- (1/M) * (-target(y) + source(x) + log|det J_F(x)|)
            y     <- resample(y, self-normalised w)
            y     <- langevin(y, target, ...)   # rejuvenate in mu_1

    No bridge potential is constructed: the weights come directly from the
    raw `source` / `target` energies and the flow Jacobian. The Langevin
    rejuvenation targets `mu_1 = exp(-target)` **directly** rather than the
    exact intermediate `pi_k`: evaluating / differentiating `log pi_k`
    would require the pushforward density of F (hence F^{-1} and its
    Jacobian gradient), which is far more expensive, and since the de-facto
    target is `mu_1` and the incremental weights already follow the `pi_k`
    path, the `mu_1` kernel introduces no essential deviation.

    Input:
        key:       PRNG key (rung k uses fold_in(key, k), split into the
                   resampling and rejuvenation keys)
        samples:   Array [N, d]      particles drawn from mu_0 (source space)
        source:    Potential         source potential U_0 = -log mu_0
        target:    Potential         target potential U_1 = -log mu_1
        flow:    Flow            the trained normalizing flow
        type:    str             'F' if the flow maps source -> target;
                                 'G' if it maps target -> source
        ladder:    int               number of annealing rungs M (>= 1). M=1 is
                                     a single reweight + resample + Langevin hop
                                     from the flow proposal to the target.
        step:      float             Langevin step size, shared across rungs
        iters:     int               Langevin steps per rung
        adjust:    bool              if True, MALA rejuvenation in mu_1
                                     (unbiased); if False, ULA (see `langevin`)
        taming:    float             if > 0, tamed Langevin drift on the target
                                     (see `langevin`)
        chunk:     int               split along dim 0 into this many chunks for
                                     the pushforward / inverse / weight passes
                                     and inside each Langevin call, to bound
                                     peak memory (statistically equivalent to
                                     chunk=1)
    Output:
        samples: Array [N, d]      particles in mu_1 (target space),
                                   approximating exp(-target)
    """
    if type not in ("F", "G"):
        raise ValueError(f"annealed_importance_sampling: type must be 'F' or 'G', got {type!r}")
    M = ladder
    # (0) push the source samples through F to obtain pi_0 = F_# mu_0.
    push = flow.__call__ if type == "F" else flow.inv
    y = jnp.concatenate(
        [push(xc) for xc in jnp.array_split(samples, chunk, axis=0)], axis=0
    )
    for k in range(1, M + 1):
        # (1) incremental weights w(y) ** (1/M): refresh each particle's
        #     latent pre-image x = F^{-1}(y) and reuse the
        #     importance_weights_log rule, scaled by 1/M for one rung.
        parts = []
        for yc in jnp.array_split(y, chunk, axis=0):
            if type == "F":
                xc = flow.inv(yc)                 # x = F^{-1}(y)
                _, ladj = flow.call_and_ladj(xc)  # log|det J_F(x)|
            else:
                xc, ladj_G = flow.call_and_ladj(yc)  # x = G(y), log|det J_G(y)|
                ladj = -ladj_G                       # log|det J_F(x)|
            parts.append((-target(yc) + source(xc) + ladj) / M)
        log_w = jnp.concatenate(parts, axis=0)
        w = jnp.exp(log_w - log_w.max())  # self-normalised, in [0, 1]
        # (2) resample onto high-weight particles, then rejuvenate in mu_1.
        key_r, key_l = jax.random.split(jax.random.fold_in(key, k))
        y = resample(key_r, y, w)
        y = langevin(key_l, y, target, step=step, iters=iters, adjust=adjust, taming=taming, chunk=chunk)
    return y


# aliases: the standard short names
smc = sequential_monte_carlo
ais = annealed_importance_sampling

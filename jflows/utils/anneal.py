"""Sequential Monte Carlo with a flow proposal for jflows.

`sequential_monte_carlo` (alias `smc`) manufactures target samples from
source samples through a trained flow: the source particles are pushed
through the flow, and every level then reweights them towards the target,
resamples, and rejuvenates under the target. Each intermediate level
rejuvenates with one Hamiltonian Monte Carlo trajectory (a random momentum,
`mc_steps` leapfrog steps of size `mc_dt`, one Metropolis decision), which
moves the resampled particles far enough for the next reweighting to act on
new positions; the last level rejuvenates with `mc_steps` Langevin steps of
size `mc_dt` (MALA by default). Built on the other
utils modules: importance reweighting + `resample` (metrics) alternating
with the rejuvenation kernels (rejuvenation), whose `taming` stabilizer is
exposed here as well.

Key convention: level k (1-indexed) uses `fold_in(key, k)`, split into
one resampling key and one rejuvenation key — so a manual composition of
`resample` + `langevin` with the same derivation reproduces the loop.

There are no temperature arguments: to anneal between tempered
distributions, pass the scaled potentials (the potential algebra covers
tempering, e.g. `beta * U`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..flow import Flow
from ..potential import Potential
from .metrics import linear_weights_from_log, resample
from .rejuvenation import hamiltonian_monte_carlo, langevin


__all__ = [
    "sequential_monte_carlo",
    "smc",
]


# ──────────────────────────────────────────────────────────────────────
# SMC — flow proposal, geometric path to the target, resampling per level,
# HMC rejuvenation on the intermediate levels and MALA on the last
# ──────────────────────────────────────────────────────────────────────

def sequential_monte_carlo(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps: int = 100,
    adjust: bool = True,
    taming: float = 0,
    chunks: int = 1,
    trace_key: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Sequential Monte Carlo with a trained flow as the proposal. The source
    `mu_0 ~ exp(-source)` and target `mu_1 ~ exp(-target)` live in the same
    space, and the flow has been trained so that the pushforward
    `F_# mu_0 ~~ mu_1`. The input `samples` are drawn from `mu_0`; the
    routine returns samples from `mu_1` together with the proposal it
    started from.

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
      (0) (y, log|J_F|) <- F(samples)    # push source samples to pi_0
          log w_full <- -target(y) + source(samples) + log|J_F|
      (1) for k = 1, ..., M:
            if k > 1:
                x <- F^{-1}(y)           # refresh after the previous move
                log w_full <- -target(y) + source(x) + log|det J_F(x)|
            log w <- (1/M) * log w_full
            y     <- resample(y, self-normalised w)
            y     <- hmc(y, target, ...)        # k < M: one trajectory in mu_1
            y     <- langevin(y, target, ...)   # k = M: rejuvenate in mu_1

    No bridge potential is constructed: the weights come directly from the
    raw `source` / `target` energies and the flow Jacobian. Every
    rejuvenation targets `mu_1 = exp(-target)` **directly** rather than the
    exact intermediate `pi_k`: evaluating / differentiating `log pi_k`
    would require the pushforward density of F (hence F^{-1} and its
    Jacobian gradient), which is far more expensive. Consequently this is a
    biased, score-free target surrogate rather than an exact SMC sampler
    on the geometric path. Each intermediate level uses one
    Metropolis-adjusted HMC trajectory: a random momentum, `mc_steps`
    leapfrog steps of size `mc_dt`, one accept/reject decision, so that the
    resampled particles move O(mc_dt * mc_steps) before the next
    reweighting instead of the O(sqrt(mc_dt)) of a Langevin step; the last
    level uses `mc_steps` Langevin steps of size `mc_dt` (MALA by default).
    Both kernels are invariant for `mu_1`; neither removes the
    intermediate-path approximation.

    Input:
        key:       PRNG key (level k uses fold_in(key, k), split into the
                   resampling and rejuvenation keys)
        samples:   Array [N, d]      particles drawn from mu_0 (source space)
        source:    Potential         source potential U_0 = -log mu_0
        target:    Potential         target potential U_1 = -log mu_1
        flow:    Flow            the trained normalizing flow
        type:    str             'F' if the flow maps source -> target;
                                 'G' if it maps target -> source
        ladder:    int               number of annealing levels M (>= 1). M=1 is
                                     a single reweight + resample + Langevin hop
                                     from the flow proposal to the target.
        mc_dt:     float             step size: leapfrog step of the HMC
                                     trajectory on levels 1 .. M-1, Langevin
                                     step on level M
        mc_steps:  int               leapfrog steps of the HMC trajectory on
                                     levels 1 .. M-1, Langevin steps on level M
        adjust:    bool              if True, MALA rejuvenation invariant for
                                     mu_1; if False, ULA (see `langevin`)
        taming:    float             if > 0, tamed Langevin drift on the target
                                     (see `langevin`)
        chunks:    int               split along dim 0 into this many execution
                                     chunks for the pushforward / inverse /
                                     weight passes and Langevin calls
                                     (statistically equivalent to chunks=1;
                                     nested compiled loops are not a strict
                                     peak-memory guarantee)
        trace_key: Array | None      optional base key for stochastic CNF
                                     log-Jacobian probes; packed trainers pass
                                     a fresh key automatically
    Output:
        samples:  Array [N, d]     particles in mu_1 (target space),
                                   approximating exp(-target)
        proposal: Array [N, d]     the pushforward particles y = F(samples)
                                   the levels started from, before any
                                   resampling or rejuvenation
        proposal_log_weights: Array [N]
                                   the full (not 1/M-scaled) log(mu_1 / F_#mu_0)
                                   on `proposal`; its ESS is the batch
                                   diagnostic the trainers report
    """
    M = ladder

    def level_flow(k: int) -> Flow:
        return flow if trace_key is None else flow.with_trace_key(
            jax.random.fold_in(trace_key, k)
        )

    # (0) Push the ORIGINAL source particles and retain the matching
    # Jacobian.  For finite-step continuous flows, numerically inverting y
    # and pushing it again is not equivalent to evaluating the actual
    # proposal F_#mu_0.  Level one therefore uses this direct pair both for
    # its incremental correction and for the optional proposal diagnostic.
    flow_1 = level_flow(1)
    y_parts = []
    initial_parts = []
    for xc in jnp.array_split(samples, chunks, axis=0):
        push = flow_1.call_and_ladj if type == "F" else flow_1.inv_and_ladj
        yc, ladj = push(xc)
        y_parts.append(yc)
        initial_parts.append(-target(yc) + source(xc) + ladj)
    proposal = jnp.concatenate(y_parts, axis=0)
    proposal_log_weights = jnp.concatenate(initial_parts, axis=0)
    y = proposal

    for k in range(1, M + 1):
        # (1) Incremental weights w(y) ** (1/M). Level one reuses the
        # direct proposal weights above. Later levels refresh each moved
        # particle's latent pre-image and use that level's trace key.
        if k == 1:
            full_log_weight = proposal_log_weights
        else:
            parts = []
            flow_k = level_flow(k)
            for yc in jnp.array_split(y, chunks, axis=0):
                if type == "F":
                    xc = flow_k.inv(yc)                 # x = F^{-1}(y)
                    _, ladj = flow_k.call_and_ladj(xc)  # log|det J_F(x)|
                else:
                    xc, ladj_G = flow_k.call_and_ladj(yc)  # x = G(y), log|det J_G(y)|
                    ladj = -ladj_G                       # log|det J_F(x)|
                parts.append(-target(yc) + source(xc) + ladj)
            full_log_weight = jnp.concatenate(parts, axis=0)
        log_w = full_log_weight / M
        w = linear_weights_from_log(log_w)
        # (2) resample onto high-weight particles, then rejuvenate in mu_1:
        #     HMC on the intermediate levels, Langevin on the last.
        key_r, key_l = jax.random.split(jax.random.fold_in(key, k))
        y = resample(key_r, y, w)
        if k < M:
            y = hamiltonian_monte_carlo(
                key_l, y, target, dt=mc_dt, leapfrog_steps=mc_steps,
                trajectories=1, chunks=chunks,
            )
        else:
            y = langevin(
                key_l, y, target, dt=mc_dt, steps=mc_steps, adjust=adjust,
                taming=taming, chunks=chunks,
            )
    return y, proposal, proposal_log_weights


# alias: the standard short name
smc = sequential_monte_carlo

"""Sequential Monte Carlo with a flow proposal for jflows.

`sequential_monte_carlo` (alias `smc`) manufactures target samples from
source samples through a trained flow: the source particles are pushed
through the flow, and every level then reweights them towards the target,
resamples, and rejuvenates at the target pi on every level with Langevin
steps of size `mc_dt` (MALA by default), `mc_steps_1` steps on the
intermediate levels and `mc_steps_2` on the last: the intermediate levels
are a target surrogate (their kernel is invariant for pi rather than for
the level's own distribution), and no level differentiates the flow. This
is the SMC of the forward KL trainers. `sequential_monte_carlo_fab` (alias
`smc_fab`) is the exact two-phase SMC of FAB: level k of M rejuvenates at
its own distribution mu_k = nu^(1 - k/M) pi^(k/M), whose potential contains
the pushforward density of the flow, and then on to pi^2 / nu along
rho_k = pi (pi / nu)^(k/M). Built on the other utils modules:
importance reweighting + `resample_index` (metrics) alternating with the
Langevin kernel (rejuvenation), whose `taming` stabilizer is exposed here
as well.

`sequential_monte_carlo_fab` (alias `smc_fab`) continues from the target
to `pi^2 / nu` in a second phase of `ladder` levels with the same weights,
the sample law that FAB (flow annealed importance sampling bootstrap,
Midgley et al. 2022) trains on, here without a replay buffer: plain SMC from
the source to `pi^2 / nu`.

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
from ..potential import Potential, potential_from
from .metrics import linear_weights_from_log, resample_index
from .rejuvenation import langevin


__all__ = [
    "sequential_monte_carlo",
    "sequential_monte_carlo_fab",
    "smc",
    "smc_fab",
]


# ──────────────────────────────────────────────────────────────────────
# SMC — flow proposal, geometric path to the target, resampling per level,
# MALA rejuvenation at the target pi on every level (the forward KL target
# surrogate); the exact levels live in `sequential_monte_carlo_fab`
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
    mc_steps_1: int = 100,
    mc_steps_2: int = 100,
    adjust: bool = True,
    taming: float = 0,
    chunks: int = 1,
    trace_key: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Sequential Monte Carlo with a trained flow as the proposal. The source
    `pi_0 ~ exp(-source)` and target `pi ~ exp(-target)` live in the same
    space, and the flow has been trained so that the pushforward
    `F_# pi_0 ~~ pi`. The input `samples` are drawn from `pi_0`; the
    routine returns samples from `pi` together with the proposal it
    started from.

    The flow acts either as the forward map F (type='F',
    source -> target) or as the inverse map G = F^{-1} (type='G',
    target -> source), the same `type` convention as `importance_weights`.

    Annealing follows the geometric path between the flow proposal and the
    target,
        mu_k(y) = pi(y) ** (k/M) * (F_# pi_0)(y) ** (1 - k/M),
        k = 0, ..., M   (M = `ladder`),
    so mu_0 = F_# pi_0 and mu_M = pi. With the change-of-variables
    pushforward density, the full proposal -> target importance weight is,
    for y with pre-image x = F^{-1}(y),
        log w(y) = -target(y) + source(x) + log|det J_F(x)|
    (cf. `importance_weights_log`), and the incremental weight along the
    geometric path is exactly its 1/M-th. With type='G' the same
    quantities are computed through G: x = G(y) and
    log|det J_F(x)| = -log|det J_G(y)|.

    Steps:
      (0) (y, log|J_F|) <- F(samples)    # push source samples to mu_0
          log w_full <- -target(y) + source(samples) + log|J_F|
      (1) for k = 1, ..., M:
            if k > 1:
                x <- F^{-1}(y)           # refresh after the previous move
                log w_full <- -target(y) + source(x) + log|det J_F(x)|
            log w <- (1/M) * log w_full
            y     <- resample(y, self-normalised w)
            y     <- langevin(y, target, ...)   # k < M: mc_steps_1 MALA in pi
            y     <- langevin(y, target, ...)   # k = M: mc_steps_2 MALA in pi

    Every level rejuvenates at the target pi: the intermediate levels are a
    target surrogate whose kernel is invariant for pi rather than for the
    level's own distribution mu_k = nu^(1 - k/M) pi^(k/M), so the routine
    is not an exact sampler on the geometric path, but no level evaluates
    or differentiates the pushforward density. This is the SMC of the
    forward KL trainers; the exact levels, whose potential contains the
    flow, are those of `sequential_monte_carlo_fab`. The start energy of
    each Langevin run is the `target(y)` the reweighting already evaluated,
    carried through the resampling; the intermediate levels use
    `mc_steps_1` Langevin steps of size `mc_dt` (MALA by default) and the
    last level runs `mc_steps_2` steps.

    Input:
        key:       PRNG key (level k uses fold_in(key, k), split into the
                   resampling and rejuvenation keys)
        samples:   Array [N, d]      particles drawn from pi_0 (source space)
        source:    Potential         source potential U_0 = -log pi_0
        target:    Potential         target potential U_1 = -log pi
        flow:    Flow            the trained normalizing flow
        type:    str             'F' if the flow maps source -> target;
                                 'G' if it maps target -> source
        ladder:    int               number of annealing levels M (>= 1). M=1 is
                                     a single reweight + resample + Langevin hop
                                     from the flow proposal to the target.
        mc_dt:     float             Langevin step size on every level
        mc_steps_1: int              Langevin steps on the intermediate levels
                                     1 .. M-1
        mc_steps_2: int              Langevin steps on the last level M (at
                                     the target alone)
        adjust:    bool              if True, MALA rejuvenation invariant for
                                     pi; if False, ULA (see `langevin`)
        taming:    float             if > 0, tamed Langevin drift on every
                                     level (see `langevin`)
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
        samples:  Array [N, d]     particles in pi (target space),
                                   approximating exp(-target)
        proposal: Array [N, d]     the pushforward particles y = F(samples)
                                   the levels started from, before any
                                   resampling or rejuvenation
        proposal_log_weights: Array [N]
                                   the full (not 1/M-scaled) log(pi / F_#pi_0)
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
    # proposal F_#pi_0.  Level one therefore uses this direct pair both for
    # its incremental correction and for the optional proposal diagnostic.
    flow_1 = level_flow(1)
    y_parts, weight_parts, energy_parts = [], [], []
    for xc in jnp.array_split(samples, chunks, axis=0):
        push = flow_1.call_and_ladj if type == "F" else flow_1.inv_and_ladj
        yc, ladj = push(xc)
        u = target(yc)
        y_parts.append(yc)
        weight_parts.append(-u + source(xc) + ladj)
        energy_parts.append(u)
    proposal = jnp.concatenate(y_parts, axis=0)
    proposal_log_weights = jnp.concatenate(weight_parts, axis=0)
    y = proposal
    full_log_weight = proposal_log_weights
    u = jnp.concatenate(energy_parts, axis=0)

    for k in range(1, M + 1):
        # (1) Incremental weights w(y) ** (1/M). Level one reuses the
        # direct proposal weights above. Later levels refresh each moved
        # particle's latent pre-image and use that level's trace key.
        flow_k = level_flow(k)
        if k > 1:
            weight_parts, energy_parts = [], []
            for yc in jnp.array_split(y, chunks, axis=0):
                if type == "F":
                    xc = flow_k.inv(yc)                 # x = F^{-1}(y)
                    _, ladj = flow_k.call_and_ladj(xc)  # log|det J_F(x)|
                else:
                    xc, ladj_G = flow_k.call_and_ladj(yc)  # x = G(y), log|det J_G(y)|
                    ladj = -ladj_G                       # log|det J_F(x)|
                u_c = target(yc)
                weight_parts.append(-u_c + source(xc) + ladj)
                energy_parts.append(u_c)
            full_log_weight = jnp.concatenate(weight_parts, axis=0)
            u = jnp.concatenate(energy_parts, axis=0)
        w = linear_weights_from_log(full_log_weight / M)
        # (2) resample onto high-weight particles, then rejuvenate with
        #     MALA at pi: mc_steps_1 steps on the intermediate levels,
        #     mc_steps_2 on the last. The start energy of the Langevin run
        #     is the target energy the reweighting already evaluated.
        key_r, key_l = jax.random.split(jax.random.fold_in(key, k))
        idx = resample_index(key_r, y, w)
        if k < M:
            y = langevin(
                key_l, y[idx], target, dt=mc_dt, steps=mc_steps_1, adjust=adjust,
                taming=taming, chunks=chunks, energy=u[idx],
            )
        else:
            y = langevin(
                key_l, y[idx], target, dt=mc_dt, steps=mc_steps_2, adjust=adjust,
                taming=taming, chunks=chunks,
            )
    return y, proposal, proposal_log_weights


def _pushforward_log_density(flow: Flow, type: str):
    """log nu(y) of the pushforward `nu = F_# pi_0` through the flow, as a function of y."""

    def log_density(y, source):
        if type == "F":
            x = flow.inv(y)
            _, ladj = flow.call_and_ladj(x)          # log|det J_F(x)|
            return -source(x) - ladj
        x, ladj = flow.call_and_ladj(y)              # x = G(y), log|det J_G(y)|
        return -source(x) + ladj

    return log_density


def _path_potential(source: Potential, target: Potential, log_density, s: float) -> Potential:
    """The potential of the path distribution pi * (pi / nu)^(-s), as a `Potential`.

    U_s(y) = target(y) + s * log w_full(y) with log w_full = log(pi / nu) =
    -target(y) - log nu(y): s = 1 - k/M gives mu_k = nu^(1 - k/M) pi^(k/M)
    of `sequential_monte_carlo`, s = -k/M gives rho_k = pi (pi / nu)^(k/M)
    of the FAB phase, s = 0 is pi and s = -1 is pi^2 / nu. The pushforward
    density enters through `log_density`, so the gradient runs through the
    flow.
    """
    return potential_from(
        lambda z: (1.0 - s) * target(z) - s * log_density(z, source)
    )


def sequential_monte_carlo_fab(
    key: Array,
    samples: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    ladder: int = 1,
    mc_dt: float = 1e-3,
    mc_steps_1: int = 100,
    mc_steps_2: int = 100,
    adjust: bool = True,
    taming: float = 0,
    chunks: int = 1,
    trace_key: Array | None = None,
) -> tuple[Array, Array, Array]:
    """
    Two-phase SMC from the flow proposal to `pi^2 / nu` (the FAB sample law).

    Phase 1 repeats the levels of `sequential_monte_carlo`: `ladder` levels
    from the pushforward `nu = F_# pi_0` to the target `pi`, MALA at the
    level's own `mu_k` on every level (`pi` on the last), with the same
    keys, weights and kernels, so its proposal and log-weights are those of
    `sequential_monte_carlo`. Phase 2 continues
    with `ladder` further levels from `pi` to `pi^2 / nu` along the
    geometric path
        rho_k(y) = pi(y) * (pi(y) / nu(y)) ** (k/M),   k = 0, ..., M,
    whose incremental weight is again the 1/M-th of the same proposal-to-
    target weight log w = log(pi / nu), refreshed at the moved particles
    through the flow. Every phase-2 level resamples and rejuvenates with
    Langevin steps of size `mc_dt` (MALA by default) at its own distribution
    rho_k,
        U_k(y) = -log rho_k(y) = target(y) - (k/M) * log w(y),
    whose gradient is taken through the flow (the pushforward density and its
    Jacobian), as on the intermediate levels of phase 1: `mc_steps_1` steps
    on the intermediate levels 1 .. M-1 and `mc_steps_2` on the last level
    (rho_M = pi^2 / nu, the end of the path). There is no replay buffer:
    this is plain SMC from the source to `pi^2 / nu`.

    Input: as `sequential_monte_carlo`; `ladder` is the number of levels of
    each phase.
    Output:
        samples:  Array [N, d]     particles approximating pi^2 / nu
        proposal: Array [N, d]     the pushforward particles phase 1 started from
        proposal_log_weights: Array [N]
                                   log(pi / nu) on `proposal`
    """
    key_1, key_2 = jax.random.split(key)
    M = ladder

    def level_flow(k: int) -> Flow:
        return flow if trace_key is None else flow.with_trace_key(
            jax.random.fold_in(trace_key, k)
        )

    # ---- phase 1: nu -> pi, the levels of `sequential_monte_carlo` ----
    # Level k rejuvenates with MALA at mu_k (U_k = target + (1 - k/M) log
    # w_full, through the flow); its start energy comes from the quantities
    # the reweighting evaluated, carried through the resampling.
    flow_1 = level_flow(1)
    y_parts, weight_parts, energy_parts = [], [], []
    for xc in jnp.array_split(samples, chunks, axis=0):
        push = flow_1.call_and_ladj if type == "F" else flow_1.inv_and_ladj
        yc, ladj = push(xc)
        u = target(yc)
        y_parts.append(yc)
        weight_parts.append(-u + source(xc) + ladj)
        energy_parts.append(u)
    proposal = jnp.concatenate(y_parts, axis=0)
    proposal_log_weights = jnp.concatenate(weight_parts, axis=0)
    y = proposal
    full_log_weight = proposal_log_weights
    u = jnp.concatenate(energy_parts, axis=0)
    for k in range(1, M + 1):
        flow_k = level_flow(k)
        if k > 1:
            weight_parts, energy_parts = [], []
            for yc in jnp.array_split(y, chunks, axis=0):
                if type == "F":
                    xc = flow_k.inv(yc)                 # x = F^{-1}(y)
                    _, ladj = flow_k.call_and_ladj(xc)  # log|det J_F(x)|
                else:
                    xc, ladj_G = flow_k.call_and_ladj(yc)  # x = G(y), log|det J_G(y)|
                    ladj = -ladj_G                       # log|det J_F(x)|
                u_c = target(yc)
                weight_parts.append(-u_c + source(xc) + ladj)
                energy_parts.append(u_c)
            full_log_weight = jnp.concatenate(weight_parts, axis=0)
            u = jnp.concatenate(energy_parts, axis=0)
        w = linear_weights_from_log(full_log_weight / M)
        key_r, key_l = jax.random.split(jax.random.fold_in(key_1, k))
        idx = resample_index(key_r, y, w)
        if k < M:
            s_k = 1.0 - k / M
            level_potential = _path_potential(
                source, target, _pushforward_log_density(flow_k, type), s_k
            )
            y = langevin(
                key_l, y[idx], level_potential, dt=mc_dt, steps=mc_steps_1,
                adjust=adjust, taming=taming, chunks=chunks,
                energy=u[idx] + s_k * full_log_weight[idx],
            )
        else:
            y = langevin(
                key_l, y[idx], target, dt=mc_dt, steps=mc_steps_2, adjust=adjust,
                taming=taming, chunks=chunks,
            )

    # ---- phase 2: pi -> pi^2 / nu along rho_k = pi (pi / nu)^{k/M} ----
    # Each level refreshes log(pi / nu) through that level's flow and
    # rejuvenates with MALA at its own rho_k (U_k = target - (k/M) log w,
    # through the flow); the start energy comes from the same target and
    # pushforward evaluations, carried through the resampling.
    for k in range(1, M + 1):
        flow_k = level_flow(M + k)
        log_density = _pushforward_log_density(flow_k, type)
        weight_parts, energy_parts = [], []
        for yc in jnp.array_split(y, chunks, axis=0):
            u_c = target(yc)
            weight_parts.append(-u_c - log_density(yc, source))   # log(pi / nu)
            energy_parts.append(u_c)
        full_log_weight = jnp.concatenate(weight_parts, axis=0)
        u = jnp.concatenate(energy_parts, axis=0)
        w = linear_weights_from_log(full_log_weight / M)
        s_k = -k / M
        level_potential = _path_potential(source, target, log_density, s_k)
        key_r, key_h = jax.random.split(jax.random.fold_in(key_2, k))
        idx = resample_index(key_r, y, w)
        y = langevin(
            key_h, y[idx], level_potential, dt=mc_dt,
            steps=mc_steps_1 if k < M else mc_steps_2,
            adjust=adjust, taming=taming, chunks=chunks,
            energy=u[idx] + s_k * full_log_weight[idx],
        )
    return y, proposal, proposal_log_weights


# aliases: the standard short names
smc = sequential_monte_carlo
smc_fab = sequential_monte_carlo_fab

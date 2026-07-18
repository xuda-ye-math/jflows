"""MCMC rejuvenation kernels for jflows — Langevin / Heun / HMC.

Split into a two-level interface:

    langevin_step / stochastic_heun_step / hmc_step / leapfrog
        — the low-level kernels: one update per call, key per call,
          returning `(x, aux)` with the tuning diagnostics (MH accept
          mask and log_alpha where applicable), for custom loops;
    langevin / stochastic_heun / hamiltonian_monte_carlo
        — the high-level loops, each exactly a lax.scan over its step.

Key convention: the loops take ONE key and derive per-chunk keys via
`jax.random.fold_in(key, chunk_index)`, then per-iteration keys via
`jax.random.split(chunk_key, n_iterations)` — so a manual composition of
the step functions with the same derivation reproduces the loop exactly.

There are no temperature arguments: to target exp(-beta * U), pass the
scaled potential `beta * U` (the potential algebra covers tempering).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax

from ..potential import Potential


__all__ = [
    "hamiltonian_monte_carlo",
    "hmc",
    "hmc_step",
    "langevin",
    "langevin_step",
    "leapfrog",
    "rejuvenation",
    "stochastic_heun",
    "stochastic_heun_step",
]


# ──────────────────────────────────────────────────────────────────────
# Langevin — overdamped Langevin / MALA / tamed variants
# ──────────────────────────────────────────────────────────────────────

def langevin_step(
    key: Array,
    x: Array,
    potential: Potential,
    dt: float = 1e-3,
    adjust: bool = True,
    taming: float = 0,
) -> tuple[Array, dict]:
    """
    One Langevin update (the loop body of `langevin`).

    Proposal (Euler-Maruyama on the overdamped Langevin SDE
    dtheta = -grad U(theta) dt + sqrt(2) dB):
        y = x - dt * grad U(x) + sqrt(2 * dt) * xi,   xi ~ N(0, I_d).

    With adjust=False the proposal is always accepted (ULA); with
    adjust=True a Metropolis-Hastings gate makes the update exact (MALA):
        log_alpha = U(x) - U(y) + log q(x|y) - log q(y|x),
        log q(z|w) = -||z - w + dt * grad U(w)||^2 / (4 * dt) + const.
    Both the energy difference AND the asymmetric-proposal correction are
    needed; the energy term alone leaves a residual O(dt) bias.

    When `taming > 0`, the raw drift grad U(x) is replaced with the tamed
    effective force
        G(x) = grad U(x) / (1 + taming * ||grad U(x)||),
    stabilizing ULA on targets whose |grad U| grows super-linearly.
    Tamed drift is incompatible with adjust=True (the MH correction
    assumes the untamed Gaussian proposal).

    Input:
        key:       PRNG key (split internally for noise and MH draw)
        x:         Array [N, d]   current particles
        potential: Potential      target potential U
        dt:        float          Euler-Maruyama step size
        adjust:    bool           if True, MALA (unbiased); if False, ULA
        taming:    float          if > 0, tamed drift (ULA only)
    Output:
        x:   Array [N, d]   updated particles
        aux: dict           {'accept': bool [N], 'log_alpha': [N]} when
                            adjust=True, else {}
    """
    if adjust and taming > 0:
        raise ValueError("langevin_step(): adjust=True and taming>0 are mutually exclusive.")
    key_noise, key_mh = jax.random.split(key)
    noise_scale = (2.0 * dt) ** 0.5

    fx = potential.grad(x)
    drift = fx / (1 + taming * jnp.linalg.norm(fx, axis=-1, keepdims=True)) if taming > 0 else fx
    y = x - dt * drift + noise_scale * jax.random.normal(key_noise, x.shape, dtype=x.dtype)
    if not adjust:
        return y, {}

    # log q(z|w) = -||z - w + dt * grad U(w)||^2 / (4 * dt) + const
    log_q_yx = -((y - x + dt * fx) ** 2).sum(axis=-1) / (4.0 * dt)  # log q(y|x)
    fy = potential.grad(y)
    log_q_xy = -((x - y + dt * fy) ** 2).sum(axis=-1) / (4.0 * dt)  # log q(x|y)
    log_alpha = potential(x) - potential(y) + log_q_xy - log_q_yx  # [N]
    accept = jnp.log(jax.random.uniform(key_mh, log_alpha.shape, dtype=x.dtype)) < log_alpha
    x_new = jnp.where(accept[:, None], y, x)
    return x_new, {"accept": accept, "log_alpha": log_alpha}


def langevin(
    key: Array,
    samples: Array,
    potential: Potential,
    dt: float = 1e-3,
    steps: int = 100,
    adjust: bool = True,
    taming: float = 0,
    chunks: int = 1,
) -> Array:
    """
    Langevin dynamics targeting exp(-U(x)) — `steps` calls of
    `langevin_step` under a lax.scan (see `langevin_step` for the ULA /
    MALA / tamed-drift semantics, and the module docstring for the exact
    key derivation that makes the loop reproducible from the steps).

    With adjust=True (default), the standard MALA scheme whose stationary
    distribution is *exactly* exp(-U) at ~2x the cost (two gradient calls
    per iteration). With adjust=False, the unadjusted Langevin algorithm
    (ULA): O(dt) bias, one gradient call per iteration.

    Exposed both as `langevin` and as the `rejuvenation` alias (in SMC
    literature, Langevin steps are the standard rejuvenation move).

    Input:
        key:       PRNG key
        samples:   Array [N, d]   initial particles
        potential: Potential      target potential U
        dt:        float          Euler-Maruyama step size
        steps:     int            number of Langevin steps
        adjust:    bool           if True, run MALA (unbiased); if False, ULA
        taming:    float          if > 0, tamed drift (ULA only)
        chunks:    int            split `samples` along dim 0 into this many
                                  execution chunks. Statistically equivalent
                                  to chunks=1 (each chunk uses independent
                                  noise); if an enclosing routine is jitted,
                                  XLA may co-schedule buffers, so this is not
                                  a strict peak-memory guarantee.
    Output:
        samples: Array [N, d]   particles after `steps` Langevin updates
    """
    if adjust and taming > 0:
        raise ValueError("langevin(): adjust=True and taming>0 are mutually exclusive.")
    out = []
    for i, x in enumerate(jnp.array_split(samples, chunks, axis=0)):
        keys = jax.random.split(jax.random.fold_in(key, i), steps)

        def body(x, k):
            x_new, _ = langevin_step(k, x, potential, dt=dt, adjust=adjust, taming=taming)
            return x_new, None

        x, _ = lax.scan(body, x, keys)
        out.append(x)
    return jnp.concatenate(out, axis=0)


# alias: in SMC literature, Langevin steps are the standard "rejuvenation" move
rejuvenation = langevin


# ──────────────────────────────────────────────────────────────────────
# Stochastic Heun — Stratonovich predictor-corrector Langevin
# ──────────────────────────────────────────────────────────────────────

def stochastic_heun_step(
    key: Array,
    x: Array,
    potential: Potential,
    dt: float = 1e-3,
) -> tuple[Array, dict]:
    """
    One stochastic-Heun update (the loop body of `stochastic_heun`).

    For the SDE dtheta = -grad U(theta) dt + sqrt(2) dB, one Heun step
    reuses a single Wiener increment dW = sqrt(2 * dt) * xi:
        predictor:  x~ = x - dt * grad U(x) + dW
        corrector:  x' = x - 0.5 * dt * (grad U(x) + grad U(x~)) + dW
    The drift is evaluated at both ends and averaged (trapezoidal rule)
    with the SAME dW, which makes the update Stratonovich-consistent;
    with additive noise, Ito and Stratonovich coincide.

    Input:
        key:       PRNG key
        x:         Array [N, d]   current particles
        potential: Potential      target potential U
        dt:        float          Heun step size
    Output:
        x:   Array [N, d]   updated particles
        aux: dict           {} (unadjusted scheme: no MH diagnostics)
    """
    dw = (2.0 * dt) ** 0.5 * jax.random.normal(key, x.shape, dtype=x.dtype)
    fx = potential.grad(x)
    x_pred = x - dt * fx + dw                        # Euler-Maruyama predictor
    fx_pred = potential.grad(x_pred)                 # force at the predictor
    return x - 0.5 * dt * (fx + fx_pred) + dw, {}    # Heun corrector (same dW)


def stochastic_heun(
    key: Array,
    samples: Array,
    potential: Potential,
    dt: float = 1e-3,
    steps: int = 100,
    chunks: int = 1,
) -> Array:
    """
    Overdamped Langevin dynamics targeting exp(-U(x)), integrated with
    the stochastic Heun (Stratonovich predictor-corrector) scheme —
    `steps` calls of `stochastic_heun_step` under a lax.scan.

    Versus Euler-Maruyama (`langevin(adjust=False)`): two gradient calls
    per iteration, but the trapezoidal drift cancels the leading O(dt)
    drift-discretization error, so the residual bias of this unadjusted
    scheme shrinks faster as dt -> 0. It remains *unadjusted*; use
    `langevin(adjust=True)` or `hmc` when you need the exactly-unbiased
    target.

    Input:
        key:       PRNG key
        samples:   Array [N, d]   initial particles
        potential: Potential      target potential U
        dt:        float          Heun step size
        steps:     int            number of Heun steps
        chunks:    int            split `samples` along dim 0 (see `langevin`)
    Output:
        samples: Array [N, d]   particles after `steps` Heun updates
    """
    out = []
    for i, x in enumerate(jnp.array_split(samples, chunks, axis=0)):
        keys = jax.random.split(jax.random.fold_in(key, i), steps)

        def body(x, k):
            x_new, _ = stochastic_heun_step(k, x, potential, dt=dt)
            return x_new, None

        x, _ = lax.scan(body, x, keys)
        out.append(x)
    return jnp.concatenate(out, axis=0)


# ──────────────────────────────────────────────────────────────────────
# HMC — Hamiltonian Monte Carlo with leapfrog + MH gate
# ──────────────────────────────────────────────────────────────────────

def leapfrog(
    x: Array,
    p: Array,
    potential: Potential,
    dt: float,
    steps: int,
) -> tuple[Array, Array]:
    """
    Deterministic leapfrog integration of the Hamiltonian flow of
    H(x, p) = U(x) + 0.5 * ||p||^2 — exposed for custom HMC variants.

    Efficient form with combined half-kicks: `steps + 1` gradient calls
    per trajectory instead of the naive 2 * steps. Exactly
    volume-preserving and time-reversible, so an MH gate on the endpoint
    needs no Jacobian term.

    Input:
        x:         Array [N, d]   positions
        p:         Array [N, d]   momenta
        potential: Potential      target potential U
        dt:        float          leapfrog step size
        steps:     int            number of leapfrog steps (>= 1)
    Output:
        x: Array [N, d]   positions after `steps` steps
        p: Array [N, d]   momenta after `steps` steps
    """
    if steps < 1:
        return x, p
    p = p - 0.5 * dt * potential.grad(x)  # leading half-kick

    def body(carry, _):
        x, p = carry
        x = x + dt * p
        p = p - dt * potential.grad(x)
        return (x, p), None

    (x, p), _ = lax.scan(body, (x, p), None, length=steps - 1)
    x = x + dt * p
    p = p - 0.5 * dt * potential.grad(x)  # trailing half-kick
    return x, p


def hmc_step(
    key: Array,
    x: Array,
    potential: Potential,
    dt: float = 1e-2,
    leapfrog_steps: int = 10,
) -> tuple[Array, dict]:
    """
    One full HMC trajectory (the loop body of `hamiltonian_monte_carlo`;
    one MCMC burn):

      1. Resample momentum p ~ N(0, I_d) — a complete refresh that
         discards any correlation with the previous trajectory.
      2. Integrate the Hamiltonian flow for `leapfrog_steps` steps of
         size `dt` (see `leapfrog`).
      3. Metropolis-Hastings accept/reject on the endpoint:
            log_alpha = U(x0) - U(x_end) + 0.5 * (||p0||^2 - ||p_end||^2).
         Divergent trajectories (non-finite log_alpha) are clamped to
         -inf and rejected, so the returned particles are always finite.

    Input:
        key:       PRNG key (split internally for momentum and MH draw)
        x:         Array [N, d]   current particles
        potential: Potential      target potential U
        dt:             float     leapfrog step size
        leapfrog_steps: int       leapfrog steps per trajectory
    Output:
        x:   Array [N, d]   updated particles
        aux: dict           {'accept': bool [N], 'log_alpha': [N]}
    """
    key_p, key_mh = jax.random.split(key)
    x_start = x
    p_start = jax.random.normal(key_p, x.shape, dtype=x.dtype)
    U_start = potential(x_start)
    K_start = 0.5 * (p_start**2).sum(axis=-1)  # [N]

    x_end, p_end = leapfrog(x_start, p_start, potential, dt, leapfrog_steps)
    U_end = potential(x_end)
    K_end = 0.5 * (p_end**2).sum(axis=-1)  # [N]

    # MH accept/reject with NaN guard: divergent trajectories produce
    # non-finite log_alpha -> -inf -> reject -> revert to x_start.
    log_alpha = (U_start - U_end) + (K_start - K_end)  # [N]
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
    accept = jnp.log(jax.random.uniform(key_mh, log_alpha.shape, dtype=x.dtype)) < log_alpha
    x_new = jnp.where(accept[:, None], x_end, x_start)
    return x_new, {"accept": accept, "log_alpha": log_alpha}


def hamiltonian_monte_carlo(
    key: Array,
    samples: Array,
    potential: Potential,
    dt: float = 1e-2,
    leapfrog_steps: int = 10,
    trajectories: int = 10,
    chunks: int = 1,
) -> Array:
    """
    Hamiltonian Monte Carlo targeting exp(-U(x)) — `trajectories` calls of
    `hmc_step` under a lax.scan (each a full momentum refresh + `leapfrog_steps`
    leapfrog steps + one MH decision).

    Compared to MALA (`langevin(adjust=True)`), HMC pays `leapfrog_steps + 1`
    gradient calls per MH decision instead of 2, but the trajectory
    moves O(dt * leapfrog_steps) per decision rather than O(sqrt(dt)), giving
    much lower autocorrelation at fixed acceptance. The MH correction
    makes the stationary distribution exactly exp(-U), and divergent
    trajectories revert to their pre-trajectory positions (NaN guard in
    `hmc_step`).

    Exposed both as `hamiltonian_monte_carlo` and as the `hmc` alias.

    Input:
        key:       PRNG key
        samples:   Array [N, d]   initial particles
        potential: Potential      target potential U
        dt:        float          leapfrog step size epsilon. Tune so the
                                  MH acceptance rate is ~0.6-0.8 (the HMC
                                  sweet spot from Beskos et al. 2013).
        leapfrog_steps: int       leapfrog steps per trajectory.
                                  Trajectory length L = dt * leapfrog_steps; pick
                                  L on the scale of the target's largest
                                  correlation length.
        trajectories: int         number of momentum refreshes / MH
                                  trajectories.
        chunks:    int            split `samples` along dim 0 (see `langevin`)
    Output:
        samples: Array [N, d]   particles after `trajectories` HMC trajectories
    """
    out = []
    for i, x in enumerate(jnp.array_split(samples, chunks, axis=0)):
        keys = jax.random.split(jax.random.fold_in(key, i), trajectories)

        def body(x, k):
            x_new, _ = hmc_step(
                k, x, potential, dt=dt, leapfrog_steps=leapfrog_steps
            )
            return x_new, None

        x, _ = lax.scan(body, x, keys)
        out.append(x)
    return jnp.concatenate(out, axis=0)


# alias: HMC is the standard short name for Hamiltonian Monte Carlo
hmc = hamiltonian_monte_carlo

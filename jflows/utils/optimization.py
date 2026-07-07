"""Batched particle optimization for jflows — L-BFGS and AdamW.

`lbfgs` (aliased `optimization`) and `adamw` are the two batched
optimizers, `adamw` an additional first-order alternative. Both follow
the two-level interface:

    lbfgs_init / lbfgs_step / LBFGS_State — the low-level L-BFGS kernel:
        an explicit state pytree and one update per call, for custom
        optimization loops;
    lbfgs (alias `optimization`)          — the high-level loop, exactly
        a lax.scan over `lbfgs_step`;
    adamw_init / adamw_step / AdamW_State — the low-level AdamW kernel;
    adamw                                 — the high-level loop, exactly
        a lax.scan over `adamw_step`.

The curvature history is stored as fixed-shape ring buffers (jit needs
static shapes): slot 0 is the oldest pair, slot -1 the newest, and empty
or curvature-violating pairs carry rho = 0, which makes the two-loop
recursion ignore them — the same masking trick applied to invalid pairs,
so the math is unchanged.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax

from ..potential import Potential


__all__ = [
    "AdamW_State",
    "LBFGS_State",
    "adamw",
    "adamw_init",
    "adamw_step",
    "lbfgs",
    "lbfgs_init",
    "lbfgs_step",
    "optimization",
]

_C1, _SHRINK, _K_MAX = 1e-4, 0.5, 6


class LBFGS_State(eqx.Module):
    """Carryable L-BFGS state (a pytree; scan/jit-friendly).

    Fields:
        x:   Array [N, d]        current particles
        g:   Array [N, d]        gradient of U at x
        s:   Array [m, N, d]     displacement ring buffer (oldest first)
        y:   Array [m, N, d]     gradient-difference ring buffer
        rho: Array [m, N]        1 / (s . y); 0 marks an empty slot or a
                                 pair violating the curvature condition
        U:   Array [N]           cached energies U(x) — maintained by
                                 `lbfgs_init` and armijo steps (stale
                                 after a non-armijo step; only the armijo
                                 line search reads it)
        k:   Array (scalar int)  iteration counter (0 = fresh state)
    """

    x: Array
    g: Array
    s: Array
    y: Array
    rho: Array
    U: Array
    k: Array


def lbfgs_init(x: Array, potential: Potential, memory: int = 6) -> LBFGS_State:
    """
    Build a fresh `LBFGS_State` at particles `x` (one batched grad and
    one batched energy evaluation).
    Input:
        x:         Array [N, d]   initial particles
        potential: Potential      target potential U
        memory:    int            curvature pairs kept per particle
    Output:
        state: LBFGS_State
    """
    N, d = x.shape
    return LBFGS_State(
        x=x,
        g=potential.grad(x),
        s=jnp.zeros((memory, N, d), dtype=x.dtype),
        y=jnp.zeros((memory, N, d), dtype=x.dtype),
        rho=jnp.zeros((memory, N), dtype=x.dtype),
        U=potential(x),
        k=jnp.zeros((), dtype=jnp.int32),
    )


def lbfgs_step(
    state: LBFGS_State,
    potential: Potential,
    step: float = 1.0,
    armijo: bool = False,
) -> LBFGS_State:
    """
    One batched L-BFGS update (the loop body of `lbfgs`).

    Per call:
      1. Two-loop recursion over the ring buffers: combine the current
         gradient g_k with the stored pairs (s_i, y_i) to get the search
         direction d_k = -H_k^{-1} g_k, where H_0^k = gamma_k * I and
         gamma_k = (s_last . y_last) / ||y_last||^2 (= 1 on the fresh
         state).
      2. Update x. With `armijo=False`, a fixed step `x - step * r`.
         With `armijo=True`, per-particle masked Armijo backtracking:
         start at alpha = step, halve until
            U(x + alpha * d) <= U(x) + C1 * alpha * (d . g),  C1 = 1e-4,
         all particles running K_MAX = 6 trials in lockstep; particles
         that never satisfy Armijo take the smallest tested step. Costs
         one batched grad + K_MAX batched energy evaluations.
      3. Evaluate the new gradient and append the curvature pair; pairs
         with s . y <= 1e-10 keep rho = 0 (ignored by the recursion).

    Input:
        state:     LBFGS_State   from `lbfgs_init` or a previous step
        potential: Potential     target potential U
        step:      float         step multiplier / initial Armijo trial
        armijo:    bool          masked backtracking line search
    Output:
        state: LBFGS_State   after one update
    """
    x, g = state.x, state.g
    m = state.s.shape[0]

    # Two-loop recursion: r approximates H_k^{-1} g
    q = g
    alphas = []
    for i in reversed(range(m)):
        a_i = state.rho[i] * (state.s[i] * q).sum(axis=-1)  # [N]
        q = q - a_i[:, None] * state.y[i]
        alphas.append(a_i)
    alphas.reverse()  # chronological order

    ys_last = (state.s[-1] * state.y[-1]).sum(axis=-1)
    yy_last = (state.y[-1] ** 2).sum(axis=-1)
    gamma = jnp.where(
        (state.k > 0) & (ys_last > 1e-10),   # only a curvature-valid newest pair scales H_0
        ys_last / jnp.maximum(yy_last, 1e-10),
        jnp.ones_like(ys_last),
    )
    r = gamma[:, None] * q
    for i in range(m):
        beta_i = state.rho[i] * (state.y[i] * r).sum(axis=-1)
        r = r + (alphas[i] - beta_i)[:, None] * state.s[i]

    if armijo:
        # Masked Armijo backtracking: per-particle alpha, all particles
        # run K_MAX trials in lockstep.
        d = -r  # search direction
        dg = (d * g).sum(axis=-1)  # [N], < 0 for a descent direction
        alpha = jnp.full(x.shape[0], step, dtype=x.dtype)
        done = jnp.zeros(x.shape[0], dtype=bool)
        for _ in range(_K_MAX):
            x_trial = x + alpha[:, None] * d
            U_trial = potential(x_trial)
            ok = U_trial <= state.U + _C1 * alpha * dg  # [N] bool
            done = done | ok
            alpha = jnp.where(done, alpha, alpha * _SHRINK)
        # x_trial encodes "accepted alpha" for done particles (alpha was
        # frozen at acceptance) and "smallest tested alpha" otherwise.
        x_new, U_new = x_trial, U_trial
    else:
        x_new, U_new = x - step * r, state.U

    g_new = potential.grad(x_new)
    s_new = x_new - x
    y_new = g_new - g
    ys = (s_new * y_new).sum(axis=-1)  # [N]
    rho_new = jnp.where(ys > 1e-10, 1.0 / ys, jnp.zeros_like(ys))

    return LBFGS_State(
        x=x_new,
        g=g_new,
        s=jnp.roll(state.s, -1, axis=0).at[-1].set(s_new),
        y=jnp.roll(state.y, -1, axis=0).at[-1].set(y_new),
        rho=jnp.roll(state.rho, -1, axis=0).at[-1].set(rho_new),
        U=U_new,
        k=state.k + 1,
    )


def lbfgs(
    samples: Array,
    potential: Potential,
    step: float = 1.0,
    iters: int = 100,
    memory: int = 6,
    armijo: bool = False,
    chunk: int = 1,
) -> Array:
    """
    Batched L-BFGS for mode-finding / MAP refinement on the target
    exp(-U(x)). Every particle in `samples` carries its own (s, y)
    history and they all step in lockstep through vectorised array ops,
    so N=2000 particles optimise as cheaply as one (modulo per-row
    arithmetic). Exposed both as `lbfgs` and as the `optimization` alias
    (the default mode-finder).

    L-BFGS builds a rank-`memory` approximation of the inverse Hessian
    from the last `memory` gradient differences, giving superlinear
    convergence on smooth potentials — contrast with Adam-style sign
    descent, which is O(init_err / step) just to reach the basin.

    Exactly a `lax.scan` over `lbfgs_step` started from
    `lbfgs_init(x, potential, memory)` — compose those two directly for
    custom loops (per-iteration step schedules, convergence monitoring).

    Input:
        samples:   Array [N, d]   initial particles
        potential: Potential      target potential U
        step:      float          multiplier on the L-BFGS direction.
                                  With armijo=False: fixed per-iter step.
                                  With armijo=True: initial trial alpha
                                  for the backtracking line search.
                                  1.0 ~ pure Newton step; reduce
                                  (e.g. 0.5, 0.1) for stiff problems.
        iters:     int            number of L-BFGS iterations
        memory:    int            curvature pairs (s, y) kept per
                                  particle (Nocedal's `m`). Typical 3-20;
                                  larger = better Hessian approximation
                                  and more memory (N * d * 2 * memory floats).
        armijo:    bool           if True, enable masked Armijo
                                  backtracking (K_MAX = 6 extra batched
                                  energy evaluations per iteration, but
                                  guaranteed sufficient decrease). Use it
                                  when the line-search-free update is
                                  unstable (first-iter blow-up, very
                                  non-convex landscapes).
        chunk:     int            split `samples` along dim 0 into this
                                  many chunks and run sequentially.
                                  Reduces peak memory at the cost of wall
                                  time; equivalent to chunk=1 (each
                                  particle's history is independent, and
                                  there is no noise).
    Output:
        samples: Array [N, d]   particles after `iters` L-BFGS updates
    """
    out = []
    for x in jnp.array_split(samples, chunk, axis=0):
        state = lbfgs_init(x, potential, memory=memory)

        def body(s, _):
            return lbfgs_step(s, potential, step=step, armijo=armijo), None

        state, _ = lax.scan(body, state, None, length=iters)
        out.append(state.x)
    return jnp.concatenate(out, axis=0)


# alias: L-BFGS is the default mode-finder / MAP-refinement routine in jflows
optimization = lbfgs


# ──────────────────────────────────────────────────────────────────────
# AdamW — first-order alternative for rough landscapes
# ──────────────────────────────────────────────────────────────────────

class AdamW_State(eqx.Module):
    """Carryable AdamW state (a pytree; scan/jit-friendly).

    Fields:
        x: Array [N, d]        current particles
        m: Array [N, d]        first-moment (gradient) running average
        v: Array [N, d]        second-moment running average
        k: Array (scalar int)  step counter (drives bias correction)
    """

    x: Array
    m: Array
    v: Array
    k: Array


def adamw_init(x: Array) -> AdamW_State:
    """
    Build a fresh `AdamW_State` at particles `x` (no potential
    evaluation — moments start at zero).
    Input:
        x: Array [N, d]   initial particles
    Output:
        state: AdamW_State
    """
    return AdamW_State(
        x=x,
        m=jnp.zeros_like(x),
        v=jnp.zeros_like(x),
        k=jnp.zeros((), dtype=jnp.int32),
    )


def adamw_step(
    state: AdamW_State,
    potential: Potential,
    step: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> AdamW_State:
    """
    One batched AdamW update (the loop body of `adamw`):

        m <- beta1 * m + (1 - beta1) * grad U(x)
        v <- beta2 * v + (1 - beta2) * grad U(x)^2
        x <- x - step * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * x ),

    with the standard bias-corrected m_hat / v_hat (Loshchilov & Hutter,
    2019 — the weight decay is decoupled from the gradient).

    Input:
        state:        AdamW_State   from `adamw_init` or a previous step
        potential:    Potential     target potential U
        step:         float         learning rate
        beta1:        float         first-moment decay
        beta2:        float         second-moment decay
        eps:          float         denominator floor
        weight_decay: float         decoupled L2 shrinkage toward the
                                    origin. NOTE: any nonzero value
                                    biases the stationary points away
                                    from the modes of U — leave at 0.0
                                    for pure mode-finding.
    Output:
        state: AdamW_State   after one update
    """
    g = potential.grad(state.x)
    k = state.k + 1
    m = beta1 * state.m + (1 - beta1) * g
    v = beta2 * state.v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1 ** k.astype(state.x.dtype))
    v_hat = v / (1 - beta2 ** k.astype(state.x.dtype))
    x = state.x - step * (m_hat / (jnp.sqrt(v_hat) + eps) + weight_decay * state.x)
    return AdamW_State(x=x, m=m, v=v, k=k)


def adamw(
    samples: Array,
    potential: Potential,
    step: float = 1e-2,
    iters: int = 100,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    chunk: int = 1,
) -> Array:
    """
    Batched AdamW descent on the target exp(-U(x)) — a first-order
    alternative to `lbfgs` for mode-finding. Every particle carries its
    own moment estimates and all step in lockstep.

    Versus `lbfgs`: one gradient call per iteration and sign-descent-like
    robustness on rough or noisy-curvature landscapes, but only linear
    convergence — expect O(init_err / step) iterations to reach the
    basin, and an O(step)-scale residual oscillation around the mode
    (shrink `step` or switch to `lbfgs` for the final refinement; the
    quench in a quench-and-temper pipeline is a natural user).

    Exactly a `lax.scan` over `adamw_step` started from
    `adamw_init(x)` — compose those directly for custom loops
    (learning-rate schedules, convergence monitoring).

    Input:
        samples:      Array [N, d]   initial particles
        potential:    Potential      target potential U
        step:         float          learning rate
        iters:        int            number of AdamW iterations
        beta1:        float          first-moment decay (default 0.9)
        beta2:        float          second-moment decay (default 0.999)
        eps:          float          denominator floor (default 1e-8)
        weight_decay: float          decoupled L2 shrinkage; nonzero
                                     values bias the stationary points
                                     away from the modes of U (default 0.0)
        chunk:        int            split `samples` along dim 0 into this
                                     many chunks and run sequentially
                                     (memory bound; equivalent to chunk=1)
    Output:
        samples: Array [N, d]   particles after `iters` AdamW updates
    """
    out = []
    for x in jnp.array_split(samples, chunk, axis=0):
        state = adamw_init(x)

        def body(s, _):
            return adamw_step(
                s, potential, step=step, beta1=beta1, beta2=beta2,
                eps=eps, weight_decay=weight_decay,
            ), None

        state, _ = lax.scan(body, state, None, length=iters)
        out.append(state.x)
    return jnp.concatenate(out, axis=0)

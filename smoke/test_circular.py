"""Standalone circular-boundary test (jflows only) — run after installation
from the repo root as `python -m smoke.test_circular`.

Tests the periodic boundary conditions of the circular machinery
separately from the generic flow properties (test_flow.py):

    1. CircularShiftTransform: identity inside [-B, B), 2B-periodic
       wrap outside, zero log-det;
    2. derivative boundary handling of MonotonicRQSTransform:
       circular=True gives d_0 == d_K exactly (one learnable seam slope,
       generically != 1); circular=False keeps d_0 = d_K = 1 (the NSF
       identity-tail convention, unchanged);
    3. CircularRQSTransform (fixed parameters): 2B-periodicity,
       value continuity across the seam (mod 2B), and C¹ seam
       continuity — including the exact left-endpoint value, derivative,
       inverse derivative, and arbitrary-period representatives;
    4. full NCSF: domain preservation, circle round-trip, ladj vs
       autodiff slogdet, trainable (unpinned) seam density, and
       pushforward-density normalisation over the torus.

Float64, on the default JAX backend (GPU when available; set
JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled.
Exits nonzero on any failure.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.core.flows import CircularRQSTransform  # noqa: E402
from jflows.core.transforms import (  # noqa: E402
    CircularShiftTransform,
    MonotonicRQSTransform,
)
from jflows.flow import NCSF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_circular.log")
PI = jnp.pi

FAILURES = 0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check(name: str, got, want, tol: float) -> None:
    global FAILURES
    got, want = np.asarray(got), np.asarray(want)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    ok = got.shape == want.shape and err <= tol
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def check_true(name: str, cond: bool, detail: str = "") -> None:
    global FAILURES
    log(f"  {name}: {detail}{' ' if detail else ''}-> {'OK' if cond else 'FAIL'}")
    if not cond:
        FAILURES += 1


def wrap(x, B=PI):
    """Reduce to the fundamental domain [-B, B)."""
    return jnp.remainder(x + B, 2 * B) - B


def main() -> None:
    global FAILURES
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_circular | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    K = 8

    # ── 1. CircularShiftTransform ──
    log("CircularShiftTransform")
    c = CircularShiftTransform(bound=PI)
    x = jnp.linspace(-PI + 1e-9, PI - 1e-9, 1001)
    check("identity inside [-B, B)", c(x), x, tol=1e-12)
    check("wrap of x + 2B", c(x + 2 * PI), x, tol=1e-9)
    check("wrap of x - 2B", c(x - 2 * PI), x, tol=1e-9)
    check("ladj == 0", c.log_abs_det_jacobian(x, c(x)), jnp.zeros_like(x), tol=0)

    # ── 2. derivative boundary handling ──
    log("MonotonicRQSTransform derivative boundaries")
    kw, kh, kd = jax.random.split(key, 3)
    w = jax.random.normal(kw, (K,))
    h = jax.random.normal(kh, (K,))
    dl = jax.random.normal(kd, (K,))
    rqs_circ = MonotonicRQSTransform(w, h, dl, bound=PI, circular=True)
    check("circular: d_0 == d_K exactly",
          rqs_circ.derivatives[..., 0], rqs_circ.derivatives[..., -1], tol=0)
    check_true("circular: seam slope is learnable (d_0 != 1 generically)",
               bool(jnp.abs(rqs_circ.derivatives[0] - 1.0) > 1e-2),
               f"d_0 = {float(rqs_circ.derivatives[0]):.4f}")
    check_true("circular: knot-derivative count == K + 1",
               rqs_circ.derivatives.shape[-1] == K + 1,
               f"shape {rqs_circ.derivatives.shape}")
    index_probes = jnp.asarray([
        rqs_circ.horizontal[0] - 1.0,
        rqs_circ.horizontal[0],
        rqs_circ.horizontal[3],
        rqs_circ.horizontal[-1],
        rqs_circ.horizontal[-1] + 1.0,
    ])
    check("first-knot fix preserves every other bin convention",
          rqs_circ.searchsorted(rqs_circ.horizontal, index_probes),
          jnp.asarray([0, 1, 3, K, K + 1]), tol=0)
    rqs_lin = MonotonicRQSTransform(w, h, dl[: K - 1], bound=PI, circular=False)
    check("non-circular: d_0 == d_K == 1 (NSF tails unchanged)",
          jnp.stack([rqs_lin.derivatives[0], rqs_lin.derivatives[-1]]),
          jnp.ones(2), tol=0)
    left = jnp.asarray([-PI])
    left_y, left_ladj = rqs_circ.call_and_ladj(left)
    left_x, left_inverse_ladj = rqs_circ.inv.call_and_ladj(left_y)
    check("circular: exact left endpoint value", left_y, left, tol=0)
    check("circular: exact left endpoint log-slope", left_ladj,
          jnp.log(rqs_circ.derivatives[:1]), tol=1e-12)
    check("circular: exact left endpoint inverse", left_x, left, tol=0)
    check("circular: exact inverse log-slope", left_inverse_ladj,
          -left_ladj, tol=1e-12)
    linear_edge_y, linear_edge_ladj = rqs_lin.call_and_ladj(
        jnp.asarray([-PI, PI])
    )
    check("non-circular: exact boundary values unchanged", linear_edge_y,
          jnp.asarray([-PI, PI]), tol=1e-12)
    check("non-circular: exact boundary log-slopes remain zero",
          linear_edge_ladj, jnp.zeros(2), tol=0)

    # ── 3. CircularRQSTransform with fixed parameters ──
    log("CircularRQSTransform (fixed parameters)")
    T = CircularRQSTransform(w, h, dl, bound=PI, slope=1e-3)
    xin = jnp.linspace(-PI + 1e-6, PI - 1e-6, 1001)
    check("2B-periodicity: T(x - 2B) == T(x)", T(x - 2 * PI), T(x), tol=1e-9)
    check("2B-periodicity: T(x + 2B) == T(x)", T(x + 2 * PI), T(x), tol=1e-9)
    check("maps into [-B, B]", jnp.clip(T(xin), -PI, PI), T(xin), tol=1e-12)
    check("round-trip on the circle", wrap(T.inv(T(xin)) - xin), jnp.zeros_like(xin), tol=1e-9)
    for eps in (1e-2, 1e-4, 1e-6):
        y_hi, l_hi = T.call_and_ladj(jnp.asarray([PI - eps]))
        y_lo, l_lo = T.call_and_ladj(jnp.asarray([-PI + eps]))
        gap_val = float(jnp.abs(wrap(y_hi - y_lo))[0])
        gap_ladj = float(jnp.abs(l_hi - l_lo)[0])
        check_true(f"seam continuity (eps={eps:g})", gap_val < 3 * eps * 10 and gap_ladj < 100 * eps,
                   f"value gap {gap_val:.2e}, log-slope gap {gap_ladj:.2e}")
    _, l_seam = T.call_and_ladj(jnp.asarray([PI - 1e-9]))
    check("seam log-slope == log d_0 (shared, learnable)",
          l_seam, jnp.log(rqs_circ.derivatives[..., :1]), tol=1e-6)
    seam_representatives = jnp.asarray([-3 * PI, -PI, PI, 3 * PI])
    seam_y, seam_ladj = T.call_and_ladj(seam_representatives)
    seam_x, seam_inverse_ladj = T.inv.call_and_ladj(seam_y)
    expected_seam_ladj = jnp.full_like(
        seam_ladj, jnp.log(rqs_circ.derivatives[0])
    )
    check("exact seam representatives have one value", seam_y,
          jnp.full_like(seam_y, -PI), tol=1e-12)
    check("exact seam representatives have learned log-slope", seam_ladj,
          expected_seam_ladj, tol=1e-12)
    check("exact seam inverse returns canonical representative", seam_x,
          jnp.full_like(seam_x, -PI), tol=1e-12)
    check("exact seam inverse log-slope", seam_inverse_ladj,
          -expected_seam_ladj, tol=1e-12)
    seam_derivative = jax.grad(lambda value: T(value[None])[0])(jnp.asarray(-PI))
    check("exact seam autodiff slope", seam_derivative,
          rqs_circ.derivatives[0], tol=1e-12)
    bound32 = jnp.asarray(np.pi, dtype=jnp.float32)
    T32 = CircularRQSTransform(
        w.astype(jnp.float32),
        h.astype(jnp.float32),
        dl.astype(jnp.float32),
        bound=bound32,
        slope=1e-3,
    )
    seam32 = jnp.asarray([-bound32, bound32], dtype=jnp.float32)
    seam_y32, seam_ladj32 = T32.call_and_ladj(seam32)
    expected_ladj32 = jnp.full_like(
        seam_ladj32, jnp.log(T32.transforms[1].derivatives[0])
    )
    check("float32 exact seam values", seam_y32,
          jnp.full_like(seam_y32, -bound32), tol=0)
    check("float32 exact seam learned log-slope", seam_ladj32,
          expected_ladj32, tol=1e-6)

    # ── 4. full NCSF ──
    log("NCSF (d=3, randmask=True)")
    pi3 = jnp.full((3,), PI)
    flow = NCSF(jax.random.key(7), -pi3, pi3, bins=K, transforms=3,
                hidden_features=(32, 16))
    F = flow.t()
    xb = jax.random.uniform(jax.random.key(8), (64, 3), minval=-PI, maxval=PI)
    y, ladj = F.call_and_ladj(xb)
    check("domain preservation", jnp.clip(y, -PI, PI), y, tol=1e-12)
    check("circle round-trip", wrap(F.inv(y) - xb), jnp.zeros_like(xb), tol=1e-9)
    J = jax.vmap(jax.jacfwd(lambda xi: F(xi[None, :])[0]))(xb)
    _, logdet = jnp.linalg.slogdet(J)
    check("ladj vs slogdet", ladj, logdet, tol=1e-9)

    eps = 1e-9
    seam = jnp.asarray([[PI - eps] * 3, [-PI + eps] * 3])
    _, ladj_seam = F.call_and_ladj(seam)
    check_true("seam density is trainable (lim ladj != 0 at seam)",
               bool(jnp.abs(ladj_seam).min() > 1e-3),
               f"ladj(seam∓eps) = {np.asarray(ladj_seam).round(4)}")
    exact_seam = jnp.asarray(
        [[-PI] * 3, [PI] * 3, [-3 * PI] * 3, [3 * PI] * 3]
    )
    exact_y, exact_ladj = F.call_and_ladj(exact_seam)
    check("NCSF exact seam representative invariance",
          wrap(exact_y - exact_y[:1]), jnp.zeros_like(exact_y), tol=1e-12)
    check("NCSF exact seam log-density invariance", exact_ladj,
          jnp.full_like(exact_ladj, exact_ladj[0]), tol=1e-12)
    check("NCSF exact seam agrees with left limit", exact_ladj[:1],
          ladj_seam[1:2], tol=1e-7)
    exact_x, exact_inverse_ladj = F.inv.call_and_ladj(exact_y)
    check("NCSF exact seam circle round-trip", wrap(exact_x - exact_seam),
          jnp.zeros_like(exact_x), tol=1e-9)
    check("NCSF exact seam inverse ladj", exact_inverse_ladj,
          -exact_ladj, tol=1e-9)

    log("NCSF pushforward normalisation on the torus (d=2, 400x400 grid)")
    flow2 = NCSF(jax.random.key(9), [-float(PI)] * 2, [float(PI)] * 2, bins=K,
                 transforms=2, hidden_features=(32, 16))
    n = 400
    g = jnp.linspace(-PI, PI, n)
    yy = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    G = flow2.t().inv
    _, ladj_g = G.call_and_ladj(yy)
    q = jnp.exp(-2 * jnp.log(2 * PI) + ladj_g).reshape(n, n)
    integral = float(jnp.trapezoid(jnp.trapezoid(q, g, axis=1), g))
    check_true("density integrates to 1", abs(integral - 1.0) < 5e-3,
               f"integral = {integral:.6f}")

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all circular-boundary tests passed")


if __name__ == "__main__":
    main()

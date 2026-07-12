"""Standalone optimization smoke test (jflows only) — run after installation
from the repo root as `python -m smoke.test_optimization`.

For lbfgs / lbfgs_init / lbfgs_step / LBFGS_State (and the adamw
counterparts, section 7):

    1. quadratic potential: convergence to the mean at near-machine
       precision in a handful of iterations (armijo on and off);
    2. Gaussian mixture: particles land on stationary points (gradient
       norm ~ 0) and the energy decreases monotonically with armijo;
    3. loop == composition: `lbfgs` equals the manual
       lbfgs_init + lbfgs_step chain exactly;
    4. chunk invariance (deterministic algorithm — exact);
    5. memory=1 still converges on the quadratic;
    6. jit: lbfgs_step carries LBFGS_State through eqx.filter_jit;
    7. adamw: quadratic convergence (up to the O(step) residual), loop ==
       composition, weight_decay pulls the solution toward the origin,
       chunk invariance.

Float64, on the default JAX backend (GPU when available; set
JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled.
Exits nonzero on any failure.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import (  # noqa: E402
    Nlog_Gaussian,
    Nlog_Gaussian_Mixture,
    potential_from,
)
from jflows.utils import (  # noqa: E402
    adamw,
    adamw_init,
    adamw_step,
    lbfgs,
    lbfgs_init,
    lbfgs_step,
    optimization,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_optimization.log")

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


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_optimization | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    N, d = 64, 3

    # ── 1. quadratic: exact mode is the mean ──
    log("quadratic convergence")
    mean = jnp.asarray([1.0, -2.0, 0.5])
    gau = Nlog_Gaussian(mean, [0.5, 2.0, 1.0])
    x0 = jax.random.normal(key, (N, d)) * 4.0
    for armijo in (False, True):
        xf = lbfgs(x0, gau, step=1.0, iters=25, memory=6, armijo=armijo)
        check(f"armijo={armijo}: reaches the mean", xf,
              jnp.broadcast_to(mean, (N, d)), tol=1e-6)
    check_true("optimization is lbfgs", optimization is lbfgs)

    # ── 2. Gaussian mixture: stationary points, monotone energy (armijo) ──
    log("Gaussian mixture")
    gmm = Nlog_Gaussian_Mixture(
        [0.4, 0.6], [[-2.0, 1.0, 0.0], [2.0, -1.0, 0.5]],
        [[0.5, 0.3, 0.4], [0.4, 0.6, 0.3]],
    )
    x0m = jax.random.normal(jax.random.key(1), (N, d)) * 2.0
    xf = lbfgs(x0m, gmm, step=0.5, iters=60, memory=6, armijo=False)
    gnorm = float(jnp.linalg.norm(gmm.grad(xf), axis=-1).max())
    check_true("stationary points (max |grad| < 1e-5)", gnorm < 1e-5, f"max|grad| = {gnorm:.2e}")

    state = lbfgs_init(x0m, gmm, memory=6)
    energies = [float(gmm(state.x).mean())]
    for _ in range(30):
        state = lbfgs_step(state, gmm, step=1.0, armijo=True)
        energies.append(float(gmm(state.x).mean()))
    check_true("armijo: mean energy non-increasing",
               all(b <= a + 1e-10 for a, b in zip(energies, energies[1:])),
               f"{energies[0]:.3f} -> {energies[-1]:.3f}")

    # ── 2b. bounded Armijo exhaustion is a rejection, never an uphill step ──
    log("strict bounded Armijo fallback")
    quartic = potential_from(lambda x: (x ** 4).sum(axis=-1))
    mixed0 = jnp.asarray([[0.1], [10.0]])
    mixed_state = lbfgs_init(mixed0, quartic, memory=3)
    mixed_next = lbfgs_step(mixed_state, quartic, step=1.0, armijo=True)
    check_true(
        "mixed acceptance: easy row moves",
        bool(jnp.abs(mixed_next.x[0] - mixed0[0]).max() > 0),
    )
    check(
        "mixed acceptance: exhausted row stays put",
        mixed_next.x[1], mixed0[1], tol=0.0,
    )
    check_true(
        "every accepted/fallback row is non-increasing",
        bool(jnp.all(mixed_next.U <= mixed_state.U)),
        f"U {np.asarray(mixed_state.U)} -> {np.asarray(mixed_next.U)}",
    )
    check("cached energy matches state", mixed_next.U, quartic(mixed_next.x), tol=0.0)
    check_true(
        "strict fallback state finite",
        bool(jnp.isfinite(mixed_next.x).all())
        and bool(jnp.isfinite(mixed_next.g).all()),
    )

    # The seventh trial is step/64. It rescues a moderately stiff row that
    # fails every formerly evaluated trial through step/32.
    stiff = potential_from(lambda x: 50.0 * (x ** 2).sum(axis=-1))
    stiff0 = jnp.asarray([[1.0]])
    stiff_state = lbfgs_init(stiff0, stiff, memory=3)
    stiff_next = lbfgs_step(stiff_state, stiff, step=1.0, armijo=True)
    check(
        "seventh trial step/64 is evaluated and accepted",
        stiff_next.x, jnp.asarray([[-0.5625]]), tol=1e-15,
    )
    check_true(
        "seventh-trial recovery decreases energy",
        bool(jnp.all(stiff_next.U < stiff_state.U)),
    )

    extreme = potential_from(lambda x: 500.0 * (x ** 2).sum(axis=-1))
    extreme0 = jnp.asarray([[1.0]])
    extreme_state = lbfgs_init(extreme0, extreme, memory=3)
    extreme_next = lbfgs_step(extreme_state, extreme, step=1.0, armijo=True)
    check(
        "extreme exhaustion leaves particle unchanged",
        extreme_next.x, extreme0, tol=0.0,
    )
    check(
        "extreme exhaustion carries next untested step/128",
        extreme_next.step_scale, jnp.asarray([1.0 / 128.0]), tol=0.0,
    )
    check("extreme cached energy remains consistent",
          extreme_next.U, extreme(extreme_next.x), tol=0.0)
    extreme_retry = lbfgs_step(extreme_next, extreme, step=1.0, armijo=True)
    check_true(
        "extreme row progresses on its next iteration",
        bool(jnp.abs(extreme_retry.x - extreme_next.x).max() > 0),
    )
    check_true(
        "extreme retry decreases energy",
        bool(jnp.all(extreme_retry.U < extreme_next.U)),
    )
    check(
        "accepted extreme retry resets Armijo scale",
        extreme_retry.step_scale, jnp.ones(1), tol=0.0,
    )

    # ── 3. loop == manual composition ──
    log("loop == lbfgs_init + lbfgs_step composition")
    state = lbfgs_init(x0m, gmm, memory=6)
    for _ in range(25):
        state = lbfgs_step(state, gmm, step=0.5, armijo=False)
    # tol: scanned loop vs eager steps differ only by XLA fusion rounding
    check("agreement (fusion rounding)", state.x,
          lbfgs(x0m, gmm, step=0.5, iters=25, memory=6, armijo=False), tol=1e-15)

    # ── 4/5. chunk invariance and memory=1 ──
    log("chunk invariance / memory=1")
    check("chunk=4 == chunk=1 (fusion rounding)", lbfgs(x0m, gmm, step=0.5, iters=25, chunk=4),
          lbfgs(x0m, gmm, step=0.5, iters=25, chunk=1), tol=1e-15)
    xf1 = lbfgs(x0, gau, step=1.0, iters=80, memory=1)
    check("memory=1 quadratic convergence", xf1, jnp.broadcast_to(mean, (N, d)), tol=1e-4)

    # ── 6. jit ──
    log("jit")
    state = lbfgs_init(x0m, gmm, memory=6)
    step_jit = eqx.filter_jit(lambda s: lbfgs_step(s, gmm, step=0.5, armijo=False))
    check("filter_jit(lbfgs_step)", step_jit(state).x,
          lbfgs_step(state, gmm, step=0.5, armijo=False).x, tol=1e-12)

    # ── 7. adamw ──
    log("adamw")
    xf = adamw(x0, gau, step=0.05, iters=1500)
    check("quadratic convergence (O(step) residual)", xf,
          jnp.broadcast_to(mean, (N, d)), tol=0.05)
    state = adamw_init(x0m)
    for _ in range(20):
        state = adamw_step(state, gmm, step=0.02)
    check("loop == composition (fusion rounding)", state.x,
          adamw(x0m, gmm, step=0.02, iters=20), tol=1e-15)
    x_wd = adamw(x0, gau, step=0.05, iters=1500, weight_decay=0.2)
    check_true("weight_decay shrinks toward the origin",
               bool(jnp.linalg.norm(x_wd.mean(0)) < jnp.linalg.norm(xf.mean(0)) - 0.1),
               f"|mean| {float(jnp.linalg.norm(x_wd.mean(0))):.3f} < "
               f"{float(jnp.linalg.norm(xf.mean(0))):.3f}")
    check("chunk=4 == chunk=1 (fusion rounding)",
          adamw(x0m, gmm, step=0.02, iters=50, chunk=4),
          adamw(x0m, gmm, step=0.02, iters=50, chunk=1), tol=1e-15)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone optimization tests passed")


if __name__ == "__main__":
    main()

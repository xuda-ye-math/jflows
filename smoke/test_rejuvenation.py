"""Standalone rejuvenation smoke test (jflows only) — run from the repo
root as `~/.envs/jax/bin/python -m smoke.test_rejuvenation`.

For langevin / stochastic_heun / hamiltonian_monte_carlo and their
low-level kernels:

    1. loop == composition: every high-level loop equals the manual
       step chain with the documented key derivation, exactly;
    2. exactness of the adjusted kernels: MALA and HMC reproduce the
       moments of a diagonal Gaussian target within Monte-Carlo error;
       the unadjusted ULA / Heun schemes match within their small-step
       bias;
    3. MH diagnostics from the step aux: acceptance -> 1 as step -> 0;
    4. tamed drift keeps a quartic-potential ULA finite where the
       untamed drift diverges; adjust=True + taming>0 raises;
    5. HMC NaN guard: an absurd step rejects everything and returns the
       (finite) initial particles;
    6. aliases and key-reproducibility.

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

from jflows.potential import Nlog_Gaussian, potential_from  # noqa: E402
from jflows.utils import (  # noqa: E402
    hamiltonian_monte_carlo,
    hmc,
    hmc_step,
    langevin,
    langevin_step,
    rejuvenation,
    stochastic_heun,
    stochastic_heun_step,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_rejuvenation.log")

FAILURES = 0
MEAN = jnp.asarray([0.5, -0.5])
VAR = jnp.asarray([0.8, 1.2])
TARGET = Nlog_Gaussian(MEAN, VAR)


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


def check_moments(name: str, x, mean_tol: float, var_rtol: float) -> None:
    m_err = float(jnp.abs(x.mean(axis=0) - MEAN).max())
    v_err = float(jnp.abs(x.var(axis=0) / VAR - 1.0).max())
    check_true(name, m_err < mean_tol and v_err < var_rtol,
               f"|mean err| {m_err:.3f} (<{mean_tol}), |var rel err| {v_err:.3f} (<{var_rtol})")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_rejuvenation | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    N = 4096
    x0 = jax.random.normal(jax.random.key(1), (N, 2))

    # ── 1. loop == manual step composition (documented key derivation) ──
    log("loop == step composition")
    for name, loop_fn, step_fn, kw_loop, kw_step in [
        ("langevin/ULA", langevin, langevin_step, dict(step=0.05, iters=7), dict(step=0.05)),
        ("langevin/MALA", langevin, langevin_step,
         dict(step=0.05, iters=7, adjust=True), dict(step=0.05, adjust=True)),
        ("stochastic_heun", stochastic_heun, stochastic_heun_step,
         dict(step=0.05, iters=7), dict(step=0.05)),
        ("hmc", hamiltonian_monte_carlo, hmc_step,
         dict(step=0.2, iters=5, burns=7), dict(step=0.2, iters=5)),
    ]:
        x_loop = loop_fn(key, x0, TARGET, **kw_loop)
        x_man = x0
        n_it = kw_loop.get("burns", kw_loop.get("iters"))
        for k in jax.random.split(jax.random.fold_in(key, 0), n_it):
            x_man, _ = step_fn(k, x_man, TARGET, **kw_step)
        # tol: scanned loop vs eager steps differ only by XLA fusion rounding
        check(f"{name}", x_loop, x_man, tol=1e-14)

    # ── 2. stationary-moment checks on the Gaussian target ──
    log("moments vs analytic Gaussian target (N=4096)")
    x = langevin(jax.random.key(2), x0, TARGET, step=0.05, iters=600, adjust=True)
    check_moments("MALA (exact)", x, mean_tol=0.10, var_rtol=0.15)
    x = hamiltonian_monte_carlo(jax.random.key(3), x0, TARGET, step=0.25, iters=10, burns=120)
    check_moments("HMC (exact)", x, mean_tol=0.10, var_rtol=0.15)
    x = langevin(jax.random.key(4), x0, TARGET, step=0.02, iters=1500, adjust=False)
    check_moments("ULA (small-step bias)", x, mean_tol=0.10, var_rtol=0.20)
    x = stochastic_heun(jax.random.key(5), x0, TARGET, step=0.02, iters=1500)
    check_moments("Heun (small-step bias)", x, mean_tol=0.10, var_rtol=0.20)

    # ── 3. acceptance diagnostics from the step aux ──
    log("MH diagnostics")
    _, aux = langevin_step(jax.random.key(6), x0, TARGET, step=1e-6, adjust=True)
    rate = float(aux["accept"].mean())
    check_true("MALA acceptance -> 1 as step -> 0", rate > 0.99, f"rate = {rate:.4f}")
    _, aux = hmc_step(jax.random.key(7), x0, TARGET, step=1e-3, iters=5)
    rate = float(aux["accept"].mean())
    check_true("HMC acceptance -> 1 as step -> 0", rate > 0.99, f"rate = {rate:.4f}")

    # ── 4. tamed drift on a quartic potential ──
    log("tamed drift")
    quartic = potential_from(lambda x: 0.25 * ((x**2).sum(-1)) ** 2)
    x_far = jnp.full((32, 2), 10.0)
    x_plain = langevin(jax.random.key(8), x_far, quartic, step=0.1, iters=200)
    x_tamed = langevin(jax.random.key(8), x_far, quartic, step=0.1, iters=200, taming=1.0)
    check_true("untamed ULA diverges (demonstrates the hazard)",
               not bool(jnp.isfinite(x_plain).all()))
    check_true("tamed ULA stays finite", bool(jnp.isfinite(x_tamed).all()),
               f"max|x| = {float(jnp.abs(x_tamed).max()):.2f}")
    try:
        langevin(key, x0, TARGET, adjust=True, taming=1.0)
        check_true("adjust+taming raises", False)
    except ValueError:
        check_true("adjust+taming raises", True)

    # ── 5. HMC NaN guard ──
    log("HMC NaN guard")
    x_guard, aux = hmc_step(jax.random.key(9), x0, quartic, step=1e3, iters=5)
    check_true("all rejected", bool(~aux["accept"].any()))
    check("particles reverted (finite)", x_guard, x0, tol=0)

    # ── 6. aliases and reproducibility ──
    log("aliases / reproducibility")
    check_true("rejuvenation is langevin", rejuvenation is langevin)
    check_true("hmc is hamiltonian_monte_carlo", hmc is hamiltonian_monte_carlo)
    check("same key -> same trajectory",
          langevin(jax.random.key(10), x0, TARGET, step=0.05, iters=20),
          langevin(jax.random.key(10), x0, TARGET, step=0.05, iters=20), tol=0)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone rejuvenation tests passed")


if __name__ == "__main__":
    main()

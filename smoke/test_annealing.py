"""Standalone annealing smoke test (jflows only) — run after installation
from the repo root as `python -m smoke.test_annealing`.

For sequential_monte_carlo / annealed_importance_sampling (type='F'/'G'):

    1. SMC transports a wide Gaussian onto a two-mode mixture: moments
       and mode proportions match the analytic target within Monte-Carlo
       error; the per-level ESS diagnostic lies in (0, 1] with a
       well-spaced ladder;
    2. loop == composition: SMC equals the manual
       reweight -> resample -> langevin chain with the documented key
       derivation;
    3. taming pass-through keeps the unadjusted rejuvenation finite on
       the same target;
    4. AIS with the identity flow (RealNVP.zeros()) reduces to a
       geometric source->target ladder and reproduces the target
       moments; the 'F'/'G' types agree given G = F.inv;
    5. reproducibility from the key.

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

from jflows.flow import RealNVP  # noqa: E402
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture  # noqa: E402
from jflows.utils import (  # noqa: E402
    ais,
    annealed_importance_sampling,
    compute_ESS_log,
    langevin,
    resample,
    sequential_monte_carlo,
    smc,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_annealing.log")

FAILURES = 0

N = 40000  # particles pushed through SMC / AIS (large enough for tight moment tolerances)

WEIGHTS = jnp.asarray([0.35, 0.65])
MEANS = jnp.asarray([[-2.5, 0.0], [2.5, 1.0]])
VARS = jnp.asarray([[0.4, 0.4], [0.5, 0.3]])
TARGET = Nlog_Gaussian_Mixture(WEIGHTS, MEANS, VARS)
SOURCE = Nlog_Gaussian([0.0, 0.0], [9.0, 9.0])  # broad enough to cover both target modes
MEAN_TRUE = (WEIGHTS[:, None] * MEANS).sum(0)
VAR_TRUE = (WEIGHTS[:, None] * (VARS + MEANS**2)).sum(0) - MEAN_TRUE**2


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


def check_target_match(name: str, x, m_tol: float = 0.15) -> None:
    m_err = float(jnp.abs(x.mean(0) - MEAN_TRUE).max())
    v_err = float(jnp.abs(x.var(0) / VAR_TRUE - 1.0).max())
    d2 = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)
    frac = float((jnp.argmin(d2, axis=1) == 1).mean())
    ok = m_err < m_tol and v_err < 0.25 and abs(frac - float(WEIGHTS[1])) < 0.06
    check_true(name, ok,
               f"|mean err| {m_err:.3f}, |var rel err| {v_err:.3f}, "
               f"mode-2 fraction {frac:.3f} (true {float(WEIGHTS[1]):.2f})")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_annealing | jax {jax.__version__} | {jax.default_backend()}")
    x0 = SOURCE.samples(jax.random.key(1), N)

    # ── 1. SMC onto the two-mode mixture ──
    log("sequential_monte_carlo (ladder=12)")
    x, ess = sequential_monte_carlo(jax.random.key(2), x0, SOURCE, TARGET,
                                    ladder=12, step=0.05, iters=120)
    check_target_match("moments & mode proportions", x)
    check_true("ess shape (M,), values in (0, 1]",
               ess.shape == (12,) and bool(((ess > 0) & (ess <= 1.0)).all()),
               f"ess = {np.asarray(ess).round(3)}")
    check_true("well-spaced ladder (all ess > 0.1)", bool((ess > 0.1).all()))

    # ── 2. loop == manual composition ──
    log("SMC == reweight -> resample -> langevin composition")
    key = jax.random.key(3)
    M = 3
    x_man = x0
    for k in range(1, M + 1):
        c = k / M
        u_k = (1.0 - c) * SOURCE + c * TARGET
        log_w = (SOURCE(x_man) - TARGET(x_man)) / M
        w = jnp.exp(log_w - log_w.max())
        key_r, key_l = jax.random.split(jax.random.fold_in(key, k))
        x_man = resample(key_r, x_man, w)
        x_man = langevin(key_l, x_man, u_k, step=0.02, iters=10)
    x_loop, _ = sequential_monte_carlo(key, x0, SOURCE, TARGET,
                                       ladder=M, step=0.02, iters=10)
    check("agreement (fusion rounding)", x_loop, x_man, tol=1e-13)

    # ── 3. taming pass-through (mild taming: active only on outlier drifts) ──
    log("taming pass-through")
    x_tamed, ess_t = sequential_monte_carlo(jax.random.key(2), x0, SOURCE, TARGET,
                                            ladder=16, step=0.05, iters=300, taming=0.05, adjust=False)
    check_true("finite", bool(jnp.isfinite(x_tamed).all()))
    # taming caps the drift, biasing the finite-step chain -> wider mean tolerance
    check_target_match("tamed moments & proportions", x_tamed, m_tol=0.2)

    # ── 4. AIS with the identity flow ──
    log("annealed_importance_sampling (identity flow, ladder=6)")
    flow_id = RealNVP(jax.random.key(5), dimension=2, transforms=2).zeros()
    y = annealed_importance_sampling(jax.random.key(6), x0, SOURCE, TARGET, flow_id, type="F",
                                     ladder=10, step=0.05, iters=150)
    check_target_match("AIS (type=F) moments & mode proportions", y)
    y_g = annealed_importance_sampling(jax.random.key(6), x0, SOURCE, TARGET, flow_id, type="G",
                                     ladder=10, step=0.05, iters=150)
    check("F/G agree for the identity flow (same key)", y_g, y, tol=1e-10)

    # weights feed the standard diagnostics
    log_w = (SOURCE(y) - TARGET(y))
    check_true("post-AIS ESS diagnostic finite",
               bool(jnp.isfinite(compute_ESS_log(log_w))))

    log("aliases")
    check_true("smc is sequential_monte_carlo", smc is sequential_monte_carlo)
    check_true("ais is annealed_importance_sampling", ais is annealed_importance_sampling)

    # ── 5. reproducibility ──
    log("reproducibility")
    x_a, _ = sequential_monte_carlo(jax.random.key(7), x0, SOURCE, TARGET, ladder=3, iters=10)
    x_b, _ = sequential_monte_carlo(jax.random.key(7), x0, SOURCE, TARGET, ladder=3, iters=10)
    check("same key -> same particles", x_a, x_b, tol=0)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone annealing tests passed")


if __name__ == "__main__":
    main()

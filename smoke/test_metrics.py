"""Standalone metrics smoke test (jflows only) — run from the repo root
as `conda activate jflows && PYTHONPATH=/mnt/projects/jflows python -m smoke.test_metrics`.

For compute_ESS / compute_ESS_log / importance_weights_* / resample:

    1. analytic ESS: uniform weights -> 1, one-hot -> 1/N, iid two-point
       weights against the closed form;
    2. invariances: ESS(c * w) == ESS(w); log/linear agreement on random
       weights; log version stays finite and correct on extreme
       log-ranges where the linear form overflows;
    3. importance weights: identity flow gives the analytic energy
       difference (and log_w == 0, ESS == 1 when target == source);
       F/G duality; linear == max-shifted exp of log; chunk-invariance;
       type dispatch; end-to-end feed into compute_ESS_log;
    4. coverage (k-NN, Naeem et al. 2020): perfect / far / half-collapsed
       candidates against a two-cluster reference, k-monotonicity,
       exact agreement with a brute-force reimplementation, jit;
    5. resample: reproducible from the key, default N, one-hot weight
       collapses to a single point, empirical frequencies match the
       weights within Monte-Carlo error, jit-compatible.

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

from jflows.flow import NSF  # noqa: E402
from jflows.potential import Nlog_Gaussian  # noqa: E402
from jflows.utils import (  # noqa: E402
    compute_ESS,
    compute_ESS_log,
    coverage,
    importance_weights,
    importance_weights_log,
    resample,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_metrics.log")

FAILURES = 0
NSAMP = 200_000


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
    log(f"START test_metrics | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    N = 512

    # ── analytic ESS values ──
    log("compute_ESS analytics")
    check("uniform weights -> 1", compute_ESS(jnp.full(N, 3.7)), jnp.asarray(1.0), tol=1e-14)
    onehot = jnp.zeros(N).at[7].set(2.0)
    check("one-hot -> 1/N", compute_ESS(onehot), jnp.asarray(1.0 / N), tol=1e-16)
    # half the weights a, half b: ESS = (a+b)^2 / (2 (a^2+b^2))
    a_, b_ = 1.0, 3.0
    two_point = jnp.where(jnp.arange(N) < N // 2, a_, b_)
    check("two-point closed form", compute_ESS(two_point),
          jnp.asarray((a_ + b_) ** 2 / (2 * (a_**2 + b_**2))), tol=1e-14)

    # ── invariances and log/linear agreement ──
    log("invariances")
    w = jax.random.uniform(key, (N,)) + 1e-3
    check("scale invariance", compute_ESS(17.3 * w), compute_ESS(w), tol=1e-13)
    check("log == linear", compute_ESS_log(jnp.log(w)), compute_ESS(w), tol=1e-13)
    logw_extreme = jax.random.normal(key, (N,)) * 400.0  # exp overflows float64
    ess_ext = compute_ESS_log(logw_extreme)
    check("extreme logs finite & shift-invariant", ess_ext,
          compute_ESS(jnp.exp(logw_extreme - logw_extreme.max())), tol=1e-13)
    check_true("extreme logs in (0, 1]", bool((ess_ext > 0) & (ess_ext <= 1.0)),
               f"ESS = {float(ess_ext):.3e}")

    # ── importance weights through a flow ──
    log("importance weights")
    box = jnp.asarray([2.0, 2.0, 2.0])
    nsf = NSF(jax.random.key(7), -box, box, bins=8, transforms=2,
              hidden_features=(32, 16))
    source = Nlog_Gaussian([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    target = Nlog_Gaussian([0.5, -0.5, 0.0], [0.7, 0.7, 0.7])
    xw = jax.random.uniform(jax.random.key(8), (64, 3), minval=-1.9, maxval=1.9)

    flow_z = nsf.zeros()  # identity flow
    check("identity flow: log_w == source - target",
          importance_weights_log(xw, source, target, flow_z, type="F"),
          source(xw) - target(xw), tol=1e-12)
    lw_same = importance_weights_log(xw, source, source, flow_z, type="F")
    check("target == source: log_w == 0", lw_same, jnp.zeros(xw.shape[0]), tol=1e-12)
    check("target == source: ESS == 1", compute_ESS_log(lw_same), jnp.asarray(1.0), tol=1e-12)

    lw = importance_weights_log(xw, source, target, nsf, type="F")
    T = nsf.t()  # definition vs the core layer, both types
    y_c, l_c = T.call_and_ladj(xw)
    check("type=F == core", lw, -target(y_c) + source(xw) + l_c, tol=0)
    lw_g = importance_weights_log(xw, source, target, nsf, type="G")
    y_g, l_g = T.inv.call_and_ladj(xw)
    check("type=G == core", lw_g, -target(y_g) + source(xw) + l_g, tol=0)
    check("linear == max-shifted exp", importance_weights(xw, source, target, nsf, type="F"),
          jnp.exp(lw - lw.max()), tol=1e-14)
    check("linear type=G", importance_weights(xw, source, target, nsf, type="G"),
          jnp.exp(lw_g - lw_g.max()), tol=1e-14)
    check("chunk invariance", importance_weights_log(xw, source, target, nsf, type="F", chunk=3),
          lw, tol=1e-12)
    try:
        importance_weights_log(xw, source, target, nsf, type="Z")
        check_true("invalid type raises", False)
    except ValueError:
        check_true("invalid type raises", True)
    ess_flow = compute_ESS_log(lw)
    check_true("end-to-end ESS in (0, 1]",
               bool((ess_flow > 0) & (ess_flow <= 1.0)), f"ESS = {float(ess_flow):.4f}")

    # ── coverage ──
    log("coverage")
    kc1, kc2 = jax.random.split(jax.random.key(9))
    ref = jnp.concatenate([  # two well-separated clusters, 256 points each
        jax.random.normal(kc1, (256, 2)) * 0.3 - 10.0,
        jax.random.normal(kc2, (256, 2)) * 0.3 + 10.0,
    ])
    check("y == x -> 1", coverage(ref, ref), jnp.asarray(1.0), tol=0)
    check("far candidates -> 0", coverage(jnp.full((128, 2), 1e6), ref),
          jnp.asarray(0.0), tol=0)
    collapsed = ref[:256] + 0.01  # candidates cover only the first cluster
    check("mode collapse -> 1/2", coverage(collapsed, ref), jnp.asarray(0.5), tol=1e-12)
    xr = jax.random.normal(jax.random.key(10), (128, 2))
    yc = jax.random.normal(jax.random.key(11), (64, 2))
    c1, c5, c10 = (float(coverage(yc, xr, k=kk)) for kk in (1, 5, 10))
    check_true("k-monotone", c1 <= c5 <= c10, f"k=1: {c1:.3f} <= k=5: {c5:.3f} <= k=10: {c10:.3f}")
    check("jit-compatible", jax.jit(lambda y, x: coverage(y, x))(yc, xr),
          coverage(yc, xr), tol=0)

    # exact agreement with an independent brute-force reimplementation
    xs_ = np.asarray(jax.random.normal(jax.random.key(12), (32, 3)))
    ys_ = np.asarray(jax.random.normal(jax.random.key(13), (48, 3)))
    k_ = 4
    covered = 0
    for i in range(32):
        d_i = sorted(float(np.linalg.norm(xs_[i] - xs_[j])) for j in range(32) if j != i)
        r_i = d_i[k_ - 1]
        covered += int(any(float(np.linalg.norm(xs_[i] - ys_[n])) < r_i for n in range(48)))
    check("chunk invariance (chunk=3)", coverage(yc, xr, k=5, chunk=3),
          coverage(yc, xr, k=5), tol=1e-12)
    check("brute-force agreement", coverage(jnp.asarray(ys_), jnp.asarray(xs_), k=k_),
          np.asarray(covered / 32), tol=1e-12)

    # ── resample ──
    log("resample")
    kx, kr = jax.random.split(jax.random.key(1))
    samples = jax.random.normal(kx, (N, 3))
    weights = jax.random.uniform(kr, (N,))
    r1 = resample(jax.random.key(2), samples, weights)
    r2 = resample(jax.random.key(2), samples, weights)
    check("reproducible from key", r1, r2, tol=0)
    check_true("default N", r1.shape == (N, 3), f"{r1.shape}")
    check_true("custom N", resample(jax.random.key(3), samples, weights, N=37).shape == (37, 3))
    r_onehot = resample(jax.random.key(4), samples, onehot, N=100)
    check("one-hot collapses to samples[7]", r_onehot,
          jnp.broadcast_to(samples[7], (100, 3)), tol=0)
    check("jit-compatible", jax.jit(lambda k: resample(k, samples, weights))(jax.random.key(2)),
          r1, tol=0)

    # empirical resampling frequencies vs normalized weights (M small, N large)
    M = 8
    s_small = jnp.arange(M, dtype=jnp.float64)[:, None]
    w_small = jnp.asarray([0.05, 0.1, 0.2, 0.05, 0.25, 0.15, 0.1, 0.1]) * 4.0  # unnormalized
    r = resample(jax.random.key(5), s_small, w_small, N=NSAMP)
    freq = jnp.bincount(r[:, 0].astype(int), length=M) / NSAMP
    check("empirical frequencies", freq, w_small / w_small.sum(), tol=5e-3)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone metrics tests passed")


if __name__ == "__main__":
    main()

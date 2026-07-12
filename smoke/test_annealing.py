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

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.flow import CNF, OTFlow, RealNVP  # noqa: E402
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture  # noqa: E402
from jflows.utils import (  # noqa: E402
    ais,
    annealed_importance_sampling,
    compute_ESS_log,
    importance_weights_log,
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


def as_float32(tree):
    """Cast array parameters only; keep static architecture fields intact."""
    return jax.tree.map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x,
        tree,
    )


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
    log("annealed_importance_sampling (identity flow, ladder=10)")
    flow_id = RealNVP(jax.random.key(5), dimension=2, transforms=2).zeros()
    y, initial_log_weight = annealed_importance_sampling(
        jax.random.key(6), x0, SOURCE, TARGET, flow_id, type="F",
        ladder=10, step=0.05, iters=150,
        return_initial_log_weights=True,
    )
    check_target_match("AIS (type=F) moments & mode proportions", y)
    expected_log_weight = importance_weights_log(
        x0, SOURCE, TARGET, flow_id, type="F"
    )
    check(
        "optional pre-AIS full log weights match proposal weights",
        initial_log_weight,
        expected_log_weight,
        tol=1e-13,
    )
    y_g = annealed_importance_sampling(jax.random.key(6), x0, SOURCE, TARGET, flow_id, type="G",
                                     ladder=10, step=0.05, iters=150)
    check("F/G agree for the identity flow (same key)", y_g, y, tol=1e-10)
    check_true(
        "pre-AIS proposal ESS finite",
        bool(jnp.isfinite(compute_ESS_log(initial_log_weight))),
    )

    # Nonidentity direction/sign regression: requesting the auxiliary must not
    # perturb the AIS trajectory, and both F/G conventions must expose the
    # same full proposal weights as the public importance helper. A ladder
    # above one catches accidental return of the incremental log weight z/M.
    log("pre-AIS weights for a nonidentity flow (F/G, ladder=3)")
    flow_probe = RealNVP(
        jax.random.key(51), dimension=2, transforms=2
    )
    probe_x = x0[:512]
    for offset, flow_type in enumerate(("F", "G")):
        probe_key = jax.random.fold_in(jax.random.key(52), offset)
        y_default = annealed_importance_sampling(
            probe_key,
            probe_x,
            SOURCE,
            TARGET,
            flow_probe,
            type=flow_type,
            ladder=3,
            step=0.01,
            iters=2,
        )
        y_optional, proposal_log_weight = annealed_importance_sampling(
            probe_key,
            probe_x,
            SOURCE,
            TARGET,
            flow_probe,
            type=flow_type,
            ladder=3,
            step=0.01,
            iters=2,
            return_initial_log_weights=True,
        )
        expected = importance_weights_log(
            probe_x, SOURCE, TARGET, flow_probe, type=flow_type
        )
        check(
            f"{flow_type}: optional return preserves samples",
            y_optional,
            y_default,
            tol=0.0,
        )
        check(
            f"{flow_type}: full proposal log weights match helper",
            proposal_log_weight,
            expected,
            tol=1e-12,
        )

    # Fixed-step continuous maps are only numerically invertible. Their
    # level-one correction must therefore retain the ORIGINAL source sample
    # and direct push Jacobian; a push -> inverse -> push round trip gives a
    # different proposal weight. Exercise both directions in float32, an
    # uneven chunk split, and a ladder above one.
    log("direct first-level weights for finite-step flows (float32)")
    source32 = Nlog_Gaussian(
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    )
    target32 = Nlog_Gaussian(
        jnp.asarray([0.5, -0.3], dtype=jnp.float32),
        jnp.asarray([0.7, 1.4], dtype=jnp.float32),
    )
    probe32 = source32.samples(jax.random.key(61), 257)
    otflow = as_float32(
        OTFlow(
            jax.random.key(62), dimension=2, hidden=16, layer=2,
            rank=3, nt=4,
        )
    )
    for offset, flow_type in enumerate(("F", "G")):
        probe_key = jax.random.fold_in(jax.random.key(63), offset)
        y_default = annealed_importance_sampling(
            probe_key, probe32, source32, target32, otflow, flow_type,
            ladder=3, iters=0, chunk=3,
        )
        y_optional, log_weight = annealed_importance_sampling(
            probe_key, probe32, source32, target32, otflow, flow_type,
            ladder=3, iters=0, chunk=3, return_initial_log_weights=True,
        )
        expected = importance_weights_log(
            probe32, source32, target32, otflow, flow_type, chunk=3
        )
        check(
            f"OTFlow {flow_type}: optional return preserves samples",
            y_optional, y_default, tol=0.0,
        )
        check(
            f"OTFlow {flow_type}: direct full proposal weights",
            log_weight, expected, tol=2e-6,
        )
        check_true(
            f"OTFlow {flow_type}: weights and ESS finite",
            bool(jnp.isfinite(log_weight).all())
            and bool(jnp.isfinite(compute_ESS_log(log_weight))),
        )

    # A stochastic CNF must use the level-one folded trace key in both the
    # sampler and the independent importance-weight oracle.
    cnf = as_float32(
        CNF(
            jax.random.key(64), dimension=2, frequency=2, nt=4,
            exact=False, hidden_features=(12, 12),
        )
    )
    trace_base = jax.random.key(65)
    trace_level_1 = jax.random.fold_in(trace_base, 1)
    for offset, flow_type in enumerate(("F", "G")):
        probe_key = jax.random.fold_in(jax.random.key(66), offset)
        y_default = annealed_importance_sampling(
            probe_key, probe32, source32, target32, cnf, flow_type,
            ladder=3, iters=0, chunk=3, trace_key=trace_base,
        )
        y_optional, log_weight = annealed_importance_sampling(
            probe_key, probe32, source32, target32, cnf, flow_type,
            ladder=3, iters=0, chunk=3, trace_key=trace_base,
            return_initial_log_weights=True,
        )
        expected = importance_weights_log(
            probe32, source32, target32, cnf, flow_type, chunk=3,
            trace_key=trace_level_1,
        )
        check(
            f"approx CNF {flow_type}: optional return preserves samples",
            y_optional, y_default, tol=0.0,
        )
        check(
            f"approx CNF {flow_type}: level-one trace-key weights",
            log_weight, expected, tol=2e-6,
        )

    # Manual three-level composition gates the later-level refresh: level one
    # uses direct weights, while levels two and three recompute the latent of
    # the already resampled/rejuvenated particles.
    manual_key = jax.random.key(67)
    y_manual, ladj = otflow.call_and_ladj(probe32)
    initial_weight = -target32(y_manual) + source32(probe32) + ladj
    for k in range(1, 4):
        if k == 1:
            full_weight = initial_weight
        else:
            parts = []
            for yc in jnp.array_split(y_manual, 3, axis=0):
                xc = otflow.inv(yc)
                _, ladj = otflow.call_and_ladj(xc)
                parts.append(-target32(yc) + source32(xc) + ladj)
            full_weight = jnp.concatenate(parts, axis=0)
        key_r, key_l = jax.random.split(jax.random.fold_in(manual_key, k))
        y_manual = resample(key_r, y_manual, jnp.exp(full_weight / 3 - jnp.max(full_weight / 3)))
        y_manual = langevin(key_l, y_manual, target32, iters=0, chunk=3)
    y_public = annealed_importance_sampling(
        manual_key, probe32, source32, target32, otflow, "F",
        ladder=3, iters=0, chunk=3,
    )
    check("OTFlow three-level manual composition", y_public, y_manual, tol=0.0)

    log("aliases")
    check_true("smc is sequential_monte_carlo", smc is sequential_monte_carlo)
    check_true("ais is annealed_importance_sampling", ais is annealed_importance_sampling)
    try:
        annealed_importance_sampling(
            jax.random.key(6), x0[:4], SOURCE, TARGET, flow_id, type="F",
            return_initial_log_weights=1,
        )
        check_true("non-Boolean optional-return flag rejected", False)
    except TypeError:
        check_true("non-Boolean optional-return flag rejected", True)

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

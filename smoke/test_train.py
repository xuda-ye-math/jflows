"""Standalone training-driver smoke test (jflows only) — run after
installation from the repo root as `python -m smoke.test_train`.

For train_reverse_KL_F / train_forward_KL_G / Monitor:

    1. backend: the test runs directly on the GPU;
    2. train_reverse_KL_F: returns (Flow, ess[steps]); the mean loss on
       the fixed set decreases; the batch-ESS history lies in (0, 1]
       and rises; the final full-set flow-IS ESS clears a sane floor;
       deterministic (two identical calls agree bit-for-bit); the
       mc_adjust=True (MALA) path trains to finite parameters;
    3. train_forward_KL_G: same contract with the target batches
       manufactured internally by single-level AIS through the CURRENT
       flow;
    4. train_forward_KLX_G: the X-regularized forward KL at a general
       coeff_lambda — same contract, deterministic;
    5. train_forward_KLXX_G: the full mixture loss at general
       (coeff_lambda, coeff_alpha, coeff_beta) — quench-and-temper pool,
       per-step hat_mu freshening, detached pushforward — same contract;
    6. Monitor: every < 1 rejected; reports exactly steps // every
       lines with the requested prefix from inside the compiled scan;
       attaching a monitor does not change the trained flow.

Default JAX backend (GPU); GPU memory preallocation is disabled.
Exits nonzero on any failure.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.flow import NSF  # noqa: E402
from jflows.loss import forward_KL_G, reverse_KL_F  # noqa: E402
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture  # noqa: E402
from jflows.train import (  # noqa: E402
    Monitor,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
    train_reverse_KL_F,
)
from jflows.utils import compute_ESS, importance_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_train.log")

FAILURES = 0

SIGMA = 2.0
N_VALID, N_BATCH, STEPS, LR = 8000, 1000, 200, 2e-3
MC_STEP, MC_ITERS, LADDER = 1e-3, 20, 1
N_POOL, MELT, OPT_STEP, OPT_ITERS = 2000, 2.0, 0.5, 100
COEFF_LAMBDA, COEFF_ALPHA, COEFF_BETA = 0.7, 0.8, 0.3


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


def max_param_delta(flow_a, flow_b) -> float:
    import equinox as eqx

    la = [x for x in jax.tree.leaves(flow_a) if eqx.is_inexact_array(x)]
    lb = [x for x in jax.tree.leaves(flow_b) if eqx.is_inexact_array(x)]
    return max(float(jnp.abs(a - b).max()) for a, b in zip(la, lb))


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_train | jax {jax.__version__} | {jax.default_backend()}")
    check_true("running on GPU", jax.default_backend() == "gpu",
               f"backend = {jax.default_backend()}")

    u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2])
    u1 = Nlog_Gaussian_Mixture(
        weights=[1.0, 1.0, 1.0],
        mean=[[0.0, 2.4], [-2.2, -1.4], [2.2, -1.4]],
        variance=[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]],
    )
    x_valid = u0.samples(jax.random.key(2), N_VALID)

    def new_flow(key):
        return NSF(key, a=[-5.0, -5.0], b=[5.0, 5.0], bins=8,
                   transforms=2, hidden_features=(32, 32)).zeros()

    # ── train_reverse_KL_F ──
    log("train_reverse_KL_F")
    f0 = new_flow(jax.random.key(0))
    loss_before = float(reverse_KL_F(x_valid, u1, f0).mean())
    flow_F, ess_F = train_reverse_KL_F(x_valid, u0, u1, f0,
                                       n_batch=N_BATCH, steps=STEPS, lr=LR,
                                       mc_step=MC_STEP, mc_iters=MC_ITERS)
    loss_after = float(reverse_KL_F(x_valid, u1, flow_F).mean())
    check_true("ess history shape", ess_F.shape == (STEPS,), f"{ess_F.shape}")
    check_true("ess history in (0, 1]",
               bool(jnp.all((ess_F > 0) & (ess_F <= 1.0))),
               f"min {float(ess_F.min()):.3f} max {float(ess_F.max()):.3f}")
    check_true("ess history rises", float(ess_F[-1]) > float(ess_F[0]),
               f"{float(ess_F[0]):.3f} -> {float(ess_F[-1]):.3f}")
    check_true("loss decreases", loss_after < loss_before,
               f"{loss_before:+.4f} -> {loss_after:+.4f}")
    ess_full = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_F, type="F")))
    check_true("final full-set ESS > 0.5", ess_full > 0.5, f"ESS = {ess_full:.4f}")
    flow_F2, _ = train_reverse_KL_F(x_valid, u0, u1, f0,
                                    n_batch=N_BATCH, steps=STEPS, lr=LR,
                                    mc_step=MC_STEP, mc_iters=MC_ITERS)
    check("deterministic (same call twice)", max_param_delta(flow_F, flow_F2), 0.0, tol=0)
    flow_A, _ = train_reverse_KL_F(x_valid, u0, u1, f0,
                                   n_batch=N_BATCH, steps=20, lr=LR,
                                   mc_step=MC_STEP, mc_iters=MC_ITERS, mc_adjust=True)
    check_true("mc_adjust=True (MALA) trains finite",
               bool(all(jnp.isfinite(x).all() for x in jax.tree.leaves(flow_A)
                        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating))))

    # ── train_forward_KL_G ──
    log("train_forward_KL_G")
    g0 = new_flow(jax.random.key(1))
    loss_before = float(forward_KL_G(x_valid, u0, g0).mean())
    flow_G, ess_G = train_forward_KL_G(x_valid, u0, u1, g0,
                                       n_batch=N_BATCH, steps=STEPS, lr=LR,
                                       ladder=LADDER, mc_step=MC_STEP, mc_iters=MC_ITERS)
    check_true("ess history shape", ess_G.shape == (STEPS,), f"{ess_G.shape}")
    check_true("ess history rises", float(ess_G[-1]) > float(ess_G[0]),
               f"{float(ess_G[0]):.3f} -> {float(ess_G[-1]):.3f}")
    ess_full_G = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_G, type="G")))
    check_true("final full-set ESS > 0.5", ess_full_G > 0.5, f"ESS = {ess_full_G:.4f}")

    # ── train_forward_KLX_G ──
    log(f"train_forward_KLX_G (coeff_lambda = {COEFF_LAMBDA})")
    h0 = new_flow(jax.random.key(3))
    flow_X, ess_X = train_forward_KLX_G(x_valid, u0, u1, h0,
                                        n_batch=N_BATCH, steps=STEPS, lr=LR,
                                        ladder=LADDER, mc_step=MC_STEP,
                                        mc_iters=MC_ITERS,
                                        coeff_lambda=COEFF_LAMBDA)
    check_true("ess history shape", ess_X.shape == (STEPS,), f"{ess_X.shape}")
    check_true("ess history in (0, 1]",
               bool(jnp.all((ess_X > 0) & (ess_X <= 1.0))),
               f"min {float(ess_X.min()):.3f} max {float(ess_X.max()):.3f}")
    check_true("ess history rises", float(ess_X[-1]) > float(ess_X[0]),
               f"{float(ess_X[0]):.3f} -> {float(ess_X[-1]):.3f}")
    ess_full_X = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_X, type="G")))
    check_true("final full-set ESS > 0.5", ess_full_X > 0.5, f"ESS = {ess_full_X:.4f}")
    flow_X2, _ = train_forward_KLX_G(x_valid, u0, u1, h0,
                                     n_batch=N_BATCH, steps=STEPS, lr=LR,
                                     ladder=LADDER, mc_step=MC_STEP,
                                     mc_iters=MC_ITERS,
                                     coeff_lambda=COEFF_LAMBDA)
    check("deterministic (same call twice)", max_param_delta(flow_X, flow_X2), 0.0, tol=0)

    # ── train_forward_KLXX_G ──
    log(f"train_forward_KLXX_G (lambda {COEFF_LAMBDA}, alpha {COEFF_ALPHA}, "
        f"beta {COEFF_BETA})")
    flow_XX, ess_XX = train_forward_KLXX_G(
        x_valid, u0, u1, h0, n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS,
        lr=LR, ladder=LADDER, melt=MELT, opt_step=OPT_STEP, opt_iters=OPT_ITERS,
        mc_step=MC_STEP, mc_iters=MC_ITERS, coeff_lambda=COEFF_LAMBDA,
        coeff_alpha=COEFF_ALPHA, coeff_beta=COEFF_BETA)
    check_true("ess history shape", ess_XX.shape == (STEPS,), f"{ess_XX.shape}")
    check_true("ess history in (0, 1]",
               bool(jnp.all((ess_XX > 0) & (ess_XX <= 1.0))),
               f"min {float(ess_XX.min()):.3f} max {float(ess_XX.max()):.3f}")
    check_true("ess history rises", float(ess_XX[-1]) > float(ess_XX[0]),
               f"{float(ess_XX[0]):.3f} -> {float(ess_XX[-1]):.3f}")
    ess_full_XX = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_XX, type="G")))
    check_true("final full-set ESS > 0.5", ess_full_XX > 0.5, f"ESS = {ess_full_XX:.4f}")
    flow_XX2, _ = train_forward_KLXX_G(
        x_valid, u0, u1, h0, n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS,
        lr=LR, ladder=LADDER, melt=MELT, opt_step=OPT_STEP, opt_iters=OPT_ITERS,
        mc_step=MC_STEP, mc_iters=MC_ITERS, coeff_lambda=COEFF_LAMBDA,
        coeff_alpha=COEFF_ALPHA, coeff_beta=COEFF_BETA)
    check("deterministic (same call twice)", max_param_delta(flow_XX, flow_XX2), 0.0, tol=0)

    # ── Monitor ──
    log("Monitor")
    try:
        Monitor(0)
        check_true("Monitor rejects every=0", False)
    except ValueError:
        check_true("Monitor rejects every=0", True)
    lines: list[str] = []
    flow_M, _ = train_reverse_KL_F(x_valid, u0, u1, f0,
                                   n_batch=N_BATCH, steps=20, lr=LR,
                                   mc_step=MC_STEP, mc_iters=MC_ITERS,
                                   monitor=Monitor(5, "[mon] ", lines.append))
    jax.effects_barrier()
    check_true("reports steps // every lines", len(lines) == 4, f"{len(lines)} lines")
    check_true("prefix and fields present",
               all(ln.startswith("[mon] step") and "loss =" in ln and "ESS =" in ln
                   for ln in lines),
               lines[0] if lines else "(no lines)")
    flow_M0, _ = train_reverse_KL_F(x_valid, u0, u1, f0,
                                    n_batch=N_BATCH, steps=20, lr=LR,
                                    mc_step=MC_STEP, mc_iters=MC_ITERS)
    check("monitor does not change training", max_param_delta(flow_M, flow_M0), 0.0, tol=0)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone training-driver tests passed")


if __name__ == "__main__":
    main()

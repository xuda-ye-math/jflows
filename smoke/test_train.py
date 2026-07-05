"""Standalone training-driver smoke test (jflows only) — run from the
repo root as `~/.envs/jax/bin/python -m smoke.test_train`.

For train_reverse_KL / train_forward_KL / Monitor:

    1. backend: the test runs directly on the GPU;
    2. train_reverse_KL: returns (Flow, ess[steps]); the mean loss on
       the fixed set decreases; the batch-ESS history lies in (0, 1]
       and rises; the final full-set flow-IS ESS clears a sane floor;
       deterministic (two identical calls agree bit-for-bit);
    3. train_forward_KL: same contract with the target batches
       manufactured internally by single-rung AIS through the CURRENT
       flow;
    4. type dispatch: both drivers reject type not in {'F', 'G'};
    5. Monitor: every < 1 rejected; reports exactly steps // every
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
from jflows.loss import forward_KL, reverse_KL  # noqa: E402
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture  # noqa: E402
from jflows.train import Monitor, train_forward_KL, train_reverse_KL  # noqa: E402
from jflows.utils import compute_ESS, importance_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_train.log")

FAILURES = 0

SIGMA = 2.0
N_VALID, N_BATCH, STEPS, LR = 8000, 1000, 200, 2e-3
MC_STEP, MC_ITERS, LADDER = 1e-3, 20, 1


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

    # ── train_reverse_KL ──
    log("train_reverse_KL")
    f0 = new_flow(jax.random.key(0))
    loss_before = float(reverse_KL(x_valid, u1, f0, type="F").mean())
    flow_F, ess_F = train_reverse_KL(x_valid, u0, u1, f0, type="F",
                                     n_batch=N_BATCH, steps=STEPS, lr=LR,
                                     mc_step=MC_STEP, mc_iters=MC_ITERS)
    loss_after = float(reverse_KL(x_valid, u1, flow_F, type="F").mean())
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
    flow_F2, _ = train_reverse_KL(x_valid, u0, u1, f0, type="F",
                                  n_batch=N_BATCH, steps=STEPS, lr=LR,
                                  mc_step=MC_STEP, mc_iters=MC_ITERS)
    check("deterministic (same call twice)", max_param_delta(flow_F, flow_F2), 0.0, tol=0)

    # ── train_forward_KL ──
    log("train_forward_KL")
    g0 = new_flow(jax.random.key(1))
    loss_before = float(forward_KL(x_valid, u0, g0, type="G").mean())
    flow_G, ess_G = train_forward_KL(x_valid, u0, u1, g0, type="G",
                                     n_batch=N_BATCH, steps=STEPS, lr=LR,
                                     ladder=LADDER, mc_step=MC_STEP, mc_iters=MC_ITERS)
    check_true("ess history shape", ess_G.shape == (STEPS,), f"{ess_G.shape}")
    check_true("ess history rises", float(ess_G[-1]) > float(ess_G[0]),
               f"{float(ess_G[0]):.3f} -> {float(ess_G[-1]):.3f}")
    ess_full_G = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_G, type="G")))
    check_true("final full-set ESS > 0.5", ess_full_G > 0.5, f"ESS = {ess_full_G:.4f}")

    # ── type dispatch ──
    log("type dispatch")
    for fn, name in ((train_reverse_KL, "train_reverse_KL"),
                     (train_forward_KL, "train_forward_KL")):
        try:
            if fn is train_reverse_KL:
                fn(x_valid, u0, u1, f0, type="X", n_batch=N_BATCH, steps=1, lr=LR,
                   mc_step=MC_STEP, mc_iters=1)
            else:
                fn(x_valid, u0, u1, f0, type="X", n_batch=N_BATCH, steps=1, lr=LR,
                   ladder=1, mc_step=MC_STEP, mc_iters=1)
            check_true(f"{name} rejects type='X'", False)
        except ValueError as e:
            check_true(f"{name} rejects type='X'", "'F' or 'G'" in str(e))

    # ── Monitor ──
    log("Monitor")
    try:
        Monitor(0)
        check_true("Monitor rejects every=0", False)
    except ValueError:
        check_true("Monitor rejects every=0", True)
    lines: list[str] = []
    flow_M, _ = train_reverse_KL(x_valid, u0, u1, f0, type="F",
                                 n_batch=N_BATCH, steps=20, lr=LR,
                                 mc_step=MC_STEP, mc_iters=MC_ITERS,
                                 monitor=Monitor(5, "[mon] ", lines.append))
    jax.effects_barrier()
    check_true("reports steps // every lines", len(lines) == 4, f"{len(lines)} lines")
    check_true("prefix and fields present",
               all(ln.startswith("[mon] step") and "loss =" in ln and "ESS =" in ln
                   for ln in lines),
               lines[0] if lines else "(no lines)")
    flow_M0, _ = train_reverse_KL(x_valid, u0, u1, f0, type="F",
                                  n_batch=N_BATCH, steps=20, lr=LR,
                                  mc_step=MC_STEP, mc_iters=MC_ITERS)
    check("monitor does not change training", max_param_delta(flow_M, flow_M0), 0.0, tol=0)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone training-driver tests passed")


if __name__ == "__main__":
    main()

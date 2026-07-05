"""Standalone Boltzmann-generator smoke test (jflows only) — run from
the repo root as `~/.envs/jax/bin/python -m smoke.test_boltzmann`.

For boltzmann_reverse_KL (the adaptive-ladder annealed BG):

    1. backend: the test runs directly on the GPU;
    2. ladder contract on a multimodal 2D target: at least one stage,
       coefficients strictly increasing and ending exactly at t = 1,
       every accepted stage clears the ESS threshold tau;
    3. per-stage records: each stage saves its incremental flow (a
       Flow instance) and its ESS — no per-stage particle sets; the
       last saved flow is bit-identical to the returned flow;
    4. final output: the advanced particle set has the full sample
       shape, is finite, and has actually moved off the source set;
    5. determinism: two identical calls agree bit for bit (same ladder,
       same trained flow);
    6. validation and monitoring: the stage printer receives training,
       validation, and acceptance lines;
    7. rejection of bad arguments: type not in {'F', 'G'}, unknown
       bg_param keys, invalid shrink_factor.

Default JAX backend (GPU); GPU memory preallocation is disabled.
Exits nonzero on any failure.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.flow import NSF  # noqa: E402
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture  # noqa: E402
from jflows.train import Monitor, boltzmann_reverse_KL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_boltzmann.log")

FAILURES = 0

N_VALID, N_BATCH, STEPS, LR = 4000, 500, 100, 2e-3
MC_STEP, MC_ITERS = 1e-3, 20
TAU = 0.5
BG = {"t_safe": 0.3, "tau": TAU}


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
    la = [x for x in jax.tree.leaves(flow_a) if eqx.is_inexact_array(x)]
    lb = [x for x in jax.tree.leaves(flow_b) if eqx.is_inexact_array(x)]
    return max(float(jnp.abs(a - b).max()) for a, b in zip(la, lb))


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_boltzmann | jax {jax.__version__} | {jax.default_backend()}")
    check_true("running on GPU", jax.default_backend() == "gpu",
               f"backend = {jax.default_backend()}")

    u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[4.0, 4.0])
    u1 = Nlog_Gaussian_Mixture(
        weights=[1.0, 1.0, 1.0],
        mean=[[0.0, 3.2], [-3.0, -1.8], [3.0, -1.8]],
        variance=[[0.06, 0.06], [0.06, 0.06], [0.06, 0.06]],
    )
    x_valid = u0.samples(jax.random.key(2), N_VALID)
    flow0 = NSF(jax.random.key(0), a=[-6.0, -6.0], b=[6.0, 6.0], bins=8,
                transforms=2, hidden_features=(32, 32)).zeros()

    # ── ladder contract ──
    log("boltzmann_reverse_KL ladder")
    lines: list[str] = []
    flow, y, stages = boltzmann_reverse_KL(
        x_valid, u0, u1, flow0, type="F",
        n_batch=N_BATCH, steps=STEPS, lr=LR, mc_step=MC_STEP, mc_iters=MC_ITERS,
        monitor=Monitor(STEPS, "[t] ", lines.append), bg_param=BG,
    )
    jax.effects_barrier()
    ts = [s["t"] for s in stages]
    check_true("at least one accepted stage", len(stages) >= 1, f"{len(stages)} stages")
    check_true("ladder strictly increasing", ts == sorted(set(ts)), f"t = {ts}")
    check_true("ladder complete (t = 1)", bool(ts) and ts[-1] == 1.0, f"t[-1] = {ts[-1] if ts else None}")
    check_true("every stage ESS >= tau", all(s["ess"] >= TAU for s in stages),
               f"ESS = {[round(s['ess'], 3) for s in stages]}")

    # ── per-stage records ──
    log("per-stage records")
    check_true("stage flows saved", all(isinstance(s["flow"], NSF) for s in stages))
    check_true("record keys are t/ess/flow",
               all(set(s.keys()) == {"t", "ess", "flow"} for s in stages))
    check("last saved flow == returned flow",
          max_param_delta(stages[-1]["flow"], flow), 0.0, tol=0)

    # ── final output ──
    check_true("particle set full-size and finite",
               y.shape == x_valid.shape and bool(jnp.isfinite(y).all()), f"{y.shape}")
    check_true("particle set advanced off the source set",
               float(jnp.abs(y - x_valid).max()) > 1.0,
               f"max move {float(jnp.abs(y - x_valid).max()):.2f}")

    # ── determinism ──
    log("determinism")
    flow2, y2, stages2 = boltzmann_reverse_KL(
        x_valid, u0, u1, flow0, type="F",
        n_batch=N_BATCH, steps=STEPS, lr=LR, mc_step=MC_STEP, mc_iters=MC_ITERS,
        bg_param=BG,
    )
    check_true("same ladder", [s["t"] for s in stages2] == ts,
               f"t = {[round(s['t'], 4) for s in stages2]}")
    check("same trained flow", max_param_delta(flow, flow2), 0.0, tol=0)
    check("same particle set", y2, y, tol=0)

    # ── monitoring ──
    log("monitoring")
    check_true("training lines observed", any("loss =" in ln for ln in lines))
    check_true("validation lines observed", any("validation" in ln for ln in lines))
    check_true("acceptance lines observed", any("ACCEPTED" in ln for ln in lines))

    # ── argument rejection ──
    log("argument rejection")
    for name, kwargs in (
        ("type='X'", dict(type="X")),
        ("unknown bg_param key", dict(type="F", bg_param={"tau_typo": 0.5})),
        ("invalid shrink_factor", dict(type="F", bg_param={"shrink_factor": 1.5})),
    ):
        try:
            boltzmann_reverse_KL(x_valid, u0, u1, flow0,
                                 n_batch=N_BATCH, steps=1, lr=LR,
                                 mc_step=MC_STEP, mc_iters=1, **kwargs)
            check_true(f"rejects {name}", False)
        except ValueError:
            check_true(f"rejects {name}", True)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone Boltzmann-generator tests passed")


if __name__ == "__main__":
    main()

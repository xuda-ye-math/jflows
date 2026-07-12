"""Standalone Boltzmann-generator smoke test (jflows only) — run after
installation from the repo root as `python -m smoke.test_boltzmann`.

For boltzmann_reverse_KL_F (the adaptive-ladder annealed BG):

    1. backend: the test runs directly on the GPU;
    2. ladder contract on a multimodal 2D target: at least one stage,
       coefficients strictly increasing and ending exactly at t = 1,
       every accepted stage clears the ESS threshold tau_ess;
    3. per-stage records: each stage saves its incremental flow (a
       Flow instance) and its ESS — no per-stage particle sets; the
       trained maps are returned ONLY through the stage records;
    4. final output: the advanced particle set has the full sample
       shape, is finite, and has actually moved off the source set;
    5. determinism: two identical calls agree bit for bit (same ladder,
       same trained flow);
    6. validation and monitoring: the stage printer receives training,
       validation, and acceptance lines;
    7. tau_smc pre-selection: a gated ladder (tau_smc > 0, MALA,
       ladder = 2) shrinks over-aggressive t_k via the multi-level SMC
       check on an n_pool-sized selection pool and still completes,
       with [select] lines reported;
    8. boltzmann_forward_KL_G: the forward KL twin (per-step AIS through
       the current flow, flow fixed as G) completes its ladder with the
       same record contract;
    9. boltzmann_forward_KLXX_G: the full-mixture generator at general
       (coeff_lambda, coeff_alpha, coeff_beta) — per-stage quench-and-
       temper pool, gated ladder — completes with the same record
       contract;
    10. adaptive KLX plus fixed-schedule KL, KLX, reverse KL, and KLXX
        interfaces all satisfy the same five-key stage-record contract;
    11. rejection of bad arguments: unknown bg_param keys, invalid
        shrink_factor.

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
from jflows.train import Monitor  # noqa: E402
from jflows.boltzmann import (  # noqa: E402
    boltzmann_forward_KL_G,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G,
    boltzmann_forward_KLXX_G_fixed,
    boltzmann_reverse_KL_F,
    boltzmann_reverse_KL_F_fixed,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_boltzmann.log")

FAILURES = 0

N_VALID, N_POOL, N_BATCH, STEPS, LR = 4000, 1000, 500, 100, 2e-3
MC_STEP, MC_ITERS = 1e-3, 20
MELT, OPT_STEP, OPT_ITERS = 2.0, 0.5, 100
COEFF_LAMBDA, COEFF_ALPHA, COEFF_BETA = 0.7, 0.8, 0.3
TAU = 0.5
BG = {"t_safe": 0.3, "tau_ess": TAU}


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
    log("boltzmann_reverse_KL_F ladder")
    lines: list[str] = []
    y, stages = boltzmann_reverse_KL_F(
        x_valid, u0, u1, flow0,
        n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS, lr=LR, ladder=1,
        mc_step=MC_STEP, mc_iters=MC_ITERS,
        monitor=Monitor(STEPS, "[t] ", lines.append), bg_param=BG,
    )
    jax.effects_barrier()
    ts = [s["t"] for s in stages]
    check_true("at least one accepted stage", len(stages) >= 1, f"{len(stages)} stages")
    check_true("ladder strictly increasing", ts == sorted(set(ts)), f"t = {ts}")
    check_true("ladder complete (t = 1)", bool(ts) and ts[-1] == 1.0, f"t[-1] = {ts[-1] if ts else None}")
    check_true("every stage ESS >= tau_ess", all(s["ess"] >= TAU for s in stages),
               f"ESS = {[round(s['ess'], 3) for s in stages]}")

    # ── per-stage records ──
    log("per-stage records")
    check_true("stage flows saved", all(isinstance(s["flow"], NSF) for s in stages))
    check_true("record keys are t/ess/flow/ess_history/imp_history",
               all(set(s.keys()) == {"t", "ess", "flow", "ess_history", "imp_history"} for s in stages))
    check_true("imp_history >= 0 (stages)",
               all(float(s["imp_history"]) >= 0.0 for s in stages))

    # ── final output ──
    check_true("particle set full-size and finite",
               y.shape == x_valid.shape and bool(jnp.isfinite(y).all()), f"{y.shape}")
    check_true("particle set advanced off the source set",
               float(jnp.abs(y - x_valid).max()) > 1.0,
               f"max move {float(jnp.abs(y - x_valid).max()):.2f}")

    # ── determinism ──
    log("determinism")
    y2, stages2 = boltzmann_reverse_KL_F(
        x_valid, u0, u1, flow0,
        n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS, lr=LR, ladder=1,
        mc_step=MC_STEP, mc_iters=MC_ITERS,
        bg_param=BG,
    )
    check_true("same ladder", [s["t"] for s in stages2] == ts,
               f"t = {[round(s['t'], 4) for s in stages2]}")
    check("same trained flow", max_param_delta(stages[-1]["flow"], stages2[-1]["flow"]), 0.0, tol=0)
    check("same particle set", y2, y, tol=0)

    # ── tau_smc pre-selection gate ──
    log("tau_smc SMC pre-selection")
    sel_lines: list[str] = []
    y3, stages3 = boltzmann_reverse_KL_F(
        x_valid, u0, u1, flow0,
        n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS, lr=LR,
        mc_step=MC_STEP, mc_iters=MC_ITERS,
        ladder=2, mc_adjust=True,
        monitor=Monitor(STEPS, "[t] ", sel_lines.append),
        bg_param={"t_safe": 0.3, "tau_ess": TAU, "tau_smc": 0.2},
    )
    jax.effects_barrier()
    check_true("gated ladder completes", bool(stages3) and stages3[-1]["t"] == 1.0,
               f"t = {[round(s['t'], 3) for s in stages3]}")
    check_true("[select] lines observed", any("[select]" in ln for ln in sel_lines),
               next((ln for ln in sel_lines if "[select]" in ln), "(none)"))

    # ── boltzmann_forward_KL_G (the forward KL twin) ──
    log("boltzmann_forward_KL_G ladder")
    yf, stages_f = boltzmann_forward_KL_G(
        x_valid, u0, u1, flow0,
        n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS, lr=LR, ladder=1,
        mc_step=MC_STEP, mc_iters=MC_ITERS, mc_adjust=True,
        bg_param=BG,
    )
    tsf = [s["t"] for s in stages_f]
    check_true("forward ladder complete (t = 1)", bool(tsf) and tsf[-1] == 1.0,
               f"t = {[round(t, 3) for t in tsf]}")
    check_true("forward stage ESS >= tau_ess", all(s["ess"] >= TAU for s in stages_f),
               f"ESS = {[round(s['ess'], 3) for s in stages_f]}")
    check_true("forward record keys are t/ess/flow/ess_history/imp_history",
               all(set(s.keys()) == {"t", "ess", "flow", "ess_history", "imp_history"} for s in stages_f))
    check_true("imp_history >= 0 (stages_f)",
               all(float(s["imp_history"]) >= 0.0 for s in stages_f))
    check_true("forward particle set full-size and finite",
               yf.shape == x_valid.shape and bool(jnp.isfinite(yf).all()), f"{yf.shape}")

    # ── boltzmann_forward_KLXX_G (the full-mixture generator) ──
    log(f"boltzmann_forward_KLXX_G ladder (lambda {COEFF_LAMBDA}, "
        f"alpha {COEFF_ALPHA}, beta {COEFF_BETA}, gated)")
    xx_lines: list[str] = []
    yxx, stages_xx = boltzmann_forward_KLXX_G(
        x_valid, u0, u1, flow0,
        n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS, lr=LR, ladder=2,
        melt=MELT, opt_step=OPT_STEP, opt_iters=OPT_ITERS,
        mc_step=MC_STEP, mc_iters=MC_ITERS,
        coeff_lambda=COEFF_LAMBDA, coeff_alpha=COEFF_ALPHA, coeff_beta=COEFF_BETA,
        monitor=Monitor(STEPS, "[t] ", xx_lines.append),
        bg_param={"t_safe": 0.3, "tau_ess": TAU, "tau_smc": 0.2},
    )
    jax.effects_barrier()
    tsxx = [s["t"] for s in stages_xx]
    check_true("KLXX ladder complete (t = 1)", bool(tsxx) and tsxx[-1] == 1.0,
               f"t = {[round(t, 3) for t in tsxx]}")
    check_true("KLXX stage ESS >= tau_ess", all(s["ess"] >= TAU for s in stages_xx),
               f"ESS = {[round(s['ess'], 3) for s in stages_xx]}")
    check_true("KLXX record keys are t/ess/flow/ess_history/imp_history",
               all(set(s.keys()) == {"t", "ess", "flow", "ess_history", "imp_history"} for s in stages_xx))
    check_true("imp_history >= 0 (stages_xx)",
               all(float(s["imp_history"]) >= 0.0 for s in stages_xx))
    check_true("KLXX particle set full-size and finite",
               yxx.shape == x_valid.shape and bool(jnp.isfinite(yxx).all()), f"{yxx.shape}")
    check_true("KLXX [select] lines observed", any("[select]" in ln for ln in xx_lines),
               next((ln for ln in xx_lines if "[select]" in ln), "(none)"))

    # ── remaining public adaptive/fixed interfaces, tiny one-level probes ──
    # These are API/logic checks rather than convergence tests. t_safe=1 and
    # tau_ess=0 make the adaptive call take exactly one accepted level.
    log("KLX adaptive and KL/KLX fixed interface probes")
    x_small = x_valid[:128]
    small = dict(n_batch=64, steps=2, lr=LR, ladder=1,
                 mc_step=MC_STEP, mc_iters=0)
    y_kx, s_kx = boltzmann_forward_KLX_G(
        x_small, u0, u1, flow0, n_pool=64, coeff_lambda=COEFF_LAMBDA,
        bg_param={"t_safe": 1.0, "tau_ess": 0.0, "max_stages": 1},
        **small,
    )
    check_true("KLX adaptive: complete 5-key finite output",
               len(s_kx) == 1 and s_kx[-1]["t"] == 1.0
               and set(s_kx[-1]) == {"t", "ess", "flow", "ess_history", "imp_history"}
               and bool(jnp.isfinite(y_kx).all()))
    for name, fn, extra in (
        ("KL fixed", boltzmann_forward_KL_G_fixed, {}),
        ("KLX fixed", boltzmann_forward_KLX_G_fixed,
         {"coeff_lambda": COEFF_LAMBDA}),
    ):
        y_probe, s_probe = fn(
            x_small, u0, u1, flow0, t_list=[1.0], **small, **extra
        )
        check_true(f"{name}: complete 5-key finite output",
                   len(s_probe) == 1 and s_probe[-1]["t"] == 1.0
                   and set(s_probe[-1]) == {"t", "ess", "flow", "ess_history", "imp_history"}
                   and bool(jnp.isfinite(y_probe).all()))

    # ── monitoring ──
    log("monitoring")
    check_true("training lines observed", any("loss =" in ln for ln in lines))
    check_true("validation lines observed", any("validation" in ln for ln in lines))
    check_true("acceptance lines observed", any("ACCEPTED" in ln for ln in lines))

    # ── argument rejection ──
    log("argument rejection")
    for name, bg in (
        ("unknown bg_param key", {"tau_typo": 0.5}),
        ("invalid shrink_factor", {"shrink_factor": 1.5}),
    ):
        try:
            boltzmann_reverse_KL_F(x_valid, u0, u1, flow0,
                                   n_pool=N_POOL, n_batch=N_BATCH, steps=1, lr=LR,
                                   ladder=1, mc_step=MC_STEP, mc_iters=1, bg_param=bg)
            check_true(f"rejects {name}", False)
        except ValueError:
            check_true(f"rejects {name}", True)

    # ── fixed-schedule variants (_fixed): fixed t_list, no SMC gate, no accept ──
    log("boltzmann_*_fixed (fixed t_list, bare step-by-step training)")
    t_list = [0.3, 0.6, 1.0]
    fx_lines: list[str] = []
    yfx, stages_fx = boltzmann_reverse_KL_F_fixed(
        x_valid, u0, u1, flow0,
        n_batch=N_BATCH, steps=STEPS, lr=LR,
        mc_step=MC_STEP, mc_iters=MC_ITERS, t_list=t_list,
        monitor=Monitor(STEPS, "[fx] ", fx_lines.append),
    )
    jax.effects_barrier()
    check_true("fixed: one stage per t_list entry", len(stages_fx) == len(t_list),
               f"{len(stages_fx)} stages")
    check_true("fixed: stage ts equal the given t_list",
               [round(s["t"], 6) for s in stages_fx] == t_list,
               f"t = {[s['t'] for s in stages_fx]}")
    check_true("fixed: record keys are the 5 keys",
               all(set(s.keys()) == {"t", "ess", "flow", "ess_history", "imp_history"}
                   for s in stages_fx))
    check_true("fixed: imp_history >= 0",
               all(float(s["imp_history"]) >= 0.0 for s in stages_fx))
    check_true("fixed: ess_history length == steps",
               all(len(s["ess_history"]) == STEPS for s in stages_fx))
    check_true("fixed: particle set finite and advanced",
               yfx.shape == x_valid.shape and bool(jnp.isfinite(yfx).all())
               and float(jnp.abs(yfx - x_valid).max()) > 1.0, f"{yfx.shape}")
    check_true("fixed: NO SMC [select] lines (bare training)",
               not any("[select]" in ln for ln in fx_lines))

    # KLXX_fixed exercises the QT-pool path along a fixed schedule
    yxfx, stages_xfx = boltzmann_forward_KLXX_G_fixed(
        x_valid, u0, u1, flow0,
        n_pool=N_POOL, n_batch=N_BATCH, steps=STEPS, lr=LR, ladder=2,
        melt=MELT, opt_step=OPT_STEP, opt_iters=OPT_ITERS,
        mc_step=MC_STEP, mc_iters=MC_ITERS, t_list=[0.5, 1.0],
        coeff_lambda=COEFF_LAMBDA, coeff_alpha=COEFF_ALPHA, coeff_beta=COEFF_BETA,
    )
    check_true("KLXX_fixed: 2 stages, complete, 5-key, imp>=0, finite",
               len(stages_xfx) == 2 and stages_xfx[-1]["t"] == 1.0
               and all(set(s.keys()) == {"t", "ess", "flow", "ess_history", "imp_history"}
                       for s in stages_xfx)
               and all(float(s["imp_history"]) >= 0.0 for s in stages_xfx)
               and bool(jnp.isfinite(yxfx).all()),
               f"t = {[round(s['t'], 3) for s in stages_xfx]}")

    # t_list validation raises on bad schedules; a leading 0 is dropped
    log("fixed: t_list validation")
    for name, tl in (("empty", []), ("non-monotone", [0.6, 0.3, 1.0]),
                     ("out of (0,1]", [0.3, 1.5])):
        try:
            boltzmann_reverse_KL_F_fixed(x_valid, u0, u1, flow0, n_batch=N_BATCH,
                                         steps=1, lr=LR, mc_step=MC_STEP, mc_iters=1,
                                         t_list=tl)
            check_true(f"fixed rejects {name} t_list", False)
        except ValueError:
            check_true(f"fixed rejects {name} t_list", True)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone Boltzmann-generator tests passed")


if __name__ == "__main__":
    main()

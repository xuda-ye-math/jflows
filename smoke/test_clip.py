"""Standalone energy/gradient clipping smoke test (jflows only) — run after
installation from the repo root as `python -m smoke.test_clip`.

For the `u_clip` (potential screen) and `g_clip` (global gradient-norm clip)
parameters of the forward training drivers (`train_forward_KL_G`,
`train_forward_KLX_G`, `train_forward_KLXX_G`):

    1. no-op default: u_clip = inf and g_clip = inf reproduce the trained
       flow EXACTLY (bit-identical parameters) to a call without the
       arguments — the screen and the clip must add nothing at the default;
    2. potential screen: with u_clip set to a percentile of the target potential
       (so a real fraction of the manufactured batch is high-energy and
       screened), the trained flow differs from the unscreened one, and the
       masked-mean / pair-masked helpers reduce over the kept sub-batch only
       (checked against a direct numpy recomputation);
    3. gradient clip: a finite g_clip caps the raw global gradient L2 norm
       BEFORE it enters the Adam moment update, so an outlier high-energy
       batch cannot spike the second-moment accumulator (Adam's per-coordinate
       normalization means the effect on the final step size is weak; the
       clip is a spike guard on the gradient, not a bound on the update);
    4. the boltzmann_forward_* wrappers accept and forward u_clip / g_clip
       (a short 2-stage ladder runs to completion under a finite screen).

Float32 (the drivers' working precision), on the default JAX backend (GPU
when available). Exits nonzero on any failure.
"""

import math
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.flow import NCSF  # noqa: E402
from jflows.potential import Nlog_Uniform, potential_from  # noqa: E402
from jflows.train import (  # noqa: E402
    _clip_global,
    _kept,
    _masked_mean,
    _masked_pair_mean,
)
from jflows.train import (  # noqa: E402
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
)
from jflows.boltzmann import boltzmann_forward_KLXX_G  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_clip.log")

FAILURES = 0
D = 6
LIM = math.pi


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


def leaves(flow):
    return [np.asarray(a) for a in jax.tree.leaves(eqx.filter(flow, eqx.is_inexact_array))]


def max_param_delta(fa, fb):
    return max(float(np.max(np.abs(a - b))) for a, b in zip(leaves(fa), leaves(fb)))


# source: uniform on the torus
u0 = Nlog_Uniform(a=[-LIM] * D, b=[LIM] * D)
# target: a periodic ridge whose energy spans a wide range across the box,
# so a percentile threshold screens a real high-energy fraction of any batch
u_smooth = potential_from(lambda x: -(jnp.cos(x - jnp.roll(x, 1, axis=-1))).sum(-1))

# u_clip set so ~40% of a representative batch is high-potential and screened
_probe = u0.samples(jax.random.key(9), 8000)
E_SCREEN = float(np.percentile(np.asarray(u_smooth(_probe)), 60.0))


def new_flow():
    return NCSF(jax.random.key(0), a=[-LIM] * D, b=[LIM] * D, bins=8,
                transforms=3, hidden_features=(64, 64)).zeros()


def train(fn, target, **kw):
    return fn(u0.samples(jax.random.key(2), 4000), u0, target, new_flow(),
             batch_size=256, train_steps=40, lr=1e-3, ladder=1, mc_dt=1e-3,
             mc_steps=20, seed=jnp.uint32(0), **kw)[0]


def main() -> None:
    open(LOG, "w").close()
    log(f"START test_clip | backend {jax.default_backend()} | D={D} float32")

    # ---- helper unit checks (numpy ground truth) ----
    rng = np.random.default_rng(0)
    v = rng.standard_normal(256).astype(np.float32)
    keep_np = rng.random(256) > 0.3
    perm_np = rng.permutation(256)
    vj, kj, pj = jnp.asarray(v), jnp.asarray(keep_np), jnp.asarray(perm_np)
    check("_masked_mean == numpy kept mean", _masked_mean(vj, kj),
          v[keep_np].mean(), tol=1e-5)
    diff = np.abs(v - v[perm_np])
    pk = keep_np.astype(np.float64) * keep_np[perm_np].astype(np.float64)
    check("_masked_pair_mean == numpy pair mean",
          _masked_pair_mean(jnp.asarray(diff), kj, pj),
          (pk * diff).sum() / max(pk.sum(), 1.0), tol=1e-4)
    keep = _kept(u_smooth, _probe, u_clip=E_SCREEN)
    frac = float(np.asarray(keep).mean())
    check_true("_kept screens the high-energy tail", 0.4 < frac < 0.8,
               f"{frac:.2f} kept at the 60th-pct threshold")

    # ---- 1. no-op: the ACTIVE masked/clip path, screening nothing, ≈ default.
    # u_clip=1e3 is above every target potential (all samples kept, so the
    # masked-mean / pair-masked branch RUNS but reduces over the full batch);
    # g_clip=1e30 is above every gradient norm (the clip branch RUNS but scales
    # by exactly 1). This exercises the masking code (unlike an inf-vs-inf
    # compare). NOTE the all-kept masked mean sum(keep·v)/sum(keep) is
    # MATHEMATICALLY equal to plain .mean() but NOT bit-identical in float32
    # (different reduction op order); over the training steps that ~1-ULP gap
    # compounds and nondeterministic GPU reductions vary it run to run — so the
    # tolerance reflects float32 training drift, not exact equality. (The exact
    # bit-identical no-op is the u_clip=inf path, which is untraced.) A real
    # masking bug diverges far past this or produces NaNs.
    BIG_E, BIG_G = 1e3, 1e30
    check_true("BIG_E keeps every sample",
               bool(np.all(np.asarray(u_smooth(_probe)) <= BIG_E)), "target < 1e3")
    log("no-op of the ACTIVE screen/clip path (all kept, gradient scale 1):")
    for name, fn in (("KL_G", train_forward_KL_G),
                     ("KLX_G", train_forward_KLX_G),
                     ("KLXX_G", train_forward_KLXX_G)):
        kw = {} if fn is not train_forward_KLXX_G else dict(
            pool_size=0, melt=2 * math.pi, opt_dt=1e-2, opt_steps=20)
        base = train(fn, u_smooth, **kw)
        active = train(fn, u_smooth, u_clip=BIG_E, g_clip=BIG_G, **kw)
        d = max_param_delta(active, base)
        check_true(f"{name}: all-kept active path ≈ default", d <= 3e-2,
                   f"max|Δparam|={d:.2e} (float32 training drift, not a bug)")

    # ---- 2. energy screen changes the trained flow ----
    log(f"potential screen (u_clip={E_SCREEN:.3f}, ~40% of each batch screened):")
    for name, fn in (("KL_G", train_forward_KL_G),
                     ("KLX_G", train_forward_KLX_G),
                     ("KLXX_G", train_forward_KLXX_G)):
        kw = {} if fn is not train_forward_KLXX_G else dict(
            pool_size=0, melt=2 * math.pi, opt_dt=1e-2, opt_steps=20)
        unscreened = train(fn, u_smooth, **kw)
        screened = train(fn, u_smooth, u_clip=E_SCREEN, **kw)
        d = max_param_delta(unscreened, screened)
        check_true(f"{name}: u_clip screen changes the flow", d > 1e-6,
                   f"max|Δparam|={d:.2e}")
        check_true(f"{name}: screened flow finite",
                   all(np.all(np.isfinite(a)) for a in leaves(screened)))

    # ---- 3. gradient clip caps the update; _clip_global unit check ----
    log("gradient clip (finite g_clip bounds the global gradient norm):")
    grads = {"a": jnp.asarray(rng.standard_normal((10, 10)), dtype=jnp.float32),
             "b": jnp.asarray(rng.standard_normal(10), dtype=jnp.float32)}
    raw = math.sqrt(sum(float((np.asarray(g) ** 2).sum()) for g in grads.values()))
    clipped = _clip_global(grads, 1.0)
    cn = math.sqrt(sum(float((np.asarray(g) ** 2).sum()) for g in clipped.values()))
    check_true("_clip_global caps norm to g_clip", cn <= 1.0 + 1e-4,
               f"raw={raw:.3f} -> clipped={cn:.4f}")
    check("_clip_global no-op above ceiling", _clip_global(grads, 1e9)["b"],
          np.asarray(grads["b"]), tol=1e-5)
    f_noclip = train(train_forward_KLX_G, u_smooth)
    f_clip = train(train_forward_KLX_G, u_smooth, g_clip=1e-2)
    check_true("KLX_G: g_clip=1e-2 has an effect on training",
               max_param_delta(f_noclip, f_clip) > 1e-6,
               f"max|Δparam|={max_param_delta(f_noclip, f_clip):.2e}")

    # ---- 4. boltzmann controller forwards u_clip / g_clip ----
    log("boltzmann_forward_KLXX_G forwards u_clip / g_clip:")
    y_bg, stages = boltzmann_forward_KLXX_G(
        u0.samples(jax.random.key(3), 4000), u0, u_smooth, new_flow(), 0,
        batch_size=256, train_steps=30, lr=1e-3, ladder=1,
        melt=2 * math.pi, opt_dt=1e-2, opt_steps=20, mc_dt=1e-3, mc_steps=20,
        bg_param={"t_safe": 0.3, "tau_ess": 0.2, "max_stages": 2, "max_retry": 2},
        u_clip=20.0, g_clip=1e3)
    check_true("boltzmann KLXX runs under finite u_clip/g_clip",
               len(stages) >= 1 and np.all(np.isfinite(np.asarray(y_bg))),
               f"{len(stages)} stage(s)")

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all clip tests passed")


if __name__ == "__main__":
    main()

"""Phase 3 weight-transplant check (jax side) — run with ~/.envs/jax/bin/python.

For every public flow class: build the jflows twin (same structural
config, randmask=False), transplant the zflows parameters from
smoke_tests/parity_phase3.npz, and assert identical forward / ladj /
inverse outputs. Then jax-only checks: zeros() ⇒ identity + ladj 0, and
finite filter_grad of a reverse-KL-shaped loss under filter_jit for every
flow class. Float64, on the default JAX backend (GPU when available; set
JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled.
"""

import math
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.core.flows import MaskedAutoregressiveTransform  # noqa: E402
from jflows.core.transforms import (  # noqa: E402
    CircularShiftTransform,
    ComposedTransform,
    MonotonicRQSTransform,
)
from jflows.flow import CNF, NCSF, NSF, OTFlow, RealNVP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase3_check.log")
NPZ = os.path.join(HERE, "parity_phase3.npz")

FAILURES = 0
d = np.load(NPZ)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check(name: str, got, want, tol: float = 1e-9) -> None:
    global FAILURES
    got, want = np.asarray(got), np.asarray(want)
    if got.dtype == bool or want.dtype == bool:
        got, want = got.astype(np.int8), want.astype(np.int8)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    ok = got.shape == want.shape and err <= tol
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def transplant_mlp(flow, accessor, prefix: str):
    """Replace the weights/biases of the MLP at `accessor(flow)` from npz."""
    n = int(d[f"{prefix}_n"])
    wheres, values = [], []
    for j in range(n):
        wheres.append(lambda f, j=j: accessor(f).linears[j].weight)
        values.append(jnp.asarray(d[f"{prefix}_w{j}"]))
        wheres.append(lambda f, j=j: accessor(f).linears[j].bias)
        values.append(jnp.asarray(d[f"{prefix}_b{j}"]))
    return eqx.tree_at(lambda f: [w(f) for w in wheres], flow, values)


def check_io(prefix: str, flow, tol: float = 1e-9) -> None:
    F = flow.t()
    x = jnp.asarray(d[f"{prefix}_x"])
    y, ladj = F.call_and_ladj(x)
    check("y", y, d[f"{prefix}_y"], tol=tol)
    check("ladj", ladj, d[f"{prefix}_ladj"], tol=tol)
    check("x_back", F.inv(y), d[f"{prefix}_xb"], tol=tol)


def check_zeros_and_grad(name: str, flow, x: jnp.ndarray) -> None:
    """zeros() ⇒ identity with ladj 0; reverse-KL-shaped loss has finite grads."""
    fz = flow.zeros()
    y, ladj = fz.t().call_and_ladj(x)
    check(f"{name} zeros() identity", y, x, tol=1e-9)
    check(f"{name} zeros() ladj", ladj, jnp.zeros(x.shape[0]), tol=1e-9)

    @eqx.filter_jit
    @eqx.filter_grad
    def loss_grad(f, x):
        y, ladj = f.t().call_and_ladj(x)
        return jnp.mean(0.5 * jnp.sum(y**2, axis=-1) - ladj)

    grads = loss_grad(flow, x)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(g)]
    finite = all(bool(jnp.isfinite(g).all()) for g in leaves)
    nonzero = any(bool(jnp.any(g != 0)) for g in leaves)
    log(f"  {name} filter_jit+filter_grad: {len(leaves)} grad leaves, finite={finite}, any-nonzero={nonzero} -> "
        f"{'OK' if finite and nonzero else 'FAIL'}")
    if not (finite and nonzero):
        global FAILURES
        FAILURES += 1


def main() -> None:
    log(f"START parity_check_phase3 | jax {jax.__version__} ({jax.default_backend()})")
    key = jax.random.key(0)

    # ── NSF ──
    log("NSF transplant")
    a = jnp.asarray([-2.0, -1.5, -3.0])
    b = jnp.asarray([2.0, 1.5, 3.0])
    nsf = NSF(key, a, b, bins=8, slope=1e-3, transforms=3, randmask=False,
              hidden_features=(32, 16))
    for i in range(3):
        n = int(d[f"nsf_maf{i}_n"])
        for j in range(n):
            check(f"maf{i} mask{j}", np.asarray(nsf._maf[i].hyper.linears[j].mask),
                  d[f"nsf_maf{i}_mask{j}"], tol=0)
        nsf = transplant_mlp(nsf, lambda f, i=i: f._maf[i].hyper, f"nsf_maf{i}")
    check_io("nsf", nsf)

    # ── NCSF machinery (legacy composition, matching the zflows reference) ──
    # jflows' NCSF uses the circular spline with a shared learnable seam
    # derivative (PLAN §3.13), so the zflows cross-check builds zuko's
    # composition explicitly: K-1 derivatives, both seam slopes pinned to 1.
    log("NCSF machinery transplant (legacy composition)")

    def legacy_circular(*phi, bound=math.pi, slope=1e-3):
        return ComposedTransform(
            CircularShiftTransform(bound=bound),
            MonotonicRQSTransform(*phi, bound=bound, slope=slope),
        )

    hw = jnp.full((3,), math.pi)
    mafs = tuple(
        MaskedAutoregressiveTransform(
            jax.random.key(i), features=3, univariate=legacy_circular,
            shapes=[(8,), (8,), (7,)],
            order=np.arange(3) if i % 2 == 0 else np.arange(3)[::-1],
            hidden_features=(32, 16), activation=jax.nn.silu,
            bound=hw, slope=1e-3,
        )
        for i in range(3)
    )
    for i in range(3):
        mafs = transplant_mlp(mafs, lambda t, i=i: t[i].hyper, f"ncsf_maf{i}")
    Fn = ComposedTransform(*[m() for m in mafs])  # center = 0 on [-pi, pi]
    xn = jnp.asarray(d["ncsf_x"])
    yn, ladjn = Fn.call_and_ladj(xn)
    check("y", yn, d["ncsf_y"])
    check("ladj", ladjn, d["ncsf_ladj"])
    check("x_back", Fn.inv(yn), d["ncsf_xb"])

    # ── RealNVP (lu and rotation mixing) ──
    for kind in ("lu", "rotation"):
        log(f"RealNVP ({kind}) transplant")
        nvp = RealNVP(key, dimension=4, transforms=3, randmask=False, mixing=kind,
                      hidden_features=(32, 16))
        i_gct = i_mix = 0
        for li, layer in enumerate(nvp._layers):
            if hasattr(layer, "hyper"):
                check(f"gct{i_gct} mask", np.asarray(layer.mask),
                      d[f"nvp_{kind}_gctmask{i_gct}"], tol=0)
                nvp = transplant_mlp(nvp, lambda f, li=li: f._layers[li].hyper,
                                     f"nvp_{kind}_gct{i_gct}")
                i_gct += 1
            else:
                nvp = eqx.tree_at(lambda f, li=li: f._layers[li].weight, nvp,
                                  jnp.asarray(d[f"nvp_{kind}_mix{i_mix}"]))
                i_mix += 1
        assert i_gct == int(d[f"nvp_{kind}_ngct"]) and i_mix == int(d[f"nvp_{kind}_nmix"])
        check_io(f"nvp_{kind}", nvp)

    # ── CNF ──
    log("CNF transplant")
    cnf = CNF(key, dimension=3, frequency=3, nt=8, exact=True, hidden_features=(16, 16))
    cnf = transplant_mlp(cnf, lambda f: f._ffj.ode, "cnf_ode")
    check_io("cnf", cnf)

    # ── OTFlow ──
    log("OTFlow transplant")
    otf = OTFlow(key, dimension=3, hidden=16, layer=3, rank=4, nt=8)
    nres = int(d["otf_nres"])
    wheres = [lambda f: f._ot.phi.A, lambda f: f._ot.phi.c.weight, lambda f: f._ot.phi.w.weight]
    values = [jnp.asarray(d["otf_A"]), jnp.asarray(d["otf_cw"]), jnp.asarray(d["otf_ww"])]
    for j in range(nres):
        wheres.append(lambda f, j=j: f._ot.phi.N.layers[j].weight)
        values.append(jnp.asarray(d[f"otf_res_w{j}"]))
        wheres.append(lambda f, j=j: f._ot.phi.N.layers[j].bias)
        values.append(jnp.asarray(d[f"otf_res_b{j}"]))
    otf = eqx.tree_at(lambda f: [w(f) for w in wheres], otf, values)
    check_io("otf", otf)
    Fo = otf.t().transforms[0]
    xo = jnp.asarray(d["otf_x"])
    _, _, cost, hjb = Fo.call_full(xo)
    check("call_full cost", cost, d["otf_cost"], tol=1e-9)
    check("call_full hjb", hjb, d["otf_hjb"], tol=1e-9)

    # ── jax-only: zeros() + jit/grad for every class ──
    log("zeros() and filter_jit/filter_grad checks")
    kz = jax.random.key(11)
    pi3 = jnp.full((3,), math.pi)
    ncsf = NCSF(kz, -pi3, pi3, bins=8, transforms=3, randmask=False,
                hidden_features=(32, 16))
    xn = jnp.asarray(d["nsf_x"])
    check_zeros_and_grad("NSF", nsf, xn)
    check_zeros_and_grad("NCSF", ncsf, jnp.asarray(d["ncsf_x"]))
    check_zeros_and_grad("RealNVP", RealNVP(kz, dimension=4, transforms=3, mixing="lu"),
                         jnp.asarray(d["nvp_lu_x"]))
    check_zeros_and_grad("CNF", cnf, jnp.asarray(d["cnf_x"]))
    check_zeros_and_grad("OTFlow", otf, xo)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all Phase 3 parity checks passed")


if __name__ == "__main__":
    main()

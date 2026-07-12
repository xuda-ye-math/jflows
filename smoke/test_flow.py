"""Standalone flow smoke test (jflows only) —
run with conda activate jflows && PYTHONPATH=/mnt/projects/jflows python.

For every public flow class (NSF, NCSF, CNF, OTFlow, RealNVP incl. both
mixing kinds), on random inputs:

    1. construct the flow and F = flow.t();
    2. inverse round-trips: F.inv(F(x)) ≈ x and F(F.inv(y)) ≈ y;
    3. log-det: F.call_and_ladj(x)[1] vs per-sample slogdet(jacfwd(F));
    4. autograd: eqx.filter_grad of a reverse KL loss under
       eqx.filter_jit — all gradient leaves finite, at least one nonzero;
    5. zeros() ⇒ identity map with ladj ≡ 0.

Float64, on the default JAX backend (GPU when available; set
JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled so
concurrently running jobs stay undisturbed. Exits nonzero on any failure.
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

from jflows.flow import CNF, NCSF, NSF, OTFlow, RealNVP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_flow.log")

FAILURES = 0


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


def exercise(name: str, flow, x, tol: float) -> None:
    """Round-trips, ladj vs autodiff, autograd, zeros() for one flow."""
    global FAILURES
    log(f"{name}")
    F = flow.t()

    # 1-2. forward / inverse round-trips
    y, ladj = F.call_and_ladj(x)
    check("call == call_and_ladj", F(x), y, tol=1e-12)
    # Flow high-level methods are thin delegations to t()
    check("flow.call_and_ladj == t()", flow.call_and_ladj(x)[1], ladj, tol=0)
    check("flow.inv == t().inv", flow.inv(y), F.inv(y), tol=0)
    check("flow.inv_and_ladj == t().inv", flow.inv_and_ladj(y)[1],
          F.inv.call_and_ladj(y)[1], tol=0)
    check("inv(F(x)) == x", F.inv(y), x, tol=tol)
    check("F(inv(y)) == y", F(F.inv(y)), y, tol=tol)

    # 3. log-det vs per-sample autodiff jacobian (batch-1 wrapper: the
    # continuous transforms take batched input only)
    J = jax.vmap(jax.jacfwd(lambda xi: F(xi[None, :])[0]))(x)
    _, logdet = jnp.linalg.slogdet(J)
    check("ladj vs slogdet", ladj, logdet, tol=max(tol, 1e-9))
    xb, ladj_inv = F.inv.call_and_ladj(y)
    check("inv ladj == -ladj", ladj_inv, -ladj, tol=max(tol, 1e-9))

    # 4. autograd through a reverse KL loss, jitted
    @eqx.filter_jit
    @eqx.filter_grad
    def loss_grad(f, x):
        y, ladj = f.t().call_and_ladj(x)
        return jnp.mean(0.5 * jnp.sum(y**2, axis=-1) - ladj)

    grads = loss_grad(flow, x)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(g)]
    finite = all(bool(jnp.isfinite(g).all()) for g in leaves)
    nonzero = any(bool(jnp.any(g != 0)) for g in leaves)
    log(f"  autograd: {len(leaves)} grad leaves, finite={finite}, any-nonzero={nonzero} -> "
        f"{'OK' if finite and nonzero else 'FAIL'}")
    if not (finite and nonzero):
        FAILURES += 1

    # 5. zeros() ⇒ identity with ladj ≡ 0
    yz, ladjz = flow.zeros().t().call_and_ladj(x)
    check("zeros() identity", yz, x, tol=1e-9)
    check("zeros() ladj", ladjz, jnp.zeros(x.shape[0]), tol=1e-9)


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_flow | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    kf, kx = jax.random.split(key)
    N = 8

    # NSF on an anisotropic box (randmask default True)
    a = jnp.asarray([-2.0, -1.5, -3.0, -1.0])
    b = jnp.asarray([2.0, 1.5, 3.0, 1.0])
    x = jax.random.uniform(kx, (N, 4), minval=-0.95, maxval=0.95) * (b - a) / 2 + (a + b) / 2
    exercise("NSF", NSF(kf, a, b, bins=8, transforms=3, hidden_features=(32, 16)), x, tol=1e-12)

    # NCSF on the 4-torus
    pi4 = jnp.full((4,), jnp.pi)
    xc = jax.random.uniform(kx, (N, 4), minval=-0.95, maxval=0.95) * jnp.pi
    exercise("NCSF", NCSF(kf, -pi4, pi4, bins=8, transforms=3, hidden_features=(32, 16)), xc, tol=1e-12)

    # RealNVP: plain, and both mixing kinds
    xg = jax.random.normal(kx, (N, 4))
    exercise("RealNVP (mixing=None)", RealNVP(kf, dimension=4, transforms=3, hidden_features=(32, 16)), xg, tol=1e-12)
    for kind in ("rotation", "lu"):
        exercise(f"RealNVP (mixing={kind})",
                 RealNVP(kf, dimension=4, transforms=3, mixing=kind, hidden_features=(32, 16)),
                 xg, tol=1e-12)

    # CNF (exact trace) — RK4-approximate inverse/ladj, tolerance set by nt
    exercise("CNF", CNF(kf, dimension=4, frequency=3, nt=24, hidden_features=(16, 16)), xg, tol=1e-6)

    # OTFlow — closed-form trace, RK4-approximate inverse; the randomly
    # initialised potential gives a stiffer drift than CNF's, so the
    # round-trip needs more steps for the same accuracy
    exercise("OTFlow", OTFlow(kf, dimension=4, hidden=16, layer=3, rank=4, nt=64), xg, tol=1e-5)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone flow tests passed")


if __name__ == "__main__":
    main()

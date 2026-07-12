"""Standalone loss smoke test (jflows only) — run after installation from
the repo root as `python -m smoke.test_loss`.

For reverse_KL_F / forward_KL_G and forward_KLX_G / forward_X_G:

    1. per-sample contract: every loss returns shape [N], and permuting
       the batch permutes the loss vector;
    2. identity flow (zeros()): reverse_KL_F == target(x) and
       forward_KL_G == source(y) exactly, per sample;
    3. definition vs the core layer: every loss reproduces the manual
       t() / t().inv computation exactly;
    4. forward_KLX_G / forward_X_G: against the manual log-ratio
       z = source(G(y)) - target(y) - ladj, forward_X_G == |z - z[perm]|
       and forward_KLX_G == z + coeff_lambda * forward_X_G at the same
       key, exactly; coeff_lambda = 0 reduces to z; X >= 0;
    5. autograd: filter_grad of the batch-mean is finite and nonzero
       (reverse_KL_F and forward_KLX_G).

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

from jflows.flow import NSF  # noqa: E402
from jflows.loss import forward_KL_G, forward_KLX_G, forward_X_G, reverse_KL_F  # noqa: E402
from jflows.potential import Nlog_Gaussian  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_loss.log")

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
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} shapes {got.shape}/{want.shape} "
        f"-> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def check_true(name: str, cond: bool, detail: str = "") -> None:
    global FAILURES
    log(f"  {name}: {detail}{' ' if detail else ''}-> {'OK' if cond else 'FAIL'}")
    if not cond:
        FAILURES += 1


def main() -> None:
    global FAILURES
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_loss | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    N = 16

    a = jnp.asarray([-2.0, -1.5, -3.0])
    b = jnp.asarray([2.0, 1.5, 3.0])
    nsf = NSF(key, a, b, bins=8, transforms=2, hidden_features=(32, 16))
    target = Nlog_Gaussian([0.0, 0.0, 0.0], [0.8, 0.8, 0.8])
    kx, ky = jax.random.split(jax.random.key(1))
    x = jax.random.uniform(kx, (N, 3), minval=-0.95, maxval=0.95) * (b - a) / 2 + (a + b) / 2
    y = jax.random.uniform(ky, (N, 3), minval=-0.95, maxval=0.95) * (b - a) / 2 + (a + b) / 2

    log("per-sample contract")
    losses = {
        "reverse_KL_F": reverse_KL_F(x, target, nsf),
        "forward_KL_G": forward_KL_G(y, target, nsf),
    }
    for name, vec in losses.items():
        check_true(f"{name} shape", vec.shape == (N,), f"{vec.shape}")
    perm = jax.random.permutation(jax.random.key(2), N)
    check("permutation equivariance", reverse_KL_F(x[perm], target, nsf),
          losses["reverse_KL_F"][perm], tol=1e-12)

    log("identity flow (zeros)")
    flow_z = nsf.zeros()
    check("reverse_KL_F == target(x)", reverse_KL_F(x, target, flow_z), target(x), tol=1e-12)
    check("forward_KL_G == source(y)", forward_KL_G(y, target, flow_z), target(y), tol=1e-12)

    log("definition vs the core layer")
    T = nsf.t()
    y_c, l_c = T.call_and_ladj(x)
    check("reverse_KL_F == core", losses["reverse_KL_F"], target(y_c) - l_c, tol=0)
    x_g, l_g2 = T.call_and_ladj(y)
    check("forward_KL_G == core", losses["forward_KL_G"], target(x_g) - l_g2, tol=0)

    log("forward_KLX_G / forward_X_G (flow fixed as G)")
    src = Nlog_Gaussian([0.2, -0.4, 0.5], [1.0, 0.9, 1.1])
    key_perm = jax.random.key(3)
    lam = 0.7
    klx = forward_KLX_G(y, src, target, nsf, key_perm, coeff_lambda=lam)
    xf = forward_X_G(y, src, target, nsf, key_perm)
    check_true("per-sample shapes [N]", klx.shape == (N,) and xf.shape == (N,),
               f"{klx.shape} / {xf.shape}")
    x_g2, l_g3 = T.call_and_ladj(y)                 # z = src(G(y)) - target(y) - ladj
    z = src(x_g2) - target(y) - l_g3
    perm_z = jax.random.permutation(key_perm, N)    # same key -> same permutation
    check("forward_X_G == |z - z[perm]|", xf, jnp.abs(z - z[perm_z]), tol=0)
    check("forward_KLX_G == z + lam * forward_X_G", klx, z + lam * xf, tol=0)
    check("coeff_lambda = 0 reduces to z",
          forward_KLX_G(y, src, target, nsf, key_perm, coeff_lambda=0.0), z, tol=0)
    check_true("X >= 0", bool(jnp.all(xf >= 0)), f"min {float(xf.min()):.2e}")
    check("z == forward_KL_G - target(y)",
          z, forward_KL_G(y, src, nsf) - target(y), tol=1e-12)

    log("autograd through the batch-mean")

    @eqx.filter_jit
    @eqx.filter_grad
    def gmean(flow, x):
        return reverse_KL_F(x, target, flow).mean()

    grads = gmean(nsf, x)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(g)]
    finite = all(bool(jnp.isfinite(g).all()) for g in leaves)
    nonzero = any(bool(jnp.any(g != 0)) for g in leaves)
    check_true("filter_grad finite & nonzero", finite and nonzero,
               f"{len(leaves)} leaves, finite={finite}, any-nonzero={nonzero}")

    @eqx.filter_jit
    @eqx.filter_grad
    def gklx(flow, y):
        return forward_KLX_G(y, src, target, flow, key_perm, coeff_lambda=lam).mean()

    grads2 = gklx(nsf, y)
    leaves2 = [g for g in jax.tree_util.tree_leaves(grads2) if eqx.is_inexact_array(g)]
    finite2 = all(bool(jnp.isfinite(g).all()) for g in leaves2)
    nonzero2 = any(bool(jnp.any(g != 0)) for g in leaves2)
    check_true("forward_KLX_G filter_grad finite & nonzero", finite2 and nonzero2,
               f"{len(leaves2)} leaves, finite={finite2}, any-nonzero={nonzero2}")

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone loss tests passed")


if __name__ == "__main__":
    main()

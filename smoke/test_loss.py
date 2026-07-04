"""Standalone loss smoke test (jflows only, no zflows fixtures) —
run from the repo root as `~/.envs/jax/bin/python -m smoke.test_loss`.

For reverse/forward_KL_{F,G} and OT_loss:

    1. per-sample contract: every loss returns shape [N], and permuting
       the batch permutes the loss vector;
    2. identity flow (zeros()): reverse_KL_F == target(x) and
       forward_KL_F == source(y) exactly, per sample;
    3. F/G duality: reverse_KL_G(x, ., F.inv) == reverse_KL_F(x, ., F)
       and forward_KL_G(y, ., F.inv) == forward_KL_F(y, ., F);
    4. no bare aliases: the module exposes only the explicit _F/_G names;
    5. OT_loss with alpha_C = alpha_R = 0 recovers reverse_KL_F;
    6. autograd: filter_grad of the batch-mean is finite and nonzero.

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

from jflows.flow import NSF, OTFlow  # noqa: E402
from jflows.loss import (  # noqa: E402
    OT_loss,
    forward_KL_F,
    forward_KL_G,
    reverse_KL_F,
    reverse_KL_G,
)
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
    F = nsf.t()
    losses = {
        "reverse_KL_F": reverse_KL_F(x, target, F),
        "reverse_KL_G": reverse_KL_G(x, target, F),
        "forward_KL_F": forward_KL_F(y, target, F),
        "forward_KL_G": forward_KL_G(y, target, F),
    }
    for name, vec in losses.items():
        check_true(f"{name} shape", vec.shape == (N,), f"{vec.shape}")
    perm = jax.random.permutation(jax.random.key(2), N)
    check("permutation equivariance", reverse_KL_F(x[perm], target, F),
          losses["reverse_KL_F"][perm], tol=1e-12)

    log("identity flow (zeros)")
    Fz = nsf.zeros().t()
    check("reverse_KL_F == target(x)", reverse_KL_F(x, target, Fz), target(x), tol=1e-12)
    check("forward_KL_F == source(y)", forward_KL_F(y, target, Fz), target(y), tol=1e-12)

    log("F/G duality and aliases")
    check("reverse_KL_G(., F.inv) == reverse_KL_F(., F)",
          reverse_KL_G(x, target, F.inv), losses["reverse_KL_F"], tol=1e-11)
    check("forward_KL_G(., F.inv) == forward_KL_F(., F)",
          forward_KL_G(y, target, F.inv), losses["forward_KL_F"], tol=1e-11)
    import jflows.loss as loss_module
    check_true("no bare aliases exported",
               not hasattr(loss_module, "reverse_KL") and not hasattr(loss_module, "forward_KL"))

    log("OT_loss")
    otf = OTFlow(jax.random.key(3), dimension=3, hidden=16, layer=3, rank=4, nt=8)
    xo = jax.random.normal(jax.random.key(4), (8, 3))
    lot = OT_loss(xo, target, otf, alpha_C=0.7, alpha_R=0.4)
    check_true("shape", lot.shape == (8,), f"{lot.shape}")
    check("alpha_C = alpha_R = 0 recovers reverse_KL_F",
          OT_loss(xo, target, otf, alpha_C=0.0, alpha_R=0.0),
          reverse_KL_F(xo, target, otf.t()), tol=1e-9)

    log("autograd through the batch-mean")

    @eqx.filter_jit
    @eqx.filter_grad
    def gmean(flow, x):
        return reverse_KL_F(x, target, flow.t()).mean()

    grads = gmean(nsf, x)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_inexact_array(g)]
    finite = all(bool(jnp.isfinite(g).all()) for g in leaves)
    nonzero = any(bool(jnp.any(g != 0)) for g in leaves)
    check_true("filter_grad finite & nonzero", finite and nonzero,
               f"{len(leaves)} leaves, finite={finite}, any-nonzero={nonzero}")

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone loss tests passed")


if __name__ == "__main__":
    main()

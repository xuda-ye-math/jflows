"""Phase 1 parity check (jax side) — run with ~/.envs/jax/bin/python.

Loads smoke_tests/parity_phase1.npz (written by parity_dump_phase1.py),
rebuilds the jflows twins, transplants parameters, and asserts value /
gradient agreement. Runs on the default JAX backend (GPU when available;
set JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled
so concurrently running jobs stay undisturbed.
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

from jflows.core.nn import MLP, Linear, MaskedMLP  # noqa: E402
from jflows.core.numerics import bisection, gauss_legendre, rk4_fixed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase1_check.log")
NPZ = os.path.join(HERE, "parity_phase1.npz")

FAILURES = 0


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
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} shapes {got.shape}/{want.shape} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def set_linears(module, Ws, bs):
    for i, (W, b) in enumerate(zip(Ws, bs, strict=True)):
        module = eqx.tree_at(lambda m, i=i: m.linears[i].weight, module, jnp.asarray(W))
        module = eqx.tree_at(lambda m, i=i: m.linears[i].bias, module, jnp.asarray(b))
    return module


def main() -> None:
    log(f"START parity_check_phase1 | jax {jax.__version__} ({jax.default_backend()}) | npz={NPZ}")
    d = np.load(NPZ)
    key = jax.random.key(0)

    # ── gauss_legendre ──
    log("gauss_legendre")
    a, b = jnp.asarray(d["gl_a"]), jnp.asarray(d["gl_b"])
    f = lambda t: t**3 - 2.0 * t + 0.5
    area = gauss_legendre(f, a, b, n=3)
    ga, gb = jax.grad(lambda a, b: gauss_legendre(f, a, b, n=3).sum(), argnums=(0, 1))(a, b)
    check("area", area, d["gl_area"])
    check("grad_a", ga, d["gl_grad_a"], tol=1e-8)
    check("grad_b", gb, d["gl_grad_b"], tol=1e-8)

    # ── bisection ──
    log("bisection")
    y = jnp.asarray(d["bi_y"])
    alpha0 = jnp.asarray(d["bi_alpha"])

    def root(y, alpha):
        return bisection(lambda x: x**3 + alpha * x, y, 0.0, 3.0, n=60)

    x_root = root(y, alpha0)
    gy, galpha = jax.grad(lambda y, al: root(y, al).sum(), argnums=(0, 1))(y, alpha0)
    check("root", x_root, d["bi_root"])
    check("grad_y", gy, d["bi_grad_y"], tol=1e-8)
    check("grad_alpha", galpha, d["bi_grad_alpha"], tol=1e-8)

    # ── rk4_fixed ──
    log("rk4_fixed")
    x0 = jnp.asarray(d["rk4_x0"])
    fode = lambda t, x: jnp.tanh(x) * jnp.cos(t)
    xT = rk4_fixed(fode, x0, 0.0, 1.0, nt=16)
    gx0 = jax.grad(lambda x: rk4_fixed(fode, x, 0.0, 1.0, nt=16).sum())(x0)
    check("xT", xT, d["rk4_xT"], tol=1e-12)
    check("grad_x0", gx0, d["rk4_grad_x0"], tol=1e-12)

    # ── Linear (stacked) ──
    log("Linear (stacked)")
    lin = Linear(key, 5, 4, bias=True, stack=3)
    lin = eqx.tree_at(lambda m: m.weight, lin, jnp.asarray(d["lin_w"]))
    lin = eqx.tree_at(lambda m: m.bias, lin, jnp.asarray(d["lin_b"]))
    check("y", lin(jnp.asarray(d["lin_x"])), d["lin_y"], tol=1e-12)

    # ── MLP ──
    log("MLP")
    n = int(d["mlp_n"])
    mlp = MLP(key, 3, 2, hidden_features=(16, 8), activation=jax.nn.silu)
    mlp = set_linears(mlp, [d[f"mlp_w{i}"] for i in range(n)], [d[f"mlp_b{i}"] for i in range(n)])
    xm = jnp.asarray(d["mlp_x"])
    check("y", mlp(xm), d["mlp_y"], tol=1e-12)
    check("grad_x", jax.grad(lambda x: mlp(x).sum())(xm), d["mlp_grad_x"], tol=1e-10)

    # ── MaskedMLP: mask structure must match exactly, then values ──
    log("MaskedMLP")
    n = int(d["mm_n"])
    mmlp = MaskedMLP(key, d["mm_adjacency"], hidden_features=(32, 16), activation=jax.nn.silu)
    assert len(mmlp.linears) == n, f"layer count {len(mmlp.linears)} != {n}"
    for i in range(n):
        check(f"mask{i}", np.asarray(mmlp.linears[i].mask, dtype=bool), d[f"mm_mask{i}"].astype(bool), tol=0)
    mmlp = set_linears(mmlp, [d[f"mm_w{i}"] for i in range(n)], [d[f"mm_b{i}"] for i in range(n)])
    check("y", mmlp(jnp.asarray(d["mm_x"])), d["mm_y"], tol=1e-12)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all Phase 1 parity checks passed")


if __name__ == "__main__":
    main()

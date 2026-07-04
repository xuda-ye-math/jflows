"""Phase 2 check (jax side) — run with ~/.envs/jax/bin/python.

Two parts:
  1. torch parity: RQS (scalar + per-coord bound), CircularShift,
     MonotonicAffine against smoke_tests/parity_phase2.npz;
  2. self-contained verification for every transform: round-trip
     inv(f(x)) ≈ x, ladj vs an autodiff Jacobian (slogdet), event-dim
     bookkeeping through ComposedTransform.

Float64, on the default JAX backend (GPU when available; set
JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.core.transforms import (  # noqa: E402
    AdditiveTransform,
    AutoregressiveTransform,
    CircularShiftTransform,
    ComposedTransform,
    CouplingTransform,
    DependentTransform,
    FreeFormJacobianTransform,
    IdentityTransform,
    LULinearTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
    RotationTransform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase2_check.log")
NPZ = os.path.join(HERE, "parity_phase2.npz")

FAILURES = 0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check(name: str, got, want, tol: float = 1e-9) -> None:
    global FAILURES
    got, want = np.asarray(got), np.asarray(want)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    ok = got.shape == want.shape and err <= tol
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def check_vector_transform(name: str, t, x: jnp.ndarray, tol: float = 1e-9) -> None:
    """Round-trip + ladj-vs-slogdet for a vector (event-dim 1) transform on x: (d,)."""
    y, ladj = t.call_and_ladj(x)
    check(f"{name} call==call_and_ladj", t(x), y, tol=0.0 if tol == 0 else 1e-12)
    check(f"{name} roundtrip", t.inv(y), x, tol=tol)
    J = jax.jacfwd(t)(x)
    _, logdet = jnp.linalg.slogdet(J)
    check(f"{name} ladj vs slogdet", ladj, logdet, tol=tol)
    xb, ladj_inv = t.inv.call_and_ladj(y)
    check(f"{name} inv ladj", ladj_inv, -ladj, tol=tol)
    check(f"{name} inv roundtrip", xb, x, tol=tol)


def main() -> None:
    log(f"START parity_check_phase2 | jax {jax.__version__} ({jax.default_backend()})")
    d = np.load(NPZ)

    # ════════ part 1 — torch parity ════════
    log("torch parity: MonotonicRQS (per-coord bound)")
    t = MonotonicRQSTransform(
        jnp.asarray(d["rqs_widths"]), jnp.asarray(d["rqs_heights"]),
        jnp.asarray(d["rqs_derivs"]), bound=jnp.asarray(d["rqs_bound_vec"]), slope=1e-3,
    )
    x = jnp.asarray(d["rqs_x"])
    y, ladj = t.call_and_ladj(x)
    check("y", y, d["rqs_vec_y"], tol=1e-12)
    check("ladj", ladj, d["rqs_vec_ladj"], tol=1e-12)
    check("x_back", t.inv(y), d["rqs_vec_xback"], tol=1e-12)
    gradx = jax.grad(lambda x: t.call_and_ladj(x)[1].sum())(x)
    check("grad ladj wrt x", gradx, d["rqs_vec_gradx"], tol=1e-10)

    log("torch parity: MonotonicRQS (scalar bound)")
    t2 = MonotonicRQSTransform(
        jnp.asarray(d["rqs_widths"]), jnp.asarray(d["rqs_heights"]),
        jnp.asarray(d["rqs_derivs"]), bound=2.0, slope=1e-3,
    )
    y2, ladj2 = t2.call_and_ladj(jnp.asarray(d["rqs_sc_x"]))
    check("y", y2, d["rqs_sc_y"], tol=1e-12)
    check("ladj", ladj2, d["rqs_sc_ladj"], tol=1e-12)

    log("torch parity: CircularShift")
    c = CircularShiftTransform(bound=jnp.asarray(d["rqs_bound_vec"]))
    check("y", c(jnp.asarray(d["cs_x"])), d["cs_y"], tol=1e-12)

    log("torch parity: MonotonicAffine")
    m = MonotonicAffineTransform(jnp.asarray(d["aff_shift"]), jnp.asarray(d["aff_scale"]), slope=1e-3)
    ym, ladjm = m.call_and_ladj(jnp.asarray(d["aff_x"]))
    check("y", ym, d["aff_y"], tol=1e-12)
    check("ladj", ladjm, d["aff_ladj"], tol=1e-12)

    # ════════ part 2 — self-contained autodiff / round-trip checks ════════
    key = jax.random.key(7)
    kx, kw, kv, ka, klu, kf = jax.random.split(key, 6)
    dim = 5
    xv = jax.random.normal(kx, (dim,))

    log("Identity / Additive (scalar-wise)")
    check("identity roundtrip", IdentityTransform().inv(IdentityTransform()(xv)), xv, tol=0)
    add = AdditiveTransform(jax.random.normal(kw, (dim,)))
    check("additive roundtrip", add.inv(add(xv)), xv, tol=1e-15)
    check("additive ladj", add.log_abs_det_jacobian(xv, add(xv)), jnp.zeros(dim), tol=0)

    log("AutoregressiveTransform (affine meta, strictly-lower conditioner)")
    W1 = jnp.tril(jax.random.normal(kw, (dim, dim)), k=-1)
    W2 = jnp.tril(jax.random.normal(kv, (dim, dim)), k=-1)

    def meta(x):
        return DependentTransform(MonotonicAffineTransform(x @ W1.T, x @ W2.T), 1)

    ar = AutoregressiveTransform(meta, passes=dim)
    check_vector_transform("AR", ar, xv, tol=1e-10)
    J = jax.jacfwd(ar)(xv)
    check("AR jacobian lower-triangular", jnp.triu(J, k=1), jnp.zeros((dim, dim)), tol=0)

    log("CouplingTransform (affine meta)")
    mask = np.array([True, False, True, False, True])
    na, nb = int(mask.sum()), int((~mask).sum())
    V1 = jax.random.normal(kw, (nb, na))
    V2 = jax.random.normal(kv, (nb, na))

    def cmeta(x_a):
        return DependentTransform(MonotonicAffineTransform(x_a @ V1.T, x_a @ V2.T), 1)

    cp = CouplingTransform(cmeta, mask)
    check_vector_transform("coupling", cp, xv, tol=1e-10)

    log("FreeFormJacobianTransform (exact trace)")
    Wf = jax.random.normal(kf, (dim, dim)) * 0.4

    def drift(t, x):
        return jnp.tanh(x @ Wf.T) * jnp.cos(t)

    ffj = FreeFormJacobianTransform(drift, t0=0.0, t1=1.0, nt=64, exact=True)
    check_vector_transform("FFJ", ffj, xv, tol=1e-6)

    log("FreeFormJacobianTransform (Hutchinson, batched: finite + unbiased-ish)")
    ffj_h = FreeFormJacobianTransform(drift, t0=0.0, t1=1.0, nt=16, exact=False, key=jax.random.key(3))
    xb = jax.random.normal(kx, (64, dim))
    yh, lh = ffj_h.call_and_ladj(xb)
    ye, le = FreeFormJacobianTransform(drift, t0=0.0, t1=1.0, nt=16, exact=True).call_and_ladj(xb)
    check("hutchinson y == exact y", yh, ye, tol=1e-12)
    log(f"  hutchinson ladj mean err {float(jnp.abs(lh - le).mean()):.3f} (stochastic, finite={bool(jnp.isfinite(lh).all())})")

    log("RotationTransform")
    A = jax.random.normal(ka, (dim, dim))
    rot = RotationTransform(A)
    check_vector_transform("rotation", rot, xv, tol=1e-9)
    check("R orthogonal", rot.R @ rot.R.T, jnp.eye(dim), tol=1e-10)

    log("LULinearTransform")
    LU = jax.random.normal(klu, (dim, dim)) * 0.3 + jnp.eye(dim)
    lu = LULinearTransform(LU)
    check_vector_transform("lu", lu, xv, tol=1e-9)

    log("ComposedTransform event-dim bookkeeping (Additive ∘ AR ∘ Additive)")
    shift = jax.random.normal(kv, (dim,))
    comp = ComposedTransform(AdditiveTransform(-shift), ar, AdditiveTransform(shift))
    check_vector_transform("composed", comp, xv, tol=1e-10)
    check("composed domain_dim", np.array(comp.domain_dim), np.array(1), tol=0)

    log("ComposedTransform.inv structure")
    ci = comp.inv
    check("composed inv roundtrip", ci(comp(xv)), xv, tol=1e-10)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all Phase 2 checks passed")


if __name__ == "__main__":
    main()

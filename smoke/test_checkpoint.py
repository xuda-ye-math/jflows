"""Equinox checkpoint and approximate-CNF contract smoke tests.

Run from the repository root as ``python -m smoke.test_checkpoint``.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp

from jflows.flow import CNF, NCSF, NSF, OTFlow, RealNVP
from jflows.potential import Nlog_Gaussian
from jflows.train import train_reverse_KL_F
from jflows.utils import importance_weights_log


FAILURES = 0


def check(name, condition):
    global FAILURES
    ok = bool(condition)
    print(f"{name}: {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def close(a, b, tol=5e-4):
    return jnp.allclose(a, b, rtol=tol, atol=tol)


def roundtrip_checkpoint(name, flow, skeleton, x, path):
    eqx.tree_serialise_leaves(path, flow)
    restored = eqx.tree_deserialise_leaves(path, skeleton)
    y0, l0 = flow.call_and_ladj(x)
    y1, l1 = restored.call_and_ladj(x)
    check(f"{name} checkpoint map", jnp.array_equal(y0, y1))
    check(f"{name} checkpoint ladj", jnp.array_equal(l0, l1))


def main():
    key = jax.random.key(0)
    x = jax.random.normal(jax.random.key(1), (6, 3)) * 0.3
    bounds = ([-2.0] * 3, [2.0] * 3)
    factories = (
        ("NSF", lambda k: NSF(k, *bounds, bins=5, transforms=2, hidden_features=(8,))),
        ("NCSF", lambda k: NCSF(k, *bounds, bins=5, transforms=2, hidden_features=(8,))),
        ("RealNVP", lambda k: RealNVP(k, 3, transforms=2, hidden_features=(8,))),
        ("CNF exact", lambda k: CNF(k, 3, nt=2, exact=True, hidden_features=(8,))),
        ("CNF approximate", lambda k: CNF(k, 3, nt=2, exact=False, hidden_features=(8,))),
        ("OTFlow", lambda k: OTFlow(k, 3, hidden=8, layer=2, rank=3, nt=2)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, factory) in enumerate(factories):
            flow = factory(jax.random.fold_in(key, i))
            # NSF/NCSF order/mask arrays are serialized and may use a
            # different-key skeleton. RealNVP's coupling masks are static
            # tuples, so its skeleton intentionally reuses the constructor key.
            skeleton_key = jax.random.fold_in(
                key, i if name == "RealNVP" else i + 100
            )
            skeleton = factory(skeleton_key)
            roundtrip_checkpoint(name, flow, skeleton, x, Path(tmp) / f"{i}.eqx")
        for i, impl in enumerate(("rbg", "unsafe_rbg"), start=len(factories)):
            flow = CNF(
                jax.random.key(0, impl=impl), 3, nt=2, exact=False,
                hidden_features=(8,),
            )
            skeleton = CNF(
                jax.random.key(99, impl=impl), 3, nt=2, exact=False,
                hidden_features=(8,),
            )
            roundtrip_checkpoint(
                f"CNF approximate ({impl})", flow, skeleton, x,
                Path(tmp) / f"{i}.eqx",
            )

    # Approximate CNF estimates must be per-sample functions: permuting the
    # batch or evaluating it in chunks cannot change a row's value.
    flow = CNF(jax.random.key(8), 3, nt=2, exact=False, hidden_features=(8,))
    y, ladj = flow.call_and_ladj(x)
    perm = jnp.asarray([4, 0, 5, 2, 1, 3])
    yp, lp = flow.call_and_ladj(x[perm])
    inv = jnp.argsort(perm)
    check("approximate CNF permutation-equivariant map", close(y, yp[inv]))
    check("approximate CNF permutation-equivariant ladj", close(ladj, lp[inv]))
    yc1, lc1 = flow.call_and_ladj(x[:2])
    yc2, lc2 = flow.call_and_ladj(x[2:])
    # Different batch shapes can choose different accelerator kernels and
    # therefore differ by ordinary floating-point reduction rounding. The
    # old content-hashed probe differed at O(1); the fixed probe contract is
    # numerical equivalence at a tight float32 tolerance, not bit identity.
    check("approximate CNF chunk-invariant map",
          close(y, jnp.concatenate([yc1, yc2])))
    check("approximate CNF chunk-invariant ladj",
          close(ladj, jnp.concatenate([lc1, lc2])))

    keyed0 = flow.with_trace_key(jax.random.key(20)).call_and_ladj(x)[1]
    keyed1 = flow.with_trace_key(jax.random.key(21)).call_and_ladj(x)[1]
    check("approximate CNF runtime trace key refreshes the estimate",
          not close(keyed0, keyed1, tol=1e-7))
    source = Nlog_Gaussian([0.0] * 3, [1.0] * 3)
    target = Nlog_Gaussian([0.2] * 3, [1.3] * 3)
    lw1 = importance_weights_log(
        x, source, target, flow, "F", chunks=1, trace_key=jax.random.key(22)
    )
    lw3 = importance_weights_log(
        x, source, target, flow, "F", chunks=3, trace_key=jax.random.key(22)
    )
    check("approximate CNF keyed importance weights are chunk-invariant",
          close(lw1, lw3))

    # Packed training supplies a fresh trace key each optimizer step.
    trained, hist = train_reverse_KL_F(
        source.samples(jax.random.key(30), 64), source, target, flow.zeros(),
        batch_size=16, train_steps=2, lr=1e-3, mc_dt=1e-3, mc_steps=0,
    )
    check("approximate CNF packed training is finite",
          jnp.isfinite(hist).all()
          and all(jnp.isfinite(a).all() for a in jax.tree.leaves(trained)
                  if eqx.is_array(a)))

    if FAILURES:
        print(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    print("DONE — all checkpoint tests passed")


if __name__ == "__main__":
    main()

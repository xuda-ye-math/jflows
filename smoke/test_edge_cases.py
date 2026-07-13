"""Focused numerical-contract smoke tests for jflows edge cases.

Run from the repository root as ``python -m smoke.test_edge_cases``.
The cases here are deliberately small: they protect behavior that ordinary
end-to-end examples rarely encounter (one-dimensional autoregressive flows,
near-singular LU parameters, invalid weights/schedules, integer constructor
inputs, and non-finite clipping masks).
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from jflows.boltzmann import _bg_parameters, _fixed_schedule
from jflows.core.transforms import LULinearTransform
from jflows.flow import CNF, NCSF, NSF, OTFlow, RealNVP
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture, Nlog_Uniform
from jflows.train import _adam_step, _clip_global, _masked_mean, _masked_pair_mean
from jflows.utils import (compute_ESS, compute_ESS_log, coverage, langevin,
                          resample, stochastic_heun)


FAILURES = 0


def check(name, condition, detail=""):
    global FAILURES
    ok = bool(condition)
    print(f"{name}: {'OK' if ok else 'FAIL'}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES += 1


def raises(name, exc, fn):
    try:
        fn()
    except exc:
        check(name, True)
    else:
        check(name, False, f"expected {exc.__name__}")


def main():
    # A one-coordinate autoregressive conditioner is unconditional but still
    # has trainable spline biases; construction and inversion must be valid.
    x1 = jnp.asarray([[-0.7], [0.2], [0.8]])
    for cls, a, b in ((NSF, [-1], [1]), (NCSF, [-jnp.pi], [jnp.pi])):
        flow = cls(
            jax.random.key(0), a, b, bins=6, transforms=2,
            hidden_features=(8,),
        )
        y, ladj = flow.call_and_ladj(x1)
        check(f"{cls.__name__} d=1 finite", jnp.isfinite(y).all() & jnp.isfinite(ladj).all())
        check(f"{cls.__name__} d=1 round-trip", jnp.allclose(flow.inv(y), x1, atol=2e-5))
        yz, lz = flow.zeros().call_and_ladj(x1)
        check(f"{cls.__name__} d=1 zeros", jnp.allclose(yz, x1, atol=2e-6)
              & jnp.allclose(lz, 0.0, atol=2e-6))
    raises("RealNVP rejects d=1", ValueError,
           lambda: RealNVP(jax.random.key(0), dimension=1))
    for cls in (NSF, NCSF):
        raises(f"{cls.__name__} rejects equal bounds", ValueError,
               lambda cls=cls: cls(jax.random.key(0), [0], [0]))
        raises(f"{cls.__name__} rejects reversed bounds", ValueError,
               lambda cls=cls: cls(jax.random.key(0), [1], [0]))
        raises(f"{cls.__name__} rejects transforms=0", ValueError,
               lambda cls=cls: cls(jax.random.key(0), [-1], [1], transforms=0))
        raises(f"{cls.__name__} rejects slope=1", ValueError,
               lambda cls=cls: cls(jax.random.key(0), [-1], [1], slope=1.0))
    raises("CNF rejects nt=0", ValueError,
           lambda: CNF(jax.random.key(0), dimension=2, nt=0))
    raises("OTFlow rejects nt=0", ValueError,
           lambda: OTFlow(jax.random.key(0), dimension=2, nt=0))

    # Forward, inverse and reported determinant must describe the same
    # floored map even when a raw diagonal parameter is zero.
    lu = LULinearTransform(jnp.asarray([[0.0, 0.2], [0.3, 1.0]]))
    x = jnp.asarray([[0.4, -0.7], [1.2, 0.1]])
    y, ladj = lu.call_and_ladj(x)
    jac = jax.jacfwd(lambda q: lu(q[None, :])[0])(x[0])
    actual = jnp.linalg.slogdet(jac)[1]
    check("LU floored determinant agrees", jnp.allclose(ladj[0], actual, atol=2e-5),
          f"reported={float(ladj[0]):.6g}, actual={float(actual):.6g}")
    check("LU floored map round-trip", jnp.allclose(lu.inv(y), x, atol=2e-5))
    raises("LU rejects unsupported float16 inverse dtype", ValueError,
           lambda: LULinearTransform(jnp.zeros((2, 2), dtype=jnp.float16)))

    # Masked non-finite values must not leak through 0 * inf/NaN.
    values = jnp.asarray([1.0, jnp.inf, jnp.nan, 3.0])
    keep = jnp.asarray([True, False, False, True])
    perm = jnp.asarray([3, 2, 1, 0])
    check("masked mean drops non-finite entries", _masked_mean(values, keep) == 2.0)
    safe_values = jnp.where(keep, values, 0.0)
    pair = _masked_pair_mean(jnp.abs(safe_values - safe_values[perm]), keep, perm)
    check("masked pair mean stays finite", jnp.isfinite(pair))
    clipped = _clip_global({"g": jnp.asarray([jnp.inf, 2.0])}, 1.0)["g"]
    check("gradient clip sanitizes infinity", jnp.isfinite(clipped).all())
    huge_grad = _clip_global({"g": jnp.full(4, 1e38)}, 1.0)["g"]
    check("gradient clip handles finite norm overflow",
          jnp.isfinite(huge_grad).all()
          & jnp.allclose(jnp.linalg.vector_norm(huge_grad), 1.0, rtol=2e-5))
    below_huge_ceiling = _clip_global({"g": jnp.asarray([1e20])}, 1e30)["g"]
    check("overflow fallback is no-op below ceiling",
          jnp.array_equal(below_huge_ceiling, jnp.asarray([1e20])))
    check("zero gradient ceiling is accepted",
          jnp.array_equal(_clip_global({"g": jnp.asarray([3.0])}, 0.0)["g"],
                          jnp.asarray([0.0])))
    empty_clip = _clip_global(
        {"empty": jnp.empty((0,)), "g": jnp.asarray([3.0, 4.0])}, 1.0
    )
    check("gradient clip permits an empty leaf",
          empty_clip["empty"].shape == (0,)
          and jnp.allclose(jnp.linalg.vector_norm(empty_clip["g"]), 1.0))

    # A rejected non-finite step must not consume Adam's bias-correction
    # counter: the following valid update equals a clean first update.
    p0 = {"p": jnp.asarray([0.0])}
    m0 = {"p": jnp.asarray([0.0])}
    v0 = {"p": jnp.asarray([0.0])}
    u0 = jnp.asarray(0, dtype=jnp.int32)
    skipped = _adam_step(
        p0, m0, v0, {"p": jnp.asarray([jnp.nan])}, jnp.asarray(jnp.nan),
        u0, 0.1, float("inf"),
    )
    after_skip = _adam_step(
        *skipped[:3], {"p": jnp.asarray([1.0])}, jnp.asarray(1.0), skipped[3],
        0.1, float("inf"),
    )
    clean_first = _adam_step(
        p0, m0, v0, {"p": jnp.asarray([1.0])}, jnp.asarray(1.0), u0,
        0.1, float("inf"),
    )
    check("skipped Adam step does not advance bias correction",
          jnp.array_equal(after_skip[0]["p"], clean_first[0]["p"])
          and after_skip[3] == clean_first[3])
    overflowed = _adam_step(
        p0, m0, v0, {"p": jnp.asarray([1e30])}, jnp.asarray(1.0), u0,
        0.1, float("inf"),
    )
    after_overflow = _adam_step(
        *overflowed[:3], {"p": jnp.asarray([1.0])}, jnp.asarray(1.0),
        overflowed[3], 0.1, float("inf"),
    )
    check("derived Adam overflow is skipped without poisoning state",
          jnp.array_equal(overflowed[0]["p"], p0["p"])
          and overflowed[3] == 0
          and jnp.array_equal(after_overflow[0]["p"], clean_first[0]["p"]))

    # Degenerate diagnostics return deliberate values, and invalid weights
    # resample uniformly instead of selecting the final particle every time.
    check("zero weights ESS -> 0", compute_ESS(jnp.zeros(4)) == 0.0)
    tiny = jnp.asarray([1e-30, 2e-30, 3e-30, 4e-30])
    check("tiny weights preserve scale-invariant ESS",
          jnp.allclose(compute_ESS(tiny), compute_ESS(tiny * 1e30), rtol=2e-6))
    check("overflowing equal weights ESS -> 1",
          jnp.allclose(compute_ESS(jnp.full(4, 1e38)), 1.0, rtol=2e-6))
    check("all -inf log ESS -> 0", compute_ESS_log(jnp.full(4, -jnp.inf)) == 0.0)
    check("two +inf log weights -> 2/N",
          jnp.allclose(compute_ESS_log(jnp.asarray([jnp.inf, 0.0, jnp.inf, -1.0])), 0.5))
    samples = jnp.arange(8).reshape(4, 2)
    draw = resample(jax.random.key(7), samples, jnp.zeros(4), N=256)
    check("zero-weight resampling uses multiple particles", jnp.unique(draw[:, 0]).size > 1)
    huge_draw = resample(
        jax.random.key(8), samples, jnp.asarray([1e38, 1e38, 1e38, 1e38]), N=64
    )
    check("overflowing finite weights resample safely", jnp.isfinite(huge_draw).all()
          & (jnp.unique(huge_draw[:, 0]).size > 1))
    invalid = jnp.asarray([jnp.inf, jnp.nan, 0.0, 0.0])
    invalid_draw = resample(jax.random.key(9), samples, invalid, N=64)
    uniform_draw = resample(jax.random.key(9), samples, jnp.zeros(4), N=64)
    check("mixed +inf/NaN weights use invalid-vector fallback",
          jnp.array_equal(invalid_draw, uniform_draw))

    # Integer convenience inputs must produce inexact samples/energies.
    potentials = (
        Nlog_Uniform([0, 0], [1, 2]),
        Nlog_Gaussian([0, 1], [1, 2]),
        Nlog_Gaussian_Mixture([1, 2], [[0, 0], [1, 1]], [[1, 1], [2, 2]]),
    )
    check("integer potential inputs promote to floating point",
          all(jnp.issubdtype(p.samples(jax.random.key(1), 3).dtype, jnp.inexact)
              for p in potentials))
    raises("Gaussian rejects zero variance", ValueError,
           lambda: Nlog_Gaussian([0], [0]))
    raises("mixture rejects all-zero weights", ValueError,
           lambda: Nlog_Gaussian_Mixture([0, 0], [[0], [1]], [[1], [1]]))
    raises("uniform rejects complex bounds", ValueError,
           lambda: Nlog_Uniform([0j], [1 + 0j]))

    # Coverage needs a genuine k-nearest-neighbor radius (self excluded).
    raises("coverage rejects k=P", ValueError,
           lambda: coverage(jnp.zeros((2, 1)), jnp.zeros((5, 1)), k=5))
    raises("coverage rejects one-point reference", ValueError,
           lambda: coverage(jnp.zeros((2, 1)), jnp.zeros((1, 1)), k=1))
    gaussian = Nlog_Gaussian([0.0], [1.0])
    particles = jnp.zeros((2, 1))
    raises("Langevin rejects negative step", ValueError,
           lambda: langevin(jax.random.key(0), particles, gaussian, dt=-1.0))
    raises("Langevin rejects NaN step", ValueError,
           lambda: langevin(jax.random.key(0), particles, gaussian,
                            dt=float("nan")))
    raises("Langevin rejects negative taming", ValueError,
           lambda: langevin(jax.random.key(0), particles, gaussian,
                            taming=-1.0, adjust=False))
    raises("Heun rejects zero step", ValueError,
           lambda: stochastic_heun(jax.random.key(0), particles, gaussian, dt=0.0))

    # Invalid adaptive/fixed schedules should fail before compilation/training.
    raises("adaptive ladder rejects zero enlarge", ValueError,
           lambda: _bg_parameters("test", {"enlarge_factor": 0.0}))
    raises("adaptive ladder rejects subnormal no-progress enlarge", ValueError,
           lambda: _bg_parameters("test", {"enlarge_factor": 5e-324}))
    raises("adaptive ladder rejects NaN", ValueError,
           lambda: _bg_parameters("test", {"tau_ess": float("nan")}))
    raises("fixed ladder rejects NaN", ValueError,
           lambda: _fixed_schedule("test", [0.2, float("nan"), 1.0]))
    raises("fixed ladder rejects duplicate level", ValueError,
           lambda: _fixed_schedule("test", [0.2, 0.2, 1.0]))

    if FAILURES:
        print(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    print("DONE — all edge-case tests passed")


if __name__ == "__main__":
    main()

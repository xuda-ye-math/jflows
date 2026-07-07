"""Standalone potential smoke test (jflows only) —
run from the repo root as `~/.envs/jax/bin/python -m smoke_tests.test_potential`.

For Nlog_Uniform, Nlog_Gaussian, Nlog_Gaussian_Mixture, and potential_from:

    1. analytic energy values (Gaussian quadratic; mixture K=1 reduces to
       the single Gaussian up to an additive constant);
    2. .grad(x) vs central finite differences (and analytic Gaussian grad);
    3. jit compatibility of __call__ and .grad;
    4. sampler moments: mean / variance / mixture proportions against the
       analytic values within Monte-Carlo error bars.

Float64, on the default JAX backend (GPU when available; set
JAX_PLATFORMS=cpu to force CPU); GPU memory preallocation is disabled.
Exits nonzero on any failure.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import (  # noqa: E402
    Nlog_Gaussian,
    Nlog_Gaussian_Mixture,
    Nlog_Uniform,
    linear_combination,
    potential_from,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_potential.log")

FAILURES = 0
NSAMP = 200_000


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


def fd_grad(u, x: jnp.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of the batched energy."""
    x = np.asarray(x)
    g = np.zeros_like(x)
    for j in range(x.shape[1]):
        xp, xm = x.copy(), x.copy()
        xp[:, j] += h
        xm[:, j] -= h
        g[:, j] = (np.asarray(u(jnp.asarray(xp))) - np.asarray(u(jnp.asarray(xm)))) / (2 * h)
    return g


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_potential | jax {jax.__version__} | {jax.default_backend()}")
    key = jax.random.key(0)
    kx, ks = jax.random.split(key)
    N, d = 32, 3
    x = jax.random.normal(kx, (N, d)) * 1.5

    # ── Nlog_Uniform ──
    log("Nlog_Uniform")
    a = jnp.asarray([-2.0, -1.0, -3.0])
    b = jnp.asarray([2.0, 1.5, 3.0])
    uni = Nlog_Uniform(a, b)
    check("U == 0", uni(x), jnp.zeros(N), tol=0)
    check("grad == 0", uni.grad(x), jnp.zeros((N, d)), tol=0)
    s = uni.samples(ks, NSAMP)
    inside = bool(jnp.all((s >= a) & (s <= b)))
    log(f"  samples inside box: {inside} -> {'OK' if inside else 'FAIL'}")
    if not inside:
        global FAILURES
        FAILURES += 1
    mean_tol = 6 * float(jnp.max(b - a)) / (12 * NSAMP) ** 0.5
    check("sample mean", s.mean(axis=0), (a + b) / 2, tol=mean_tol)
    check("sample var (rel 5%)", s.var(axis=0) / ((b - a) ** 2 / 12), jnp.ones(d), tol=0.05)

    # ── Nlog_Gaussian ──
    log("Nlog_Gaussian")
    mean = jnp.asarray([0.5, -1.0, 2.0])
    var = jnp.asarray([0.7, 1.3, 0.4])
    gau = Nlog_Gaussian(mean, var)
    check("U analytic", gau(x), 0.5 * ((x - mean) ** 2 / var).sum(-1), tol=1e-14)
    check("grad analytic", gau.grad(x), (x - mean) / var, tol=1e-14)
    check("grad vs finite diff", gau.grad(x), fd_grad(gau, x), tol=1e-8)
    check("jit(U)", jax.jit(lambda x: gau(x))(x), gau(x), tol=1e-12)
    check("jit(grad)", jax.jit(lambda x: gau.grad(x))(x), gau.grad(x), tol=1e-12)
    s = gau.samples(ks, NSAMP)
    check("sample mean", s.mean(axis=0), mean,
          tol=6 * float(jnp.sqrt(var.max())) / NSAMP**0.5)
    check("sample var (rel 5%)", s.var(axis=0) / var, jnp.ones(d), tol=0.05)

    # ── Nlog_Gaussian_Mixture ──
    log("Nlog_Gaussian_Mixture")
    w = jnp.asarray([0.2, 0.5, 0.3])
    means = jnp.asarray([[-3.0, 0.0, 1.0], [2.0, 1.0, -1.0], [0.0, -2.0, 3.0]])
    variances = jnp.asarray([[0.4, 0.6, 0.5], [0.8, 0.3, 0.7], [0.5, 0.5, 0.5]])
    gmm = Nlog_Gaussian_Mixture(w, means, variances)
    check("grad vs finite diff", gmm.grad(x), fd_grad(gmm, x), tol=1e-8)
    check("jit(U)", jax.jit(lambda x: gmm(x))(x), gmm(x), tol=1e-12)

    # K=1 mixture reduces to the single Gaussian up to an additive constant
    g1 = Nlog_Gaussian_Mixture(jnp.asarray([2.5]), mean[None], var[None])
    diff = g1(x) - gau(x)
    check("K=1 == Gaussian + const", diff - diff[0], jnp.zeros(N), tol=1e-12)

    # unnormalized weights are normalized internally
    gmm2 = Nlog_Gaussian_Mixture(7.0 * w, means, variances)
    check("weight normalisation", gmm2(x), gmm(x), tol=1e-12)

    s = gmm.samples(ks, NSAMP)
    check("sample mean", s.mean(axis=0), (w[:, None] * means).sum(0), tol=0.05)
    # component proportions via nearest mixture center
    d2 = ((s[:, None, :] - means[None]) ** 2 / variances[None]).sum(-1)
    frac = jnp.bincount(jnp.argmin(d2, axis=1), length=3) / NSAMP
    check("component proportions", frac, w, tol=0.01)

    # ── potential_from ──
    log("potential_from")
    u = potential_from(lambda x: 0.5 * ((x - mean) ** 2 / var).sum(-1))
    check("U == Nlog_Gaussian", u(x), gau(x), tol=0)
    check("grad == Nlog_Gaussian", u.grad(x), gau.grad(x), tol=0)

    # ── linear_combination / potential algebra ──
    log("linear_combination algebra")
    U1 = Nlog_Gaussian([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    U2 = Nlog_Gaussian([1.0, -1.0, 0.5], [0.5, 2.0, 1.0])
    U3 = Nlog_Gaussian([-2.0, 0.5, 1.0], [1.5, 0.7, 0.9])

    V1 = 0.5 * U1 + 0.5 * U2
    V2 = 0.5 * U2 + 0.5 * U3
    W = V1 + V2
    check_true("V1+V2 has 3 identity-merged terms",
               len(W.terms) == 3
               and W.terms[0] is U1 and W.terms[1] is U2 and W.terms[2] is U3,
               f"{len(W.terms)} terms")
    check("merged coeffs (0.5, 1.0, 0.5)", W.coeffs, jnp.asarray([0.5, 1.0, 0.5]), tol=0)
    check("value", W(x), 0.5 * U1(x) + U2(x) + 0.5 * U3(x), tol=1e-13)
    check("grad (linearity)", W.grad(x),
          0.5 * U1.grad(x) + U2.grad(x) + 0.5 * U3.grad(x), tol=1e-13)

    U2b = Nlog_Gaussian([1.0, -1.0, 0.5], [0.5, 2.0, 1.0])  # equal params, distinct instance
    check_true("identity (not equality) merging",
               len((V1 + 0.5 * U2b).terms) == 3,
               f"{len((V1 + 0.5 * U2b).terms)} terms")

    check("factory == operators", linear_combination([U1, U2, U3], [0.5, 1.0, 0.5])(x), W(x), tol=0)
    check("uniform default", linear_combination([U1, U2])(x), 0.5 * (U1(x) + U2(x)), tol=1e-13)
    check("sum() builtin", sum([V1, V2])(x), W(x), tol=0)
    check("U - U == 0", (U1 - U1)(x), jnp.zeros(x.shape[0]), tol=0)
    check("2*U/2 == U", (2.0 * U1 / 2.0)(x), U1(x), tol=1e-15)
    check("scalar * nested flattens", (0.5 * (2.0 * U1) + U2)(x), U1(x) + U2(x), tol=1e-13)

    # bridge with a traced coefficient: one trace, retuned without recompile
    bridge = jax.jit(lambda c, x: linear_combination([U1, U3], [1 - c, c])(x))
    check("bridge c=0.3", bridge(0.3, x), 0.7 * U1(x) + 0.3 * U3(x), tol=1e-13)
    check("bridge c=0.9", bridge(0.9, x), 0.1 * U1(x) + 0.9 * U3(x), tol=1e-13)
    check("jit(W.grad)", jax.jit(lambda x: W.grad(x))(x), W.grad(x), tol=1e-12)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all standalone potential tests passed")


if __name__ == "__main__":
    main()

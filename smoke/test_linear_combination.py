"""Standalone linear_combination test with a visual interpolation (jflows only) —
run after installation from the repo root as
`python -m smoke.test_linear_combination`.

Builds the interpolation U_c = (1 - c) * U_uniform + c * U_gmm in 2D
with the potential algebra, checks it quantitatively at every point, and
renders the normalized densities exp(-U_c) as a sequence of heatmaps
(smoke/test_linear_combination.png) so the uniform -> Gaussian-
mixture transition is visible as a smooth concentration of mass.

This is the standard linear potential interpolation used by staged Boltzmann
generators: SMC / AIS traverse exactly such an interpolation, with the
coefficient c retuned per inner level (no recompile — the coefficients are an
array leaf).

Checks per interpolation point:
    1. algebra: U_c(x) == (1 - c) * U_uniform(x) + c * U_gmm(x);
    2. structure: two identity-merged terms, coeffs (1 - c, c);
    3. endpoints reproduce the pure potentials exactly;
    4. each panel's density normalizes to 1 on the plotting box.

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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jflows.potential import (  # noqa: E402
    Nlog_Gaussian_Mixture,
    Nlog_Uniform,
    linear_combination,
    potential_from,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_linear_combination.log")
PNG = os.path.join(HERE, "test_linear_combination.png")

FAILURES = 0
BOX = 4.0
LEVELS = [0.0, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1.0]


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


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START test_linear_combination | jax {jax.__version__} | {jax.default_backend()}")

    u0 = Nlog_Uniform([-BOX, -BOX], [BOX, BOX])
    u1 = Nlog_Gaussian_Mixture(
        weights=[0.4, 0.35, 0.25],
        mean=[[-2.0, -2.0], [2.0, 1.5], [0.5, -2.5]],
        variance=[[0.5, 0.3], [0.4, 0.6], [0.25, 0.25]],
    )

    n = 400
    g = jnp.linspace(-BOX, BOX, n)
    grid = jnp.stack(jnp.meshgrid(g, g, indexing="xy"), axis=-1).reshape(-1, 2)
    xs = jax.random.uniform(jax.random.key(0), (64, 2), minval=-BOX, maxval=BOX)

    log(f"interpolation U_c = (1 - c) * U_uniform + c * U_gmm, points c = {np.round(LEVELS, 3)}")
    densities = []
    for c in LEVELS:
        bridge = (1.0 - c) * u0 + c * u1
        check(f"c={c:.3f} algebra", bridge(xs), (1.0 - c) * u0(xs) + c * u1(xs), tol=1e-14)
        check_true(
            f"c={c:.3f} structure",
            len(bridge.terms) == 2
            and bridge.terms[0] is u0
            and bridge.terms[1] is u1
            and np.allclose(np.asarray(bridge.coeffs), [1.0 - c, c], atol=0),
            f"terms 2, coeffs {np.asarray(bridge.coeffs).round(3)}",
        )
        q = np.array(jnp.exp(-bridge(grid)).reshape(n, n))
        Z = np.trapezoid(np.trapezoid(q, np.asarray(g), axis=1), np.asarray(g))
        q /= Z
        integral = float(np.trapezoid(np.trapezoid(q, np.asarray(g), axis=1), np.asarray(g)))
        check(f"c={c:.3f} normalisation", np.asarray(integral), np.asarray(1.0), tol=1e-12)
        densities.append(q)

    check("endpoint c=0 == uniform", densities[0], np.full((n, n), 1.0 / (2 * BOX) ** 2), tol=1e-15)
    q_gmm = np.array(jnp.exp(-u1(grid)).reshape(n, n))
    q_gmm /= np.trapezoid(np.trapezoid(q_gmm, np.asarray(g), axis=1), np.asarray(g))
    check("endpoint c=1 == mixture", densities[-1], q_gmm, tol=1e-14)

    # Exact zero coefficients remain in the pytree so an interpolation can retune
    # without recompilation. They must neutralize nonfinite hard-wall values
    # without changing ordinary differentiation with respect to the scalar
    # coefficient.
    log("zero-coefficient nonfinite safety")
    hard_wall = potential_from(
        lambda x: jnp.where(x[:, 0] > 0.0, jnp.inf, x[:, 0] ** 2)
    )
    hard_x = jnp.asarray([[1.0, 0.0], [-1.0, 0.0]])
    cancelled = hard_wall - hard_wall
    check("hard-wall cancellation U - U", cancelled(hard_x), jnp.zeros(2), tol=0.0)
    check(
        "hard-wall cancellation under jit",
        jax.jit(lambda x: cancelled(x))(hard_x), jnp.zeros(2), tol=0.0,
    )
    endpoint = linear_combination([hard_wall, u1], [0.0, 1.0])
    check("hard-wall interpolation endpoint", endpoint(hard_x), u1(hard_x), tol=1e-14)
    active = linear_combination([hard_wall], [1.0])(hard_x)
    check_true(
        "nonzero hard-wall coefficient still propagates infinity",
        bool(jnp.isposinf(active[0])) and bool(jnp.isfinite(active[1])),
    )
    finite_term = potential_from(lambda x: 2.0 + x[:, 0])
    finite_value = finite_term(hard_x).sum()
    coeff_grad = jax.grad(
        lambda c: linear_combination([finite_term], [c])(hard_x).sum()
    )(jnp.asarray(0.0))
    check(
        "finite zero-coefficient derivative is preserved",
        coeff_grad, finite_value, tol=0.0,
    )
    check_true(
        "cancellation keeps one merged zero-coefficient term",
        len(cancelled.terms) == 1
        and cancelled.terms[0] is hard_wall
        and float(cancelled.coeffs[0]) == 0.0,
    )

    log(f"rendering heatmaps -> {PNG}")
    vmax = max(q.max() for q in densities)
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2), constrained_layout=True)
    for ax, c, q in zip(axes.ravel(), LEVELS, densities):
        im = ax.imshow(
            q, origin="lower", extent=[-BOX, BOX, -BOX, BOX],
            cmap="magma", vmin=0.0, vmax=vmax, interpolation="bilinear",
        )
        ax.set_title(f"c = {c:.3f}", fontsize=11)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        ax.tick_params(labelsize=8)
    fig.suptitle(
        r"$\exp(-U_c)$,  $U_c = (1-c)\,U_{\mathrm{uniform}} + c\,U_{\mathrm{mixture}}$",
        fontsize=13,
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=30)
    cbar.set_label("normalized density", fontsize=10)
    fig.savefig(PNG, dpi=150)
    plt.close(fig)
    check_true("figure written", os.path.isfile(PNG) and os.path.getsize(PNG) > 10000,
               f"{os.path.getsize(PNG)} bytes")

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log(f"DONE — all linear_combination tests passed; figure at {PNG}")


if __name__ == "__main__":
    main()

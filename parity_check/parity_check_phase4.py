"""Phase 4a parity check (jax side) — run from the repo root as
`~/.envs/jax/bin/python -m smoke_tests.parity_check_phase4`.

Loads smoke_tests/parity_phase4.npz (written by parity_dump_phase4.py),
builds the Nlog_* twins, and asserts energy / gradient agreement.
Float32 end-to-end, matching the zflows potentials' float32 buffers
(float64 analytic checks live in test_potential.py). Runs on the default
JAX backend (GPU when available; set JAX_PLATFORMS=cpu to force CPU);
GPU memory preallocation is disabled.
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
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase4_check.log")
NPZ = os.path.join(HERE, "parity_phase4.npz")

FAILURES = 0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check(name: str, got, want, tol: float = 2e-5) -> None:
    global FAILURES
    got, want = np.asarray(got), np.asarray(want)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    ok = got.shape == want.shape and err <= tol
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def main() -> None:
    log(f"START parity_check_phase4 | jax {jax.__version__} ({jax.default_backend()}) | npz={NPZ}")
    d = np.load(NPZ)
    x = jnp.asarray(d["x"])

    log("Nlog_Uniform")
    uni = Nlog_Uniform(jnp.asarray(d["uni_a"]), jnp.asarray(d["uni_b"]))
    check("U", uni(x), d["uni_U"])
    check("grad (constant energy)", uni.grad(x), np.zeros_like(d["x"]))

    log("Nlog_Gaussian")
    gau = Nlog_Gaussian(jnp.asarray(d["gau_mean"]), jnp.asarray(d["gau_var"]))
    check("U", gau(x), d["gau_U"])
    check("grad", gau.grad(x), d["gau_grad"])

    log("Nlog_Gaussian_Mixture")
    gmm = Nlog_Gaussian_Mixture(
        jnp.asarray(d["gmm_w"]), jnp.asarray(d["gmm_mean"]), jnp.asarray(d["gmm_var"])
    )
    check("U", gmm(x), d["gmm_U"])
    check("grad", gmm.grad(x), d["gmm_grad"])

    log("linear_combination (vs zflows Linear_Combination)")
    lin = 0.7 * gau + (-0.3) * gmm
    check("U", lin(x), d["lc_U"])
    check("grad", lin.grad(x), d["lc_grad"])
    nested = 0.5 * (2.0 * gau) + gmm  # flat in jflows, nested in zflows
    check("nested-flattened U", nested(x), d["lcn_U"])

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all Phase 4 parity checks passed")


if __name__ == "__main__":
    main()

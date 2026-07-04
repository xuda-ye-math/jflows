"""Phase 6a parity check (jax side) — run from the repo root as
`~/.envs/jax/bin/python -m parity_check.parity_check_phase6`.

Compares jflows compute_ESS / compute_ESS_log and the
importance_weights_{F,G} (+log) family (on a transplanted NSF) against
the zflows values in parity_check/parity_phase6.npz. Extended as the
remaining Phase 6 modules land. Float64, default backend.
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
from jflows.potential import Nlog_Gaussian  # noqa: E402
from jflows.utils import (  # noqa: E402
    compute_ESS,
    compute_ESS_log,
    importance_weights_F,
    importance_weights_log_F,
    importance_weights_log_G,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase6_check.log")
NPZ = os.path.join(HERE, "parity_phase6.npz")

FAILURES = 0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check(name: str, got, want, tol: float = 1e-14) -> None:
    global FAILURES
    got, want = np.asarray(got), np.asarray(want)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    ok = got.shape == want.shape and err <= tol
    log(f"  {name}: max|Δ|={err:.3e} tol={tol:.1e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def main() -> None:
    log(f"START parity_check_phase6 | jax {jax.__version__} ({jax.default_backend()})")
    d = np.load(NPZ)

    log("compute_ESS / compute_ESS_log")
    w, logw = jnp.asarray(d["w"]), jnp.asarray(d["logw"])
    check("ESS", compute_ESS(w), d["ess"])
    check("ESS_log", compute_ESS_log(logw), d["ess_log"])
    check("ESS(exp(logw))", compute_ESS(jnp.exp(logw)), d["ess_from_logw"])

    log("importance weights on transplanted NSF")
    nsf = NSF(jax.random.key(0), jnp.asarray([-2.0, -1.5, -3.0]),
              jnp.asarray([2.0, 1.5, 3.0]), bins=8, slope=1e-3, transforms=3,
              randmask=False, hidden_features=(32, 16))
    for i in range(3):
        n = int(d[f"nsf_maf{i}_n"])
        wheres, values = [], []
        for j in range(n):
            wheres.append(lambda f, i=i, j=j: f._maf[i].hyper.linears[j].weight)
            values.append(jnp.asarray(d[f"nsf_maf{i}_w{j}"]))
            wheres.append(lambda f, i=i, j=j: f._maf[i].hyper.linears[j].bias)
            values.append(jnp.asarray(d[f"nsf_maf{i}_b{j}"]))
        nsf = eqx.tree_at(lambda f: [w_(f) for w_ in wheres], nsf, values)

    source = Nlog_Gaussian(jnp.asarray(d["src_mean"]), jnp.asarray(d["src_var"]))
    target = Nlog_Gaussian(jnp.asarray(d["tgt_mean"]), jnp.asarray(d["tgt_var"]))
    x = jnp.asarray(d["x"])
    F = nsf.t()
    check("iw_log_F", importance_weights_log_F(x, source, target, F), d["iw_log_F"], tol=1e-9)
    check("iw_log_G", importance_weights_log_G(x, source, target, F), d["iw_log_G"], tol=1e-9)
    check("iw_F (linear)", importance_weights_F(x, source, target, F), d["iw_F"], tol=1e-12)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all Phase 6a parity checks passed")


if __name__ == "__main__":
    main()

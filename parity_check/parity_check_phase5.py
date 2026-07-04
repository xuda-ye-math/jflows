"""Phase 5 parity check (jax side) — run from the repo root as
`~/.envs/jax/bin/python -m smoke_tests.parity_check_phase5`.

Transplants the zflows NSF / OTFlow from smoke_tests/parity_phase5.npz
and asserts, for every loss: the jflows per-sample vector has shape [N]
and its batch-mean equals the zflows scalar. Float64, default backend.
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
LOG = os.path.join(HERE, "parity_phase5_check.log")
NPZ = os.path.join(HERE, "parity_phase5.npz")

FAILURES = 0
d = np.load(NPZ)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check_loss(name: str, vec, scalar, N: int, tol: float = 1e-9) -> None:
    global FAILURES
    vec = np.asarray(vec)
    err = abs(float(vec.mean()) - float(scalar))
    ok = vec.shape == (N,) and err <= tol
    log(f"  {name}: shape {vec.shape} (want ({N},)), |mean-zflows|={err:.3e} tol={tol:.1e} "
        f"-> {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES += 1


def main() -> None:
    log(f"START parity_check_phase5 | jax {jax.__version__} ({jax.default_backend()})")
    key = jax.random.key(0)

    nsf = NSF(key, jnp.asarray([-2.0, -1.5, -3.0]), jnp.asarray([2.0, 1.5, 3.0]),
              bins=8, slope=1e-3, transforms=3, randmask=False, hidden_features=(32, 16))
    for i in range(3):
        n = int(d[f"nsf_maf{i}_n"])
        wheres, values = [], []
        for j in range(n):
            wheres.append(lambda f, i=i, j=j: f._maf[i].hyper.linears[j].weight)
            values.append(jnp.asarray(d[f"nsf_maf{i}_w{j}"]))
            wheres.append(lambda f, i=i, j=j: f._maf[i].hyper.linears[j].bias)
            values.append(jnp.asarray(d[f"nsf_maf{i}_b{j}"]))
        nsf = eqx.tree_at(lambda f: [w(f) for w in wheres], nsf, values)

    target = Nlog_Gaussian(jnp.asarray(d["tgt_mean"]), jnp.asarray(d["tgt_var"]))
    x, y = jnp.asarray(d["x"]), jnp.asarray(d["y"])
    N = x.shape[0]

    log("KL losses on transplanted NSF")
    F = nsf.t()
    check_loss("reverse_KL_F", reverse_KL_F(x, target, F), d["rkl_F"], N)
    check_loss("reverse_KL_G", reverse_KL_G(x, target, F), d["rkl_G"], N)
    check_loss("forward_KL_F", forward_KL_F(y, target, F), d["fkl_F"], N)
    check_loss("forward_KL_G", forward_KL_G(y, target, F), d["fkl_G"], N)

    log("OT_loss on transplanted OTFlow")
    otf = OTFlow(key, dimension=3, hidden=16, layer=3, rank=4, nt=8)
    nres = int(d["otf_nres"])
    wheres = [lambda f: f._ot.phi.A, lambda f: f._ot.phi.c.weight, lambda f: f._ot.phi.w.weight]
    values = [jnp.asarray(d["otf_A"]), jnp.asarray(d["otf_cw"]), jnp.asarray(d["otf_ww"])]
    for j in range(nres):
        wheres.append(lambda f, j=j: f._ot.phi.N.layers[j].weight)
        values.append(jnp.asarray(d[f"otf_res_w{j}"]))
        wheres.append(lambda f, j=j: f._ot.phi.N.layers[j].bias)
        values.append(jnp.asarray(d[f"otf_res_b{j}"]))
    otf = eqx.tree_at(lambda f: [w(f) for w in wheres], otf, values)
    xo = jnp.asarray(d["xo"])
    check_loss("OT_loss", OT_loss(xo, target, otf, alpha_C=0.7, alpha_R=0.4),
               d["ot_loss"], xo.shape[0])

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all Phase 5 parity checks passed")


if __name__ == "__main__":
    main()

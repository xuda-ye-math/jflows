"""Standalone chunking smoke test (jflows only) — run after installation
from the repo root as `python -m smoke.test_chunk`.

For the chunked full-set evaluations (`importance_weights_log` and the
eager per-chunk wrapper used by the Boltzmann drivers):

    1. correctness: the eager per-chunk loop reproduces the single
       full-set call to float32 precision (different compiled programs
       reassociate reductions, ~1e-5 relative), for both flow
       directions ('F' and 'G'), for even and uneven chunk counts;
    2. memory: the eager per-chunk loop MUST reduce the peak device
       memory — its high-water mark stays well below the single-jit
       full-set call on the heavy 'G' (autoregressive inverse) path;
    3. negative control: chunking INSIDE one jit does not bound the
       peak (the XLA scheduler overlaps the data-independent chunk
       subgraphs), which is why the wrapper chunks eagerly.

Float32 (the drivers' working precision), on the default JAX backend
(GPU when available); GPU memory preallocation is disabled so the
allocator's high-water mark tracks the real working set. Exits nonzero
on any failure.
"""

import math
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jflows.flow import NCSF  # noqa: E402
from jflows.potential import Nlog_Uniform, potential_from  # noqa: E402
from jflows.boltzmann import (  # noqa: E402
    _chunked_log_weights,
)
from jflows.utils import importance_weights_log  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_chunk.log")

FAILURES = 0
NSAMP = 200000
D = 32
LIM = math.pi


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


def peak_gib():
    stats = jax.local_devices()[0].memory_stats()
    return None if stats is None else stats["peak_bytes_in_use"] / 2**30


def main() -> None:
    open(LOG, "w").close()
    measured = peak_gib() is not None
    sample_count = NSAMP if measured else 20000
    log(f"START test_chunk | backend {jax.default_backend()} | "
        f"N={sample_count} D={D} float32")

    u0 = Nlog_Uniform(a=[-LIM] * D, b=[LIM] * D)
    u1 = potential_from(lambda x: -(jnp.cos(x - jnp.roll(x, 1, axis=-1))).sum(-1))
    flow = NCSF(jax.random.key(0), a=[-LIM] * D, b=[LIM] * D, bins=8,
                transforms=4, hidden_features=(64, 64))
    x = u0.samples(jax.random.key(2), sample_count)
    x = jax.block_until_ready(x)
    p0 = peak_gib()
    if measured:
        log(f"baseline peak after data: {p0:.3f} GiB")

    # 1 — correctness: eager chunks == single full-set call, both directions,
    # even (8 | 200k) and uneven (7) chunk counts, on a small subset
    xs = x[:9973]                       # prime size -> every split is uneven
    for type_ in ("F", "G"):
        ref = jax.block_until_ready(
            importance_weights_log(xs, u0, u1, flow, type_))
        for c in (2, 7):
            got = jax.block_until_ready(
                _chunked_log_weights(
                    xs, u0, u1, flow, type_, chunks=c
                ))
            check(f"eager chunks={c} == full call (type {type_})", got, ref, tol=1e-3)

    # 2 — memory: the eager per-chunk loop runs FIRST (smallest working
    # set), so every later high-water mark is attributable to the in-jit
    # calls it is compared against
    lw_eager = jax.block_until_ready(_chunked_log_weights(
        x, u0, u1, flow, "G", chunks=8
    ))
    p_eager = peak_gib()
    if measured:
        eager_delta = p_eager - p0
        log(f"peak after EAGER chunks=8  'G' on {sample_count}: "
            f"{p_eager:.3f} GiB (delta {eager_delta:.3f})")

    # negative control: the SAME chunk count inside one jit — the XLA
    # scheduler overlaps the chunk subgraphs, so the peak must land far
    # above the eager loop's
    full_jit = eqx.filter_jit(importance_weights_log)
    lw_injit = jax.block_until_ready(full_jit(x, u0, u1, flow, "G", chunks=8))
    p_injit = peak_gib()
    check("eager == in-jit values", lw_eager, lw_injit, tol=1e-3)
    if measured:
        injit_delta = p_injit - p0
        log(f"peak after IN-JIT chunks=8 'G' on {sample_count}: "
            f"{p_injit:.3f} GiB (delta {injit_delta:.3f})")
        check_true("eager chunking reduces peak memory",
                   injit_delta >= 2.0 * eager_delta,
                   f"in-jit/eager peak-delta ratio = "
                   f"{injit_delta / max(eager_delta, 1e-9):.1f} (need >= 2)")
    else:
        log("  peak-memory assertion skipped: backend exposes no memory stats")

    # 3 — the unchunked full-set call agrees in value (its peak is
    # already covered by the control above)
    lw_full = jax.block_until_ready(full_jit(x, u0, u1, flow, "G"))
    check("in-jit chunks=8 == unchunked full call", lw_injit, lw_full, tol=1e-3)

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log("DONE — all chunking tests passed")


if __name__ == "__main__":
    main()

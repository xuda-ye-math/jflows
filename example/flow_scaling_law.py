"""Flow map scaling law — forward vs inverse latency across dimension.

jflows has no torch.compile, so this measures the pure jitted map latency
of the two fused maps of an NSF across dimension and conditioner width:

    forward + ladj:   flow.call_and_ladj(x)   -> (y, log|det J_F|)
    inverse + ladj:   flow.inv_and_ladj(y)    -> (x, log|det J_{F^-1}|)

Each map is `eqx.filter_jit`-compiled once per dimension (warmup absorbs the
compile), then timed over a fixed batch. The inverse of a MAF-style spline flow
is autoregressive — d sequential coordinate solves — while the forward is a
single parallel pass, so the inverse latency grows with dimension where the
forward stays flat; this sweep traces that scaling from d = 4 to d = 128.

Run from the repo root:  conda activate jflows && PYTHONPATH=/mnt/projects/jflows python -m example.flow_scaling_law
"""

import csv
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax

from jflows.flow import NSF

HERE = Path(__file__).resolve().parent
LOG = HERE / "flow_scaling_law.log"

# dimension sweep
DIMS = [4, 8, 16, 32, 64, 128]  # feature dimensions swept

# NSF architecture (bins / transforms fixed; conditioner width swept)
NSF_LIM: float = 3.0    # box half-width; the flow acts on [-NSF_LIM, NSF_LIM]^d
BINS: int = 12          # rational-quadratic spline bins per transform
TRANSFORMS: int = 4     # autoregressive transforms stacked in the flow
HIDDEN_FEATURES_GRID = [(64, 64), (128, 128), (256, 256)]  # conditioner MLP widths swept

# timing
BATCH: int = 2000       # fixed batch size pushed through each map
WARMUP: int = 20        # untimed calls to absorb the jit compile + retrace
TIMED: int = 50         # timed calls; latency = mean wall time per call


@eqx.filter_jit
def forward_map(flow, x):
    """Fused forward map (y, log|det J_F|)."""
    return flow.call_and_ladj(x)


@eqx.filter_jit
def inverse_map(flow, y):
    """Fused inverse map (x, log|det J_{F^-1}|)."""
    return flow.inv_and_ladj(y)


def time_map(fn, flow, pts):
    """Mean wall-clock ms per jitted map call over the fixed batch."""
    for _ in range(WARMUP):
        out = fn(flow, pts)
    jax.block_until_ready(out)             # finish warmup (compile + retrace)
    t0 = time.perf_counter()
    for _ in range(TIMED):
        out = fn(flow, pts)
    jax.block_until_ready(out)             # one sync after all timed calls
    return (time.perf_counter() - t0) * 1000.0 / TIMED


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main():
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START flow_scaling_law | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"dims={DIMS} hidden_grid={HIDDEN_FEATURES_GRID} bins={BINS} transforms={TRANSFORMS} "
        f"BATCH={BATCH} warmup={WARMUP} timed={TIMED}")

    results = {}
    for hf in HIDDEN_FEATURES_GRID:
        hfs = "x".join(map(str, hf))
        for d in DIMS:
            flow = NSF(jax.random.key(0), a=[-NSF_LIM] * d, b=[NSF_LIM] * d,
                       bins=BINS, transforms=TRANSFORMS, hidden_features=hf)
            x = jax.random.normal(jax.random.fold_in(jax.random.key(1), d), (BATCH, d))       # source points
            y = jax.random.normal(jax.random.fold_in(jax.random.key(2), d), (BATCH, d)) * 1.5  # points to invert
            fwd_ms = time_map(forward_map, flow, x)
            inv_ms = time_map(inverse_map, flow, y)
            results[(hfs, d)] = (fwd_ms, inv_ms)
            log(f"[hf={hfs:<7} d={d:>3}] forward {fwd_ms:8.3f} ms   inverse {inv_ms:8.3f} ms   "
                f"(inv/fwd = {inv_ms / fwd_ms:5.1f})")

    # tidy CSV: one row per (hidden_features, dimension) cell
    csv_path = HERE / "flow_scaling_law.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hidden_features", "dimension", "forward_ms", "inverse_ms", "inv_over_fwd"])
        for hf in HIDDEN_FEATURES_GRID:
            hfs = "x".join(map(str, hf))
            for d in DIMS:
                fwd_ms, inv_ms = results[(hfs, d)]
                writer.writerow([hfs, d, f"{fwd_ms:.3f}", f"{inv_ms:.3f}", f"{inv_ms / fwd_ms:.2f}"])
    log(f"csv saved -> {csv_path}")
    log(f"DONE — flow scaling law complete ({len(HIDDEN_FEATURES_GRID)} x {len(DIMS)} cells)")


if __name__ == "__main__":
    main()

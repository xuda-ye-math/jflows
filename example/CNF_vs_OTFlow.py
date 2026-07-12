"""Multi-well (8-mode) reverse-KL benchmark: CNF vs OTFlow across dimension.

Both continuous flows are trained by the SAME objective — plain reverse KL
through `train_reverse_KL_F` (no rejuvenation, `MC_ITERS = 0`) — against the
same target, so the comparison isolates the one variable that differs:
CNF's free-form MLP velocity with an O(d) augmented-Jacobian trace vs
OTFlow's potential-gradient velocity with a closed-form trace. Each cell
reports the importance-sampling effective sample size (ESS) of the trained
proposal.

Target potential (factorized, `dimension`-dependent):
    - the first N_WELL coordinates are symmetric DOUBLE WELLS,
    - the remaining coordinates are standard Gaussian (quadratic).
N_WELL independent double wells give 2**N_WELL modes; the Gaussian tail only
raises the dimension. The wells are deliberately shallow (a low barrier) so
mode-seeking reverse KL does not collapse onto a subset of the modes.

Run after installation from the repo root:  python -m example.CNF_vs_OTFlow
"""

import csv
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax

from jax import Array
from jflows.flow import CNF, OTFlow
from jflows.potential import Nlog_Gaussian, Potential
from jflows.train import Monitor, train_reverse_KL_F
from jflows.utils import compute_ESS_log, importance_weights_log

HERE = Path(__file__).resolve().parent
LOG = HERE / "CNF_vs_OTFlow.log"

# dimension sweep and multi-well target
DIMS = [4, 8, 16, 32, 64, 128]  # feature dimensions swept (modes stay fixed)
N_WELL: int = 3        # double-well coordinates -> 2**N_WELL = 8 modes
WELL_SEP: float = 1.5  # double-well minima at +/- WELL_SEP (energy 0)
WELL_BARRIER: float = 1.5  # barrier height at the origin (shallow on purpose)

# continuous-flow architecture (shared budget: same nt, same ODE-net width)
NT: int = 12           # fixed RK4 integration steps (identical for both flows)
HIDDEN: int = 64       # ODE-net width (CNF MLP / OTFlow ResNet)
FREQUENCY: int = 4     # CNF time-embedding frequencies
LAYER: int = 3         # OTFlow ResNet depth
RANK: int = 10         # OTFlow low-rank quadratic (clamped to d + 1 per cell)

# training parameters (plain reverse KL)
N_VALID: int = 40000   # fixed source pool per dimension (subsampled per Adam step)
N_BATCH: int = 512     # source samples per Adam step
STEPS: int = 1000      # Adam steps per cell (one compiled call)
LR: float = 2e-3       # Adam learning rate
MONITOR_EVERY: int = 100  # print loss + batch ESS every MONITOR_EVERY steps
MC_STEP: float = 1e-3  # Langevin step size (unused: MC_ITERS = 0)
MC_ITERS: int = 0      # 0 -> no Langevin rejuvenation (plain reverse KL)
CHECKPOINT: bool = True  # rematerialize the CNF exact-trace forward pass in the backward
CHUNK: int = 4         # split the full-set ESS evaluation to bound peak memory

# evaluation
N_ESS: int = 20000     # held-out source set for the final ESS estimate


class Multi_Well(Potential):
    """First `n_well` coordinates are symmetric double wells, the rest Gaussian.

        U(x) = sum_{i < n_well} barrier * ((x_i / sep)^2 - 1)^2
             + sum_{i >= n_well} (1/2) x_i^2

    Each double well has minima at +/- sep (energy 0) and a barrier of height
    `barrier` at the origin, so the joint has 2**n_well modes; the extra
    Gaussian coordinates add no modes and only raise the dimension.
    """

    dimension: int = eqx.field(static=True)
    n_well: int = eqx.field(static=True)
    sep: float = eqx.field(static=True)
    barrier: float = eqx.field(static=True)

    def __init__(self, dimension: int, n_well: int = N_WELL,
                 sep: float = WELL_SEP, barrier: float = WELL_BARRIER):
        self.dimension = dimension
        self.n_well = n_well
        self.sep = sep
        self.barrier = barrier

    def __call__(self, x: Array) -> Array:   # Array [N, d] -> Array [N]
        dw = x[:, : self.n_well]
        gauss = x[:, self.n_well :]
        u_dw = (self.barrier * ((dw / self.sep) ** 2 - 1.0) ** 2).sum(axis=-1)
        u_gauss = 0.5 * (gauss ** 2).sum(axis=-1)
        return u_dw + u_gauss


def new_cnf(key, d):
    """FFJORD CNF at the identity map (exact O(d) augmented-Jacobian trace)."""
    return CNF(key, dimension=d, frequency=FREQUENCY, nt=NT, exact=True,
               hidden_features=(HIDDEN, HIDDEN)).zeros()


def new_otflow(key, d):
    """OT-Flow at the identity map (closed-form trace; rank clamped to d + 1)."""
    return OTFlow(key, dimension=d, hidden=HIDDEN, layer=LAYER,
                  rank=min(RANK, d + 1), nt=NT).zeros()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START CNF_vs_OTFlow | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"dims={DIMS} NT={NT} hidden={HIDDEN} layer={LAYER} steps={STEPS} "
        f"batch={N_BATCH} lr={LR} N_VALID={N_VALID} N_ESS={N_ESS} "
        f"well(n={N_WELL}, sep={WELL_SEP}, barrier={WELL_BARRIER})")

    results = {"CNF": {}, "OTFlow": {}}
    timings = {"CNF": {}, "OTFlow": {}}
    for d in DIMS:
        u0 = Nlog_Gaussian(mean=[0.0] * d, variance=[1.0] * d)   # standard Gaussian source
        u1 = Multi_Well(dimension=d)                              # 8-mode multi-well target
        x_valid = u0.samples(jax.random.fold_in(jax.random.key(0), d), N_VALID)  # shared pool
        x_ess = u0.samples(jax.random.fold_in(jax.random.key(1), d), N_ESS)      # held-out eval set
        for name, build, fkey in (
            ("CNF", new_cnf, jax.random.fold_in(jax.random.key(2), d)),
            ("OTFlow", new_otflow, jax.random.fold_in(jax.random.key(3), d)),
        ):
            flow = build(fkey, d)
            t0 = time.perf_counter()
            flow, _ = train_reverse_KL_F(
                x_valid, u0, u1, flow,
                n_batch=N_BATCH, steps=STEPS, lr=LR,
                mc_step=MC_STEP, mc_iters=MC_ITERS, checkpoint=CHECKPOINT,
                monitor=Monitor(MONITOR_EVERY, f"[d={d:>2} {name:<6}] ", log),
            )
            flow = jax.block_until_ready(flow)   # real wall time: wait for the device
            secs = time.perf_counter() - t0
            log_w = importance_weights_log(x_ess, u0, u1, flow, type="F", chunk=CHUNK)
            ess = float(compute_ESS_log(log_w))
            results[name][d] = ess
            timings[name][d] = secs
            log(f"[d={d:>2} {name:<6}] final ESS = {ess:.4f}   ({secs:.1f}s, N_ESS={N_ESS})")

    # tidy CSV: one row per flow x dimension (flow, dimension, ess, train_seconds)
    csv_path = HERE / "CNF_vs_OTFlow.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["flow", "dimension", "ess", "train_seconds"])
        for name in ("CNF", "OTFlow"):
            for d in DIMS:
                writer.writerow([name, d, f"{results[name][d]:.4f}", f"{timings[name][d]:.2f}"])
    log(f"csv saved -> {csv_path}")
    log(f"DONE — CNF vs OTFlow sweep complete ({len(DIMS)} dims x 2 flows)")


if __name__ == "__main__":
    main()

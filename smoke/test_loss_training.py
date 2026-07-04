"""Standalone loss-training smoke test (jflows only) — run from the repo
root as `~/.envs/jax/bin/python -m smoke_tests.test_loss_training`.

Trains four flow architectures on the SAME 2D three-mode Gaussian-mixture
target with the per-sample `reverse_KL` loss (type='F') (mean-reduced at the call
site), and saves the four loss curves to smoke_tests/test_loss_training.png:

    NSF      — spline flow on a box, uniform source
    RealNVP  — affine couplings + LU mixing, Gaussian source
    CNF      — FFJORD, exact trace, Gaussian source
    OTFlow   — OT potential flow, Gaussian source

Checks per architecture: every recorded loss is finite, and the mean of
the last 30 steps sits at least 0.5 nats below the mean of the first 30
(training actually progressed). Float32 (the training dtype), on the
default JAX backend; GPU memory preallocation is disabled. Exits nonzero
on any failure.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jflows.flow import CNF, NSF, OTFlow, RealNVP  # noqa: E402
from jflows.loss import reverse_KL  # noqa: E402
from jflows.potential import Nlog_Gaussian_Mixture  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "test_loss_training.log")
PNG = os.path.join(HERE, "test_loss_training.png")

FAILURES = 0
STEPS = 300
BATCH = 256
LR = 2e-3
BOX = 4.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def check_true(name: str, cond: bool, detail: str = "") -> None:
    global FAILURES
    log(f"  {name}: {detail}{' ' if detail else ''}-> {'OK' if cond else 'FAIL'}")
    if not cond:
        FAILURES += 1


TARGET = Nlog_Gaussian_Mixture(
    weights=[0.4, 0.35, 0.25],
    mean=[[-2.0, -2.0], [2.0, 1.5], [0.5, -2.5]],
    variance=[[0.5, 0.3], [0.4, 0.6], [0.25, 0.25]],
)


def sample_uniform(key):
    return jax.random.uniform(key, (BATCH, 2), minval=-BOX, maxval=BOX)


def sample_normal(key):
    return jax.random.normal(key, (BATCH, 2))


def train(name: str, flow, sample_fn, seed: int) -> np.ndarray:
    """Adam on mean(reverse_KL); returns the batch-mean loss curve."""
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)

    @eqx.filter_jit
    def step(params, m, v, t, key):
        def loss_fn(p):
            f = eqx.combine(p, static)
            return reverse_KL(sample_fn(key), TARGET, f.t(), type="F").mean()

        loss, g = jax.value_and_grad(loss_fn)(params)
        m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, m, g)
        v = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, v, g)
        mh = jax.tree.map(lambda x: x / (1 - 0.9**t), m)
        vh = jax.tree.map(lambda x: x / (1 - 0.999**t), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - LR * mh / (jnp.sqrt(vh) + 1e-8), params, mh, vh
        )
        return params, m, v, loss

    key = jax.random.key(seed)
    curve = np.empty(STEPS)
    t0 = time.time()
    for i in range(1, STEPS + 1):
        key, k = jax.random.split(key)
        # step index as a traced array — a python int would be a static
        # argument under filter_jit and recompile the step every iteration
        params, m, v, loss = step(params, m, v, jnp.asarray(i, dtype=jnp.float32), k)
        curve[i - 1] = float(loss)
        if i % 100 == 0 or i == 1:
            log(f"    {name}: step {i:3d}/{STEPS}  loss {curve[i - 1]:+.4f}")
    log(f"    {name}: done in {time.time() - t0:.1f}s")
    return curve


def main() -> None:
    log(f"START test_loss_training | jax {jax.__version__} | {jax.default_backend()}")
    kf = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(kf, 4)
    box = jnp.asarray([BOX, BOX])

    flows = [
        ("NSF", NSF(k1, -box, box, bins=8, transforms=3, hidden_features=(48, 48)),
         sample_uniform),
        ("RealNVP (lu)", RealNVP(k2, dimension=2, transforms=4, mixing="lu",
                                 hidden_features=(48, 48)), sample_normal),
        ("CNF", CNF(k3, dimension=2, frequency=3, nt=8, hidden_features=(48, 48)),
         sample_normal),
        ("OTFlow", OTFlow(k4, dimension=2, hidden=32, layer=3, rank=3, nt=8),
         sample_normal),
    ]

    curves: dict[str, np.ndarray] = {}
    for seed, (name, flow, sample_fn) in enumerate(flows, start=10):
        log(f"training {name}")
        curves[name] = train(name, flow, sample_fn, seed)

    log("pass criteria")
    for name, c in curves.items():
        finite = bool(np.isfinite(c).all())
        drop = float(c[:30].mean() - c[-30:].mean())
        check_true(f"{name} finite", finite)
        check_true(f"{name} loss decreased", drop > 0.5,
                   f"first30 {c[:30].mean():+.3f} -> last30 {c[-30:].mean():+.3f} (drop {drop:.3f})")

    log(f"rendering loss curves -> {PNG}")
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(np.arange(1, STEPS + 1), c, lw=1.6, color=f"C{i}", label=name)
    ax.set_xlabel("training step")
    ax.set_ylabel("batch-mean reverse KL (+ const)")
    ax.set_title("reverse KL training across flow architectures — 2D 3-mode Gaussian mixture")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(PNG, dpi=150)
    plt.close(fig)
    check_true("figure written", os.path.isfile(PNG) and os.path.getsize(PNG) > 10_000,
               f"{os.path.getsize(PNG)} bytes")

    if FAILURES:
        log(f"DONE — {FAILURES} FAILURE(S)")
        sys.exit(1)
    log(f"DONE — all loss-training tests passed; figure at {PNG}")


if __name__ == "__main__":
    main()

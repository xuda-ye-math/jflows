"""3D periodic target — single-stage reverse KL with reweighting.

A von Mises ridge mixture on the 3-torus [-NCSF_LIM, NCSF_LIM]^3,

    U_target(x) = -log[ exp(k cos(x1 - x2)) + exp(k cos(x2 - x3))
                      + exp(k cos(x3 - x1)) ],

sampled from the uniform source with BOTH objectives, each in one
packed stage — reverse KL (`train_reverse_KL`, type='F') and forward KL
(`train_forward_KL`, type='G', target data manufactured per Adam step
by single-rung AIS) — followed by the same reweighting pipeline:
importance weights -> ESS -> multinomial resampling -> MALA
rejuvenation at the target. The figure compares them side by side
(left: reverse KL; right: forward KL). The periodic domain requires
the NCSF (Neural Circular Spline Flow).

Run from the repo root:  ~/.envs/jax/bin/python -m example.3D_periodic
"""

import math
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})

from jax import Array
from jflows.flow import NCSF
from jflows.potential import Nlog_Uniform, Potential
from jflows.train import Monitor, train_forward_KL, train_reverse_KL
from jflows.utils import compute_ESS_log, importance_weights_log, langevin, resample

HERE = Path(__file__).resolve().parent
LOG = HERE / "3D_periodic.log"

# periodic box and target
NCSF_LIM: float = math.pi  # half-width of the periodic box [-NCSF_LIM, NCSF_LIM]^3
KAPPA: float = 4.0         # von Mises concentration of the ridge mixture

# NCSF flow architecture on the torus
BINS: int = 8                 # rational-quadratic spline bins per transform
TRANSFORMS: int = 4           # autoregressive transforms stacked in the flow
HIDDEN_FEATURES = (128, 128)  # hidden widths of each conditioner MLP

# training parameters
N_VALID: int = 40000   # the fixed source set (training pool + final ESS evaluation)
N_BATCH: int = 2000    # batch drawn from the fixed set per Adam step
STEPS: int = 200       # Adam steps (one compiled call)
LR: float = 1e-3       # Adam learning rate
MONITOR_EVERY: int = 20  # print loss + batch ESS every MONITOR_EVERY steps

# Langevin rejuvenation (training batches + the post-resample refresh)
LADDER: int = 1        # AIS rungs of the forward KL data manufacturing
MC_STEP: float = 1e-3  # Langevin step size
MC_ITERS: int = 100    # Langevin steps

# figure
N_PLOT: int = 10000    # subsample for a less crowded 3D scatter


# source: uniform on the periodic box
u0 = Nlog_Uniform(a=[-NCSF_LIM] * 3, b=[NCSF_LIM] * 3)


class U_Target(Potential):
    """Von Mises ridge mixture on the 3-torus (three pairwise ridges)."""

    kappa: float = eqx.field(static=True)

    def __init__(self, kappa: float = KAPPA):
        self.kappa = kappa

    def __call__(self, x: Array) -> Array:   # Array [N, 3] -> Array [N]
        t1, t2, t3 = x[:, 0], x[:, 1], x[:, 2]
        logits = jnp.stack([
            self.kappa * jnp.cos(t1 - t2),
            self.kappa * jnp.cos(t2 - t3),
            self.kappa * jnp.cos(t3 - t1),
        ], axis=-1)                                    # [N, 3]
        return -jax.scipy.special.logsumexp(logits, axis=-1)  # [N]


u1 = U_Target()


def new_flow(key):
    flow = NCSF(key, a=[-NCSF_LIM] * 3, b=[NCSF_LIM] * 3, bins=BINS,
                transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES)
    return flow.zeros()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START 3D_periodic | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"N_VALID={N_VALID} N_BATCH={N_BATCH} STEPS={STEPS} LR={LR} "
        f"MC={MC_STEP}x{MC_ITERS} kappa={KAPPA}")
    x_valid = u0.samples(jax.random.key(2), N_VALID)  # the fixed N_VALID source set

    results = {}
    for row, name in enumerate(("reverse KL", "forward KL")):
        log(f"[{name}] training (packed single stage) ...")
        if name == "reverse KL":
            flow, hist = train_reverse_KL(x_valid, u0, u1, new_flow(jax.random.key(0)),
                                          type="F", n_batch=N_BATCH, steps=STEPS, lr=LR,
                                          mc_step=MC_STEP, mc_iters=MC_ITERS,
                                          monitor=Monitor(MONITOR_EVERY, f"[{name}] ", log))
            tp, y = "F", flow(x_valid)                # pushforward F(x)
        else:
            flow, hist = train_forward_KL(x_valid, u0, u1, new_flow(jax.random.key(1)),
                                          type="G", n_batch=N_BATCH, steps=STEPS, lr=LR,
                                          ladder=LADDER, mc_step=MC_STEP, mc_iters=MC_ITERS,
                                          monitor=Monitor(MONITOR_EVERY, f"[{name}] ", log))
            tp, y = "G", flow.inv(x_valid)            # G's inverse pushes source forward
        log(f"[{name}] {STEPS} steps done   batch ESS "
            f"{float(hist[0]):.3f} -> {float(hist[STEPS // 2]):.3f} -> {float(hist[-1]):.3f}")

        # reweighting pipeline: importance weights -> ESS -> resample -> MALA
        log_w = importance_weights_log(x_valid, u0, u1, flow, type=tp, chunk=2)
        ess = float(compute_ESS_log(log_w))
        log(f"[{name}] final ESS = {ess:.4f}   (N_VALID = {N_VALID})")
        key_res, key_mc = jax.random.split(jax.random.key(10 + row))
        y = resample(key_res, y, jnp.exp(log_w - log_w.max()))
        y = langevin(key_mc, y, u1, step=MC_STEP, iters=MC_ITERS, chunk=4)
        log(f"[{name}] particle set resampled + rejuvenated at the target")
        results[name] = (y, ess)

    # figure (1, 2): left reverse KL, right forward KL — 3D scatters
    fig = plt.figure(figsize=(7.6, 3.4))
    for col, (name, color) in enumerate((("reverse KL", "#1F77B4"),
                                         ("forward KL", "#D62728")), start=1):
        y, ess = results[name]
        idx = jax.random.choice(jax.random.key(4), y.shape[0], (N_PLOT,), replace=False)
        y_np = np.asarray(y[idx])
        ax = fig.add_subplot(1, 2, col, projection="3d")
        ax.scatter(y_np[:, 0], y_np[:, 1], y_np[:, 2], s=0.8, alpha=0.6, color=color)
        ax.set_xlim(-NCSF_LIM, NCSF_LIM)
        ax.set_ylim(-NCSF_LIM, NCSF_LIM)
        ax.set_zlim(-NCSF_LIM, NCSF_LIM)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_zlabel(r"$\theta_3$", labelpad=1)
        ax.set_title(f"{name}, ESS = $\\mathbf{{{ess:.4f}}}$", pad=0)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.92, bottom=0.06, wspace=0.18)
    png = HERE / "3D_periodic.png"
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    log(f"DONE — figure at {png}")


if __name__ == "__main__":
    main()

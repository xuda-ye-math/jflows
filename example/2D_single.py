"""2D single-stage flow training — reverse KL vs forward KL.

Trains one NSF on a simple 2D potential with each objective, neither of
which touches the inverse map during training (each transform is
differentiated in its native direction only):

    reverse KL            : the flow acts as F (source -> target);
                            `train_reverse_KL_F` draws an BATCH_SIZE subset of
                            the fixed source set every Adam step and
                            freshens it with Langevin rejuvenation at the
                            source.
    forward KL            : the flow acts as G (target -> source);
                            `train_forward_KL_G` manufactures its target
                            batch every Adam step by single-level SMC
                            through the CURRENT flow (pushforward ->
                            reweight -> resample -> Langevin rejuvenation
                            at the target).

X-regularization conventions: one fixed set of VALID_SIZE source samples
serves training and evaluation — the packed trainers regenerate their
BATCH_SIZE training data inside every Adam step, so no frozen batch is ever
reused; the final ESS is computed on the full VALID_SIZE set through the
flow importance weights.

Run after installation from the repo root:  python -m example.2D_single
"""

import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture
from jflows.train import Monitor, train_forward_KL_G, train_reverse_KL_F
from jflows.utils import compute_ESS, importance_weights

HERE = Path(__file__).resolve().parent
LOG = HERE / "2D_single.log"

# boundary of the domain
SIGMA = 2.0            # standard deviation of the isotropic Gaussian source pi_0
PLT_LIM = 5.0          # half-width of the plot window; axes span [-PLT_LIM, +PLT_LIM]
NSF_LIM = 5.0          # half-width of the NSF spline domain; the flow acts on [-NSF_LIM, +NSF_LIM]^2

# NSF flow architecture
BINS: int = 16                      # number of rational-quadratic spline bins per transform
TRANSFORMS: int = 4                 # number of autoregressive transforms stacked in the flow
HIDDEN_FEATURES = (64, 64)          # widths of the hidden layers in each conditioner MLP

# training parameters
VALID_SIZE: int = 40000   # the fixed source set (training pool + final ESS evaluation)
BATCH_SIZE: int = 2000    # batch drawn from the fixed set per Adam step
STEPS_TOTAL: int = 200       # Adam steps (one compiled call per method)
LR: float = 1e-3       # Adam learning rate
MONITOR_EVERY: int = 10  # print loss + proposal ESS every MONITOR_EVERY steps

# Langevin rejuvenation (reverse KL batches + the single-level SMC)
LADDER: int = 1        # one reweight + resample + rejuvenation hop
MC_DT: float = 1e-3  # Langevin rejuvenation step size
MC_STEPS_1: int = 100  # MALA steps on the intermediate SMC levels only (through the flow)
MC_STEPS_2: int = 100  # MALA steps of every other rejuvenation: the last SMC level at the target, the reverse KL source batch


# source: Gaussian U0
u0 = Nlog_Gaussian(mean=[0.0, 0.0], variance=[SIGMA**2, SIGMA**2])

# target: three-mode Gaussian mixture placed like the Julia three-dot sign
# (one mode on top, two below, forming an upward triangle)
u1 = Nlog_Gaussian_Mixture(
    weights=[1.0, 1.0, 1.0],
    mean=[[0.0, 2.4], [-2.2, -1.4], [2.2, -1.4]],
    variance=[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]],
)


def new_flow(key):
    flow = NSF(key, a=[-NSF_LIM, -NSF_LIM], b=[+NSF_LIM, +NSF_LIM], bins=BINS,
               transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES)
    return flow.zeros()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START 2D_single | jax {jax.__version__} | backend {jax.default_backend()} | "
        f"VALID_SIZE={VALID_SIZE} BATCH_SIZE={BATCH_SIZE} STEPS_TOTAL={STEPS_TOTAL} LR={LR} "
        f"MC={MC_DT}x{MC_STEPS_1}")
    x_valid = u0.samples(jax.random.key(2), VALID_SIZE)  # the fixed VALID_SIZE source set

    log("[reverse KL] training (packed single stage) ...")
    flow_F, hist_F = train_reverse_KL_F(x_valid, u0, u1, new_flow(jax.random.key(0)),
                                        batch_size=BATCH_SIZE, steps_total=STEPS_TOTAL, lr=LR,
                                      mc_dt=MC_DT, mc_steps_2=MC_STEPS_2,
                                      monitor=Monitor(MONITOR_EVERY, "[reverse KL] ", log))
    log(f"[reverse KL] {STEPS_TOTAL} steps done   proposal ESS "
        f"{float(hist_F[0]):.3f} -> {float(hist_F[STEPS_TOTAL // 2]):.3f} -> {float(hist_F[-1]):.3f}")

    log("[forward KL] training (packed single stage) ...")
    flow_G, hist_G = train_forward_KL_G(x_valid, u0, u1, new_flow(jax.random.key(1)),
                                        batch_size=BATCH_SIZE, steps_total=STEPS_TOTAL, lr=LR,
                                      ladder=LADDER, mc_dt=MC_DT, mc_steps_1=MC_STEPS_1,
                                      mc_steps_2=MC_STEPS_2,
                                      monitor=Monitor(MONITOR_EVERY, "[forward KL] ", log))
    log(f"[forward KL] {STEPS_TOTAL} steps done   proposal ESS "
        f"{float(hist_G[0]):.3f} -> {float(hist_G[STEPS_TOTAL // 2]):.3f} -> {float(hist_G[-1]):.3f}")

    # final ESS on the full fixed set
    ess_F = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_F, type="F")))
    ess_G = float(compute_ESS(importance_weights(x_valid, u0, u1, flow_G, type="G")))
    log(f"[reverse KL] final ESS = {ess_F:.4f}   (VALID_SIZE = {VALID_SIZE})")
    log(f"[forward KL] final ESS = {ess_G:.4f}   (VALID_SIZE = {VALID_SIZE})")

    # figure: ESS history, then the two pushforward panels
    n = 300
    g = jnp.linspace(-PLT_LIM, PLT_LIM, n)
    X1, X2 = jnp.meshgrid(g, g, indexing="xy")
    U_np = np.asarray(u1(jnp.stack([X1.ravel(), X2.ravel()], axis=-1)).reshape(n, n))
    levels = np.linspace(U_np.min(), U_np.min() + 12.0, 50)
    cmap = LinearSegmentedColormap.from_list("light_yellow_red", ["#fffefa", "#ffe5e5"])

    y_F = flow_F(x_valid)              # F pushes source samples forward
    y_G = flow_G.inv(x_valid)          # G's inverse pushes source samples forward
    prior = np.asarray(u0.samples(jax.random.key(3), 5000))  # source Gaussian, visual context

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)

    steps_axis = np.arange(1, STEPS_TOTAL + 1)
    axes[0].plot(steps_axis, np.asarray(hist_F), color="#1F77B4", lw=1.2, label="reverse KL")
    axes[0].plot(steps_axis, np.asarray(hist_G), color="#D62728", lw=1.2, label="forward KL")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("ESS")
    axes[0].set_xlim(0, STEPS_TOTAL)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("proposal ESS history")
    axes[0].legend(loc="lower right")
    axes[0].set_box_aspect(1.0)

    for ax, y, color, name, ess in (
        (axes[1], y_F, "#1F77B4A0", "reverse KL", ess_F),
        (axes[2], y_G, "#D62728A0", "forward KL", ess_G),
    ):
        ax.contourf(np.asarray(X1), np.asarray(X2), U_np, levels=levels,
                    cmap=cmap.reversed(), extend="max")
        ax.contour(np.asarray(X1), np.asarray(X2), U_np, levels=levels,
                   colors="gray", linewidths=0.2, alpha=0.2)
        ax.scatter(prior[:, 0], prior[:, 1], s=0.04, alpha=0.3, color="gray", zorder=5)
        ax.scatter(np.asarray(y[:, 0]), np.asarray(y[:, 1]), s=0.2, alpha=0.3,
                   color=color, zorder=10, rasterized=True)
        ax.set_title(f"{name}, ESS = $\\mathbf{{{ess:.4f}}}$")
        ax.set_xlim(-PLT_LIM, PLT_LIM)
        ax.set_ylim(-PLT_LIM, PLT_LIM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    png = HERE / "2D_single.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE — figure at {png}")


if __name__ == "__main__":
    main()

"""4D annealed Boltzmann generator — reverse KL with an ADAPTIVE ladder.

The 4D two-charge target of zflows' `tests/4D_Boltzmann_generator.py`:
x = (x1, x2), x_i in R^2, confined to a soft annulus and repelling via a
regularized 3D Coulomb interaction,

    U_target(x) = a [ (|x1|^2 - r0^2)^2 + (|x2|^2 - r0^2)^2 ]
                + q2 / sqrt(|x1 - x2|^2 + eps^2).

A direct flow proposal from the 4D Gaussian source has ESS ~ 0, so the
flow is trained along the bridge ladder U_t = (1 - t) U_0 + t U_1 by
`boltzmann_reverse_KL`: per stage, packed reverse KL training against
the bridge (warm-started), full-set ESS validation with
rejection/shrink, then importance resampling + Langevin rejuvenation of
the particle set. The ladder coefficient t is ADAPTIVE (safe start,
enlarge-factor extrapolation, shrink on rejection) — the fixed
c_k = k / 12 schedule of the original is replaced by the ESS-gated
selection.

Run from the repo root:  ~/.envs/jax/bin/python -m example.4D_Boltzmann_generator
"""

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
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, Potential
from jflows.train import Monitor, boltzmann_reverse_KL

HERE = Path(__file__).resolve().parent
LOG = HERE / "4D_Boltzmann_generator.log"

# target physics (identical to the zflows original)
R0: float = 2.0        # annulus radius of the soft trap
A: float = 1.0         # trap stiffness
Q2: float = 4.0        # Coulomb coupling
EPS: float = 1e-3      # Coulomb regularization

# NSF flow architecture on the box [-NSF_LIM, NSF_LIM]^4
NSF_LIM = 3.0          # covers the Gaussian source (~3 sigma) and the annulus (r0 = 2)
BINS: int = 8          # rational-quadratic spline bins per transform
TRANSFORMS: int = 6    # autoregressive transforms stacked in the flow
HIDDEN_FEATURES = (64, 64)   # hidden widths of each conditioner MLP

# training parameters
N_VALID: int = 120000   # the fixed source set (training pool + validation ESS)
N_BATCH: int = 2000    # batch drawn from the fixed set per Adam step
STEPS: int = 500       # Adam steps per stage attempt (one compiled call)
LR: float = 1e-4       # Adam learning rate
MONITOR_EVERY: int = 20  # print loss + batch ESS every MONITOR_EVERY steps

# Langevin rejuvenation (training batches + the per-stage particle refresh)
MC_STEP: float = 1e-3  # Langevin rejuvenation step size
MC_ITERS: int = 100    # Langevin rejuvenation steps

# adaptive ladder (bg_param of boltzmann_reverse_KL)
BG_PARAM = {
    "t_safe": 0.1,        # stage-1 coefficient (the safe start)
    "shrink_factor": 0.7,  # rejected stage: t_k <- t_prev + shrink (t_k - t_prev)
    "enlarge_factor": 1.5, # accepted stage: extrapolation growth
    "tau": 0.6,            # full-set ESS acceptance threshold
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# source: 4D standard Gaussian
u0 = Nlog_Gaussian(mean=[0.0] * 4, variance=[1.0] * 4)


class U_Target(Potential):
    """Two charges on a soft annulus in R^2, regularized 3D Coulomb repulsion."""

    r0: float = eqx.field(static=True)
    a: float = eqx.field(static=True)
    q2: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, r0: float = R0, a: float = A, q2: float = Q2, eps: float = EPS):
        self.r0 = r0
        self.a = a
        self.q2 = q2
        self.eps = eps

    def __call__(self, x: Array) -> Array:   # Array [N, 4] -> Array [N]
        x1, x2 = x[:, :2], x[:, 2:]
        sq1 = (x1**2).sum(axis=-1)
        sq2 = (x2**2).sum(axis=-1)
        conf = self.a * ((sq1 - self.r0**2) ** 2 + (sq2 - self.r0**2) ** 2)
        d2 = ((x1 - x2) ** 2).sum(axis=-1)
        coul = self.q2 / jnp.sqrt(d2 + self.eps**2)
        return conf + coul


u1 = U_Target()


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START 4D_Boltzmann_generator | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | N_VALID={N_VALID} N_BATCH={N_BATCH} "
        f"STEPS={STEPS} LR={LR} MC={MC_STEP}x{MC_ITERS} bg={BG_PARAM}")
    x_valid = u0.samples(jax.random.key(2), N_VALID)  # the fixed N_VALID source set
    flow = NSF(jax.random.key(0), a=[-NSF_LIM] * 4, b=[NSF_LIM] * 4, bins=BINS,
               transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES).zeros()

    t0 = time.time()
    flow, y, stages = boltzmann_reverse_KL(
        x_valid, u0, u1, flow, type="F",
        n_batch=N_BATCH, steps=STEPS, lr=LR, mc_step=MC_STEP, mc_iters=MC_ITERS,
        monitor=Monitor(MONITOR_EVERY, "[train] ", log), bg_param=BG_PARAM,
    )
    ts = [s["t"] for s in stages]
    log(f"ladder done in {time.time() - t0:.1f}s: "
        f"t = {[round(t, 4) for t in ts]}  ESS = {[round(s['ess'], 3) for s in stages]}  "
        f"({'COMPLETE' if ts and ts[-1] == 1.0 else 'INCOMPLETE'})")

    # incremental stage quality (the generator's sample output is y)
    if stages:
        log(f"stage ESS: min = {min(s['ess'] for s in stages):.4f}   "
            f"last = {stages[-1]['ess']:.4f}   (N_VALID = {N_VALID})")

    # figure: adaptive ladder | particle-1 marginal at t = 1 | relative angle
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)

    axes[0].plot(range(1, len(ts) + 1), ts, "o-", color="#1F77B4", lw=1.2, label=r"$t_k$")
    axes[0].plot(range(1, len(ts) + 1), [s["ess"] for s in stages], "s--",
                 color="#D62728", lw=1.2, label="stage ESS")
    axes[0].set_xlabel("stage $k$")
    axes[0].set_xticks(range(1, len(ts) + 1))
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("adaptive ladder")
    axes[0].legend(loc="lower right")
    axes[0].set_box_aspect(1.0)

    y_np = np.asarray(y)
    theta = np.linspace(-np.pi, np.pi, 400)
    axes[1].scatter(y_np[:, 0], y_np[:, 1], s=0.2, alpha=0.3, color="#1F77B4A0",
                    rasterized=True)
    axes[1].plot(R0 * np.cos(theta), R0 * np.sin(theta), color="gray", lw=0.8, ls="--")
    axes[1].set_title("particle 1 at $t = 1$")
    axes[1].set_xlim(-NSF_LIM, NSF_LIM)
    axes[1].set_ylim(-NSF_LIM, NSF_LIM)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")

    dtheta = np.arctan2(y_np[:, 3], y_np[:, 2]) - np.arctan2(y_np[:, 1], y_np[:, 0])
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    axes[2].hist(dtheta, bins=80, density=True, color="#D62728A0")
    axes[2].set_title(r"relative angle $\Delta\theta$")
    axes[2].set_xlabel(r"$\Delta\theta$")
    axes[2].set_xlim(-np.pi, np.pi)
    axes[2].set_box_aspect(1.0)

    png = HERE / "4D_Boltzmann_generator.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE — figure at {png}")


if __name__ == "__main__":
    main()

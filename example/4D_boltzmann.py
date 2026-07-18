"""4D annealed Boltzmann generator — reverse KL vs forward KL, ADAPTIVE ladder.

The 4D two-charge target:
x = (x1, x2), x_i in R^2, confined to a soft annulus and repelling via a
regularized 3D Coulomb interaction,

    U_target(x) = a [ (|x1|^2 - r0^2)^2 + (|x2|^2 - r0^2)^2 ]
                + q2 / sqrt(|x1 - x2|^2 + eps^2).

A direct flow proposal from the 4D Gaussian source has ESS ~ 0, so both
generators anneal along the bridge ladder U_t = (1 - t) U_0 + t U_1 with
the ADAPTIVE coefficient (SMC-gated safe start, enlarge-factor
extrapolation, ESS-gated rejection/shrink) replacing the fixed
c_k = k / 12 schedule of the original:

    boltzmann_reverse_KL_F: each stage trains the increment by
        reverse KL on Langevin-freshened batches of the particle set;
    boltzmann_forward_KL_G: each stage trains the increment by
        forward KL on target batches manufactured per Adam step by AIS
        through the CURRENT flow (SMC gate and AIS share LADDER).

Both advance the particle set by reweight -> resample -> MALA and are
compared row by row in the figure (top: reverse KL; bottom: forward KL).

Run after installation from the repo root:  python -m example.4D_boltzmann
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
from jflows.train import (
    Monitor,
    boltzmann_forward_KL_G,
    boltzmann_reverse_KL_F,
)
from jflows.artifacts import inspect_run

HERE = Path(__file__).resolve().parent
LOG = HERE / "4D_boltzmann.log"
RUN_ROOT = Path(os.environ.get("JFLOWS_RUN_ROOT", "run/4D_boltzmann"))

# target physics
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
N_VALID: int = 120000   # exact population for selection, training, and validation
BATCH_SIZE: int = 2000    # batch drawn from the fixed set per Adam step
TRAIN_STEPS: int = 500       # Adam steps per stage attempt (one compiled call)
LR: float = 1e-4       # Adam learning rate
MONITOR_EVERY: int = 20  # print loss + proposal ESS every MONITOR_EVERY steps

# Langevin rejuvenation (training batches + the per-stage particle refresh)
LADDER: int = 1        # SMC levels of the tau_smc selection gate
MC_DT: float = 1e-3  # Langevin rejuvenation step size
MC_STEPS: int = 100    # Langevin rejuvenation steps (MALA default: rejects Coulomb-wall proposals)

# adaptive ladder (bg_param of boltzmann_reverse_KL_F)
BG_PARAM = {
    "t_safe": 0.2,        # stage-1 coefficient (the safe start)
    "shrink_factor": 0.7,  # rejected stage: t_k <- t_prev + shrink (t_k - t_prev)
    "enlarge_factor": 1.5, # accepted stage: extrapolation growth
    "tau_smc": 0.2,        # SMC pre-selection gate on t_k
    "tau_ess": 0.6,        # incremental ESS acceptance threshold
}


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


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    open(LOG, "w").close()   # fresh log per run (no appending)
    log(f"START 4D_boltzmann | jax {jax.__version__} | "
        f"backend {jax.default_backend()} | N_VALID={N_VALID} "
        f"BATCH_SIZE={BATCH_SIZE} TRAIN_STEPS={TRAIN_STEPS} LR={LR} MC={MC_DT}x{MC_STEPS} bg={BG_PARAM}")
    x_valid = u0.samples(jax.random.key(2), N_VALID)  # the fixed N_VALID source set

    def new_flow(key):
        return NSF(key, a=[-NSF_LIM] * 4, b=[NSF_LIM] * 4, bins=BINS,
                   transforms=TRANSFORMS, hidden_features=HIDDEN_FEATURES).zeros()

    results = {}
    for name, key_f in (
        ("reverse KL", jax.random.key(0)),
        ("forward KL", jax.random.key(1)),
    ):
        run_dir = RUN_ROOT / name.lower().replace(" ", "_")
        resume = (run_dir / "run.json").is_file()
        run_samples = None if resume else x_valid
        if resume:
            log(f"[{name}] resuming durable run at {run_dir}")
        t0 = time.time()
        if name == "reverse KL":
            y, stages = boltzmann_reverse_KL_F(
                run_samples, u0, u1, new_flow(key_f),
                batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, lr=LR, ladder=LADDER,
                mc_dt=MC_DT, mc_steps=MC_STEPS,
                monitor=Monitor(MONITOR_EVERY, f"[{name}] ", log), bg_param=BG_PARAM,
                run_dir=run_dir, resume=resume, problem_id="4d-two-charge-v1",
            )
        else:
            y, stages = boltzmann_forward_KL_G(
                run_samples, u0, u1, new_flow(key_f),
                batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, lr=LR, ladder=LADDER,
                mc_dt=MC_DT, mc_steps=MC_STEPS,
                monitor=Monitor(MONITOR_EVERY, f"[{name}] ", log), bg_param=BG_PARAM,
                run_dir=run_dir, resume=resume, problem_id="4d-two-charge-v1",
            )
        manifest = inspect_run(run_dir)
        log(
            f"[{name}] durable lifecycle={manifest['lifecycle']} "
            f"training_time={manifest['training_elapsed_seconds_total']:.1f}s"
        )
        ts = [s["t"] for s in stages]
        log(f"[{name}] ladder done in {time.time() - t0:.1f}s: "
            f"t = {[round(t, 4) for t in ts]}  "
            f"ESS = {[round(s['valid_selected_ess'], 3) for s in stages]}  "
            f"({'COMPLETE' if ts and ts[-1] == 1.0 else 'INCOMPLETE'})")
        if not ts or ts[-1] != 1.0:
            raise RuntimeError(
                f"{name} ladder stopped before the target; refusing to label or plot "
                f"the particles as t=1 (last t={ts[-1] if ts else 0.0:.4f})"
            )
        if stages:
            log(
                f"[{name}] stage ESS: min = "
                f"{min(s['valid_selected_ess'] for s in stages):.4f}   "
                f"last = {stages[-1]['valid_selected_ess']:.4f}   "
                f"(N_VALID = {N_VALID})"
            )
        results[name] = (y, stages)

    # figure (2, 3): per row — adaptive ladder | particle-1 marginal | relative angle
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0), constrained_layout=True)
    theta = np.linspace(-np.pi, np.pi, 400)
    for row, (name, scatter_c, hist_c) in enumerate(
        (("reverse KL", "#1F77B4A0", "#1F77B4A0"),
         ("forward KL", "#D62728A0", "#D62728A0"))
    ):
        y, stages = results[name]
        ts = [s["t"] for s in stages]
        y_np = np.asarray(y)

        ax = axes[row, 0]
        ax.plot(range(1, len(ts) + 1), ts, "o-", color="#1F77B4", lw=1.2, label=r"$t_k$")
        ax.plot(range(1, len(ts) + 1),
                [s["valid_selected_ess"] for s in stages], "s--",
                color="#D62728", lw=1.2, label="stage ESS")
        ax.set_xlabel("stage $k$")
        ax.set_xticks(range(1, len(ts) + 1))
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{name}: adaptive ladder")
        ax.legend(loc="lower right")
        ax.set_box_aspect(1.0)

        ax = axes[row, 1]
        ax.scatter(y_np[:, 0], y_np[:, 1], s=0.2, alpha=0.3, color=scatter_c,
                   rasterized=True)
        ax.plot(R0 * np.cos(theta), R0 * np.sin(theta), color="gray", lw=0.8, ls="--")
        ax.set_title(f"{name}: particle 1 at $t = 1$")
        ax.set_xlim(-NSF_LIM, NSF_LIM)
        ax.set_ylim(-NSF_LIM, NSF_LIM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

        ax = axes[row, 2]
        dtheta = np.arctan2(y_np[:, 3], y_np[:, 2]) - np.arctan2(y_np[:, 1], y_np[:, 0])
        dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
        ax.hist(dtheta, bins=80, density=True, color=hist_c)
        ax.set_title(f"{name}: relative angle $\\Delta\\theta$")
        ax.set_xlabel(r"$\Delta\theta$")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_box_aspect(1.0)

    png = HERE / "4D_boltzmann.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"DONE — figure at {png}")


if __name__ == "__main__":
    main()

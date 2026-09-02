"""Tiny trainer and generator smoke test — run from the repo root as
`python -m smoke.test_train`.

Every public trainer runs two Adam steps on a 2D Gaussian pair with a
batch of 16 rows, and the KLXX adaptive-staging generator completes one
two-stage schedule, so that the signatures, the compiled scans, the batch
ESS histories, the ``mc_steps_1`` / ``mc_steps_2`` split, ``coeff_qt``,
the HMC trajectory of an intermediate SMC level (forward KL at
``ladder=2``), and the policy keys are exercised once. Numbers are not
benchmark results.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from jflows.boltzmann import boltzmann_forward_KL_G_fixed, boltzmann_forward_KLXX_G
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.train import (
    Monitor,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
    train_reverse_KL_F,
)

VALID_SIZE = 64
BATCH_SIZE = 16
STEPS_TOTAL = 2
MC_STEPS_1 = 1
MC_STEPS_2 = 2


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    print(f"{name}: OK")


def main():
    source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    target = Nlog_Gaussian([0.3, -0.2], [0.8, 1.2])
    flow = NSF(
        jax.random.key(0), [-4.0, -4.0], [4.0, 4.0], bins=4,
        transforms=1, hidden_features=(8,),
    ).zeros()
    x_valid = source.samples(jax.random.key(1), VALID_SIZE)
    monitor = Monitor(1, "[smoke] ")

    runs = {
        "reverse KL": train_reverse_KL_F(
            x_valid, source, target, flow, BATCH_SIZE, STEPS_TOTAL, 1e-3,
            1e-3, MC_STEPS_1, monitor=monitor,
        ),
        "forward KL": train_forward_KL_G(
            x_valid, source, target, flow, BATCH_SIZE, STEPS_TOTAL, 1e-3,
            2, 1e-3, MC_STEPS_1, u_clip=50.0,
        ),
        "KLX": train_forward_KLX_G(
            x_valid, source, target, flow, BATCH_SIZE, STEPS_TOTAL, 1e-3,
            1, 1e-3, MC_STEPS_1, coeff_lambda=0.5,
        ),
        "KLXX": train_forward_KLXX_G(
            x_valid, source, target, flow, 0, BATCH_SIZE, STEPS_TOTAL, 1e-3,
            1, 0.5, 1.0, 2, 1e-3, MC_STEPS_1, MC_STEPS_2, coeff_qt=0.5,
            u_clip=50.0, chunks=2,
        ),
    }
    for name, (trained, history) in runs.items():
        check(f"{name} ESS history shape", history.shape == (STEPS_TOTAL,))
        check(f"{name} ESS history finite", jnp.all(jnp.isfinite(history)))
        check(f"{name} returns a flow", isinstance(trained, NSF))

    _, stages = boltzmann_forward_KLXX_G(
        x_valid, source, target, flow, 0, BATCH_SIZE, STEPS_TOTAL, 1e-3,
        1, 0.5, 1.0, 1, 1e-3, MC_STEPS_1, MC_STEPS_2, coeff_qt=0.3,
        bg_param={"t_safe": 0.5, "tau_valid": 0.0}, chunks=2,
    )
    check("KLXX adaptive schedule reaches t = 1", stages[-1]["t"] == 1.0)
    check("stage record has no selection history", "selection_history" not in stages[0])
    _, fixed = boltzmann_forward_KL_G_fixed(
        x_valid, source, target, flow, BATCH_SIZE, STEPS_TOTAL, 1e-3, 1, 1e-3,
        MC_STEPS_1, MC_STEPS_2, [0.5, 1.0],
    )
    check("fixed schedule follows t_list", [s["t"] for s in fixed] == [0.5, 1.0])
    try:
        boltzmann_forward_KLXX_G(
            x_valid, source, target, flow, 0, BATCH_SIZE, STEPS_TOTAL, 1e-3,
            1, 0.5, 1.0, 1, 1e-3, MC_STEPS_1, MC_STEPS_2, bg_param={"tau_ess": 0.5},
        )
    except KeyError:
        check("retired policy key is rejected", True)
    else:
        raise AssertionError("retired policy key is rejected")


if __name__ == "__main__":
    main()

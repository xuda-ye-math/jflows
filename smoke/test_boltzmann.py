"""Focused numerical checks for the Boltzmann computation API."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.boltzmann import (
    boltzmann_forward_KL_G,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G,
    boltzmann_forward_KLXX_G_fixed,
    boltzmann_reverse_KL_F,
    boltzmann_reverse_KL_F_fixed,
)


VALID_SZIE = 64
POOL_SIZE = 32
BATCH_SZIE = 16
TRAIN_STEPS = 2
LR = 1e-3
MC_DT = 1e-3
MC_STEPS = 0

STAGE_KEYS = {
    "t",
    "t_start",
    "valid_selected_ess",
    "valid_trained_ess",
    "valid_identity_ess",
    "valid_sample_count",
    "selected",
    "flow",
    "continuation_flow",
    "t_hist",
    "batch_ess_hist",
    "valid_trained_ess_hist",
    "valid_identity_ess_hist",
    "attempt_status_hist",
    "selection_history",
    "elapsed_seconds",
    "selected_flow_path",
    "continuation_flow_path",
    "validation_samples_path",
}


def _check(result, samples):
    population, stages = result
    assert population.shape == samples.shape
    assert bool(jnp.isfinite(population).all())
    assert len(stages) == 1 and stages[0]["t"] == 1.0
    assert set(stages[0]) == STAGE_KEYS
    assert stages[0]["valid_sample_count"] == VALID_SZIE
    assert stages[0]["selected"] in ("trained", "identity")
    assert stages[0]["batch_ess_hist"].shape == (1, TRAIN_STEPS)
    return population, stages


def main():
    source = Nlog_Gaussian([0.0, 0.0], [2.0, 2.0])
    target = Nlog_Gaussian([1.0, -1.0], [1.0, 1.0])
    samples = source.samples(jax.random.key(1), VALID_SZIE)
    flow = NSF(
        jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
        bins=4, transforms=1, hidden_features=(8,),
    ).zeros()
    adaptive = {"t_safe": 1.0, "tau_ess": 0.0, "max_stages": 1}

    reverse = dict(
        batch_size=BATCH_SZIE, train_steps=TRAIN_STEPS, lr=LR,
        mc_dt=MC_DT, mc_steps=MC_STEPS,
    )
    forward = dict(reverse, ladder=1)
    klxx = dict(
        forward,
        melt=0.0,
        opt_dt=0.5,
        opt_steps=0,
    )

    first, first_stages = _check(
        boltzmann_reverse_KL_F(
            samples, source, target, flow, ladder=1,
            bg_param=adaptive, **reverse,
        ),
        samples,
    )
    _check(
        boltzmann_forward_KL_G(
            samples, source, target, flow, bg_param=adaptive, **forward,
        ),
        samples,
    )
    _check(
        boltzmann_forward_KLX_G(
            samples, source, target, flow, bg_param=adaptive, **forward,
        ),
        samples,
    )
    _check(
        boltzmann_forward_KLXX_G(
            samples, source, target, flow, POOL_SIZE,
            bg_param=adaptive, **klxx,
        ),
        samples,
    )
    _check(
        boltzmann_reverse_KL_F_fixed(
            samples, source, target, flow, t_list=[1.0], **reverse,
        ),
        samples,
    )
    _check(
        boltzmann_forward_KL_G_fixed(
            samples, source, target, flow, t_list=[1.0], **forward,
        ),
        samples,
    )
    _check(
        boltzmann_forward_KLX_G_fixed(
            samples, source, target, flow, t_list=[1.0], **forward,
        ),
        samples,
    )
    _check(
        boltzmann_forward_KLXX_G_fixed(
            samples, source, target, flow, 0,
            t_list=[1.0], **klxx,
        ),
        samples,
    )

    repeated, repeated_stages = boltzmann_reverse_KL_F(
        samples, source, target, flow, ladder=1,
        bg_param=adaptive, **reverse,
    )
    assert bool(jnp.array_equal(first, repeated))
    assert first_stages[0]["t"] == repeated_stages[0]["t"]
    print("boltzmann computation: OK")


if __name__ == "__main__":
    main()

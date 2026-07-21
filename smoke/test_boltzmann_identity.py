"""Identity-only adaptive Boltzmann computation smoke test."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import jflows.boltzmann as boltzmann_api
from jflows.boltzmann import boltzmann_identity
from jflows.potential import Nlog_Gaussian, linear_combination
from jflows.utils import compute_ESS_log


VALID_SZIE = 64
CONTROLS = {
    "ladder": 1,
    "mc_dt": 1e-3,
    "mc_steps": 0,
    "chunks": 4,
    "seed": 3,
    "bg_param": {
        "t_safe": 1.0,
        "shrink_factor": 0.5,
        "enlarge_factor": 1.0,
        "tau_smc": 0.0,
        "tau_ess": 0.8,
        "max_stages": 2,
        "max_retry": 3,
    },
}
STAGE_KEYS = {
    "t",
    "t_start",
    "valid_selected_ess",
    "valid_identity_ess",
    "valid_sample_count",
    "selected",
    "t_hist",
    "valid_identity_ess_hist",
    "attempt_status_hist",
    "selection_history",
    "elapsed_seconds",
    "validation_samples_path",
}


def main():
    parameters = inspect.signature(boltzmann_identity).parameters
    assert tuple(parameters) == (
        "x_valid",
        "source",
        "target",
        "ladder",
        "mc_dt",
        "mc_steps",
        "mc_adjust",
        "monitor",
        "bg_param",
        "chunks",
        "seed",
    )
    assert not {
        "flow", "batch_size", "train_steps", "lr", "checkpoint",
        "initialize_from_identity",
    } & set(parameters)

    source = Nlog_Gaussian([0.0, 0.0], [2.0, 2.0])
    target = Nlog_Gaussian([0.5, -0.5], [1.0, 1.0])
    samples = source.samples(jax.random.key(1), VALID_SZIE)
    bridge = linear_combination([target, source], [0.5, 0.5])
    expected_first_ess = compute_ESS_log(source(samples) - bridge(samples))

    original = boltzmann_api._train_attempt

    def unexpected_training(*args, **kwargs):
        del args, kwargs
        raise AssertionError("boltzmann_identity called a flow trainer")

    boltzmann_api._train_attempt = unexpected_training
    try:
        population, stages = boltzmann_identity(
            samples, source, target, **CONTROLS
        )
        repeated, repeated_stages = boltzmann_identity(
            samples, source, target, **CONTROLS
        )
        _, gated_stages = boltzmann_identity(
            samples,
            source,
            source,
            ladder=2,
            mc_dt=1e-3,
            mc_steps=0,
            bg_param={
                "t_safe": 1.0,
                "tau_smc": 0.9,
                "tau_ess": 0.9,
                "max_stages": 1,
            },
            chunks=4,
            seed=3,
        )
    finally:
        boltzmann_api._train_attempt = original

    assert population.shape == samples.shape
    assert bool(jnp.isfinite(population).all())
    assert bool(jnp.array_equal(population, repeated))
    assert [stage["t"] for stage in stages] == [0.5, 1.0]
    assert [stage["t"] for stage in repeated_stages] == [0.5, 1.0]
    assert jnp.isclose(stages[0]["valid_identity_ess"], expected_first_ess)
    assert stages[0]["t_hist"].tolist() == [1.0, 0.5]
    assert stages[0]["attempt_status_hist"] == ("rejected", "accepted")
    assert gated_stages[0]["selection_history"][0]["decision"] == "accepted"
    assert gated_stages[0]["selection_history"][0]["minimum_smc_ess"] >= 0.9
    for index, stage in enumerate(stages):
        assert set(stage) == STAGE_KEYS
        assert stage["selected"] == "identity"
        assert stage["valid_sample_count"] == VALID_SZIE
        assert stage["valid_selected_ess"] == stage["valid_identity_ess"]
        attempts = 2 if index == 0 else 1
        assert stage["t_hist"].shape == (attempts,)
        assert stage["valid_identity_ess_hist"].shape == (attempts,)
        assert stage["selection_history"][0]["reason"] == "tau_smc_zero"
        assert not any("flow" in key or "trained" in key for key in stage)
        assert "batch_ess_hist" not in stage
    print("identity Boltzmann computation: OK")


if __name__ == "__main__":
    main()

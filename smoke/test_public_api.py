"""Focused checks for the direct public training surface."""

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp

import jflows.train as training
import jflows.boltzmann as boltzmann_api
import jflows.boltzmann.load as boltzmann_load
import jflows.boltzmann.write as boltzmann_write
import jflows.artifacts as artifacts_api
import jflows.train as train_api
from jflows.boltzmann import (
    boltzmann_identity,
    boltzmann_forward_KL_G,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G,
    boltzmann_forward_KLXX_G_fixed,
    boltzmann_reverse_KL_F,
    boltzmann_reverse_KL_F_fixed,
)
from jflows.flow import NSF, OTFlow
from jflows.potential import Nlog_Gaussian
from jflows.train import (
    Monitor,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
    train_reverse_KL_F,
)


TRAINERS = (
    train_reverse_KL_F,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
)

GENERATORS = (
    boltzmann_reverse_KL_F,
    boltzmann_forward_KL_G,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
    boltzmann_reverse_KL_F_fixed,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G_fixed,
)


def _problem():
    potential = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    samples = potential.samples(jax.random.key(1), 16)
    flow = NSF(
        jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
        bins=4, transforms=1, hidden_features=(8,),
    )
    return potential, samples, flow


def test_signatures():
    assert Path(boltzmann_api.__file__).name == "__init__.py"
    assert Path(boltzmann_api.__file__).parent.name == "boltzmann"
    assert train_api.__all__ == [
        "Monitor",
        "train_forward_KL_G",
        "train_forward_KLX_G",
        "train_forward_KLXX_G",
        "train_reverse_KL_F",
    ]
    assert boltzmann_api.__all__ == [
        "boltzmann_identity",
        "boltzmann_forward_KL_G",
        "boltzmann_forward_KL_G_fixed",
        "boltzmann_forward_KLX_G",
        "boltzmann_forward_KLX_G_fixed",
        "boltzmann_forward_KLXX_G",
        "boltzmann_forward_KLXX_G_fixed",
        "boltzmann_reverse_KL_F",
        "boltzmann_reverse_KL_F_fixed",
    ]
    assert Monitor is train_api.Monitor
    assert not any(name.startswith("boltzmann_") for name in train_api.__all__)
    assert not any(name.startswith("train_") for name in boltzmann_api.__all__)
    assert boltzmann_load.__all__ == [
        "fork",
        "fork_run",
        "inspect_run",
        "load",
        "load_stage_flow",
        "load_training_history",
        "load_validation_samples",
        "manifest",
        "run",
        "validate",
        "validate_run",
    ]
    assert boltzmann_write.__all__ == ["create", "finish", "stage"]
    assert artifacts_api.__all__ == [
        "load_flow",
        "load_history",
        "load_samples",
        "save_flow",
        "save_history",
        "save_samples",
    ]
    for function in TRAINERS:
        parameters = inspect.signature(function).parameters
        assert "initialize_from_identity" in parameters
        assert "t_start" in parameters and "t_end" in parameters
    for function in GENERATORS:
        parameters = inspect.signature(function).parameters
        assert parameters["initialize_from_identity"].default is True
        assert "run_dir" not in parameters
        assert "resume" not in parameters
    identity_parameters = inspect.signature(boltzmann_identity).parameters
    assert not {
        "flow", "batch_size", "train_steps", "lr", "checkpoint",
        "initialize_from_identity", "run_dir", "resume",
    } & set(identity_parameters)
    for function in (
        train_forward_KLXX_G,
        boltzmann_forward_KLXX_G,
        boltzmann_forward_KLXX_G_fixed,
    ):
        parameters = inspect.signature(function).parameters
        assert "pool_size" in parameters
        assert "chunks" in parameters


def test_identity_start():
    potential, samples, flow = _problem()
    trained, history = train_reverse_KL_F(
        samples, potential, potential, flow,
        batch_size=8, train_steps=1, lr=0.0,
        mc_dt=1e-3, mc_steps=0,
        initialize_from_identity=True,
    )
    assert history.shape == (1,)
    assert type(trained) is type(flow)
    ot = OTFlow(jax.random.key(3), 2, hidden=8, layer=2, rank=2, nt=2)
    identity, _ = train_reverse_KL_F(
        samples, potential, potential, ot,
        batch_size=8, train_steps=1, lr=0.0,
        mc_dt=1e-3, mc_steps=0,
        initialize_from_identity=True,
    )
    assert float(jnp.linalg.norm(identity._ot.phi.A)) > 0.0


def test_pool_semantics():
    potential, samples, flow = _problem()
    observed = []
    original = training.quench_and_temper

    def quench(key, values, target, melt, opt_dt, opt_steps,
               mc_dt, mc_steps, mc_adjust, chunks=1):
        del key, target, melt, opt_dt, opt_steps, mc_dt, mc_steps, mc_adjust
        observed.append((values.shape[0], chunks))
        return values

    training.quench_and_temper = quench
    try:
        for pool_size in (0, 8):
            trained, history = train_forward_KLXX_G(
                samples, potential, potential, flow.zeros(), pool_size,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                melt=0.0, opt_dt=0.1, opt_steps=0,
                mc_dt=1e-3, mc_steps=0, chunks=4,
            )
            assert type(trained) is type(flow) and history.shape == (1,)
    finally:
        training.quench_and_temper = original
    assert observed == [(samples.shape[0], 4), (8, 4)]


def main():
    test_signatures()
    test_identity_start()
    test_pool_semantics()
    print("public training API: OK")


if __name__ == "__main__":
    main()

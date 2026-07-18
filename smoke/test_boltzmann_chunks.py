"""Verify that KLXX Boltzmann ``chunks`` reaches quench-and-temper.

Run from the repository root as ``python -m smoke.test_boltzmann_chunks``.
"""

import inspect
import importlib
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.train import (
    boltzmann_forward_KLXX_G,
    boltzmann_forward_KLXX_G_fixed,
)
from jflows.training import drivers
from jflows.utils.quench import quench_and_temper


CHUNKS: int = 2


def problem():
    source = Nlog_Gaussian(mean=[0.0, 0.0], variance=[1.0, 1.0])
    target = Nlog_Gaussian(mean=[1.0, -1.0], variance=[1.0, 1.0])
    x_valid = source.samples(jax.random.key(2), 8)
    flow = NSF(
        jax.random.key(0),
        a=[-4.0, -4.0],
        b=[4.0, 4.0],
        bins=4,
        transforms=1,
        hidden_features=(8,),
    )
    return source, target, x_valid, flow


def common():
    return {
        "batch_size": 4,
        "train_steps": 1,
        "lr": 1e-3,
        "ladder": 1,
        "melt": 0.0,
        "opt_dt": 0.5,
        "opt_steps": 1,
        "mc_dt": 1e-3,
        "mc_steps": 1,
        "chunks": CHUNKS,
    }


def verify_result(samples, stages, shape) -> None:
    assert samples.shape == shape
    assert bool(jnp.isfinite(samples).all())
    assert len(stages) == 1
    assert stages[0]["t"] == 1.0


def main() -> None:
    for function in (boltzmann_forward_KLXX_G, boltzmann_forward_KLXX_G_fixed):
        parameters = [
            name for name in inspect.signature(function).parameters if "chunk" in name
        ]
        assert parameters == ["chunks"], (
            f"{function.__name__} chunk controls are {parameters!r}"
        )

    parameters = [
        name
        for name in inspect.signature(quench_and_temper).parameters
        if "chunk" in name
    ]
    assert parameters == ["chunks"], (
        f"quench_and_temper chunk controls are {parameters!r}"
    )

    quench_module = importlib.import_module("jflows.utils.quench")
    observed = []
    original_quench = drivers.quench_and_temper
    original_lbfgs = quench_module.lbfgs
    original_langevin = quench_module.langevin

    def record_quench(*args, **kwargs):
        observed.append(("quench", kwargs.get("chunks"), isinstance(args[1], jax.core.Tracer)))
        return original_quench(*args, **kwargs)

    def record_lbfgs(*args, **kwargs):
        observed.append(("lbfgs", kwargs.get("chunks"), isinstance(args[0], jax.core.Tracer)))
        return original_lbfgs(*args, **kwargs)

    def record_langevin(*args, **kwargs):
        observed.append(("langevin", kwargs.get("chunks"), isinstance(args[1], jax.core.Tracer)))
        return original_langevin(*args, **kwargs)

    expected = [
        ("quench", CHUNKS, False),
        ("lbfgs", CHUNKS, False),
        ("langevin", CHUNKS, False),
    ]
    drivers.quench_and_temper = record_quench
    quench_module.lbfgs = record_lbfgs
    quench_module.langevin = record_langevin
    try:
        source, target, x_valid, flow = problem()
        samples, stages = boltzmann_forward_KLXX_G(
            x_valid,
            source,
            target,
            flow,
            bg_param={"t_safe": 1.0, "tau_ess": 0.0},
            **common(),
        )
        jax.block_until_ready(samples)
        verify_result(samples, stages, x_valid.shape)
        assert observed == expected, observed

        observed.clear()
        jax.clear_caches()

        source, target, x_valid, flow = problem()
        samples, stages = boltzmann_forward_KLXX_G_fixed(
            x_valid,
            source,
            target,
            flow,
            t_list=[1.0],
            **common(),
        )
        jax.block_until_ready(samples)
        verify_result(samples, stages, x_valid.shape)
        assert observed == expected, observed

        for invalid in (0, True, x_valid.shape[0] + 1):
            try:
                quench_and_temper(
                    jax.random.key(3), x_valid, target, 0.0,
                    opt_steps=0, mc_steps=0, chunks=invalid,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid chunks={invalid!r}")
    finally:
        drivers.quench_and_temper = original_quench
        quench_module.lbfgs = original_lbfgs
        quench_module.langevin = original_langevin
        jax.clear_caches()

    print(
        "test_boltzmann_chunks: OK — adaptive and fixed KLXX ran concrete "
        f"quench-and-temper, L-BFGS, and Langevin with chunks={CHUNKS}"
    )


if __name__ == "__main__":
    main()

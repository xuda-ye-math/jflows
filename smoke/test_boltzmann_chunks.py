"""Verify that the single ``chunks`` control reaches KLXX quench-and-temper."""

import inspect

import jax
import jax.numpy as jnp

import jflows.boltzmann as boltzmann
import jflows.train as training
import jflows.utils.quench as quench_module
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.train import train_forward_KLXX_G


def main():
    potential = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    samples = potential.samples(jax.random.key(1), 16)
    flow = NSF(
        jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
        bins=4, transforms=1, hidden_features=(8,),
    ).zeros()

    chunk_names = [
        name for name in inspect.signature(train_forward_KLXX_G).parameters
        if "chunk" in name
    ]
    assert chunk_names == ["chunks"]
    assert inspect.isfunction(train_forward_KLXX_G)

    utility_calls = []
    original_lbfgs = quench_module.lbfgs
    original_langevin = quench_module.langevin

    def lbfgs(values, target, alpha, steps, armijo, chunks):
        del target, alpha, steps, armijo
        utility_calls.append(("lbfgs", chunks))
        return values

    def langevin(key, values, target, dt, steps, adjust, chunks):
        del key, target, dt, steps, adjust
        utility_calls.append(("langevin", chunks))
        return values

    quench_module.lbfgs = lbfgs
    quench_module.langevin = langevin
    try:
        quench_module.quench_and_temper(
            jax.random.key(3), samples, potential, 0.0,
            opt_dt=0.1, opt_steps=0, mc_dt=1e-3, mc_steps=0, chunks=4,
        )
    finally:
        quench_module.lbfgs = original_lbfgs
        quench_module.langevin = original_langevin
    assert utility_calls == [("lbfgs", 4), ("langevin", 4)]

    quench_calls = []
    original_quench = training.quench_and_temper

    def quench(key, values, target, melt, opt_dt, opt_steps,
               mc_dt, mc_steps, mc_adjust, chunks=1):
        del key, target, melt, opt_dt, opt_steps, mc_dt, mc_steps, mc_adjust
        quench_calls.append((values.shape[0], chunks))
        return values

    training.quench_and_temper = quench
    try:
        train_forward_KLXX_G(
            samples, potential, potential, flow, 0,
            batch_size=8, train_steps=1, lr=0.0, ladder=1,
            melt=0.0, opt_dt=0.1, opt_steps=0,
            mc_dt=1e-3, mc_steps=0, chunks=4,
        )
    finally:
        training.quench_and_temper = original_quench
    assert quench_calls == [(16, 4)]

    forwarded = []
    original_train = boltzmann.train_forward_KLXX_G

    def train(*args, **kwargs):
        forwarded.append(kwargs["chunks"])
        return args[3], jnp.ones((args[6],))

    boltzmann.train_forward_KLXX_G = train
    try:
        boltzmann.boltzmann_forward_KLXX_G_fixed(
            samples, potential, potential, flow, 8,
            batch_size=8, train_steps=1, lr=0.0, ladder=1,
            melt=0.0, opt_dt=0.1, opt_steps=0,
            mc_dt=1e-3, mc_steps=0, t_list=[1.0], chunks=4,
        )
    finally:
        boltzmann.train_forward_KLXX_G = original_train
    assert forwarded == [4]
    print("KLXX chunks forwarding: OK")


if __name__ == "__main__":
    main()

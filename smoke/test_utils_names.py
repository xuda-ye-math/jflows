"""Focused smoke test for canonical utility keywords and legacy aliases."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.utils import (
    adamw,
    annealed_importance_sampling,
    coverage,
    hamiltonian_monte_carlo,
    importance_weights_log,
    langevin,
    lbfgs,
    quench_and_temper,
    sequential_monte_carlo,
)


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    print(f"{name}: OK")


def same(name, canonical, legacy):
    check(name, jnp.array_equal(canonical, legacy))


def main():
    key = jax.random.key(0)
    x = jax.random.normal(jax.random.key(1), (8, 2))
    source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    target = Nlog_Gaussian([0.2, -0.1], [0.8, 1.2])
    flow = NSF(
        jax.random.key(2), [-3.0, -3.0], [3.0, 3.0], bins=4,
        transforms=1, hidden_features=(4,),
    ).zeros()

    signatures = {
        langevin: ("dt", "steps", "chunks"),
        hamiltonian_monte_carlo: (
            "dt", "leapfrog_steps", "trajectories", "chunks",
        ),
        sequential_monte_carlo: ("ladder", "mc_dt", "mc_steps", "chunks"),
        annealed_importance_sampling: (
            "ladder", "mc_dt", "mc_steps", "chunks",
        ),
        lbfgs: ("alpha", "steps", "chunks"),
        adamw: ("lr", "steps", "chunks"),
        quench_and_temper: (
            "opt_alpha", "opt_steps", "mc_dt", "mc_steps", "chunks",
        ),
    }
    for function, expected in signatures.items():
        names = inspect.signature(function).parameters
        check(
            f"{function.__name__} canonical signature",
            all(name in names for name in expected),
        )
        check(
            f"{function.__name__} documents aliases",
            "Deprecated keyword aliases" in function.__doc__,
        )

    same(
        "langevin old/new equivalence",
        langevin(key, x, target, dt=0.01, steps=2, adjust=False, chunks=2),
        langevin(key, x, target, step=0.01, iters=2, adjust=False, chunk=2),
    )
    same(
        "HMC old/new equivalence",
        hamiltonian_monte_carlo(
            key, x, target, dt=0.05, leapfrog_steps=2,
            trajectories=2, chunks=2,
        ),
        hamiltonian_monte_carlo(
            key, x, target, step=0.05, iters=2, burns=2, chunk=2,
        ),
    )
    smc_new = sequential_monte_carlo(
        key, x, source, target, ladder=2, mc_dt=0.01, mc_steps=1, chunks=2,
    )
    smc_old = sequential_monte_carlo(
        key, x, source, target, ladder=2, step=0.01, iters=1, chunk=2,
    )
    same("SMC samples old/new equivalence", smc_new[0], smc_old[0])
    same("SMC ESS old/new equivalence", smc_new[1], smc_old[1])
    same(
        "AIS old/new equivalence",
        annealed_importance_sampling(
            key, x, source, target, flow, "G", ladder=2,
            mc_dt=0.01, mc_steps=1, chunks=2,
        ),
        annealed_importance_sampling(
            key, x, source, target, flow, "G", ladder=2,
            step=0.01, iters=1, chunk=2,
        ),
    )
    same(
        "L-BFGS old/new equivalence",
        lbfgs(x, target, alpha=0.1, steps=2, chunks=2),
        lbfgs(x, target, step=0.1, iters=2, chunk=2),
    )
    same(
        "AdamW old/new equivalence",
        adamw(x, target, lr=0.01, steps=2, chunks=2),
        adamw(x, target, step=0.01, iters=2, chunk=2),
    )
    same(
        "quench-and-temper old/new equivalence",
        quench_and_temper(
            key, x, target, 0.1, opt_alpha=0.1, opt_steps=1,
            mc_dt=0.01, mc_steps=1, chunks=2,
        ),
        quench_and_temper(
            key, x, target, 0.1, opt_step=0.1, opt_iters=1,
            mc_step=0.01, mc_iters=1, chunk=2,
        ),
    )
    same(
        "importance weights old/new equivalence",
        importance_weights_log(x, source, target, flow, "G", chunks=2),
        importance_weights_log(x, source, target, flow, "G", chunk=2),
    )
    same(
        "coverage old/new equivalence",
        coverage(x[:4], x, k=2, chunks=2),
        coverage(x[:4], x, k=2, chunk=2),
    )

    try:
        langevin(key, x, target, dt=0.01, step=0.01)
    except TypeError as exc:
        check("duplicate old/new rejection", "both 'dt'" in str(exc))
    else:
        raise AssertionError("duplicate old/new rejection")


if __name__ == "__main__":
    main()

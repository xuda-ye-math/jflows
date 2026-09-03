"""Focused smoke test for the canonical utility API."""

import inspect
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian
from jflows.utils import (
    adamw,
    coverage,
    hamiltonian_monte_carlo,
    importance_weights_log,
    langevin,
    lbfgs,
    quench_and_temper,
    sequential_monte_carlo,
    sequential_monte_carlo_fab,
    smc,
    smc_fab,
)


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    print(f"{name}: OK")


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
        sequential_monte_carlo: (
            "ladder", "mc_dt", "mc_steps_1", "mc_steps_2", "chunks",
        ),
        lbfgs: ("alpha", "steps", "chunks"),
        adamw: ("lr", "steps", "chunks"),
        quench_and_temper: (
            "opt_dt", "opt_steps", "mc_dt", "mc_steps", "chunks", "source",
            "coeff_qt",
        ),
    }
    for function, expected in signatures.items():
        names = inspect.signature(function).parameters
        check(
            f"{function.__name__} canonical signature",
            all(name in names for name in expected),
        )
        check(f"{function.__name__} has no variadic keyword shim", "kwargs" not in names)

    check("smc is the sequential-Monte-Carlo alias", smc is sequential_monte_carlo)
    check("smc_fab is the FAB alias", smc_fab is sequential_monte_carlo_fab)
    check("smc_fab shares the SMC signature",
          list(inspect.signature(sequential_monte_carlo_fab).parameters)
          == list(inspect.signature(sequential_monte_carlo).parameters))

    try:
        langevin(key, x, target, step=0.01)
    except TypeError:
        check("retired integration keyword is rejected", True)
    else:
        raise AssertionError("retired integration keyword is rejected")

    y_langevin = langevin(
        key, x, target, dt=0.01, steps=2, adjust=False, chunks=2,
    )
    y_hmc = hamiltonian_monte_carlo(
        key, x, target, dt=0.05, leapfrog_steps=2,
        trajectories=2, chunks=2,
    )
    y_smc, y_proposal, log_w_smc = smc(
        key, x, source, target, flow, "G", ladder=2,
        mc_dt=0.05, mc_steps_1=4, mc_steps_2=4, chunks=2,
    )
    y_smc_one, _, _ = smc(
        key, x, source, target, flow, "G", ladder=1,
        mc_dt=0.05, mc_steps_1=4, mc_steps_2=4, chunks=2,
    )
    y_lbfgs = lbfgs(x, target, alpha=0.1, steps=2, chunks=2)
    y_adamw = adamw(x, target, lr=0.01, steps=2, chunks=2)
    y_qt = quench_and_temper(
        key, x, target, 0.1, opt_dt=0.1, opt_steps=1,
        mc_dt=0.01, mc_steps=1, chunks=2,
    )
    y_qt_partial = quench_and_temper(
        key, x, target, 0.1, opt_dt=0.1, opt_steps=1,
        mc_dt=0.01, mc_steps=1, chunks=2, source=source, coeff_qt=0.5,
    )
    try:
        quench_and_temper(
            key, x, target, 0.1, opt_dt=0.1, opt_steps=1,
            mc_dt=0.01, mc_steps=1, coeff_qt=0.5,
        )
    except ValueError:
        check("coeff_qt > 0 without a source is rejected", True)
    else:
        raise AssertionError("coeff_qt > 0 without a source is rejected")
    log_w = importance_weights_log(x, source, target, flow, "G", chunks=2)
    cov = coverage(x[:4], x, k=2, chunks=2)

    for name, value, shape in (
        ("langevin", y_langevin, x.shape),
        ("HMC", y_hmc, x.shape),
        ("SMC", y_smc, x.shape),
        ("SMC proposal", y_proposal, x.shape),
        ("SMC proposal log weights", log_w_smc, (x.shape[0],)),
        ("L-BFGS", y_lbfgs, x.shape),
        ("AdamW", y_adamw, x.shape),
        ("quench and temper", y_qt, x.shape),
        ("quench and temper, coeff_qt = 0.5", y_qt_partial, x.shape),
        ("importance weights", log_w, (x.shape[0],)),
    ):
        check(f"{name} finite", value.shape == shape and jnp.all(jnp.isfinite(value)))
    check("identity-flow SMC proposal is the input",
          jnp.allclose(y_proposal, x, atol=1e-6))
    check("single-level SMC finite", jnp.all(jnp.isfinite(y_smc_one)))
    check("intermediate SMC level moves the particles",
          float(jnp.abs(y_smc - x).mean()) > float(jnp.abs(y_smc_one - x).mean()))
    check("coverage finite", jnp.isfinite(cov))


if __name__ == "__main__":
    main()

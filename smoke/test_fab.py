"""Tiny smoke test of `sequential_monte_carlo_fab` — run from the repo root as
`python -m smoke.test_fab`.

With the identity flow the pushforward is the source. For the Gaussian pair
source N(0, s^2 I) and target N(0, t^2 I), the FAB law pi^2 / nu is the
Gaussian with variance 1 / (2 / t^2 - 1 / s^2) per coordinate, so the sample
variance after the two phases is checked against that value (s^2 = 2,
t^2 = 1: variance 2/3), while the phase-1 samples are checked against t^2 = 1.
Shapes, finiteness, the alias, the F/G directions, and chunking are covered.
`train_FAB_G` and `train_FABX_G` (FAB loss plus the KLXX mixture variation)
are then run on the three-mode target of example/2D_single: the training
loss reported by the Monitor must decrease and the batch and population ESS
must rise.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import re

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, Nlog_Gaussian_Mixture
from jflows.train import Monitor, train_FAB_G, train_FABX_G
from jflows.utils import (
    compute_ESS,
    importance_weights,
    sequential_monte_carlo,
    sequential_monte_carlo_fab,
    smc_fab,
)

N, D = 4000, 2
SOURCE_VARIANCE, TARGET_VARIANCE = 2.0, 1.0
FAB_VARIANCE = 1.0 / (2.0 / TARGET_VARIANCE - 1.0 / SOURCE_VARIANCE)


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    print(f"{name}: OK")


def main():
    source = Nlog_Gaussian([0.0] * D, [SOURCE_VARIANCE] * D)
    target = Nlog_Gaussian([0.0] * D, [TARGET_VARIANCE] * D)
    flow = NSF(
        jax.random.key(0), [-6.0] * D, [6.0] * D, bins=4, transforms=1,
        hidden_features=(8,),
    ).zeros()
    x = source.samples(jax.random.key(1), N)
    check("smc_fab is the alias", smc_fab is sequential_monte_carlo_fab)

    y_pi, proposal, log_w = sequential_monte_carlo(
        jax.random.key(2), x, source, target, flow, "G", ladder=4,
        mc_dt=0.1, mc_steps=10,
    )
    y_fab, proposal_fab, log_w_fab = sequential_monte_carlo_fab(
        jax.random.key(2), x, source, target, flow, "G", ladder=4,
        mc_dt=0.1, mc_steps=10,
    )
    check("shapes", y_fab.shape == x.shape and proposal_fab.shape == x.shape
          and log_w_fab.shape == (N,))
    check("finite", jnp.all(jnp.isfinite(y_fab)))
    check("phase 1 is sequential_monte_carlo", jnp.array_equal(proposal_fab, proposal)
          and jnp.array_equal(log_w_fab, log_w))
    variance_pi = float(jnp.var(y_pi, axis=0).mean())
    variance_fab = float(jnp.var(y_fab, axis=0).mean())
    print(f"variance: source {SOURCE_VARIANCE}, target samples {variance_pi:.3f} "
          f"(expected {TARGET_VARIANCE}), FAB samples {variance_fab:.3f} "
          f"(expected {FAB_VARIANCE:.3f})")
    check("phase-1 samples have the target variance", abs(variance_pi - TARGET_VARIANCE) < 0.1)
    check("FAB samples have the pi^2/nu variance", abs(variance_fab - FAB_VARIANCE) < 0.1)

    y_f, _, _ = sequential_monte_carlo_fab(
        jax.random.key(2), x, source, target, flow, "F", ladder=2,
        mc_dt=0.1, mc_steps=5, chunks=2,
    )
    check("F direction and chunks finite", y_f.shape == x.shape and jnp.all(jnp.isfinite(y_f)))
    check("F direction reaches the pi^2/nu variance",
          abs(float(jnp.var(y_f, axis=0).mean()) - FAB_VARIANCE) < 0.15)

    # training on the three-mode target of example/2D_single with its flow
    # (source N(0, 4 I), NSF with 16 bins, 4 transforms, (64, 64)); the loss
    # reported by the Monitor must decrease and the batch and population ESS rise
    source_2d = Nlog_Gaussian([0.0, 0.0], [4.0, 4.0])
    target_2d = Nlog_Gaussian_Mixture(
        weights=[1.0, 1.0, 1.0],
        mean=[[0.0, 2.4], [-2.2, -1.4], [2.2, -1.4]],
        variance=[[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]],
    )
    flow_2d = NSF(
        jax.random.key(1), [-5.0, -5.0], [5.0, 5.0], bins=16, transforms=4,
        hidden_features=(64, 64),
    ).zeros()
    x_2d = source_2d.samples(jax.random.key(2), 40000)
    lines = []
    trained, history = train_FAB_G(
        x_2d, source_2d, target_2d, flow_2d, 2000, 200, 1e-3, 1, 1e-3, 100,
        monitor=Monitor(20, "[fab] ", lines.append),
    )
    losses = [float(re.search(r"loss = ([+-][0-9.]+e[+-][0-9]+)", line).group(1)) for line in lines]
    print("loss at steps 20..200:", " ".join(f"{value:.3f}" for value in losses))
    print(f"batch ESS: first 20 steps {float(history[:20].mean()):.3f}, "
          f"last 20 steps {float(history[-20:].mean()):.3f}")
    check("loss decreases", sum(losses[-3:]) / 3 < losses[0] - 0.2)
    check("batch ESS rises", float(history[-20:].mean()) > float(history[:20].mean()) + 0.2)
    final_ess = float(compute_ESS(importance_weights(x_2d, source_2d, target_2d, trained, type="G")))
    initial_ess = float(compute_ESS(importance_weights(x_2d, source_2d, target_2d, flow_2d, type="G")))
    print(f"population ESS: identity {initial_ess:.3f}, trained {final_ess:.3f}")
    check("trained flow beats the identity on the population", final_ess > initial_ess + 0.2)

    lines = []
    trained_x, history_x = train_FABX_G(
        x_2d, source_2d, target_2d, flow_2d, 0, 2000, 200, 1e-3, 1,
        2.0, 0.5, 50, 1e-3, 100, 100, coeff_theta=1.0, coeff_alpha=0.5,
        monitor=Monitor(20, "[fabx] ", lines.append), chunks=4,
    )
    losses_x = [float(re.search(r"loss = ([+-][0-9.]+e[+-][0-9]+)", line).group(1)) for line in lines]
    print("FABX loss at steps 20..200:", " ".join(f"{value:.3f}" for value in losses_x))
    print(f"FABX batch ESS: first 20 steps {float(history_x[:20].mean()):.3f}, "
          f"last 20 steps {float(history_x[-20:].mean()):.3f}")
    check("FABX loss decreases", sum(losses_x[-3:]) / 3 < losses_x[0] - 0.2)
    check("FABX batch ESS rises", float(history_x[-20:].mean()) > float(history_x[:20].mean()) + 0.2)
    final_x = float(compute_ESS(importance_weights(x_2d, source_2d, target_2d, trained_x, type="G")))
    print(f"FABX population ESS: trained {final_x:.3f}")
    check("FABX trained flow beats the identity on the population", final_x > initial_ess + 0.2)


if __name__ == "__main__":
    main()

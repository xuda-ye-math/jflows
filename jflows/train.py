"""Single-stage training drivers for jflows.

Packs one training stage into a single compiled call: Adam on the
flow's array leaves, the whole step loop under one `lax.scan`
(one compilation regardless of `steps`), returning the trained flow
and the per-step ESS history. Only the flow is updated, and the
buffer-like leaves (the NSF/NCSF center shifts, spline bounds,
time-embedding frequencies) are gradient-protected and stay fixed.

Both drivers implement the X-regularization data pipeline on a fixed
source set: every Adam step draws a fresh `n_batch`-sized subset and
regenerates its training data on the fly — no frozen batch is ever
reused, so a single packed call does not memorize per-sample
corrections.

Public API:
    train_reverse_KL — reverse KL; each step Langevin-freshens its
                       source batch at the source potential
    train_forward_KL — forward KL; each step manufactures its target
                       batch by AIS through the CURRENT flow
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from .flow import Flow
from .loss import forward_KL, reverse_KL
from .potential import Potential
from .utils.annealing import annealed_importance_sampling
from .utils.metrics import compute_ESS_log
from .utils.rejuvenation import langevin


__all__ = ["train_forward_KL", "train_reverse_KL"]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def train_reverse_KL(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    n_batch: int,
    steps: int,
    lr: float,
    mc_step: float,
    mc_iters: int,
) -> tuple[Flow, Array]:
    """
    Single-stage reverse KL training of a flow on a fixed source set:
    minimize `reverse_KL(x, target, flow, type).mean()` with Adam for
    `steps` iterations and return the trained flow with the per-step
    ESS history.

    Every Adam iteration draws a fresh `n_batch`-sized subset of
    `x_valid` (without replacement) and freshens it with `mc_iters`
    Langevin steps at the source potential before the gradient step.
    Deterministic (no PRNG key): the per-step subsampling and
    rejuvenation keys are derived internally from a fixed seed. The
    loop runs under a single `lax.scan`, so the whole call compiles
    once; the flow is differentiated in its native direction only (see
    `reverse_KL`), and the update touches only the flow's array leaves.

    The ESS history is the flow-proposal importance-sampling ESS of
    each step's rejuvenated batch (computed from the per-sample losses
    at no extra flow evaluations: log w = source(x) - loss).

    Input:
        x_valid:  Array [N, d]   fixed set of source samples
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train
        type:     str            'F' if the flow maps source -> target;
                                 'G' if it maps target -> source
        n_batch:  int            samples drawn from the fixed set per Adam step
        steps:    int            number of Adam optimization steps
        lr:       float          Adam learning rate
        mc_step:  float          Langevin rejuvenation step size
        mc_iters: int            Langevin rejuvenation steps per batch
    Output:
        flow: Flow          the trained flow
        ess:  Array [steps] per-step batch ESS (before that step's update)
    """
    if type not in ("F", "G"):
        raise ValueError(f"train_reverse_KL: type must be 'F' or 'G', got {type!r}")

    key = jax.random.key(0)  # fixed internal seed (deterministic interface)
    N = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    def body(carry, t):
        params, m, v = carry
        key_idx, key_mc = jax.random.split(jax.random.fold_in(key, t))
        x = x_valid[jax.random.choice(key_idx, N, (n_batch,), replace=False)]
        x = langevin(key_mc, x, source, step=mc_step, iters=mc_iters)

        def loss_fn(p):
            losses = reverse_KL(x, target, eqx.combine(p, static), type)
            return losses.mean(), losses

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        ess = compute_ESS_log(source(x) - losses)  # log w = -target(y) + source(x) + ladj
        t_f = t.astype(x_valid.dtype)  # bias-correction exponent
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1**t_f), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2**t_f), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), ess

    ts = jnp.arange(1, steps + 1)  # traced step counter (per-step keys + bias correction)
    (params, _, _), ess = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static), ess


def train_forward_KL(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    type: str,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    mc_step: float,
    mc_iters: int,
) -> tuple[Flow, Array]:
    """
    Single-stage forward KL training of a flow on a fixed source set:
    minimize `forward_KL(y, source, flow, type).mean()` with Adam for
    `steps` iterations and return the trained flow with the per-step
    ESS history.

    The target samples y ~ mu_1 are manufactured internally: every Adam
    iteration draws a fresh `n_batch`-sized subset of `x_valid` (without
    replacement) and runs annealed importance sampling through the
    CURRENT flow (`ladder` rungs, Langevin rejuvenation at the target
    with `mc_iters` steps of size `mc_step`). No gradient flows through
    the data generation; the flow is differentiated in its native
    direction only (see `forward_KL`). Deterministic (no PRNG key): the
    per-step keys are derived internally from a fixed seed. The loop
    runs under a single `lax.scan`, so the whole call compiles once.

    The ESS history is the flow-proposal importance-sampling ESS of
    each step's manufactured batch (computed from the per-sample losses
    at no extra flow evaluations: log w = -target(y) + loss).

    Input:
        x_valid:  Array [N, d]   fixed set of source samples
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train
        type:     str            'F' if the flow maps source -> target;
                                 'G' if it maps target -> source
        n_batch:  int            samples drawn from the fixed set per Adam step
        steps:    int            number of Adam optimization steps
        lr:       float          Adam learning rate
        ladder:   int            AIS rungs per manufactured batch
        mc_step:  float          Langevin rejuvenation step size
        mc_iters: int            Langevin rejuvenation steps per rung
    Output:
        flow: Flow          the trained flow
        ess:  Array [steps] per-step batch ESS (before that step's update)
    """
    if type not in ("F", "G"):
        raise ValueError(f"train_forward_KL: type must be 'F' or 'G', got {type!r}")

    key = jax.random.key(0)  # fixed internal seed (deterministic interface)
    N = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    def body(carry, t):
        params, m, v = carry
        key_idx, key_ais = jax.random.split(jax.random.fold_in(key, t))
        x = x_valid[jax.random.choice(key_idx, N, (n_batch,), replace=False)]
        y = annealed_importance_sampling(
            key_ais, x, source, target, eqx.combine(params, static), type,
            ladder=ladder, step=mc_step, iters=mc_iters,
        )

        def loss_fn(p):
            losses = forward_KL(y, source, eqx.combine(p, static), type)
            return losses.mean(), losses

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        ess = compute_ESS_log(-target(y) + losses)  # log w = -target(y) + source(x) + ladj
        t_f = t.astype(x_valid.dtype)  # bias-correction exponent
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1**t_f), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2**t_f), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), ess

    ts = jnp.arange(1, steps + 1)  # traced step counter (per-step keys + bias correction)
    (params, _, _), ess = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static), ess

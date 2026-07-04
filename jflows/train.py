"""Single-stage training drivers for jflows.

Packs one training stage into a single compiled call: Adam on the
flow's array leaves, the whole step loop under one `lax.scan`
(one compilation regardless of `steps`), returning the trained flow.
Only the flow is updated — the potential and the input batch stay
fixed, and the buffer-like leaves (the NSF/NCSF center shifts, spline
bounds, time-embedding frequencies) are gradient-protected and stay
fixed as well.

Public API:
    train_reverse_KL — reverse KL over a fixed source batch
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from .flow import Flow
from .loss import reverse_KL
from .potential import Potential


__all__ = ["train_reverse_KL"]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def train_reverse_KL(
    x: Array,
    target: Potential,
    flow: Flow,
    type: str,
    steps: int,
    lr: float,
) -> Flow:
    """
    Single-stage reverse KL training of a flow on a FIXED source
    batch: minimize `reverse_KL(x, target, flow, type).mean()` with
    Adam for `steps` iterations and return the trained flow.

    Deterministic (no PRNG key): the same batch `x` is used at every
    step — draw a fresh batch and call again for multi-stage schedules
    (e.g. per rung of an annealed ladder). The loop runs under a single
    `lax.scan`, so the whole call compiles once; the flow is
    differentiated in its native direction only (see `reverse_KL`), and
    the update touches only the flow's array leaves.

    Pass a generously sized batch (an N_TRAIN-style pool) and keep
    `steps` proportionate: on a small fixed batch, long training
    memorizes per-sample corrections — visible as a collapsing
    IN-SAMPLE ESS while the fresh-batch ESS stays healthy — so evaluate
    the trained flow on fresh source samples (the N_VALID
    convention).

    Input:
        x:      Array [N, d]   samples drawn from the source distribution
        target: Potential      negative log-density of the target (up to const)
        flow:   Flow           the normalizing flow to train
        type:   str            'F' if the flow maps source -> target;
                               'G' if it maps target -> source
        steps:  int            number of Adam optimization steps
        lr:     float          Adam learning rate
    Output:
        flow: Flow   the trained flow
    """
    if type not in ("F", "G"):
        raise ValueError(f"train_reverse_KL: type must be 'F' or 'G', got {type!r}")

    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    def loss_fn(p):
        return reverse_KL(x, target, eqx.combine(p, static), type).mean()

    def body(carry, t):
        params, m, v = carry
        grads = jax.grad(loss_fn)(params)
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1**t), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2**t), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), None

    ts = jnp.arange(1, steps + 1, dtype=x.dtype)  # traced step counter (bias correction)
    (params, _, _), _ = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static)

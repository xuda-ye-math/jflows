"""Direct single-stage flow trainers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from .flow import Flow
from .loss import forward_KL_G, reverse_KL_F
from .potential import Potential
from .utils.anneal import annealed_importance_sampling
from .utils.metrics import compute_ESS_log, resample
from .utils.quench import quench_and_temper
from .utils.rejuvenation import langevin

__all__ = [
    "Monitor",
    "train_forward_KL_G",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
    "train_reverse_KL_F",
]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _training_identity(flow: Flow) -> Flow:
    """Return a trainable identity parameterization."""
    near_identity = getattr(flow, "near_identity", None)
    return flow.zeros() if near_identity is None else near_identity()


def _kept(target: Potential, samples: Array, u_clip: float) -> Array:
    """Stop-gradient mask for finite samples below the potential screen."""
    energy = target(samples)
    return lax.stop_gradient(jnp.isfinite(energy) & (energy <= u_clip))


def _masked_mean(values: Array, keep: Array) -> Array:
    """Mean over retained entries, returning zero when all are screened."""
    count = keep.astype(values.dtype).sum()
    values = jnp.where(keep, values, jnp.zeros_like(values))
    return values.sum() / jnp.maximum(count, 1.0)


def _masked_pair_mean(values: Array, keep: Array, permutation: Array) -> Array:
    """Mean over permutation pairs whose two entries are retained."""
    pair_keep = keep & keep[permutation]
    count = pair_keep.astype(values.dtype).sum()
    values = jnp.where(pair_keep, values, jnp.zeros_like(values))
    return values.sum() / jnp.maximum(count, 1.0)


def _clip_global(grads, limit: float):
    """Clip one gradient pytree by its stable global L2 norm."""
    clean = jax.tree.map(
        lambda value: jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value)),
        grads,
    )
    leaves = [value for value in jax.tree.leaves(clean) if value.size > 0]
    if not leaves:
        return clean
    square = sum(jnp.sum(jnp.square(value)) for value in leaves)
    direct = jnp.sqrt(square)
    scale = jnp.max(jnp.stack([jnp.max(jnp.abs(value)) for value in leaves]))

    def normalized(value):
        return jnp.where(
            value != 0,
            jnp.sign(value) * jnp.exp(jnp.log(jnp.abs(value)) - jnp.log(scale)),
            0.0,
        )

    scaled = jax.tree.map(normalized, clean)
    scaled_leaves = [value for value in jax.tree.leaves(scaled) if value.size > 0]
    scaled_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(value)) for value in scaled_leaves)
    )
    direct_factor = jnp.minimum(1.0, limit / (direct + _EPS))
    stable_factor = limit / (scaled_norm + _EPS)
    needs_stable_clip = jnp.log(scale) + jnp.log(scaled_norm) > jnp.log(limit)
    return jax.tree.map(
        lambda value, normalized_value: jnp.where(
            jnp.isfinite(direct),
            value * direct_factor,
            jnp.where(
                needs_stable_clip,
                normalized_value * stable_factor,
                value,
            ),
        ),
        clean,
        scaled,
    )


def _adam_step(params, first, second, grads, loss, updates, lr, g_clip):
    """Apply one guarded Adam update to the trainable flow leaves."""
    finite = jnp.isfinite(loss)
    for grad in jax.tree.leaves(grads):
        finite = finite & jnp.all(jnp.isfinite(grad))
    clean = jax.tree.map(
        lambda value: jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value)),
        grads,
    )
    if g_clip != float("inf"):
        clean = _clip_global(clean, g_clip)
    candidate_updates = updates + finite.astype(updates.dtype)
    bias_t = jnp.maximum(candidate_updates, 1)
    first_new = jax.tree.map(
        lambda old, grad: _BETA1 * old + (1 - _BETA1) * grad,
        first,
        clean,
    )
    second_new = jax.tree.map(
        lambda old, grad: _BETA2 * old + (1 - _BETA2) * grad * grad,
        second,
        clean,
    )
    first_hat = jax.tree.map(
        lambda value: value / (1 - _BETA1 ** bias_t.astype(value.dtype)),
        first_new,
    )
    second_hat = jax.tree.map(
        lambda value: value / (1 - _BETA2 ** bias_t.astype(value.dtype)),
        second_new,
    )
    params_new = jax.tree.map(
        lambda value, m, v: value - lr * m / (jnp.sqrt(v) + _EPS),
        params,
        first_hat,
        second_hat,
    )
    commit = finite
    for tree in (params_new, first_new, second_new):
        for leaf in jax.tree.leaves(tree):
            commit = commit & jnp.all(jnp.isfinite(leaf))
    params = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), params_new, params
    )
    first = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), first_new, first
    )
    second = jax.tree.map(
        lambda new, old: jnp.where(commit, new, old), second_new, second
    )
    return params, first, second, updates + commit.astype(updates.dtype)


class Monitor:
    """Report loss and optimizer-batch ESS from a compiled training scan."""

    def __init__(self, every: int, prefix: str = "", printer=print):
        self.every = every
        self.prefix = prefix
        self.printer = printer

    def _emit(self, step, loss, ess, t_start, t_end) -> None:
        self.printer(
            f"{self.prefix}[t: {float(t_start):.6f} -> "
            f"{float(t_end):.6f}] step {int(step):>5d}   "
            f"loss = {float(loss):+.4e}   ESS = {float(ess):.4f}"
        )

    def report(self, step, loss, ess, train_steps, t_start, t_end) -> None:
        """Emit the first, last, and configured intermediate steps."""
        lax.cond(
            (step == 1) | (step == train_steps) | (step % self.every == 0),
            lambda: jax.debug.callback(
                self._emit, step, loss, ess, t_start, t_end, ordered=True
            ),
            lambda: None,
        )


@eqx.filter_jit
def train_reverse_KL_F(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    train_steps,
    lr,
    mc_dt,
    mc_steps,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
):
    """Train one reverse-KL F map and return flow plus batch ESS history."""
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(1), seed)
    count = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        key_index, key_mc = jax.random.split(step_key)
        trace_key = jax.random.fold_in(step_key, 101)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        x = langevin(
            key_mc, x, source, dt=mc_dt, steps=mc_steps, adjust=mc_adjust
        )

        def loss_fn(values):
            losses = reverse_KL_F(
                x, target, eqx.combine(values, static), trace_key=trace_key
            )
            return losses.mean(), losses

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, losses), grads = jax.value_and_grad(
            evaluated, has_aux=True
        )(params)
        ess = compute_ESS_log(source(x) - losses)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, float("inf")
        )
        return state, ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history


@eqx.filter_jit
def train_forward_KL_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
):
    """Train one forward-KL G map and return flow plus batch ESS history."""
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(2), seed)
    count = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        key_index, key_ais = jax.random.split(step_key)
        trace_key = jax.random.fold_in(step_key, 101)
        ais_trace_key = jax.random.fold_in(step_key, 102)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, proposal_log_weight = annealed_importance_sampling(
            key_ais,
            x,
            source,
            target,
            eqx.combine(params, static),
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps,
            adjust=mc_adjust,
            trace_key=ais_trace_key,
            return_initial_log_weights=True,
        )
        keep = _kept(target, y, u_clip) if u_clip != float("inf") else None

        def loss_fn(values):
            losses = forward_KL_G(
                y, source, eqx.combine(values, static), trace_key=trace_key
            )
            if keep is None:
                return losses.mean(), losses
            valid = keep & lax.stop_gradient(jnp.isfinite(losses))
            return _masked_mean(losses, valid), losses

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history


@eqx.filter_jit
def train_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
):
    """Train one KLX G map and return flow plus batch ESS history."""
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(5), seed)
    count = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        key_index, key_ais, key_permutation = jax.random.split(step_key, 3)
        trace_key = jax.random.fold_in(step_key, 101)
        ais_trace_key = jax.random.fold_in(step_key, 102)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, proposal_log_weight = annealed_importance_sampling(
            key_ais,
            x,
            source,
            target,
            eqx.combine(params, static),
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps,
            adjust=mc_adjust,
            trace_key=ais_trace_key,
            return_initial_log_weights=True,
        )
        permutation = jax.random.permutation(key_permutation, batch_size)
        keep = _kept(target, y, u_clip) if u_clip != float("inf") else None

        def loss_fn(values):
            z = forward_KL_G(
                y, source, eqx.combine(values, static), trace_key=trace_key
            ) - target(y)
            difference = jnp.abs(z - z[permutation])
            if keep is None:
                return (z + coeff_lambda * difference).mean(), z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            z_safe = jnp.where(valid, z, jnp.zeros_like(z))
            loss = _masked_mean(z, valid) + coeff_lambda * _masked_pair_mean(
                jnp.abs(z_safe - z_safe[permutation]), valid, permutation
            )
            return loss, z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history


def train_forward_KLXX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    train_steps,
    lr,
    ladder,
    melt,
    opt_dt,
    opt_steps,
    mc_dt,
    mc_steps,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    *,
    chunks=1,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
):
    """Train one KLXX G map and return flow plus batch ESS history."""
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(7), seed)
    if pool_size == 0:
        pool = x_valid
        key_qt = jax.random.fold_in(key, 0)
    else:
        key_pool, key_qt = jax.random.split(jax.random.fold_in(key, 0))
        pool = x_valid[
            jax.random.randint(key_pool, (pool_size,), 0, x_valid.shape[0])
        ]
    hat_pool = quench_and_temper(
        key_qt,
        pool,
        target,
        melt,
        opt_dt,
        opt_steps,
        mc_dt,
        mc_steps,
        mc_adjust,
        chunks=chunks,
    )
    count = x_valid.shape[0]
    pool_count = hat_pool.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)
    mixture_weights = jnp.concatenate([
        jnp.full(batch_size, coeff_alpha),
        jnp.full(batch_size, coeff_beta),
    ])

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        (
            key_index,
            key_ais,
            key_permutation,
            key_hat,
            key_hat_mc,
            key_mixture,
            key_mixture_permutation,
        ) = jax.random.split(step_key, 7)
        trace_key = jax.random.fold_in(step_key, 101)
        mixture_trace_key = jax.random.fold_in(step_key, 102)
        ais_trace_key = jax.random.fold_in(step_key, 103)
        flow_now = eqx.combine(params, static)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, proposal_log_weight = annealed_importance_sampling(
            key_ais,
            x,
            source,
            target,
            flow_now,
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps,
            adjust=mc_adjust,
            trace_key=ais_trace_key,
            return_initial_log_weights=True,
        )
        permutation = jax.random.permutation(key_permutation, batch_size)
        y_hat = hat_pool[
            jax.random.randint(key_hat, (batch_size,), 0, pool_count)
        ]
        y_hat = langevin(
            key_hat_mc,
            y_hat,
            target,
            dt=mc_dt,
            steps=mc_steps,
            adjust=mc_adjust,
        )
        y_bar = flow_now.inv(x)
        y_mixture = resample(
            key_mixture,
            jnp.concatenate([y_hat, y_bar]),
            mixture_weights,
            N=batch_size,
        )
        mixture_permutation = jax.random.permutation(
            key_mixture_permutation, batch_size
        )
        screen = u_clip != float("inf")
        keep = _kept(target, y, u_clip) if screen else None
        keep_mixture = _kept(target, y_mixture, u_clip) if screen else None

        def loss_fn(values):
            current = eqx.combine(values, static)
            z = forward_KL_G(
                y, source, current, trace_key=trace_key
            ) - target(y)
            z_mixture = forward_KL_G(
                y_mixture,
                source,
                current,
                trace_key=mixture_trace_key,
            ) - target(y_mixture)
            mixture_scale = (coeff_alpha + coeff_beta) ** 2
            if keep is None:
                loss = (
                    z + coeff_lambda * jnp.abs(z - z[permutation])
                ).mean() + mixture_scale * jnp.abs(
                    z_mixture - z_mixture[mixture_permutation]
                ).mean()
                return loss, z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            valid_mixture = keep_mixture & lax.stop_gradient(
                jnp.isfinite(z_mixture)
            )
            z_safe = jnp.where(valid, z, jnp.zeros_like(z))
            mixture_safe = jnp.where(
                valid_mixture, z_mixture, jnp.zeros_like(z_mixture)
            )
            loss = (
                _masked_mean(z, valid)
                + coeff_lambda * _masked_pair_mean(
                    jnp.abs(z_safe - z_safe[permutation]), valid, permutation
                )
                + mixture_scale * _masked_pair_mean(
                    jnp.abs(
                        mixture_safe - mixture_safe[mixture_permutation]
                    ),
                    valid_mixture,
                    mixture_permutation,
                )
            )
            return loss, z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, train_steps, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, train_steps + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history

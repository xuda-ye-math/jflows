"""Direct single-stage flow trainers.

Every trainer runs ``steps_total`` Adam steps in one compiled scan and
regenerates its batch inside every step. Two Langevin budgets: ``mc_steps_1``
rejuvenates batch-sized sets (the reverse KL source batch, every level of the
SMC, and the quench-and-temper rows of the KLXX mixture batch);
``mc_steps_2`` is the temper of the whole quench-and-temper pool,
which is built once before the scan. Inside the SMC every level rejuvenates
with ``mc_steps_1`` MALA steps of size ``mc_dt`` at its own distribution of
the geometric path (the target on the last level).

The X functional of KLX and KLXX is the exact batch Gini mean difference of
the log-ratio, evaluated by one sort (``_variation``); there is no random
pairing.

``train_FAB_G`` is the FAB surrogate loss without a replay buffer: the
log-ratio averaged over a batch drawn from ``pi^2 / nu`` by
``sequential_monte_carlo_fab`` (two phases of ``ladder`` levels each), whose
gradient is that of the alpha = 2 divergence estimated from those samples.
``train_FABX_G`` adds ``coeff_theta`` times the mixture variation of KLXX to
that loss (no target variation: the FAB loss already drives the weights
towards uniformity). ``train_forward_KLL1_G`` is the forward KL with the
centered L1 log-dispersion of LDR-L1 (``_dispersion``: the mean absolute
deviation of the log-ratio from its batch mean) in place of the variation.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from .flow import Flow
from .loss import forward_KL_G, reverse_KL_F
from .potential import Potential
from .utils.anneal import sequential_monte_carlo, sequential_monte_carlo_fab
from .utils.metrics import compute_ESS_log, resample
from .utils.quench import quench_and_temper
from .utils.rejuvenation import langevin

__all__ = [
    "Monitor",
    "train_FAB_G",
    "train_FABX_G",
    "train_forward_KL_G",
    "train_forward_KLL1_G",
    "train_forward_KLX_G",
    "train_forward_KLXX_G",
    "train_reverse_KL_F",
]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _training_identity(flow: Flow) -> Flow:
    """Return a trainable identity parameterization."""
    near_identity = getattr(flow, "near_identity", None)
    return flow.zeros() if near_identity is None else near_identity()


def _kept(energy: Array, u_clip: float) -> Array:
    """Stop-gradient mask for finite target energies below the potential screen."""
    return lax.stop_gradient(jnp.isfinite(energy) & (energy <= u_clip))


def _masked_mean(values: Array, keep: Array) -> Array:
    """Mean over retained entries, returning zero when all are screened."""
    count = keep.astype(values.dtype).sum()
    values = jnp.where(keep, values, jnp.zeros_like(values))
    return values.sum() / jnp.maximum(count, 1.0)


def _variation(z: Array, keep: Array | None = None) -> Array:
    """Exact mean of |z_i - z_j| over all pairs i != j (the Gini mean difference).

    With z sorted ascending, sum_{i<j} (z_j - z_i) = sum_k (2 k - n - 1) z_(k), so
    the unbiased pairwise mean is 2 / (n (n - 1)) times that sum: one sort, no
    extra energy or flow evaluation, and no random pairing. Screened rows
    (``keep`` false) are excluded; fewer than two retained rows give zero.
    """
    n = z.shape[0]
    position = jnp.arange(1, n + 1, dtype=z.dtype)
    if keep is None:
        return 2.0 * jnp.sum((2.0 * position - n - 1.0) * jnp.sort(z)) / (n * (n - 1.0))
    count = keep.astype(z.dtype).sum()
    ordered = jnp.sort(jnp.where(keep, z, jnp.inf))            # retained rows first
    coefficient = jnp.where(position <= count, 2.0 * position - count - 1.0, 0.0)
    ordered = jnp.where(jnp.isfinite(ordered), ordered, 0.0)
    return 2.0 * jnp.sum(coefficient * ordered) / jnp.maximum(count * (count - 1.0), 1.0)


def _dispersion(z: Array, keep: Array | None = None) -> Array:
    """Mean absolute deviation of z from its batch mean (the L1 log-dispersion of LDR-L1).

    Screened rows (``keep`` false) are excluded from the mean and the deviation;
    the center is differentiated through.
    """
    if keep is None:
        return jnp.abs(z - z.mean()).mean()
    center = _masked_mean(z, keep)
    return _masked_mean(jnp.abs(z - center), keep)


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

    def report(self, step, loss, ess, steps_total, t_start, t_end) -> None:
        """Emit every ``every``-th step and the last step."""
        lax.cond(
            (step == steps_total) | (step % self.every == 0),
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
    steps_total,
    lr,
    mc_dt,
    mc_steps_1,
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
            key_mc, x, source, dt=mc_dt, steps=mc_steps_1, adjust=mc_adjust
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
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, float("inf")
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
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
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
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
    """Train one forward KL G map and return flow plus batch ESS history."""
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
        key_index, key_smc = jax.random.split(step_key)
        trace_key = jax.random.fold_in(step_key, 101)
        smc_trace_key = jax.random.fold_in(step_key, 102)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, _, proposal_log_weight = sequential_monte_carlo(
            key_smc,
            x,
            source,
            target,
            eqx.combine(params, static),
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps_1,
            adjust=mc_adjust,
            trace_key=smc_trace_key,
        )
        keep = _kept(target(y), u_clip) if u_clip != float("inf") else None

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
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history


@eqx.filter_jit
def train_FAB_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
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
    """Train one G map with the FAB surrogate loss and return flow plus batch ESS history.

    Every step draws a source batch, runs ``sequential_monte_carlo_fab`` through
    the current flow (phase 1 to the target, phase 2 on to ``pi^2 / nu``, each
    of ``ladder`` levels with ``mc_steps_1`` steps per level), and minimizes the
    mean log-ratio ``z = source(G(y)) - target(y) - log|det J_G(y)|`` over the
    ``pi^2 / nu`` batch, the gradient of the alpha = 2 divergence. No replay
    buffer. The reported batch ESS is the proposal ESS of phase 1.
    """
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(3), seed)
    count = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        key_index, key_smc = jax.random.split(step_key)
        trace_key = jax.random.fold_in(step_key, 101)
        smc_trace_key = jax.random.fold_in(step_key, 102)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, _, proposal_log_weight = sequential_monte_carlo_fab(
            key_smc,
            x,
            source,
            target,
            eqx.combine(params, static),
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps_1,
            adjust=mc_adjust,
            trace_key=smc_trace_key,
        )
        energy = target(y)
        keep = _kept(energy, u_clip) if u_clip != float("inf") else None

        def loss_fn(values):
            z = forward_KL_G(
                y, source, eqx.combine(values, static), trace_key=trace_key
            ) - energy
            if keep is None:
                return z.mean(), z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            return _masked_mean(z, valid), z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
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
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
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
        key_index, key_smc = jax.random.split(step_key)
        trace_key = jax.random.fold_in(step_key, 101)
        smc_trace_key = jax.random.fold_in(step_key, 102)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, _, proposal_log_weight = sequential_monte_carlo(
            key_smc,
            x,
            source,
            target,
            eqx.combine(params, static),
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps_1,
            adjust=mc_adjust,
            trace_key=smc_trace_key,
        )
        energy = target(y)
        keep = _kept(energy, u_clip) if u_clip != float("inf") else None

        def loss_fn(values):
            z = forward_KL_G(
                y, source, eqx.combine(values, static), trace_key=trace_key
            ) - energy
            if keep is None:
                return z.mean() + coeff_lambda * _variation(z), z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            return _masked_mean(z, valid) + coeff_lambda * _variation(z, valid), z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history


@eqx.filter_jit
def train_forward_KLL1_G(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    steps_total,
    lr,
    ladder,
    mc_dt,
    mc_steps_1,
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
    """Train one LDR-L1 G map (forward KL + L1 log-dispersion) and return flow plus batch ESS history.

    Identical to ``train_forward_KLX_G`` except that the target-batch term is
    ``coeff_lambda`` times the mean absolute deviation of the log-ratio from
    its batch mean (``_dispersion``) instead of the pairwise variation.
    """
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(13), seed)
    count = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        key_index, key_smc = jax.random.split(step_key)
        trace_key = jax.random.fold_in(step_key, 101)
        smc_trace_key = jax.random.fold_in(step_key, 102)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        y, _, proposal_log_weight = sequential_monte_carlo(
            key_smc,
            x,
            source,
            target,
            eqx.combine(params, static),
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps_1,
            adjust=mc_adjust,
            trace_key=smc_trace_key,
        )
        energy = target(y)
        keep = _kept(energy, u_clip) if u_clip != float("inf") else None

        def loss_fn(values):
            z = forward_KL_G(
                y, source, eqx.combine(values, static), trace_key=trace_key
            ) - energy
            if keep is None:
                return z.mean() + coeff_lambda * _dispersion(z), z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            return _masked_mean(z, valid) + coeff_lambda * _dispersion(z, valid), z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
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
    steps_total,
    lr,
    ladder,
    melt,
    opt_dt,
    opt_steps,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
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
    """Train one KLXX G map and return flow plus batch ESS history.

    The loss is the forward KL plus ``coeff_lambda`` times the variation of
    the log-ratio over the target batch plus ``coeff_theta`` times its
    variation over the mixture batch, drawn with proportion ``coeff_alpha``
    from the rejuvenated quench-and-temper pool and ``1 - coeff_alpha`` from
    the detached pushforward of the source batch. ``coeff_qt > 0`` resamples
    the pool by the partial importance weights exp(coeff_qt * (U_0 - U))
    and rejuvenates it before the quench and temper (see
    ``quench_and_temper``).
    """
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
        mc_steps_2,
        mc_adjust,
        chunks=chunks,
        source=source,
        coeff_qt=coeff_qt,
    )
    count = x_valid.shape[0]
    pool_count = hat_pool.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)
    mixture_weights = jnp.concatenate([
        jnp.full(batch_size, coeff_alpha),
        jnp.full(batch_size, 1.0 - coeff_alpha),
    ])

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        (
            key_index,
            key_smc,
            key_hat,
            key_hat_mc,
            key_mixture,
        ) = jax.random.split(step_key, 5)
        trace_key = jax.random.fold_in(step_key, 101)
        mixture_trace_key = jax.random.fold_in(step_key, 102)
        smc_trace_key = jax.random.fold_in(step_key, 103)
        flow_now = eqx.combine(params, static)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        # y_bar is the SMC proposal itself: the pushforward of the source
        # batch through the current G^{-1}, detached (computed outside the
        # differentiated loss), so no second inverse pass is needed.
        y, y_bar, proposal_log_weight = sequential_monte_carlo(
            key_smc,
            x,
            source,
            target,
            flow_now,
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps_1,
            adjust=mc_adjust,
            trace_key=smc_trace_key,
        )
        y_hat = hat_pool[
            jax.random.randint(key_hat, (batch_size,), 0, pool_count)
        ]
        y_hat = langevin(
            key_hat_mc,
            y_hat,
            target,
            dt=mc_dt,
            steps=mc_steps_1,
            adjust=mc_adjust,
        )
        y_mixture = resample(
            key_mixture,
            jnp.concatenate([y_hat, y_bar]),
            mixture_weights,
            N=batch_size,
        )
        energy = target(y)
        energy_mixture = target(y_mixture)
        screen = u_clip != float("inf")
        keep = _kept(energy, u_clip) if screen else None
        keep_mixture = _kept(energy_mixture, u_clip) if screen else None

        def loss_fn(values):
            current = eqx.combine(values, static)
            z = forward_KL_G(
                y, source, current, trace_key=trace_key
            ) - energy
            z_mixture = forward_KL_G(
                y_mixture,
                source,
                current,
                trace_key=mixture_trace_key,
            ) - energy_mixture
            if keep is None:
                loss = (
                    z.mean()
                    + coeff_lambda * _variation(z)
                    + coeff_theta * _variation(z_mixture)
                )
                return loss, z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            valid_mixture = keep_mixture & lax.stop_gradient(
                jnp.isfinite(z_mixture)
            )
            loss = (
                _masked_mean(z, valid)
                + coeff_lambda * _variation(z, valid)
                + coeff_theta * _variation(z_mixture, valid_mixture)
            )
            return loss, z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history


def train_FABX_G(
    x_valid,
    source,
    target,
    flow,
    pool_size,
    batch_size,
    steps_total,
    lr,
    ladder,
    melt,
    opt_dt,
    opt_steps,
    mc_dt,
    mc_steps_1,
    mc_steps_2,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
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
    """Train one G map with the FAB loss plus the mixture variation and return flow plus batch ESS history.

    The loss is the mean log-ratio over the ``pi^2 / nu`` batch of
    ``sequential_monte_carlo_fab`` (the FAB surrogate, as in ``train_FAB_G``)
    plus ``coeff_theta`` times the exact variation of the log-ratio over the
    mixture batch of KLXX, drawn with proportion ``coeff_alpha`` from the
    rejuvenated quench-and-temper pool and ``1 - coeff_alpha`` from the
    detached pushforward of the source batch. There is no target variation:
    the FAB loss already drives the importance weights towards uniformity.
    The pool, ``coeff_qt``, ``mc_steps_1`` and ``mc_steps_2`` are as in
    ``train_forward_KLXX_G``.
    """
    x_valid = jnp.asarray(x_valid)
    if initialize_from_identity:
        flow = _training_identity(flow)
    t_start, t_end = jnp.asarray(t_start), jnp.asarray(t_end)
    key = jax.random.fold_in(jax.random.key(11), seed)
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
        mc_steps_2,
        mc_adjust,
        chunks=chunks,
        source=source,
        coeff_qt=coeff_qt,
    )
    count = x_valid.shape[0]
    pool_count = hat_pool.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)
    updates = jnp.asarray(0, dtype=jnp.int32)
    mixture_weights = jnp.concatenate([
        jnp.full(batch_size, coeff_alpha),
        jnp.full(batch_size, 1.0 - coeff_alpha),
    ])

    def step_fn(state, step):
        params, first, second, updates = state
        step_key = jax.random.fold_in(key, step)
        (
            key_index,
            key_smc,
            key_hat,
            key_hat_mc,
            key_mixture,
        ) = jax.random.split(step_key, 5)
        trace_key = jax.random.fold_in(step_key, 101)
        mixture_trace_key = jax.random.fold_in(step_key, 102)
        smc_trace_key = jax.random.fold_in(step_key, 103)
        flow_now = eqx.combine(params, static)
        x = x_valid[
            jax.random.choice(key_index, count, (batch_size,), replace=False)
        ]
        # y_bar is the phase-1 proposal: the pushforward of the source batch
        # through the current G^{-1}, detached.
        y, y_bar, proposal_log_weight = sequential_monte_carlo_fab(
            key_smc,
            x,
            source,
            target,
            flow_now,
            "G",
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps_1,
            adjust=mc_adjust,
            trace_key=smc_trace_key,
        )
        y_hat = hat_pool[
            jax.random.randint(key_hat, (batch_size,), 0, pool_count)
        ]
        y_hat = langevin(
            key_hat_mc,
            y_hat,
            target,
            dt=mc_dt,
            steps=mc_steps_1,
            adjust=mc_adjust,
        )
        y_mixture = resample(
            key_mixture,
            jnp.concatenate([y_hat, y_bar]),
            mixture_weights,
            N=batch_size,
        )
        energy = target(y)
        energy_mixture = target(y_mixture)
        screen = u_clip != float("inf")
        keep = _kept(energy, u_clip) if screen else None
        keep_mixture = _kept(energy_mixture, u_clip) if screen else None

        def loss_fn(values):
            current = eqx.combine(values, static)
            z = forward_KL_G(
                y, source, current, trace_key=trace_key
            ) - energy
            z_mixture = forward_KL_G(
                y_mixture,
                source,
                current,
                trace_key=mixture_trace_key,
            ) - energy_mixture
            if keep is None:
                return z.mean() + coeff_theta * _variation(z_mixture), z
            valid = keep & lax.stop_gradient(jnp.isfinite(z))
            valid_mixture = keep_mixture & lax.stop_gradient(
                jnp.isfinite(z_mixture)
            )
            loss = _masked_mean(z, valid) + coeff_theta * _variation(
                z_mixture, valid_mixture
            )
            return loss, z

        evaluated = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, _), grads = jax.value_and_grad(evaluated, has_aux=True)(params)
        ess = compute_ESS_log(proposal_log_weight)
        if monitor is not None:
            monitor.report(step, loss, ess, steps_total, t_start, t_end)
        state = _adam_step(
            params, first, second, grads, loss, updates, lr, g_clip
        )
        return state, ess

    steps = jnp.arange(1, steps_total + 1)
    (params, _, _, _), ess_history = lax.scan(
        step_fn, (params, first, second, updates), steps
    )
    return eqx.combine(params, static), ess_history

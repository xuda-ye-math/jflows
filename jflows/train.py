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
    train_reverse_KL_F — reverse KL (flow fixed as F, source -> target);
                       each step Langevin-freshens its source batch at the
                       source potential
    train_forward_KL_G — forward KL (flow fixed as G, target -> source);
                       each step manufactures its target batch by AIS
                       through the CURRENT flow
    train_forward_KLX_G — forward KL + X functional (flow fixed as G,
                       target -> source): train_forward_KL_G with the
                       coeff_lambda-weighted X regularizer added
    train_forward_KLXX_G — the full X-regularized forward KL:
                       KL + coeff_lambda * X_mu
                       + X_{coeff_alpha * hat_mu + coeff_beta * bar_nu},
                       with hat_mu a quench-and-temper pool of size
                       n_pool and bar_nu the detached pushforward
    Monitor          — live training-status reporter (loss + batch ESS
                       every `every` steps, from inside the compiled loop)

The annealed Boltzmann generators that chain these stage trainers into the
full adaptive-temperature ladder live in `jflows.boltzmann`.

Both trainers are `eqx.filter_jit`-compiled: python scalars (`n_batch`,
`steps`, `lr`, `mc_*`, `type`) and the Monitor instance are static, so
repeated calls with the same configuration — every stage of a
boltzmann ladder, retuned bridge coefficients included — reuse one XLA
executable. Memory caveat: training a MAF-style flow (NSF/NCSF) in its
NON-native direction, or a CNF with `exact=True`, keeps large autodiff
residuals per step; pass `checkpoint=True` to rematerialize the loss
forward pass during the backward (more compute, much less memory).
"""

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


__all__ = ["Monitor", "train_forward_KL_G", "train_forward_KLX_G",
           "train_forward_KLXX_G", "train_reverse_KL_F"]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _mask_keep(target: Potential, y: Array, e_clip: float) -> Array:
    """Per-sample keep mask of the energy screen: True where the target
    energy is admissible (target(y) <= e_clip), stop-gradient. Screened
    samples are the near-singular particles the loss must not see."""
    return lax.stop_gradient(target(y) <= e_clip)


def _masked_mean(v: Array, keep: Array) -> Array:
    """Mean of `v` over the kept samples only (screened entries drop from
    numerator and denominator); an all-screened batch yields 0."""
    kf = keep.astype(v.dtype)
    return (kf * v).sum() / jnp.maximum(kf.sum(), 1.0)


def _masked_pair_mean(diff: Array, keep: Array, perm: Array) -> Array:
    """Mean of a permutation-paired term |z - z[perm]| over pairs whose
    BOTH members are kept, so a screened sample contributes to neither its
    own nor its partner's term."""
    kf = keep.astype(diff.dtype)
    pk = kf * kf[perm]
    return (pk * diff).sum() / jnp.maximum(pk.sum(), 1.0)


def _clip_global(grads, g_clip: float):
    """Scale the whole gradient pytree so its global L2 norm is at most
    g_clip (no-op when the norm is already below the ceiling)."""
    gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    return jax.tree.map(lambda g: g * jnp.minimum(1.0, g_clip / (gnorm + _EPS)), grads)


class Monitor:
    """
    Live training-status monitor for the packed train_* drivers.

    Reports the Adam step, the mean loss, and the batch ESS every
    `every` steps through `printer`. The drivers invoke it inside the
    compiled `lax.scan` via `jax.debug.callback`, so the status appears
    WHILE the packed call is running (the non-reporting steps skip the
    callback entirely via `lax.cond`).

    Input:
        every:   int        report every `every` Adam steps (e.g. 10)
        prefix:  str        line prefix, e.g. '[reverse KL] ' (default '')
        printer: callable   line consumer (default print; pass a log
                            function to also write a status file)
    """

    def __init__(self, every: int, prefix: str = "", printer=print):
        if every < 1:
            raise ValueError(f"Monitor: every must be a positive int, got {every!r}")
        self.every = int(every)
        self.prefix = prefix
        self.printer = printer

    def _emit(self, t, loss, ess) -> None:
        self.printer(f"{self.prefix}step {int(t):>5d}   "
                     f"loss = {float(loss):+.4e}   ESS = {float(ess):.4f}")

    def report(self, t: Array, loss: Array, ess: Array) -> None:
        """Called by the drivers inside the scan body (traced values)."""
        lax.cond(
            t % self.every == 0,
            lambda: jax.debug.callback(self._emit, t, loss, ess),
            lambda: None,
        )


@eqx.filter_jit
def train_reverse_KL_F(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    n_batch: int,
    steps: int,
    lr: float,
    mc_step: float,
    mc_iters: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    seed: int | Array = 0,
    checkpoint: bool = False,
) -> tuple[Flow, Array]:
    """
    Single-stage reverse KL training of a flow on a fixed source set,
    with the flow fixed as the forward map F (source -> target): minimize
    `reverse_KL_F(x, target, flow).mean()` with Adam for `steps`
    iterations and return the trained flow with the per-step ESS history.

    Every Adam iteration draws a fresh `n_batch`-sized subset of
    `x_valid` (without replacement; note the draw sorts the full pool
    per step — at pools beyond ~1e6 prefer an external batching scheme)
    and freshens it with `mc_iters` Langevin steps at the source
    potential before the gradient step. Deterministic: the per-step
    keys derive from this driver's own base stream folded with `seed`
    (distinct base streams per driver — no cross-driver collisions);
    pass distinct `seed` values (a traced array avoids recompilation)
    to decorrelate repeated runs. The loop runs under a single
    `lax.scan` and the call is `filter_jit`-compiled, so one XLA
    executable serves all same-configuration calls; the flow is
    differentiated in its native F direction only (see `reverse_KL_F`),
    and the update touches only the flow's array leaves.

    The ESS history is the flow-proposal importance-sampling ESS of
    each step's rejuvenated batch (computed from the per-sample losses
    at no extra flow evaluations: log w = source(x) - loss).

    Input:
        x_valid:  Array [N, d]   fixed set of source samples
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train, applied as F
                                 (source -> target)
        n_batch:  int            samples drawn from the fixed set per Adam step
        steps:    int            number of Adam optimization steps
        lr:       float          Adam learning rate
        mc_step:  float          Langevin rejuvenation step size
        mc_iters: int            Langevin rejuvenation steps per batch
        mc_adjust: bool          False: unadjusted ULA; True: MALA — the
                                 Metropolis gate rejects proposals into steep
                                 walls (the reference guard for near-singular
                                 targets like regularized Coulomb)
        monitor:  Monitor        optional live status reporter (step, loss,
                                 batch ESS every `monitor.every` steps)
        seed:     int | Array    extra fold into the internal PRNG stream
                                 (default 0; pass a traced array to avoid
                                 recompilation across many seeds)
        checkpoint: bool         rematerialize the loss forward pass in the
                                 backward (jax.checkpoint) — use for CNF
                                 exact-trace or non-native-direction training
    Output:
        flow: Flow          the trained flow
        ess:  Array [steps] per-step batch ESS (before that step's update)
    """
    key = jax.random.fold_in(jax.random.key(1), seed)  # driver-specific base stream
    N = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    def body(carry, t):
        params, m, v = carry
        key_idx, key_mc = jax.random.split(jax.random.fold_in(key, t))
        x = x_valid[jax.random.choice(key_idx, N, (n_batch,), replace=False)]
        x = langevin(key_mc, x, source, step=mc_step, iters=mc_iters, adjust=mc_adjust)

        def loss_fn(p):
            losses = reverse_KL_F(x, target, eqx.combine(p, static))
            return losses.mean(), losses

        loss_eval = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, losses), grads = jax.value_and_grad(loss_eval, has_aux=True)(params)
        ess = compute_ESS_log(source(x) - losses)  # log w = -target(y) + source(x) + ladj
        if monitor is not None:
            monitor.report(t, loss, ess)
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1 ** t.astype(a.dtype)), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2 ** t.astype(a.dtype)), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), ess

    ts = jnp.arange(1, steps + 1)  # traced step counter (per-step keys + bias correction)
    (params, _, _), ess = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static), ess


@eqx.filter_jit
def train_forward_KL_G(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    mc_step: float,
    mc_iters: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
) -> tuple[Flow, Array]:
    """
    Single-stage forward KL training of a flow on a fixed source set,
    with the flow fixed as the inverse map G (target -> source): minimize
    `forward_KL_G(y, source, flow).mean()` with Adam for `steps`
    iterations and return the trained flow with the per-step ESS history.

    The target samples y ~ mu_1 are manufactured internally: every Adam
    iteration draws a fresh `n_batch`-sized subset of `x_valid` (without
    replacement) and runs annealed importance sampling through the
    CURRENT flow (`ladder` rungs, Langevin rejuvenation at the target
    with `mc_iters` steps of size `mc_step`). No gradient flows through
    the data generation; the flow is differentiated in its native G
    direction only (see `forward_KL_G`). Deterministic (no PRNG key): the
    per-step keys are derived internally from a fixed seed. The loop
    runs under a single `lax.scan`, so the whole call compiles once.

    The ESS history evaluates the flow-vs-target overlap on each
    step's manufactured batch (log w = -target(y) + loss, at no extra
    flow evaluations). Since y is (approximately) target-distributed,
    this is the reverse-direction chi^2 overlap — in (0, 1], equal to 1
    iff proposal == target, a valid convergence monitor — but its value
    is not numerically comparable to the proposal-side
    `importance_weights` -> `compute_ESS` on the same flow.

    Input:
        x_valid:  Array [N, d]   fixed set of source samples
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train, applied as G
                                 (target -> source)
        n_batch:  int            samples drawn from the fixed set per Adam step
        steps:    int            number of Adam optimization steps
        lr:       float          Adam learning rate
        ladder:   int            AIS rungs per manufactured batch
        mc_step:  float          Langevin rejuvenation step size
        mc_iters: int            Langevin rejuvenation steps per rung
        mc_adjust: bool          False: unadjusted ULA in the AIS
                                 rejuvenation; True: MALA (the Metropolis
                                 gate for near-singular targets)
        monitor:  Monitor        optional live status reporter (step, loss,
                                 batch ESS every `monitor.every` steps)
        seed:     int | Array    extra fold into the internal PRNG stream
                                 (default 0; traced array avoids recompiles)
        checkpoint: bool         rematerialize the loss forward pass in the
                                 backward (jax.checkpoint)
        e_clip:   float          energy screen: manufactured samples with
                                 target(y) > e_clip make no contribution to
                                 the loss (masked mean over the kept batch).
                                 Default inf keeps every sample (exact
                                 unscreened behaviour, no overhead)
        g_clip:   float          global gradient-norm clip: the raw gradient
                                 is scaled to L2 norm at most g_clip BEFORE
                                 the Adam moment update — a spike guard that
                                 keeps an outlier high-energy batch from
                                 corrupting Adam's running moments (Adam's
                                 per-coordinate normalization makes the effect
                                 on the final step size weak). Default inf
                                 leaves the gradient untouched
    Output:
        flow: Flow          the trained flow
        ess:  Array [steps] per-step batch ESS (before that step's update)
    """
    key = jax.random.fold_in(jax.random.key(2), seed)  # driver-specific base stream
    N = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    def body(carry, t):
        params, m, v = carry
        key_idx, key_ais = jax.random.split(jax.random.fold_in(key, t))
        x = x_valid[jax.random.choice(key_idx, N, (n_batch,), replace=False)]
        y = annealed_importance_sampling(
            key_ais, x, source, target, eqx.combine(params, static), "G",
            ladder=ladder, step=mc_step, iters=mc_iters, adjust=mc_adjust,
        )
        keep = _mask_keep(target, y, e_clip) if e_clip != float("inf") else None

        def loss_fn(p):
            losses = forward_KL_G(y, source, eqx.combine(p, static))
            loss = losses.mean() if keep is None else _masked_mean(losses, keep)
            return loss, losses

        loss_eval = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, losses), grads = jax.value_and_grad(loss_eval, has_aux=True)(params)
        if g_clip != float("inf"):
            grads = _clip_global(grads, g_clip)
        ess = compute_ESS_log(-target(y) + losses)  # log w = -target(y) + source(x) + ladj
        if monitor is not None:
            monitor.report(t, loss, ess)
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1 ** t.astype(a.dtype)), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2 ** t.astype(a.dtype)), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), ess

    ts = jnp.arange(1, steps + 1)  # traced step counter (per-step keys + bias correction)
    (params, _, _), ess = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static), ess


@eqx.filter_jit
def train_forward_KLX_G(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    mc_step: float,
    mc_iters: int,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
) -> tuple[Flow, Array]:
    """
    Single-stage X-regularized forward KL training of a flow on a fixed
    source set, with the flow fixed as the inverse map G (target -> source):
    minimize `forward_KLX_G(y, source, target, flow, key, coeff_lambda).mean()`
    with Adam for `steps` iterations and return the trained flow with the
    per-step ESS history.

    Identical to `train_forward_KL_G` except the objective adds the
    X functional. Writing the per-sample log-ratio
    z = source(G(y)) - target(y) - log|det J_G(y)|, the loss is
        mean(z) + coeff_lambda * mean(|z - z[perm]|)
    for a fresh random permutation `perm` of each batch. The forward KL term
    reuses the same forward pass, so the X functional costs only a permutation
    and an element-wise difference (see `forward_KLX_G`).

    The target samples y ~ mu_1 are manufactured internally: every Adam
    iteration draws a fresh `n_batch`-sized subset of `x_valid` (without
    replacement) and runs annealed importance sampling through the CURRENT
    flow (`ladder` rungs, Langevin rejuvenation at the target with `mc_iters`
    steps of size `mc_step`). No gradient flows through the data generation;
    the flow is differentiated in its native G direction only. Deterministic:
    the per-step keys (batch draw, AIS, permutation) derive from this driver's
    own base stream folded with `seed`. The loop runs under a single
    `lax.scan`, so the whole call compiles once.

    The ESS history is the honest flow-vs-target overlap of each step's
    manufactured batch, compute_ESS_log(z) with the log-ratio z above (at no
    extra flow evaluations — z falls out of the loss forward pass), so its
    value is unaffected by coeff_lambda. Since y is
    (approximately) target-distributed, this is the reverse-direction chi^2
    overlap — in (0, 1], equal to 1 iff proposal == target.

    Input:
        x_valid:  Array [N, d]   fixed set of source samples
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train, applied as G
        n_batch:  int            samples drawn from the fixed set per Adam step
        steps:    int            number of Adam optimization steps
        lr:       float          Adam learning rate
        ladder:   int            AIS rungs per manufactured batch
        mc_step:  float          Langevin rejuvenation step size
        mc_iters: int            Langevin rejuvenation steps per rung
        coeff_lambda: float      weight of the X functional term
        mc_adjust: bool          False: unadjusted ULA in the AIS
                                 rejuvenation; True: MALA (the Metropolis
                                 gate for near-singular targets)
        monitor:  Monitor        optional live status reporter (step, loss,
                                 batch ESS every `monitor.every` steps)
        seed:     int | Array    extra fold into the internal PRNG stream
                                 (default 0; traced array avoids recompiles)
        checkpoint: bool         rematerialize the loss forward pass in the
                                 backward (jax.checkpoint)
        e_clip:   float          energy screen: manufactured samples with
                                 target(y) > e_clip make no contribution to
                                 the KL term nor, as a permutation partner,
                                 to the X term. Default inf keeps every
                                 sample (exact unscreened behaviour)
        g_clip:   float          global gradient-norm clip on the raw gradient
                                 before the Adam moment update — a spike guard
                                 (default inf: none)
    Output:
        flow: Flow          the trained flow
        ess:  Array [steps] per-step batch ESS (before that step's update)
    """
    key = jax.random.fold_in(jax.random.key(5), seed)  # driver-specific base stream
    N = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    def body(carry, t):
        params, m, v = carry
        key_idx, key_ais, key_perm = jax.random.split(jax.random.fold_in(key, t), 3)
        x = x_valid[jax.random.choice(key_idx, N, (n_batch,), replace=False)]
        y = annealed_importance_sampling(
            key_ais, x, source, target, eqx.combine(params, static), "G",
            ladder=ladder, step=mc_step, iters=mc_iters, adjust=mc_adjust,
        )
        perm = jax.random.permutation(key_perm, n_batch)
        keep = _mask_keep(target, y, e_clip) if e_clip != float("inf") else None

        def loss_fn(p):
            # z: per-sample log-ratio log(mu/nu); the returned vector equals
            # forward_KLX_G(y, source, target, flow, key_perm, coeff_lambda).
            z = forward_KL_G(y, source, eqx.combine(p, static)) - target(y)
            if keep is None:
                loss = (z + coeff_lambda * jnp.abs(z - z[perm])).mean()
            else:
                loss = (_masked_mean(z, keep)
                        + coeff_lambda * _masked_pair_mean(jnp.abs(z - z[perm]), keep, perm))
            return loss, z

        loss_eval = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, z), grads = jax.value_and_grad(loss_eval, has_aux=True)(params)
        if g_clip != float("inf"):
            grads = _clip_global(grads, g_clip)
        ess = compute_ESS_log(z)  # log w = -target(y) + source(x) - ladj (no extra flow eval)
        if monitor is not None:
            monitor.report(t, loss, ess)
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1 ** t.astype(a.dtype)), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2 ** t.astype(a.dtype)), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), ess

    ts = jnp.arange(1, steps + 1)  # traced step counter (per-step keys + bias correction)
    (params, _, _), ess = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static), ess


@eqx.filter_jit
def train_forward_KLXX_G(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    n_pool: int,
    n_batch: int,
    steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_step: float,
    opt_iters: int,
    mc_step: float,
    mc_iters: int,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    seed: int | Array = 0,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
) -> tuple[Flow, Array]:
    """
    Single-stage X-regularized forward KL training with the full mixture
    loss, the flow fixed as the inverse map G (target -> source): minimize

        KL + coeff_lambda * X_mu + X_{coeff_alpha hat_mu + coeff_beta bar_nu}

    with Adam for `steps` iterations and return the trained flow with the
    per-step ESS history. Writing the per-sample log-ratio
    z = source(G(y)) - target(y) - log|det J_G(y)|, each Adam step computes

        mean(z) + coeff_lambda * mean(|z - z[perm]|)                 on the mu batch
        + (coeff_alpha + coeff_beta)^2 * mean(|z - z[perm']|)        on the mixture batch,

    the mixture scaling being the normalizer of the X functional under the
    unnormalized weight coeff_alpha * hat_mu + coeff_beta * bar_nu (equal to
    1 at the default coeff_alpha = coeff_beta = 1/2).

    The three sampling measures are supplied as follows. The mu batch is
    manufactured per step exactly as in `train_forward_KLX_G`: a fresh
    `n_batch`-sized subset of `x_valid` pushed by `ladder`-rung AIS through
    the CURRENT flow. The wide-coverage measure hat_mu is an `n_pool`-sized
    pool built ONCE per call by `quench_and_temper` on `n_pool` source
    samples (melt scale `melt`, armijo L-BFGS quench `opt_step` x
    `opt_iters`, Langevin temper `mc_step` x `mc_iters`); every step
    resamples `n_batch` particles from the pool (with replacement) and
    freshens them with `mc_iters` Langevin steps at the target. The frozen
    pushforward bar_nu is free: the step's source subset pushed through the
    CURRENT flow outside the loss gradient, a stop-gradient copy. The
    mixture batch draws `n_batch` particles from the hat/bar stack by
    multinomial resampling with weights coeff_alpha (hat half) and
    coeff_beta (bar half). No gradient flows through any data generation;
    the flow is differentiated in its native G direction only.

    Deterministic: the per-step keys (batch draw, AIS, permutations, pool
    draws, mixture resampling) derive from this driver's own base stream
    folded with `seed`. The step loop runs under a single `lax.scan` and
    the call is `filter_jit`-compiled; the quench-and-temper pool is part
    of the same compiled call, ahead of the scan.

    The ESS history is the flow-vs-target overlap of each step's mu batch,
    compute_ESS_log(z) at no extra flow evaluations, so its value is
    unaffected by all three coefficients.

    Input:
        x_valid:  Array [N, d]   fixed set of source samples
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train, applied as G
        n_pool:   int            quench-and-temper pool size (hat_mu particles,
                                 built once per call)
        n_batch:  int            samples per Adam step (mu batch and mixture
                                 batch alike)
        steps:    int            number of Adam optimization steps
        lr:       float          Adam learning rate
        ladder:   int            AIS rungs per manufactured mu batch
        melt:     float          quench-and-temper melt scale (std of the
                                 Gaussian scatter)
        opt_step: float          L-BFGS trial alpha of the quench (armijo)
        opt_iters: int           L-BFGS iterations of the quench
        mc_step:  float          Langevin step size (AIS rejuvenation, the
                                 temper, AND the per-step hat_mu freshening)
        mc_iters: int            Langevin steps (same uses)
        coeff_lambda: float      weight of the X_mu term
        coeff_alpha: float       hat_mu weight of the mixture term
        coeff_beta: float        bar_nu weight of the mixture term
        mc_adjust: bool          False: unadjusted ULA; True: MALA in the
                                 AIS rejuvenation, the temper, and the
                                 hat_mu freshening
        monitor:  Monitor        optional live status reporter (step, loss,
                                 batch ESS every `monitor.every` steps)
        seed:     int | Array    extra fold into the internal PRNG stream
                                 (default 0; traced array avoids recompiles)
        checkpoint: bool         rematerialize the loss forward pass in the
                                 backward (jax.checkpoint)
        e_clip:   float          energy screen: samples with target > e_clip
                                 make no contribution — applied to the mu
                                 batch y (KL and X_mu terms) and independently
                                 to the mixture batch y_mix (X_mix term), each
                                 as a masked mean with partner screening.
                                 Default inf keeps every sample
        g_clip:   float          global gradient-norm clip on the raw gradient
                                 before the Adam moment update — a spike guard
                                 (default inf: none)
    Output:
        flow: Flow          the trained flow
        ess:  Array [steps] per-step batch ESS (before that step's update)
    """
    key = jax.random.fold_in(jax.random.key(7), seed)  # driver-specific base stream
    N = x_valid.shape[0]
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)

    # hat_mu: the quench-and-temper wide-coverage pool, built once per call
    key_qt_idx, key_qt = jax.random.split(jax.random.fold_in(key, 0))
    pool = x_valid[jax.random.randint(key_qt_idx, (n_pool,), 0, N)]
    hat_pool = quench_and_temper(key_qt, pool, target, melt, opt_step, opt_iters,
                                 mc_step, mc_iters, mc_adjust)

    # mixture weights of the hat/bar stack (constant across steps)
    w_mix = jnp.concatenate([jnp.full(n_batch, coeff_alpha),
                             jnp.full(n_batch, coeff_beta)])

    def body(carry, t):
        params, m, v = carry
        key_idx, key_ais, key_perm, key_hat, key_hat_mc, key_mix, key_perm2 = \
            jax.random.split(jax.random.fold_in(key, t), 7)
        flow_now = eqx.combine(params, static)
        x = x_valid[jax.random.choice(key_idx, N, (n_batch,), replace=False)]
        y = annealed_importance_sampling(
            key_ais, x, source, target, flow_now, "G",
            ladder=ladder, step=mc_step, iters=mc_iters, adjust=mc_adjust,
        )
        perm = jax.random.permutation(key_perm, n_batch)
        # mixture batch: freshened hat_mu draw + detached pushforward bar_nu,
        # resampled by the (coeff_alpha, coeff_beta) weights
        y_hat = hat_pool[jax.random.randint(key_hat, (n_batch,), 0, n_pool)]
        y_hat = langevin(key_hat_mc, y_hat, target, step=mc_step, iters=mc_iters,
                         adjust=mc_adjust)
        y_bar = flow_now.inv(x)          # stop-gradient copy (outside the loss grad)
        y_mix = resample(key_mix, jnp.concatenate([y_hat, y_bar], axis=0),
                         w_mix, N=n_batch)
        perm2 = jax.random.permutation(key_perm2, n_batch)
        screen = e_clip != float("inf")
        keep = _mask_keep(target, y, e_clip) if screen else None
        keep_mix = _mask_keep(target, y_mix, e_clip) if screen else None

        def loss_fn(p):
            # per-sample log-ratios log(mu/nu); the terms equal
            # forward_KLX_G on y plus (a+b)^2 * forward_X_G on y_mix.
            fl = eqx.combine(p, static)
            z = forward_KL_G(y, source, fl) - target(y)
            zx = forward_KL_G(y_mix, source, fl) - target(y_mix)
            ab2 = (coeff_alpha + coeff_beta) ** 2
            if keep is None:
                loss = (z + coeff_lambda * jnp.abs(z - z[perm])).mean() \
                    + ab2 * jnp.abs(zx - zx[perm2]).mean()
            else:
                loss = (_masked_mean(z, keep)
                        + coeff_lambda * _masked_pair_mean(jnp.abs(z - z[perm]), keep, perm)
                        + ab2 * _masked_pair_mean(jnp.abs(zx - zx[perm2]), keep_mix, perm2))
            return loss, z

        loss_eval = jax.checkpoint(loss_fn) if checkpoint else loss_fn
        (loss, z), grads = jax.value_and_grad(loss_eval, has_aux=True)(params)
        if g_clip != float("inf"):
            grads = _clip_global(grads, g_clip)
        ess = compute_ESS_log(z)  # log w = -target(y) + source(x) - ladj (no extra flow eval)
        if monitor is not None:
            monitor.report(t, loss, ess)
        m = jax.tree.map(lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, m, grads)
        v = jax.tree.map(lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, v, grads)
        m_hat = jax.tree.map(lambda a: a / (1 - _BETA1 ** t.astype(a.dtype)), m)
        v_hat = jax.tree.map(lambda a: a / (1 - _BETA2 ** t.astype(a.dtype)), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + _EPS), params, m_hat, v_hat
        )
        return (params, m, v), ess

    ts = jnp.arange(1, steps + 1)  # traced step counter (per-step keys + bias correction)
    (params, _, _), ess = lax.scan(body, (params, m0, v0), ts)
    return eqx.combine(params, static), ess



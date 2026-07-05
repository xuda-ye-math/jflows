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
    boltzmann_reverse_KL — annealed Boltzmann generator on the bridge
                       ladder U_t = (1-t) U_0 + t U_1: per stage train ->
                       ESS evaluation -> rejection -> importance weights ->
                       rejuvenation, with adaptive t selection (bg_param)
    Monitor          — live training-status reporter (loss + batch ESS
                       every `every` steps, from inside the compiled loop)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from .flow import Flow
from .loss import forward_KL, reverse_KL
from .potential import Potential, linear_combination
from .utils.annealing import annealed_importance_sampling
from .utils.metrics import compute_ESS_log, importance_weights_log, resample
from .utils.rejuvenation import langevin


__all__ = ["Monitor", "boltzmann_reverse_KL", "train_forward_KL", "train_reverse_KL"]

_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


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
    monitor: Monitor | None = None,
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
        monitor:  Monitor        optional live status reporter (step, loss,
                                 batch ESS every `monitor.every` steps)
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

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        ess = compute_ESS_log(source(x) - losses)  # log w = -target(y) + source(x) + ladj
        if monitor is not None:
            monitor.report(t, loss, ess)
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
    monitor: Monitor | None = None,
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
        monitor:  Monitor        optional live status reporter (step, loss,
                                 batch ESS every `monitor.every` steps)
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

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        ess = compute_ESS_log(-target(y) + losses)  # log w = -target(y) + source(x) + ladj
        if monitor is not None:
            monitor.report(t, loss, ess)
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


_BG_DEFAULTS = {
    "t_safe": 0.25,        # stage-1 bridge coefficient (the safe start)
    "shrink_factor": 0.5,  # rejected stage: t_k <- t_prev + shrink_factor (t_k - t_prev)
    "enlarge_factor": 2.0, # accepted stage: t_init extrapolation growth factor
    "tau": 0.5,            # ESS acceptance threshold of a trained stage
    "t_tol": 0.1,          # snap the initial guess to t = 1 when 1 - t_k < t_tol
    "max_stages": 30,      # ladder-length safety cap
    "max_retry": 6,        # training attempts per stage before giving up
}


def boltzmann_reverse_KL(
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
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
) -> tuple[Flow, Array, list[dict]]:
    """
    Annealed Boltzmann generator on the reverse KL, as in the zflows
    4D reference: advance a particle set along the bridge ladder

        U_t = (1 - t) U_0 + t U_1,        0 < t_1 < ... < t_K = 1,

    where t is the linear-combination coefficient, by training the
    warm-started flow as the INCREMENTAL map from the current particle
    distribution mu_{t_{k-1}} to the next bridge mu_{t_k} — each stage
    only ever learns a small deformation. Stage k runs

        1. training:        `train_reverse_KL(y, U_{t_{k-1}}, U_{t_k},
                            flow, type, ...)` — a packed reverse KL
                            stage on batches drawn from the CURRENT
                            particle set (the trainer's internal batch
                            rejuvenation runs at U_{t_{k-1}}, the very
                            distribution the set follows);
        2. ESS evaluation:  incremental importance-sampling ESS of the
                            trained increment over the full particle
                            set, from U_{t_{k-1}} to U_{t_k};
        3. rejection:       ESS < tau rejects the stage — shrink
                            t_k <- t_prev + shrink_factor (t_k - t_prev)
                            and retrain from the SAME warm start (at most
                            `max_retry` attempts, then the ladder stops);
        4. advance:         on acceptance, push the particle set through
                            the increment, reweight by the stage
                            importance weights, multinomially resample,
                            and freshen with (unadjusted) Langevin steps
                            at U_{t_k} — the advanced set is the next
                            stage's training data (not saved per stage).

    The stage coefficient starts at `t_safe` (the leading increment faces
    the largest deformation), thereafter extrapolates as
    t_init = min(t_prev + enlarge_factor (t_prev - t_pprev), 1), snapped
    to 1 when within `t_tol`. Every bridge shares one pytree structure
    (`linear_combination([target, source], [t, 1 - t])`), so a single
    compilation of the packed trainer serves the whole ladder.
    Deterministic (no PRNG key), like the stage trainer it wraps.

    Both processes are monitored: the per-step training status streams
    through `monitor` (loss + batch ESS from inside the compiled loop),
    and every attempt's validation ESS — the acceptance metric — is
    reported through the same printer in the per-stage ladder lines.

    Input:
        x_valid:  Array [N, d]   fixed set of source samples (the stage-0
                                 particle set)
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train
        type:     str            'F' if the flow maps source -> target;
                                 'G' if it maps target -> source
        n_batch:  int            samples drawn from the particle set per Adam step
        steps:    int            Adam steps per stage attempt
        lr:       float          Adam learning rate
        mc_step:  float          Langevin step size (batch rejuvenation inside
                                 training AND the particle-set advance)
        mc_iters: int            Langevin steps (same two uses)
        monitor:  Monitor        optional live status reporter; its printer
                                 also carries the per-stage ladder lines
        bg_param: dict           overrides of the ladder parameters
                                 {t_safe, shrink_factor, enlarge_factor,
                                 tau, t_tol, max_stages, max_retry}
    Output:
        flow:   Flow         the LAST incremental map (mu_{t_{K-1}} -> target;
                             == stages[-1]["flow"]). The generator's sample
                             output is `y`, not a single global flow.
        y:      Array [N, d] the advanced particle set — at the target on a
                             complete ladder, at the last accepted bridge on
                             an incomplete one
        stages: list[dict]   one record per accepted stage:
                             {"t": float, "ess": float, "flow": Flow} —
                             the coefficient, the incremental ESS, and the
                             SAVED stage flow. The ladder is complete iff
                             stages[-1]["t"] == 1.
    """
    if type not in ("F", "G"):
        raise ValueError(f"boltzmann_reverse_KL: type must be 'F' or 'G', got {type!r}")
    p = dict(_BG_DEFAULTS)
    if bg_param:
        unknown = set(bg_param) - set(p)
        if unknown:
            raise ValueError(f"boltzmann_reverse_KL: unknown bg_param keys {sorted(unknown)}")
        p.update(bg_param)
    if not (0.0 < p["shrink_factor"] < 1.0 and 0.0 < p["t_safe"] <= 1.0 and p["max_retry"] >= 1):
        raise ValueError(f"boltzmann_reverse_KL: invalid bg_param {p!r}")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(0)  # fixed internal seed (deterministic interface)

    y = x_valid                        # particle set at t = 0 (= mu_0)
    stages: list[dict] = []
    t_prev = 0.0
    while t_prev < 1.0 and len(stages) < p["max_stages"]:
        k = len(stages) + 1
        # initial guess: t_safe on stage 1, thereafter the enlarge-factor
        # extrapolation over the last two accepted coefficients
        hist = [0.0] + [s["t"] for s in stages]
        t_k = p["t_safe"] if not stages else min(
            hist[-1] + p["enlarge_factor"] * (hist[-1] - hist[-2]), 1.0
        )
        if 1.0 - t_k < p["t_tol"]:
            t_k = 1.0
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        accepted = False
        for attempt in range(1, p["max_retry"] + 1):
            u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
            status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} "
                   f"(attempt {attempt}/{p['max_retry']}) training the increment ...")
            cand, _ = train_reverse_KL(y, u_prev, u_k, flow, type,
                                       n_batch, steps, lr, mc_step, mc_iters, monitor)
            log_w = importance_weights_log(y, u_prev, u_k, cand, type)
            ess_k = float(compute_ESS_log(log_w))
            status(f"[stage {k}] t={t_k:.4f} validation: incremental ESS = {ess_k:.3f} "
                   f"(tau = {p['tau']:.2f})")
            if ess_k >= p["tau"]:
                accepted = True
                break
            status(f"[stage {k}] t={t_k:.4f} REJECTED -> shrink")
            t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        if not accepted:
            status(f"[stage {k}] gave up after {p['max_retry']} attempts "
                   f"(last t={t_k:.4f}, ESS = {ess_k:.3f}) — ladder INCOMPLETE")
            break
        flow = cand                    # warm start of every later stage
        # advance the particle set: push through the increment, reweight,
        # resample, and freshen with (unadjusted) Langevin at U_{t_k}
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_push = flow(y) if type == "F" else flow.inv(y)
        y = resample(key_res, y_push, jnp.exp(log_w - log_w.max()))
        y = langevin(key_mc, y, u_k, step=mc_step, iters=mc_iters)
        stages.append({"t": t_k, "ess": ess_k, "flow": flow})
        status(f"[stage {k}] t={t_k:.4f} ACCEPTED (attempt {attempt}): "
               f"stage flow saved; particle set advanced")
        t_prev = t_k
    if t_prev < 1.0:
        status(f"boltzmann_reverse_KL: ladder INCOMPLETE at t = {t_prev:.4f} "
               f"({len(stages)} accepted stages); particle set at the last accepted bridge")
    else:
        status(f"boltzmann_reverse_KL: ladder COMPLETE ({len(stages)} stages); "
               f"particle set at the target")
    return flow, y, stages

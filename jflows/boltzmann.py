"""Annealed Boltzmann generators for jflows.

The high-level API: `boltzmann_reverse_KL_F`, `boltzmann_forward_KL_G`,
`boltzmann_forward_KLX_G`, and `boltzmann_forward_KLXX_G` chain the packed
stage trainers of `jflows.train` into the full adaptive-temperature ladder
U_t = (1 - t) U_0 + t U_1. Each stage selects t_k (SMC gate), trains the
warm-started increment mu_{t_{k-1}} -> mu_{t_k}, keeps whichever of the
trained flow and the identity map (pure SMC) scores the higher incremental
ESS, and advances the particle set by reweight -> resample -> Langevin.
The stage trainer, weight evaluation, and advance are `filter_jit`-compiled
once and reused across the whole ladder.

Each generator also has a `_fixed` twin (`boltzmann_reverse_KL_F_fixed`, etc.)
that walks a caller-supplied fixed `t_list` from 0 to 1 with the adaptive
machinery removed — no SMC pre-selection, no acceptance/rejection, just bare
step-by-step training of each increment. The per-stage identity check and the
output records are identical, so `(y_valid, stages)` is consumed the same way.

Stage records expose unambiguous full-validation ESS values and aligned
attempt histories. Passing ``flow_dir`` additionally writes every trained
candidate (including rejected and identity-losing attempts), the accepted
stage map, metadata, and an atomic run manifest. Persistence is eager and does
not participate in JIT compilation or PRNG handling. The old ``ess``,
``ess_history``, and ``imp_history`` record keys remain temporary aliases.
"""

from __future__ import annotations

import math
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ._artifacts import FlowArtifactWriter
from .flow import Flow
from .potential import Potential, linear_combination
from .utils._compat import inherit_implementation_doc, legacy_keywords
from .utils.anneal import sequential_monte_carlo
from .utils.metrics import (_linear_weights_from_log, compute_ESS_log,
                            importance_weights_log, resample)
from .utils.rejuvenation import langevin
from .train import (Monitor, train_forward_KL_G, train_forward_KLX_G,
                    train_forward_KLXX_G, train_reverse_KL_F)


__all__ = ["boltzmann_forward_KL_G", "boltzmann_forward_KLX_G",
           "boltzmann_forward_KLXX_G", "boltzmann_reverse_KL_F",
           "boltzmann_forward_KL_G_fixed", "boltzmann_forward_KLX_G_fixed",
           "boltzmann_forward_KLXX_G_fixed", "boltzmann_reverse_KL_F_fixed"]


_iw_log_kernel = eqx.filter_jit(importance_weights_log)


def _iw_log_jit(samples, source, target, flow, type, chunks=1):
    """Full-set log importance weights, chunked eagerly: the compiled
    per-chunk kernel is looped in Python, so the device frees each chunk's
    buffers before the next runs and peak memory is one chunk's graph
    (an in-graph chunk loop is overlapped by the XLA scheduler, which
    reconstitutes the full-set peak). Statistically and numerically
    identical to a single full-set call."""
    return jnp.concatenate([
        _iw_log_kernel(x, source, target, flow, type)
        for x in jnp.array_split(samples, chunks, axis=0)
    ], axis=0)


_pot_diff_kernel = eqx.filter_jit(lambda y, source, target: source(y) - target(y))


def _iw_log_identity(samples, source, target, chunks=1):
    """Identity-map log importance weights source(y) - target(y), chunked
    eagerly. This is the increment's SMC (identity-map) reweighting from
    mu_{k-1} to mu_k: with the flow fixed to the identity the Jacobian term
    vanishes and the log-weight is just the potential difference, in BOTH
    flow directions. It evaluates two potentials and NEVER touches the flow
    inverse — the whole point is that the identity check costs no
    `inv_and_ladj` (the autoregressive inverse is the stage's most expensive
    op). Only the trained flow pays for its inverse via `_iw_log_jit`."""
    return jnp.concatenate([
        _pot_diff_kernel(x, source, target)
        for x in jnp.array_split(samples, chunks, axis=0)
    ], axis=0)


def _trainable_identity(flow: Flow) -> Flow:
    """Identity-like warm start after an exact identity fallback.

    Ordinary flow families train normally from ``zeros()``. OTFlow's exact
    zero PSD factor is absorbing, so its public ``near_identity()`` keeps the
    next stage trainable while the selected stage map itself remains the exact
    identity used for weighting, advancement, and records.
    """
    near_identity = getattr(flow, "near_identity", None)
    return flow.zeros() if near_identity is None else near_identity()


_smc_jit = eqx.filter_jit(sequential_monte_carlo)
_langevin_jit = eqx.filter_jit(langevin)   # per-stage selection-pool rejuvenation


def _new_attempt_hist() -> dict[str, list]:
    """Host-side attempt diagnostics; never enters a compiled computation."""
    return {
        "t": [],
        "batch_ess": [],
        "valid_trained_ess": [],
        "valid_identity_ess": [],
        "status": [],
        "trained_flow_path": [],
    }


def _record_attempt(
    hist: dict[str, list],
    artifacts: FlowArtifactWriter,
    *,
    stage: int,
    attempt: int,
    t: float,
    batch_ess: Array,
    candidate: Flow,
    status: str,
    selected: str,
    valid_selected_ess: float,
    valid_trained_ess: float,
    valid_identity_ess: float,
) -> None:
    """Append one attempt and, when enabled, persist its trained candidate."""
    path = artifacts.save_attempt(
        stage, attempt, t, candidate, status=status, selected=selected,
        valid_selected_ess=valid_selected_ess,
        valid_trained_ess=valid_trained_ess,
        valid_identity_ess=valid_identity_ess,
        batch_ess=batch_ess,
    )
    hist["t"].append(float(t))
    hist["batch_ess"].append(batch_ess)
    hist["valid_trained_ess"].append(float(valid_trained_ess))
    hist["valid_identity_ess"].append(float(valid_identity_ess))
    hist["status"].append(status)
    hist["trained_flow_path"].append(path)


def _stage_record(
    *,
    t: float,
    valid_selected_ess: float,
    valid_trained_ess: float,
    valid_identity_ess: float,
    selected: str,
    flow: Flow,
    hist: dict[str, list],
    selected_flow_path: str | None,
) -> dict:
    """Canonical stage record plus temporary aliases for old consumers."""
    batch_ess_hist = jnp.stack(hist["batch_ess"], axis=0)
    record = {
        "t": float(t),
        "valid_selected_ess": float(valid_selected_ess),
        "valid_trained_ess": float(valid_trained_ess),
        "valid_identity_ess": float(valid_identity_ess),
        "selected": selected,
        "flow": flow,
        "t_hist": jnp.asarray(hist["t"]),
        "batch_ess_hist": batch_ess_hist,
        "valid_trained_ess_hist": jnp.asarray(hist["valid_trained_ess"]),
        "valid_identity_ess_hist": jnp.asarray(hist["valid_identity_ess"]),
        "attempt_status_hist": tuple(hist["status"]),
        "trained_flow_path_hist": tuple(hist["trained_flow_path"]),
        "selected_flow_path": selected_flow_path,
    }
    # Compatibility aliases. New code should use the explicit names above.
    record["ess"] = record["valid_selected_ess"]
    record["ess_history"] = batch_ess_hist[-1]
    record["imp_history"] = max(
        0.0, record["valid_trained_ess"] - record["valid_identity_ess"]
    )
    return record


@eqx.filter_jit
def _bg_advance(key_res, key_mc, y, log_w, flow, type, u_k, mc_dt, mc_steps,
                mc_adjust, chunks):
    """Stage advance: push the particle set through the increment (chunked),
    reweight by the stage log-weights, resample, Langevin-freshen at U_k
    (MALA when mc_adjust)."""
    push = flow.__call__ if type == "F" else flow.inv
    y_push = jnp.concatenate(
        [push(c) for c in jnp.array_split(y, chunks, axis=0)], axis=0
    )
    y_new = resample(key_res, y_push, _linear_weights_from_log(log_w))
    return langevin(key_mc, y_new, u_k, dt=mc_dt, steps=mc_steps,
                    adjust=mc_adjust, chunks=chunks)


_BG_DEFAULTS = {
    "t_safe": 0.2,        # stage-1 bridge coefficient (the safe start)
    "shrink_factor": 0.7,  # rejected stage: t_k <- t_prev + shrink_factor (t_k - t_prev)
    "enlarge_factor": 1.5, # accepted stage: t_init extrapolation growth factor
    "tau_smc": 0.0,        # SMC pre-selection gate on t_k (0.0 by default: no SMC gate)
    "tau_ess": 0.6,        # ESS acceptance threshold of a trained stage
    "t_tol": 0.01,         # snap the initial guess to t = 1 when 1 - t_k < t_tol
    "max_stages": 30,      # ladder-length safety cap
    "max_retry": 6,        # training attempts per stage before giving up
}


def _bg_parameters(name: str, bg_param: dict | None) -> dict:
    """Merge and validate one adaptive-ladder parameter dictionary."""
    p = dict(_BG_DEFAULTS)
    if bg_param:
        unknown = set(bg_param) - set(p)
        if unknown:
            raise ValueError(f"{name}: unknown bg_param keys {sorted(unknown)}")
        p.update(bg_param)

    real_names = (
        "t_safe", "shrink_factor", "enlarge_factor", "tau_smc", "tau_ess", "t_tol"
    )
    try:
        values = {key: float(p[key]) for key in real_names}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: non-numeric bg_param {p!r}") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{name}: bg_param values must be finite, got {p!r}")
    if not (
        0.0 < values["t_safe"] <= 1.0
        and 0.0 < values["shrink_factor"] < 1.0
        and values["enlarge_factor"] > 0.0
        and 0.0 <= values["tau_smc"] <= 1.0
        and 0.0 <= values["tau_ess"] <= 1.0
        and 0.0 <= values["t_tol"] <= 1.0
    ):
        raise ValueError(f"{name}: invalid bg_param {p!r}")
    if values["t_safe"] < 1.0 and min(
        values["t_safe"] * (1.0 + values["enlarge_factor"]), 1.0
    ) <= values["t_safe"]:
        raise ValueError(
            f"{name}: enlarge_factor={values['enlarge_factor']!r} is too small "
            "to advance the ladder in floating-point arithmetic"
        )
    for key in ("max_stages", "max_retry"):
        try:
            value = operator.index(p[key])
        except TypeError as exc:
            raise ValueError(f"{name}: {key} must be a positive int, got {p[key]!r}") from exc
        if isinstance(p[key], bool) or value < 1:
            raise ValueError(f"{name}: {key} must be a positive int, got {p[key]!r}")
        p[key] = value
    p.update(values)
    return p


def _fixed_schedule(name: str, t_list) -> list[float]:
    """Normalize and validate a caller-supplied fixed bridge schedule."""
    try:
        schedule = [float(t) for t in t_list]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: t_list must contain real numbers") from exc
    if schedule and schedule[0] == 0.0:
        schedule = schedule[1:]
    if not schedule:
        raise ValueError(f"{name}: t_list is empty")
    if not all(math.isfinite(t) for t in schedule):
        raise ValueError(f"{name}: t_list values must be finite")
    if any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError(f"{name}: t_list must be strictly increasing")
    if not (0.0 < schedule[0] and schedule[-1] <= 1.0):
        raise ValueError(f"{name}: t_list must lie in (0, 1]")
    return schedule


def _boltzmann_reverse_KL_F_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    Annealed Boltzmann generator on the reverse KL, with the flow fixed
    as the forward map F (source -> target): advance a particle set along
    the bridge ladder

        U_t = (1 - t) U_0 + t U_1,        0 < t_1 < ... < t_K = 1,

    where t is the linear-combination coefficient, by training the
    warm-started flow as the INCREMENTAL map from the current particle
    distribution mu_{t_{k-1}} to the next bridge mu_{t_k} — each stage
    only ever learns a small deformation. Stage k runs

        1. selection:       when tau_smc > 0, a `ladder`-level SMC check
                            from U_{t_{k-1}} to the candidate bridge on an
                            `pool_size`-sized selection pool — drawn with
                            replacement from the particle set and
                            Langevin-rejuvenated at U_{t_{k-1}} — shrinks
                            t_k until its MINIMUM per-level ESS clears
                            tau_smc (at most 60 shrinks, as in the
                            reference adaptive_step) — SMC is far cheaper
                            than a training attempt, so the gate filters
                            over-aggressive t_k for free;
        2. training:        `train_reverse_KL_F(y, U_{t_{k-1}}, U_{t_k},
                            flow, ...)` — a packed reverse KL
                            stage on batches drawn from the CURRENT
                            particle set (the trainer's internal batch
                            rejuvenation runs at U_{t_{k-1}}, the very
                            distribution the set follows);
        3. ESS evaluation:  incremental importance-sampling ESS of the
                            trained increment over the full particle
                            set, from U_{t_{k-1}} to U_{t_k};
        4. rejection:       ESS < tau_ess rejects the stage — shrink
                            t_k <- t_prev + shrink_factor (t_k - t_prev)
                            and retrain from the SAME warm start (at most
                            `max_retry` attempts, then the ladder stops);
        5. advance:         on acceptance, push the particle set through
                            the increment, reweight by the stage
                            importance weights, multinomially resample,
                            and freshen with Langevin steps at U_{t_k}
                            (MALA when `mc_adjust` — the reference guard
                            that keeps near-singular tails out of the
                            set) — the advanced set is the next stage's
                            training data (not saved per stage).

    The stage coefficient starts at `t_safe` (the leading increment faces
    the largest deformation), thereafter extrapolates as
    t_init = min(t_prev + enlarge_factor (t_prev - t_pprev), 1), snapped
    to 1 when within `t_tol`. Every bridge shares one pytree structure
    (`linear_combination([target, source], [t, 1 - t])`) and the stage
    trainer, weight evaluation, and advance are `filter_jit`-compiled,
    so a single compilation of each serves the whole ladder.
    Deterministic (no PRNG key): the ladder uses its own base stream,
    and every stage attempt threads a distinct seed into the trainer,
    so retries and stages explore fresh randomness.

    Both processes are monitored: the per-step training status streams
    through `monitor` (loss + batch ESS from inside the compiled loop),
    and every attempt's validation ESS — the acceptance metric — is
    reported through the same printer in the per-stage ladder lines.

    Input:
        x_valid:  Array [N, d]   fixed set of source samples (the stage-0
                                 particle set)
        source:   Potential      negative log-density of the source (up to const)
        target:   Potential      negative log-density of the target (up to const)
        flow:     Flow           the normalizing flow to train, applied as F
                                 (source -> target)
        pool_size:   int            selection-pool size: the tau_smc gate runs on
                                 this many particles drawn (with replacement)
                                 from the particle set per stage
        batch_size:  int            samples drawn from the particle set per Adam step
        train_steps: int            Adam updates per stage attempt
        lr:       float          Adam learning rate
        ladder:   int            SMC levels of the tau_smc selection gate (the
                                 gate accepts on the minimum per-level ESS)
        mc_dt:  float          Langevin step size (batch rejuvenation inside
                                 training, the SMC gate, AND the particle-set
                                 advance)
        mc_steps: int            Langevin steps (same uses)
        mc_adjust: bool          False: unadjusted ULA; True: MALA in the
                                 trainer's batch rejuvenation and the
                                 particle-set advance — rejects proposals into
                                 steep walls, keeping the particle set free of
                                 the near-singular tails that crater the
                                 acceptance ESS
        monitor:  Monitor        optional live status reporter; its printer
                                 also carries the per-stage ladder lines
        bg_param: dict           overrides of the ladder parameters
                                 {t_safe, shrink_factor, enlarge_factor,
                                 tau_smc, tau_ess, t_tol, max_stages,
                                 max_retry}
        chunks:    int            split full-set stage operations into this many
                                 execution chunks. Weight evaluation iterates
                                 eagerly and bounds that graph; push/Langevin
                                 run inside the compiled advancement and are
                                 not a strict peak-memory guarantee
        checkpoint: bool         forwarded to the stage trainer: rematerialize
                                 the loss forward pass in the backward
                                 (jax.checkpoint) — worthwhile for deep flows,
                                 CNF exact-trace, or non-native-direction
                                 training
    Output:
        y_valid: Array [N, d] the advanced validation/particle set — the
                             generator's sample output: at the target on a
                             complete ladder, at the last accepted bridge on
                             an incomplete one
        stages:  list[dict]  one record per accepted stage. Canonical scalar
                             fields are ``t``, ``valid_selected_ess``,
                             ``valid_trained_ess``, ``valid_identity_ess``,
                             ``selected``, and ``flow``. Attempt-aligned fields
                             are ``t_hist``, ``batch_ess_hist``,
                             ``valid_trained_ess_hist``,
                             ``valid_identity_ess_hist``,
                             ``attempt_status_hist``, and
                             ``trained_flow_path_hist``. ``selected_flow_path``
                             names the separately saved accepted map, or is
                             None when ``flow_dir`` is disabled. The ladder is
                             complete iff stages[-1]["t"] == 1.

                             Identity check: after training, each stage keeps
                             whichever of the trained flow and the identity map
                             (pure SMC reweighting) has the higher incremental
                             ESS on the particle set. So
                             ``valid_selected_ess`` is
                             max(trained ESS, identity ESS) — the ACCEPTED map's
                             ESS, not necessarily the trained flow's — "flow" is
                             the identity map when the fallback wins, and
                             the identity ESS costs two potential evaluations
                             and no flow inverse. ``ess``, ``ess_history``, and
                             ``imp_history`` are temporary aliases for old
                             consumers.
    """
    p = _bg_parameters("boltzmann_reverse_KL_F", bg_param)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_reverse_KL_F")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(3)  # the ladder's own base stream (distinct from the trainers)

    y_valid = x_valid                  # validation/particle set at t = 0 (= mu_0)
    identity_flow = flow.zeros()       # identity map — the SMC fallback, shared by all stages
    stages: list[dict] = []
    t_prev = 0.0
    while t_prev < 1.0 and len(stages) < p["max_stages"]:
        k = len(stages) + 1
        artifacts.start_stage(k, t_prev)
        # initial guess: t_safe on stage 1, thereafter the enlarge-factor
        # extrapolation over the last two accepted coefficients
        hist = [0.0] + [s["t"] for s in stages]
        t_k = p["t_safe"] if not stages else min(
            hist[-1] + p["enlarge_factor"] * (hist[-1] - hist[-2]), 1.0
        )
        if 1.0 - t_k < p["t_tol"]:
            t_k = 1.0
        if not (t_prev < t_k <= 1.0):
            status(f"[stage {k}] candidate t={t_k!r} does not advance past "
                   f"t={t_prev!r} — ladder INCOMPLETE")
            artifacts.fail_stage(k, "nonadvancing_candidate", t=t_k)
            break
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        # (1) selection pool + SMC pre-selection of t_k (the reference
        # adaptive_step): the gate runs on a pool_size-sized pool drawn with
        # replacement from the particle set and rejuvenated at U_{t_{k-1}}
        # (the set is exact mu_0 at t = 0); shrink t_k until the pool's
        # minimum per-level SMC ESS clears tau_smc
        if p["tau_smc"] > 0.0:
            key_pool, key_mc_pool = jax.random.split(jax.random.fold_in(key, 20_000 + k))
            pool = y_valid[jax.random.randint(key_pool, (pool_size,), 0, y_valid.shape[0])]
            if t_prev > 0.0:
                pool = _langevin_jit(key_mc_pool, pool, u_prev, dt=mc_dt,
                                     steps=mc_steps, adjust=mc_adjust, chunks=chunks)
            smc_base = jax.random.fold_in(key, 10_000 + k)  # disjoint from the advance keys
            ok_smc = False
            for s_i in range(60):                           # max_shrinks, as in the reference
                if not t_k > t_prev:
                    break
                u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
                _, smc_ess = _smc_jit(jax.random.fold_in(smc_base, s_i), pool,
                                      u_prev, u_k, ladder=ladder, mc_dt=mc_dt,
                                      mc_steps=mc_steps, adjust=mc_adjust, chunks=chunks)
                ess_smc = float(smc_ess.min())
                ok_smc = ess_smc >= p["tau_smc"]
                status(f"[stage {k}] [select] t_k={t_k:.4f}  SMC ESS = {ess_smc:.3f} "
                       f"({'accept' if ok_smc else 'shrink'})")
                if ok_smc:
                    break
                t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
            if not ok_smc:
                status(f"[stage {k}] SMC selection failed after 60 shrinks "
                       f"(last t={t_k:.6g}) — ladder INCOMPLETE")
                artifacts.fail_stage(k, "smc_selection_failed", t=t_k)
                break
        accepted = False
        attempt_hist = _new_attempt_hist()
        ess_k = float("nan")
        for attempt in range(1, p["max_retry"] + 1):
            if not t_k > t_prev:
                status(f"[stage {k}] rejected-step shrink made no floating-point "
                       "progress — ladder INCOMPLETE")
                break
            u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
            status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} "
                   f"(attempt {attempt}/{p['max_retry']}) training the increment ...")
            seed = jnp.uint32(k * p["max_retry"] + attempt)  # fresh stream per attempt
            cand, ess_hist = train_reverse_KL_F(y_valid, u_prev, u_k, flow,
                                                batch_size, train_steps, lr, mc_dt, mc_steps,
                                                mc_adjust, monitor,
                                                seed=seed, checkpoint=checkpoint)
            log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "F", chunks=chunks)
            ess_tr = float(compute_ESS_log(log_w_tr))
            log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
            ess_id = float(compute_ESS_log(log_w_id))
            jax.effects_barrier()  # keep monitor lines ahead of the stage status
            # identity check: keep the better of the trained flow and the
            # identity map (pure SMC), so a stage is never worse than SMC
            if ess_tr >= ess_id:
                stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
                selected = "trained"
            else:
                stage_flow, next_flow, log_w, ess_k = (
                    identity_flow, _trainable_identity(flow), log_w_id, ess_id
                )
                selected = "identity"
            imp = ess_k - ess_id  # improvement over identity, always >= 0
            status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
                   f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f}; "
                   f"tau_ess = {p['tau_ess']:.2f})")
            attempt_status = "accepted" if ess_k >= p["tau_ess"] else "rejected"
            _record_attempt(
                attempt_hist, artifacts, stage=k, attempt=attempt, t=t_k,
                batch_ess=ess_hist, candidate=cand, status=attempt_status,
                selected=selected, valid_selected_ess=ess_k,
                valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
            )
            if attempt_status == "accepted":
                accepted = True
                break
            status(f"[stage {k}] t={t_k:.4f} REJECTED -> shrink")
            t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        if not accepted:
            failed_t = attempt_hist["t"][-1] if attempt_hist["t"] else t_k
            artifacts.fail_stage(k, "failed", t=failed_t, valid_selected_ess=ess_k)
            status(f"[stage {k}] gave up after {len(attempt_hist['t'])} attempts "
                   f"(last attempted t={failed_t:.4f}, ESS = {ess_k:.3f}) "
                   "— ladder INCOMPLETE")
            break
        # advance the particle set: push through the increment, reweight,
        # resample, and freshen with Langevin (MALA when mc_adjust) at U_{t_k}
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "F", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage, not the next
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} ACCEPTED (attempt {attempt}): "
               f"stage flow saved; particle set advanced")
        t_prev = t_k
    if t_prev < 1.0:
        status(f"boltzmann_reverse_KL_F: ladder INCOMPLETE at t = {t_prev:.4f} "
               f"({len(stages)} accepted stages); particle set at the last accepted bridge")
    else:
        status(f"boltzmann_reverse_KL_F: ladder COMPLETE ({len(stages)} stages); "
               f"particle set at the target")
    artifacts.finish(complete=t_prev == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_forward_KL_G_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    Annealed Boltzmann generator on the forward KL, with the flow fixed
    as the inverse map G (target -> source) — the twin of
    `boltzmann_reverse_KL_F` with the same ladder machinery and parameter
    setting: advance a particle set along the bridge ladder

        U_t = (1 - t) U_0 + t U_1,        0 < t_1 < ... < t_K = 1,

    where t is the linear-combination coefficient, by training the
    warm-started flow as the INCREMENTAL map from the current particle
    distribution mu_{t_{k-1}} to the next bridge mu_{t_k}. Stage k runs

        1. selection:       when tau_smc > 0, a `ladder`-level SMC check
                            from U_{t_{k-1}} to the candidate bridge on an
                            `pool_size`-sized selection pool drawn from the
                            particle set shrinks t_k until its MINIMUM
                            per-level ESS clears tau_smc (at most 60 shrinks);
        2. training:        `train_forward_KL_G(y, U_{t_{k-1}}, U_{t_k},
                            flow, ...)` — each Adam step draws an
                            `batch_size` subset of the particle set and
                            manufactures its target batch by
                            `ladder`-level AIS through the CURRENT flow
                            (SMC gate and AIS share the same `ladder`);
        3. ESS evaluation:  incremental importance-sampling ESS of the
                            trained increment over the full particle
                            set, from U_{t_{k-1}} to U_{t_k};
        4. rejection:       ESS < tau_ess rejects the stage — shrink
                            t_k <- t_prev + shrink_factor (t_k - t_prev)
                            and retrain from the SAME warm start (at most
                            `max_retry` attempts, then the ladder stops);
        5. advance:         on acceptance, push the particle set through
                            the increment, reweight by the stage
                            importance weights, multinomially resample,
                            and freshen with Langevin steps at U_{t_k}
                            (MALA when `mc_adjust`).

    `tau_smc`, the t schedule (t_safe / shrink_factor / enlarge_factor /
    t_tol), acceptance, records, monitoring, determinism, and the
    compile-once behaviour are identical to `boltzmann_reverse_KL_F`; the
    ladder uses its own base PRNG stream and threads a distinct seed
    into the trainer per stage attempt.

    Input:  as `boltzmann_reverse_KL_F`, with `mc_adjust` applying to both
            the stage trainer's internal AIS rejuvenation and the
            particle-set advance, and `e_clip` / `g_clip` (the energy screen
            and global gradient-norm clip) forwarded to every stage's
            `train_forward_KL_G` (default inf / inf: no screen, no clip).
    Output: as `boltzmann_reverse_KL_F` — (y_valid, stages) with the common
            full-validation/attempt-history stage schema.
    """
    p = _bg_parameters("boltzmann_forward_KL_G", bg_param)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_forward_KL_G")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(4)  # this ladder's own base stream

    y_valid = x_valid                  # validation/particle set at t = 0 (= mu_0)
    identity_flow = flow.zeros()       # identity map — the SMC fallback, shared by all stages
    stages: list[dict] = []
    t_prev = 0.0
    while t_prev < 1.0 and len(stages) < p["max_stages"]:
        k = len(stages) + 1
        artifacts.start_stage(k, t_prev)
        hist = [0.0] + [s["t"] for s in stages]
        t_k = p["t_safe"] if not stages else min(
            hist[-1] + p["enlarge_factor"] * (hist[-1] - hist[-2]), 1.0
        )
        if 1.0 - t_k < p["t_tol"]:
            t_k = 1.0
        if not (t_prev < t_k <= 1.0):
            status(f"[stage {k}] candidate t={t_k!r} does not advance past "
                   f"t={t_prev!r} — ladder INCOMPLETE")
            artifacts.fail_stage(k, "nonadvancing_candidate", t=t_k)
            break
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        # (1) SMC pre-selection of t_k on a pool_size-sized selection pool
        # (drawn with replacement, rejuvenated at U_{t_{k-1}} when t > 0)
        if p["tau_smc"] > 0.0:
            key_pool, key_mc_pool = jax.random.split(jax.random.fold_in(key, 20_000 + k))
            pool = y_valid[jax.random.randint(key_pool, (pool_size,), 0, y_valid.shape[0])]
            if t_prev > 0.0:
                pool = _langevin_jit(key_mc_pool, pool, u_prev, dt=mc_dt,
                                     steps=mc_steps, adjust=mc_adjust, chunks=chunks)
            smc_base = jax.random.fold_in(key, 10_000 + k)  # disjoint from the advance keys
            ok_smc = False
            for s_i in range(60):                           # max_shrinks, as in the reference
                if not t_k > t_prev:
                    break
                u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
                _, smc_ess = _smc_jit(jax.random.fold_in(smc_base, s_i), pool,
                                      u_prev, u_k, ladder=ladder, mc_dt=mc_dt,
                                      mc_steps=mc_steps, adjust=mc_adjust, chunks=chunks)
                ess_smc = float(smc_ess.min())
                ok_smc = ess_smc >= p["tau_smc"]
                status(f"[stage {k}] [select] t_k={t_k:.4f}  SMC ESS = {ess_smc:.3f} "
                       f"({'accept' if ok_smc else 'shrink'})")
                if ok_smc:
                    break
                t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
            if not ok_smc:
                status(f"[stage {k}] SMC selection failed after 60 shrinks "
                       f"(last t={t_k:.6g}) — ladder INCOMPLETE")
                artifacts.fail_stage(k, "smc_selection_failed", t=t_k)
                break
        accepted = False
        attempt_hist = _new_attempt_hist()
        ess_k = float("nan")
        for attempt in range(1, p["max_retry"] + 1):
            if not t_k > t_prev:
                status(f"[stage {k}] rejected-step shrink made no floating-point "
                       "progress — ladder INCOMPLETE")
                break
            u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
            status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} "
                   f"(attempt {attempt}/{p['max_retry']}) training the increment ...")
            seed = jnp.uint32(k * p["max_retry"] + attempt)  # fresh stream per attempt
            cand, ess_hist = train_forward_KL_G(y_valid, u_prev, u_k, flow,
                                                batch_size, train_steps, lr, ladder, mc_dt,
                                                mc_steps, mc_adjust, monitor,
                                                seed=seed, checkpoint=checkpoint,
                                                e_clip=e_clip, g_clip=g_clip)
            log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunks=chunks)
            ess_tr = float(compute_ESS_log(log_w_tr))
            log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
            ess_id = float(compute_ESS_log(log_w_id))
            jax.effects_barrier()  # keep monitor lines ahead of the stage status
            # identity check: keep the better of the trained flow and the
            # identity map (pure SMC), so a stage is never worse than SMC
            if ess_tr >= ess_id:
                stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
                selected = "trained"
            else:
                stage_flow, next_flow, log_w, ess_k = (
                    identity_flow, _trainable_identity(flow), log_w_id, ess_id
                )
                selected = "identity"
            imp = ess_k - ess_id  # improvement over identity, always >= 0
            status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
                   f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f}; "
                   f"tau_ess = {p['tau_ess']:.2f})")
            attempt_status = "accepted" if ess_k >= p["tau_ess"] else "rejected"
            _record_attempt(
                attempt_hist, artifacts, stage=k, attempt=attempt, t=t_k,
                batch_ess=ess_hist, candidate=cand, status=attempt_status,
                selected=selected, valid_selected_ess=ess_k,
                valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
            )
            if attempt_status == "accepted":
                accepted = True
                break
            status(f"[stage {k}] t={t_k:.4f} REJECTED -> shrink")
            t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        if not accepted:
            failed_t = attempt_hist["t"][-1] if attempt_hist["t"] else t_k
            artifacts.fail_stage(k, "failed", t=failed_t, valid_selected_ess=ess_k)
            status(f"[stage {k}] gave up after {len(attempt_hist['t'])} attempts "
                   f"(last attempted t={failed_t:.4f}, ESS = {ess_k:.3f}) "
                   "— ladder INCOMPLETE")
            break
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "G", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage, not the next
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} ACCEPTED (attempt {attempt}): stage flow saved")
        t_prev = t_k
    if t_prev < 1.0:
        status(f"boltzmann_forward_KL_G: ladder INCOMPLETE at t = {t_prev:.4f} "
               f"({len(stages)} accepted stages); particle set at the last accepted bridge")
    else:
        status(f"boltzmann_forward_KL_G: ladder COMPLETE ({len(stages)} stages); "
               f"particle set at the target")
    artifacts.finish(complete=t_prev == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_forward_KLX_G_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    X-regularized forward KL Boltzmann generator — `boltzmann_forward_KL_G`
    with `train_forward_KLX_G` as the stage trainer, so every stage adds the
    coeff_lambda-weighted X functional on top of the forward KL. The flow is
    fixed as the inverse map G (target -> source); the bridge ladder, the SMC
    selection gate, the t schedule (t_safe / shrink_factor / enlarge_factor /
    t_tol), acceptance, per-stage records, monitoring, determinism, and the
    compile-once behaviour are identical to `boltzmann_forward_KL_G`. The ladder
    uses its own base PRNG stream and threads a distinct seed into the trainer
    per stage attempt.

    Input:  as `boltzmann_forward_KL_G`, with
            `coeff_lambda` — the X functional weight — and
            `e_clip` / `g_clip` (energy screen and gradient-norm clip)
            forwarded to every stage's `train_forward_KLX_G`.
    Output: as `boltzmann_forward_KL_G` — (y_valid, stages) with the common
            full-validation/attempt-history stage schema.
    """
    p = _bg_parameters("boltzmann_forward_KLX_G", bg_param)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_forward_KLX_G")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(6)  # this ladder's own base stream

    y_valid = x_valid                  # validation/particle set at t = 0 (= mu_0)
    identity_flow = flow.zeros()       # identity map — the SMC fallback, shared by all stages
    stages: list[dict] = []
    t_prev = 0.0
    while t_prev < 1.0 and len(stages) < p["max_stages"]:
        k = len(stages) + 1
        artifacts.start_stage(k, t_prev)
        hist = [0.0] + [s["t"] for s in stages]
        t_k = p["t_safe"] if not stages else min(
            hist[-1] + p["enlarge_factor"] * (hist[-1] - hist[-2]), 1.0
        )
        if 1.0 - t_k < p["t_tol"]:
            t_k = 1.0
        if not (t_prev < t_k <= 1.0):
            status(f"[stage {k}] candidate t={t_k!r} does not advance past "
                   f"t={t_prev!r} — ladder INCOMPLETE")
            artifacts.fail_stage(k, "nonadvancing_candidate", t=t_k)
            break
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        # (1) SMC pre-selection of t_k on a pool_size-sized selection pool
        # (drawn with replacement, rejuvenated at U_{t_{k-1}} when t > 0)
        if p["tau_smc"] > 0.0:
            key_pool, key_mc_pool = jax.random.split(jax.random.fold_in(key, 20_000 + k))
            pool = y_valid[jax.random.randint(key_pool, (pool_size,), 0, y_valid.shape[0])]
            if t_prev > 0.0:
                pool = _langevin_jit(key_mc_pool, pool, u_prev, dt=mc_dt,
                                     steps=mc_steps, adjust=mc_adjust, chunks=chunks)
            smc_base = jax.random.fold_in(key, 10_000 + k)  # disjoint from the advance keys
            ok_smc = False
            for s_i in range(60):                           # max_shrinks, as in the reference
                if not t_k > t_prev:
                    break
                u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
                _, smc_ess = _smc_jit(jax.random.fold_in(smc_base, s_i), pool,
                                      u_prev, u_k, ladder=ladder, mc_dt=mc_dt,
                                      mc_steps=mc_steps, adjust=mc_adjust, chunks=chunks)
                ess_smc = float(smc_ess.min())
                ok_smc = ess_smc >= p["tau_smc"]
                status(f"[stage {k}] [select] t_k={t_k:.4f}  SMC ESS = {ess_smc:.3f} "
                       f"({'accept' if ok_smc else 'shrink'})")
                if ok_smc:
                    break
                t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
            if not ok_smc:
                status(f"[stage {k}] SMC selection failed after 60 shrinks "
                       f"(last t={t_k:.6g}) — ladder INCOMPLETE")
                artifacts.fail_stage(k, "smc_selection_failed", t=t_k)
                break
        accepted = False
        attempt_hist = _new_attempt_hist()
        ess_k = float("nan")
        for attempt in range(1, p["max_retry"] + 1):
            if not t_k > t_prev:
                status(f"[stage {k}] rejected-step shrink made no floating-point "
                       "progress — ladder INCOMPLETE")
                break
            u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
            status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} "
                   f"(attempt {attempt}/{p['max_retry']}) training the increment ...")
            seed = jnp.uint32(k * p["max_retry"] + attempt)  # fresh stream per attempt
            cand, ess_hist = train_forward_KLX_G(y_valid, u_prev, u_k, flow,
                                                 batch_size, train_steps, lr, ladder, mc_dt,
                                                 mc_steps, coeff_lambda, mc_adjust, monitor,
                                                 seed=seed, checkpoint=checkpoint,
                                                 e_clip=e_clip, g_clip=g_clip)
            log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunks=chunks)
            ess_tr = float(compute_ESS_log(log_w_tr))
            log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
            ess_id = float(compute_ESS_log(log_w_id))
            jax.effects_barrier()  # keep monitor lines ahead of the stage status
            # identity check: keep the better of the trained flow and the
            # identity map (pure SMC), so a stage is never worse than SMC
            if ess_tr >= ess_id:
                stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
                selected = "trained"
            else:
                stage_flow, next_flow, log_w, ess_k = (
                    identity_flow, _trainable_identity(flow), log_w_id, ess_id
                )
                selected = "identity"
            imp = ess_k - ess_id  # improvement over identity, always >= 0
            status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
                   f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f}; "
                   f"tau_ess = {p['tau_ess']:.2f})")
            attempt_status = "accepted" if ess_k >= p["tau_ess"] else "rejected"
            _record_attempt(
                attempt_hist, artifacts, stage=k, attempt=attempt, t=t_k,
                batch_ess=ess_hist, candidate=cand, status=attempt_status,
                selected=selected, valid_selected_ess=ess_k,
                valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
            )
            if attempt_status == "accepted":
                accepted = True
                break
            status(f"[stage {k}] t={t_k:.4f} REJECTED -> shrink")
            t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        if not accepted:
            failed_t = attempt_hist["t"][-1] if attempt_hist["t"] else t_k
            artifacts.fail_stage(k, "failed", t=failed_t, valid_selected_ess=ess_k)
            status(f"[stage {k}] gave up after {len(attempt_hist['t'])} attempts "
                   f"(last attempted t={failed_t:.4f}, ESS = {ess_k:.3f}) "
                   "— ladder INCOMPLETE")
            break
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "G", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage, not the next
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} ACCEPTED (attempt {attempt}): stage flow saved")
        t_prev = t_k
    if t_prev < 1.0:
        status(f"boltzmann_forward_KLX_G: ladder INCOMPLETE at t = {t_prev:.4f} "
               f"({len(stages)} accepted stages); particle set at the last accepted bridge")
    else:
        status(f"boltzmann_forward_KLX_G: ladder COMPLETE ({len(stages)} stages); "
               f"particle set at the target")
    artifacts.finish(complete=t_prev == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_forward_KLXX_G_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps: int,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    X-regularized forward KL Boltzmann generator with the full mixture
    loss — `boltzmann_forward_KL_G` with `train_forward_KLXX_G` as the stage
    trainer, so every stage minimizes

        KL + coeff_lambda * X_mu + X_{coeff_alpha hat_mu + coeff_beta bar_nu}

    on the increment mu_{t_{k-1}} -> mu_{t_k}. Each stage attempt rebuilds
    its quench-and-temper pool on the stage bridge U_{t_k} (melt scale
    `melt`, armijo L-BFGS `opt_alpha` x `opt_steps`) from pool_size particles,
    so mode discovery tracks the deforming bridge; `pool_size` sizes both this
    hat_mu pool and the tau_smc selection pool. The flow is fixed as the
    inverse map G (target -> source); the bridge ladder, the SMC selection
    gate, the t schedule (t_safe / shrink_factor / enlarge_factor / t_tol),
    acceptance, per-stage records, monitoring, determinism, and the
    compile-once behaviour are identical to `boltzmann_forward_KL_G`. The
    ladder uses its own base PRNG stream and threads a distinct seed into
    the trainer per stage attempt.

    Input:  as `boltzmann_forward_KL_G`, with
            the quench-and-temper family (`melt`, `opt_alpha`, `opt_steps`),
            the loss weights (`coeff_lambda`, `coeff_alpha`, `coeff_beta`),
            and `e_clip` / `g_clip` (energy screen and gradient-norm clip)
            forwarded to every stage's `train_forward_KLXX_G`.
    Output: as `boltzmann_forward_KL_G` — (y_valid, stages) with the common
            full-validation/attempt-history stage schema.
    """
    if not (0.0 <= coeff_alpha < float("inf")):
        raise ValueError(
            "boltzmann_forward_KLXX_G: coeff_alpha must be finite and non-negative"
        )
    if not (0.0 <= coeff_beta < float("inf")):
        raise ValueError(
            "boltzmann_forward_KLXX_G: coeff_beta must be finite and non-negative"
        )
    p = _bg_parameters("boltzmann_forward_KLXX_G", bg_param)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_forward_KLXX_G")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(8)  # this ladder's own base stream

    y_valid = x_valid                  # validation/particle set at t = 0 (= mu_0)
    identity_flow = flow.zeros()       # identity map — the SMC fallback, shared by all stages
    stages: list[dict] = []
    t_prev = 0.0
    while t_prev < 1.0 and len(stages) < p["max_stages"]:
        k = len(stages) + 1
        artifacts.start_stage(k, t_prev)
        hist = [0.0] + [s["t"] for s in stages]
        t_k = p["t_safe"] if not stages else min(
            hist[-1] + p["enlarge_factor"] * (hist[-1] - hist[-2]), 1.0
        )
        if 1.0 - t_k < p["t_tol"]:
            t_k = 1.0
        if not (t_prev < t_k <= 1.0):
            status(f"[stage {k}] candidate t={t_k!r} does not advance past "
                   f"t={t_prev!r} — ladder INCOMPLETE")
            artifacts.fail_stage(k, "nonadvancing_candidate", t=t_k)
            break
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        # (1) SMC pre-selection of t_k on a pool_size-sized selection pool
        # (drawn with replacement, rejuvenated at U_{t_{k-1}} when t > 0)
        if p["tau_smc"] > 0.0:
            key_pool, key_mc_pool = jax.random.split(jax.random.fold_in(key, 20_000 + k))
            pool = y_valid[jax.random.randint(key_pool, (pool_size,), 0, y_valid.shape[0])]
            if t_prev > 0.0:
                pool = _langevin_jit(key_mc_pool, pool, u_prev, dt=mc_dt,
                                     steps=mc_steps, adjust=mc_adjust, chunks=chunks)
            smc_base = jax.random.fold_in(key, 10_000 + k)  # disjoint from the advance keys
            ok_smc = False
            for s_i in range(60):                           # max_shrinks, as in the reference
                if not t_k > t_prev:
                    break
                u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
                _, smc_ess = _smc_jit(jax.random.fold_in(smc_base, s_i), pool,
                                      u_prev, u_k, ladder=ladder, mc_dt=mc_dt,
                                      mc_steps=mc_steps, adjust=mc_adjust, chunks=chunks)
                ess_smc = float(smc_ess.min())
                ok_smc = ess_smc >= p["tau_smc"]
                status(f"[stage {k}] [select] t_k={t_k:.4f}  SMC ESS = {ess_smc:.3f} "
                       f"({'accept' if ok_smc else 'shrink'})")
                if ok_smc:
                    break
                t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
            if not ok_smc:
                status(f"[stage {k}] SMC selection failed after 60 shrinks "
                       f"(last t={t_k:.6g}) — ladder INCOMPLETE")
                artifacts.fail_stage(k, "smc_selection_failed", t=t_k)
                break
        accepted = False
        attempt_hist = _new_attempt_hist()
        ess_k = float("nan")
        for attempt in range(1, p["max_retry"] + 1):
            if not t_k > t_prev:
                status(f"[stage {k}] rejected-step shrink made no floating-point "
                       "progress — ladder INCOMPLETE")
                break
            u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
            status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} "
                   f"(attempt {attempt}/{p['max_retry']}) training the increment ...")
            seed = jnp.uint32(k * p["max_retry"] + attempt)  # fresh stream per attempt
            cand, ess_hist = train_forward_KLXX_G(y_valid, u_prev, u_k, flow,
                                                  pool_size, batch_size, train_steps, lr, ladder,
                                                  melt, opt_alpha, opt_steps, mc_dt,
                                                  mc_steps, coeff_lambda, coeff_alpha,
                                                  coeff_beta, mc_adjust, monitor,
                                                  seed=seed, checkpoint=checkpoint,
                                                  e_clip=e_clip, g_clip=g_clip)
            log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunks=chunks)
            ess_tr = float(compute_ESS_log(log_w_tr))
            log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
            ess_id = float(compute_ESS_log(log_w_id))
            jax.effects_barrier()  # keep monitor lines ahead of the stage status
            # identity check: keep the better of the trained flow and the
            # identity map (pure SMC), so a stage is never worse than SMC
            if ess_tr >= ess_id:
                stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
                selected = "trained"
            else:
                stage_flow, next_flow, log_w, ess_k = (
                    identity_flow, _trainable_identity(flow), log_w_id, ess_id
                )
                selected = "identity"
            imp = ess_k - ess_id  # improvement over identity, always >= 0
            status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
                   f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f}; "
                   f"tau_ess = {p['tau_ess']:.2f})")
            attempt_status = "accepted" if ess_k >= p["tau_ess"] else "rejected"
            _record_attempt(
                attempt_hist, artifacts, stage=k, attempt=attempt, t=t_k,
                batch_ess=ess_hist, candidate=cand, status=attempt_status,
                selected=selected, valid_selected_ess=ess_k,
                valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
            )
            if attempt_status == "accepted":
                accepted = True
                break
            status(f"[stage {k}] t={t_k:.4f} REJECTED -> shrink")
            t_k = t_prev + p["shrink_factor"] * (t_k - t_prev)
        if not accepted:
            failed_t = attempt_hist["t"][-1] if attempt_hist["t"] else t_k
            artifacts.fail_stage(k, "failed", t=failed_t, valid_selected_ess=ess_k)
            status(f"[stage {k}] gave up after {len(attempt_hist['t'])} attempts "
                   f"(last attempted t={failed_t:.4f}, ESS = {ess_k:.3f}) "
                   "— ladder INCOMPLETE")
            break
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "G", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage, not the next
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} ACCEPTED (attempt {attempt}): stage flow saved")
        t_prev = t_k
    if t_prev < 1.0:
        status(f"boltzmann_forward_KLXX_G: ladder INCOMPLETE at t = {t_prev:.4f} "
               f"({len(stages)} accepted stages); particle set at the last accepted bridge")
    else:
        status(f"boltzmann_forward_KLXX_G: ladder COMPLETE ({len(stages)} stages); "
               f"particle set at the target")
    artifacts.finish(complete=t_prev == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_reverse_KL_F_fixed_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    mc_dt: float,
    mc_steps: int,
    t_list,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    Fixed-schedule variant of `boltzmann_reverse_KL_F`: train the increment
    for each coefficient in `t_list` in turn, with NO SMC pre-selection and
    NO acceptance/rejection — bare step-by-step training on a caller-supplied
    ladder. Every stage still keeps the better of the trained flow and the
    identity map (the cheap identity check, no flow inverse), advances the
    particle set, and records the common full-validation/attempt-history
    schema, so `(y_valid, stages)` is consumed identically downstream.

    Input:  as `boltzmann_reverse_KL_F` but without the adaptive machinery
            (`pool_size`, `ladder`, `bg_param` dropped) and with
            `t_list` — the fixed schedule, a strictly increasing sequence in
            (0, 1] (a leading 0 is dropped); the last value should be 1.0 for
            a complete ladder, else the particle set stops at the last bridge.
    Output: as `boltzmann_reverse_KL_F` — (y_valid, stages).
    """
    t_list = _fixed_schedule("boltzmann_reverse_KL_F_fixed", t_list)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_reverse_KL_F_fixed")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(3)  # same base stream as boltzmann_reverse_KL_F
    if t_list[-1] != 1.0:
        status(f"boltzmann_reverse_KL_F_fixed: t_list ends at {t_list[-1]:.4f} < 1 "
               f"— incomplete fixed ladder; particle set stops at the last bridge")

    y_valid = x_valid
    identity_flow = flow.zeros()
    stages: list[dict] = []
    t_prev = 0.0
    for k, t_k in enumerate(t_list, start=1):
        artifacts.start_stage(k, t_prev)
        attempt_hist = _new_attempt_hist()
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
        status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} training the increment (fixed) ...")
        seed = jnp.uint32(k)
        cand, ess_hist = train_reverse_KL_F(y_valid, u_prev, u_k, flow,
                                            batch_size, train_steps, lr, mc_dt, mc_steps,
                                            mc_adjust, monitor,
                                            seed=seed, checkpoint=checkpoint)
        log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "F", chunks=chunks)
        ess_tr = float(compute_ESS_log(log_w_tr))
        log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
        ess_id = float(compute_ESS_log(log_w_id))
        jax.effects_barrier()  # keep monitor lines ahead of the stage status
        # identity check: keep the better of the trained flow and the
        # identity map (pure SMC), so a stage is never worse than SMC
        if ess_tr >= ess_id:
            stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
            selected = "trained"
        else:
            stage_flow, next_flow, log_w, ess_k = (
                identity_flow, _trainable_identity(flow), log_w_id, ess_id
            )
            selected = "identity"
        imp = ess_k - ess_id  # improvement over identity, always >= 0
        status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
               f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f})")
        _record_attempt(
            attempt_hist, artifacts, stage=k, attempt=1, t=t_k,
            batch_ess=ess_hist, candidate=cand, status="accepted",
            selected=selected, valid_selected_ess=ess_k,
            valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
        )
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "F", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} DONE: stage flow saved; particle set advanced")
        t_prev = t_k
    status(f"boltzmann_reverse_KL_F_fixed: fixed ladder DONE "
           f"({len(stages)} stages, {'COMPLETE' if t_list[-1] == 1.0 else 'INCOMPLETE'})")
    artifacts.finish(complete=t_list[-1] == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_forward_KL_G_fixed_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    Fixed-schedule variant of `boltzmann_forward_KL_G`: bare step-by-step
    forward KL training along the caller-supplied `t_list`, with NO SMC
    pre-selection and NO acceptance/rejection. `ladder` is retained (it feeds
    the trainer's per-step AIS); `pool_size` and `bg_param` are dropped. Every
    stage keeps the better of the trained flow and the identity map and
    records the common full-validation/attempt-history schema, so
    `(y_valid, stages)` is consumed identically downstream.

    Input:  as `boltzmann_forward_KL_G`, minus `pool_size`/`bg_param`, plus
            `t_list` (strictly increasing in (0, 1], ideally ending at 1.0).
    Output: as `boltzmann_forward_KL_G` — (y_valid, stages).
    """
    t_list = _fixed_schedule("boltzmann_forward_KL_G_fixed", t_list)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_forward_KL_G_fixed")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(4)  # same base stream as boltzmann_forward_KL_G
    if t_list[-1] != 1.0:
        status(f"boltzmann_forward_KL_G_fixed: t_list ends at {t_list[-1]:.4f} < 1 "
               f"— incomplete fixed ladder; particle set stops at the last bridge")

    y_valid = x_valid
    identity_flow = flow.zeros()
    stages: list[dict] = []
    t_prev = 0.0
    for k, t_k in enumerate(t_list, start=1):
        artifacts.start_stage(k, t_prev)
        attempt_hist = _new_attempt_hist()
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
        status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} training the increment (fixed) ...")
        seed = jnp.uint32(k)
        cand, ess_hist = train_forward_KL_G(y_valid, u_prev, u_k, flow,
                                            batch_size, train_steps, lr, ladder, mc_dt,
                                            mc_steps, mc_adjust, monitor,
                                            seed=seed, checkpoint=checkpoint,
                                            e_clip=e_clip, g_clip=g_clip)
        log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunks=chunks)
        ess_tr = float(compute_ESS_log(log_w_tr))
        log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
        ess_id = float(compute_ESS_log(log_w_id))
        jax.effects_barrier()  # keep monitor lines ahead of the stage status
        # identity check: keep the better of the trained flow and the
        # identity map (pure SMC), so a stage is never worse than SMC
        if ess_tr >= ess_id:
            stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
            selected = "trained"
        else:
            stage_flow, next_flow, log_w, ess_k = (
                identity_flow, _trainable_identity(flow), log_w_id, ess_id
            )
            selected = "identity"
        imp = ess_k - ess_id  # improvement over identity, always >= 0
        status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
               f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f})")
        _record_attempt(
            attempt_hist, artifacts, stage=k, attempt=1, t=t_k,
            batch_ess=ess_hist, candidate=cand, status="accepted",
            selected=selected, valid_selected_ess=ess_k,
            valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
        )
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "G", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} DONE: stage flow saved; particle set advanced")
        t_prev = t_k
    status(f"boltzmann_forward_KL_G_fixed: fixed ladder DONE "
           f"({len(stages)} stages, {'COMPLETE' if t_list[-1] == 1.0 else 'INCOMPLETE'})")
    artifacts.finish(complete=t_list[-1] == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_forward_KLX_G_fixed_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    Fixed-schedule variant of `boltzmann_forward_KLX_G`: bare step-by-step
    X-regularized forward KL training along `t_list`, no SMC pre-selection and
    no acceptance/rejection. `ladder` and `coeff_lambda` are retained (the
    trainer's AIS and X weight); `pool_size` and `bg_param` are dropped. Same
    stage records and output as the adaptive version.

    Input:  as `boltzmann_forward_KLX_G`, minus `pool_size`/`bg_param`, plus
            `t_list` (strictly increasing in (0, 1], ideally ending at 1.0).
    Output: as `boltzmann_forward_KLX_G` — (y_valid, stages).
    """
    t_list = _fixed_schedule("boltzmann_forward_KLX_G_fixed", t_list)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_forward_KLX_G_fixed")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(6)  # same base stream as boltzmann_forward_KLX_G
    if t_list[-1] != 1.0:
        status(f"boltzmann_forward_KLX_G_fixed: t_list ends at {t_list[-1]:.4f} < 1 "
               f"— incomplete fixed ladder; particle set stops at the last bridge")

    y_valid = x_valid
    identity_flow = flow.zeros()
    stages: list[dict] = []
    t_prev = 0.0
    for k, t_k in enumerate(t_list, start=1):
        artifacts.start_stage(k, t_prev)
        attempt_hist = _new_attempt_hist()
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
        status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} training the increment (fixed) ...")
        seed = jnp.uint32(k)
        cand, ess_hist = train_forward_KLX_G(y_valid, u_prev, u_k, flow,
                                             batch_size, train_steps, lr, ladder, mc_dt,
                                             mc_steps, coeff_lambda, mc_adjust, monitor,
                                             seed=seed, checkpoint=checkpoint,
                                             e_clip=e_clip, g_clip=g_clip)
        log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunks=chunks)
        ess_tr = float(compute_ESS_log(log_w_tr))
        log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
        ess_id = float(compute_ESS_log(log_w_id))
        jax.effects_barrier()  # keep monitor lines ahead of the stage status
        # identity check: keep the better of the trained flow and the
        # identity map (pure SMC), so a stage is never worse than SMC
        if ess_tr >= ess_id:
            stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
            selected = "trained"
        else:
            stage_flow, next_flow, log_w, ess_k = (
                identity_flow, _trainable_identity(flow), log_w_id, ess_id
            )
            selected = "identity"
        imp = ess_k - ess_id  # improvement over identity, always >= 0
        status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
               f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f})")
        _record_attempt(
            attempt_hist, artifacts, stage=k, attempt=1, t=t_k,
            batch_ess=ess_hist, candidate=cand, status="accepted",
            selected=selected, valid_selected_ess=ess_k,
            valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
        )
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "G", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} DONE: stage flow saved; particle set advanced")
        t_prev = t_k
    status(f"boltzmann_forward_KLX_G_fixed: fixed ladder DONE "
           f"({len(stages)} stages, {'COMPLETE' if t_list[-1] == 1.0 else 'INCOMPLETE'})")
    artifacts.finish(complete=t_list[-1] == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages


def _boltzmann_forward_KLXX_G_fixed_impl(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """
    Fixed-schedule variant of `boltzmann_forward_KLXX_G`: bare step-by-step
    training of the full mixture loss along `t_list`, no SMC pre-selection and
    no acceptance/rejection. The quench-and-temper family (`pool_size`, `melt`,
    `opt_alpha`, `opt_steps`), `ladder`, and the loss weights are retained (all
    feed the stage trainer); only `bg_param` is dropped. Same stage records
    and output as the adaptive version.

    Input:  as `boltzmann_forward_KLXX_G`, minus `bg_param`, plus `t_list`
            (strictly increasing in (0, 1], ideally ending at 1.0).
    Output: as `boltzmann_forward_KLXX_G` — (y_valid, stages).
    """
    if not (0.0 <= coeff_alpha < float("inf")):
        raise ValueError(
            "boltzmann_forward_KLXX_G_fixed: coeff_alpha must be finite and non-negative"
        )
    if not (0.0 <= coeff_beta < float("inf")):
        raise ValueError(
            "boltzmann_forward_KLXX_G_fixed: coeff_beta must be finite and non-negative"
        )
    t_list = _fixed_schedule("boltzmann_forward_KLXX_G_fixed", t_list)
    artifacts = FlowArtifactWriter(flow_dir, "boltzmann_forward_KLXX_G_fixed")

    status = monitor.printer if monitor is not None else print
    key = jax.random.key(8)  # same base stream as boltzmann_forward_KLXX_G
    if t_list[-1] != 1.0:
        status(f"boltzmann_forward_KLXX_G_fixed: t_list ends at {t_list[-1]:.4f} < 1 "
               f"— incomplete fixed ladder; particle set stops at the last bridge")

    y_valid = x_valid
    identity_flow = flow.zeros()
    stages: list[dict] = []
    t_prev = 0.0
    for k, t_k in enumerate(t_list, start=1):
        artifacts.start_stage(k, t_prev)
        attempt_hist = _new_attempt_hist()
        u_prev = linear_combination([target, source], [t_prev, 1.0 - t_prev])
        u_k = linear_combination([target, source], [t_k, 1.0 - t_k])
        status(f"[stage {k}] t={t_prev:.4f} -> {t_k:.4f} training the increment (fixed) ...")
        seed = jnp.uint32(k)
        cand, ess_hist = train_forward_KLXX_G(y_valid, u_prev, u_k, flow,
                                              pool_size, batch_size, train_steps, lr, ladder,
                                              melt, opt_alpha, opt_steps, mc_dt,
                                              mc_steps, coeff_lambda, coeff_alpha,
                                              coeff_beta, mc_adjust, monitor,
                                              seed=seed, checkpoint=checkpoint,
                                              e_clip=e_clip, g_clip=g_clip)
        log_w_tr = _iw_log_jit(y_valid, u_prev, u_k, cand, "G", chunks=chunks)
        ess_tr = float(compute_ESS_log(log_w_tr))
        log_w_id = _iw_log_identity(y_valid, u_prev, u_k, chunks=chunks)
        ess_id = float(compute_ESS_log(log_w_id))
        jax.effects_barrier()  # keep monitor lines ahead of the stage status
        # identity check: keep the better of the trained flow and the
        # identity map (pure SMC), so a stage is never worse than SMC
        if ess_tr >= ess_id:
            stage_flow, next_flow, log_w, ess_k = cand, cand, log_w_tr, ess_tr
            selected = "trained"
        else:
            stage_flow, next_flow, log_w, ess_k = (
                identity_flow, _trainable_identity(flow), log_w_id, ess_id
            )
            selected = "identity"
        imp = ess_k - ess_id  # improvement over identity, always >= 0
        status(f"[stage {k}] t={t_k:.4f} validation: ESS = {ess_k:.3f} "
               f"(trained {ess_tr:.3f} / identity {ess_id:.3f}, imp {imp:+.3f})")
        _record_attempt(
            attempt_hist, artifacts, stage=k, attempt=1, t=t_k,
            batch_ess=ess_hist, candidate=cand, status="accepted",
            selected=selected, valid_selected_ess=ess_k,
            valid_trained_ess=ess_tr, valid_identity_ess=ess_id,
        )
        key_res, key_mc = jax.random.split(jax.random.fold_in(key, k))
        y_valid = _bg_advance(key_res, key_mc, y_valid, log_w, stage_flow, "G", u_k,
                              mc_dt, mc_steps, mc_adjust, chunks)
        y_valid = jax.block_until_ready(y_valid)  # errors surface at THIS stage
        selected_path = artifacts.accept_stage(
            k, t_k, stage_flow, selected=selected, valid_selected_ess=ess_k
        )
        stages.append(_stage_record(
            t=t_k, valid_selected_ess=ess_k, valid_trained_ess=ess_tr,
            valid_identity_ess=ess_id, selected=selected, flow=stage_flow,
            hist=attempt_hist, selected_flow_path=selected_path,
        ))
        flow = next_flow
        status(f"[stage {k}] t={t_k:.4f} DONE: stage flow saved; particle set advanced")
        t_prev = t_k
    status(f"boltzmann_forward_KLXX_G_fixed: fixed ladder DONE "
           f"({len(stages)} stages, {'COMPLETE' if t_list[-1] == 1.0 else 'INCOMPLETE'})")
    artifacts.finish(complete=t_list[-1] == 1.0, last_t=t_prev,
                     accepted_stages=len(stages))
    return y_valid, stages
@legacy_keywords(
    n_pool="pool_size", n_batch="batch_size", steps="train_steps",
    mc_step="mc_dt", mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_reverse_KL_F(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Adaptive reverse-KL Boltzmann generator."""
    return _boltzmann_reverse_KL_F_impl(
        x_valid, source, target, flow, pool_size, batch_size, train_steps, lr,
        ladder, mc_dt, mc_steps, mc_adjust, monitor, bg_param, chunks,
        checkpoint, flow_dir,
    )


@legacy_keywords(
    n_pool="pool_size", n_batch="batch_size", steps="train_steps",
    mc_step="mc_dt", mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_forward_KL_G(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Adaptive forward-KL Boltzmann generator."""
    return _boltzmann_forward_KL_G_impl(
        x_valid, source, target, flow, pool_size, batch_size, train_steps, lr,
        ladder, mc_dt, mc_steps, mc_adjust, monitor, bg_param, chunks,
        checkpoint, e_clip, g_clip, flow_dir,
    )


@legacy_keywords(
    n_pool="pool_size", n_batch="batch_size", steps="train_steps",
    mc_step="mc_dt", mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_forward_KLX_G(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Adaptive X-regularized forward-KL Boltzmann generator."""
    return _boltzmann_forward_KLX_G_impl(
        x_valid, source, target, flow, pool_size, batch_size, train_steps, lr,
        ladder, mc_dt, mc_steps, coeff_lambda, mc_adjust, monitor, bg_param,
        chunks, checkpoint, e_clip, g_clip, flow_dir,
    )


@legacy_keywords(
    n_pool="pool_size", n_batch="batch_size", steps="train_steps",
    opt_step="opt_alpha", opt_iters="opt_steps", mc_step="mc_dt",
    mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_forward_KLXX_G(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps: int,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Adaptive full-mixture forward-KLXX Boltzmann generator."""
    return _boltzmann_forward_KLXX_G_impl(
        x_valid, source, target, flow, pool_size, batch_size, train_steps, lr,
        ladder, melt, opt_alpha, opt_steps, mc_dt, mc_steps, coeff_lambda,
        coeff_alpha, coeff_beta, mc_adjust, monitor, bg_param, chunks,
        checkpoint, e_clip, g_clip, flow_dir,
    )


@legacy_keywords(
    n_batch="batch_size", steps="train_steps", mc_step="mc_dt",
    mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_reverse_KL_F_fixed(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    mc_dt: float,
    mc_steps: int,
    t_list,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Fixed-schedule reverse-KL Boltzmann generator."""
    return _boltzmann_reverse_KL_F_fixed_impl(
        x_valid, source, target, flow, batch_size, train_steps, lr, mc_dt,
        mc_steps, t_list, mc_adjust, monitor, chunks, checkpoint, flow_dir,
    )


@legacy_keywords(
    n_batch="batch_size", steps="train_steps", mc_step="mc_dt",
    mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_forward_KL_G_fixed(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Fixed-schedule forward-KL Boltzmann generator."""
    return _boltzmann_forward_KL_G_fixed_impl(
        x_valid, source, target, flow, batch_size, train_steps, lr, ladder,
        mc_dt, mc_steps, t_list, mc_adjust, monitor, chunks, checkpoint,
        e_clip, g_clip, flow_dir,
    )


@legacy_keywords(
    n_batch="batch_size", steps="train_steps", mc_step="mc_dt",
    mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_forward_KLX_G_fixed(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Fixed-schedule X-regularized forward-KL Boltzmann generator."""
    return _boltzmann_forward_KLX_G_fixed_impl(
        x_valid, source, target, flow, batch_size, train_steps, lr, ladder,
        mc_dt, mc_steps, t_list, coeff_lambda, mc_adjust, monitor, chunks,
        checkpoint, e_clip, g_clip, flow_dir,
    )


@legacy_keywords(
    n_pool="pool_size", n_batch="batch_size", steps="train_steps",
    opt_step="opt_alpha", opt_iters="opt_steps", mc_step="mc_dt",
    mc_iters="mc_steps", chunk="chunks",
)
def boltzmann_forward_KLXX_G_fixed(
    x_valid: Array,
    source: Potential,
    target: Potential,
    flow: Flow,
    pool_size: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_alpha: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    e_clip: float = float("inf"),
    g_clip: float = float("inf"),
    flow_dir=None,
) -> tuple[Array, list[dict]]:
    """Fixed-schedule full-mixture forward-KLXX Boltzmann generator."""
    return _boltzmann_forward_KLXX_G_fixed_impl(
        x_valid, source, target, flow, pool_size, batch_size, train_steps, lr,
        ladder, melt, opt_alpha, opt_steps, mc_dt, mc_steps, t_list,
        coeff_lambda, coeff_alpha, coeff_beta, mc_adjust, monitor, chunks,
        checkpoint, e_clip, g_clip, flow_dir,
    )


for _public, _implementation in (
    (boltzmann_reverse_KL_F, _boltzmann_reverse_KL_F_impl),
    (boltzmann_forward_KL_G, _boltzmann_forward_KL_G_impl),
    (boltzmann_forward_KLX_G, _boltzmann_forward_KLX_G_impl),
    (boltzmann_forward_KLXX_G, _boltzmann_forward_KLXX_G_impl),
    (boltzmann_reverse_KL_F_fixed, _boltzmann_reverse_KL_F_fixed_impl),
    (boltzmann_forward_KL_G_fixed, _boltzmann_forward_KL_G_fixed_impl),
    (boltzmann_forward_KLX_G_fixed, _boltzmann_forward_KLX_G_fixed_impl),
    (boltzmann_forward_KLXX_G_fixed, _boltzmann_forward_KLXX_G_fixed_impl),
):
    inherit_implementation_doc(_public, _implementation)
del _public, _implementation

"""Private adaptive and fixed-schedule Boltzmann implementation.

Every controller uses the complete validation population, records explicit
temperature transitions, selects the better trained/identity increment by
full-validation ESS, and can persist/resume stage transactions.
"""

from __future__ import annotations

import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .run_store import RunStore
from ..flow import Flow
from ..potential import Potential, linear_combination
from .drivers import (
    Monitor,
    _train_forward_KL_G_stage,
    _train_forward_KLX_G_stage,
    _train_forward_KLXX_G_stage,
    _train_reverse_KL_F_stage,
    _training_identity,
    _u_clip,
)
from ..utils.anneal import sequential_monte_carlo
from ..utils.metrics import (
    compute_ESS_log,
    importance_weights_log,
    linear_weights_from_log,
    resample,
)
from ..utils.rejuvenation import langevin
from .validation import (
    require_boolean as _boolean,
    require_nonnegative_integer as _nonnegative_integer,
    require_positive_integer as _positive_integer,
    require_real_control as _real_control,
    require_seed as _seed,
)
from .spec import (
    FORWARD_KL_G,
    FORWARD_KLX_G,
    FORWARD_KLXX_G,
    REVERSE_KL_F,
    ObjectiveSpec,
    SELECTION_PROPOSAL_LIMIT,
    normalize_adaptive_policy,
    normalize_fixed_schedule,
)
from .records import StageAttemptResult, build_stage_result
from .state import RunLifecycle, StagePhase
from .signatures import structure_signature


_iw_log_kernel = eqx.filter_jit(importance_weights_log)


def _chunked_log_importance_weights(
    samples, source, target, flow, direction, chunks=1,
):
    result = jnp.concatenate([
        _iw_log_kernel(part, source, target, flow, direction)
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    return jax.block_until_ready(result)


_pot_diff_kernel = eqx.filter_jit(lambda y, source, target: source(y) - target(y))


def _identity_log_importance_weights(samples, source, target, chunks=1):
    result = jnp.concatenate([
        _pot_diff_kernel(part, source, target)
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    return jax.block_until_ready(result)


_smc_jit = eqx.filter_jit(sequential_monte_carlo)


def _bounded_ess(value) -> float:
    """Normalize a diagnostic ESS to its durable mathematical range."""
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return min(max(result, 0.0), 1.0)


@eqx.filter_jit
def _advance_stage_samples(
    key_res,
    key_mc,
    y,
    log_w,
    flow,
    direction,
    u_k,
    mc_dt,
    mc_steps,
    mc_adjust,
    chunks,
):
    push = flow.__call__ if direction == "F" else flow.inv
    proposal = jnp.concatenate([
        push(part) for part in jnp.array_split(y, chunks, axis=0)
    ])
    advanced = resample(
        key_res, proposal, linear_weights_from_log(log_w), N=y.shape[0]
    )
    return langevin(
        key_mc, advanced, u_k, dt=mc_dt, steps=mc_steps,
        adjust=mc_adjust, chunks=chunks,
    )


def _operation_key(base_key, namespace: int, stage: int, attempt: int = 0,
                   index: int = 0):
    key = jax.random.fold_in(base_key, namespace)
    key = jax.random.fold_in(key, stage)
    key = jax.random.fold_in(key, attempt)
    return jax.random.fold_in(key, index)


def _select_adaptive_endpoint(
    *,
    base_key,
    stage,
    y_valid,
    source,
    target,
    t_start,
    t_end,
    parameters,
    ladder,
    mc_dt,
    mc_steps,
    mc_adjust,
    chunks,
    emit_status,
    history=None,
    persist=None,
):
    history = list(history or [])
    if history and history[-1].get("decision") == "accepted":
        return float(history[-1]["t_end"]), history, True, None
    if history and history[-1].get("status") == "skipped":
        return float(history[-1]["t_end"]), history, True, None
    if parameters["tau_smc"] == 0.0:
        event = {
            "index": 0, "t_start": t_start, "t_end": t_end,
            "status": "skipped", "reason": "tau_smc_zero",
            "validation_sample_count": int(y_valid.shape[0]),
        }
        history.append(event)
        if persist is not None:
            persist(event, t_end, None)
        return t_end, history, True, None
    source_bridge = linear_combination(
        [target, source], [t_start, 1.0 - t_start]
    )
    for index in range(len(history), SELECTION_PROPOSAL_LIMIT):
        if not t_end > t_start:
            break
        target_bridge = linear_combination(
            [target, source], [t_end, 1.0 - t_end]
        )
        key = _operation_key(base_key, 201, stage, index=index)
        started = time.perf_counter()
        _, per_level = _smc_jit(
            key, y_valid, source_bridge, target_bridge,
            ladder=ladder, mc_dt=mc_dt, mc_steps=mc_steps,
            adjust=mc_adjust, chunks=chunks,
        )
        per_level = jax.block_until_ready(per_level)
        elapsed = time.perf_counter() - started
        minimum = float(jnp.min(per_level))
        accepted = minimum >= parameters["tau_smc"]
        event = {
            "index": index + 1,
            "t_start": t_start,
            "t_end": t_end,
            "smc_ess": np.asarray(per_level).tolist(),
            "minimum_smc_ess": minimum,
            "decision": "accepted" if accepted else "shrink",
            "raw_key": np.asarray(jax.random.key_data(key)).tolist(),
            "elapsed_seconds": elapsed,
            "validation_sample_count": int(y_valid.shape[0]),
        }
        history.append(event)
        emit_status(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"selection min SMC ESS={minimum:.4f} "
            f"({'accept' if accepted else 'shrink'})"
        )
        if accepted:
            if persist is not None:
                persist(event, t_end, None)
            return t_end, history, True, None
        next_t = t_start + parameters["shrink_factor"] * (t_end - t_start)
        if not t_start < next_t < t_end:
            if persist is not None:
                persist(event, t_end, "temperature_resolution")
            return t_end, history, False, "temperature_resolution"
        if persist is not None:
            persist(event, next_t, None)
        t_end = next_t
    return t_end, history, False, "selection_limit"


def _train_attempt(
    spec: ObjectiveSpec,
    y_valid,
    source,
    target,
    flow,
    *,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    mc_adjust,
    monitor,
    seed,
    checkpoint,
    u_clip,
    g_clip,
    coeff_lambda,
    coeff_alpha,
    coeff_beta,
    melt,
    opt_dt,
    opt_steps,
    t_start,
    t_end,
    stage,
    attempt,
):
    common = dict(
        initialize_from_identity=False,
        t_start=t_start,
        t_end=t_end,
        stage_index=stage,
        attempt_index=attempt,
        _trusted_inputs=True,
    )
    if spec.objective == "reverse_kl":
        return _train_reverse_KL_F_stage(
            y_valid, source, target, flow, batch_size, train_steps, lr,
            mc_dt, mc_steps, mc_adjust, monitor, seed, checkpoint, **common,
        )
    if spec.objective == "forward_kl":
        return _train_forward_KL_G_stage(
            y_valid, source, target, flow, batch_size, train_steps, lr,
            ladder, mc_dt, mc_steps, mc_adjust, monitor, seed, checkpoint,
            u_clip, g_clip, **common,
        )
    if spec.objective == "forward_klx":
        return _train_forward_KLX_G_stage(
            y_valid, source, target, flow, batch_size, train_steps, lr,
            ladder, mc_dt, mc_steps, coeff_lambda, mc_adjust, monitor, seed,
            checkpoint, u_clip, g_clip, **common,
        )
    if spec.objective == "forward_klxx":
        return _train_forward_KLXX_G_stage(
            y_valid, source, target, flow, batch_size, train_steps, lr,
            ladder, melt, opt_dt, opt_steps, mc_dt, mc_steps,
            coeff_lambda, coeff_alpha, coeff_beta, mc_adjust, monitor, seed,
            checkpoint, u_clip, g_clip, **common,
        )
    raise ValueError(f"unsupported Boltzmann objective: {spec.objective!r}")


def _ephemeral_stage_record(
    *, t_start, t_end, selected, selected_flow, continuation_flow,
    validation_count, attempts, selection_history, elapsed_seconds,
):
    normalized = [
        StageAttemptResult(
            t_end=entry["t"],
            loss_history=entry["loss"],
            batch_ess_history=entry["batch_ess"],
            valid_trained_ess=entry["valid_trained_ess"],
            valid_identity_ess=entry["valid_identity_ess"],
            valid_selected_ess=entry["valid_selected_ess"],
            status=entry["status"],
        )
        for entry in attempts
    ]
    return build_stage_result(
        t_start=t_start,
        t_end=t_end,
        selected=selected,
        selected_flow=selected_flow,
        continuation_flow=continuation_flow,
        validation_sample_count=validation_count,
        attempts=normalized,
        selected_flow_path=None,
        continuation_flow_path=None,
        validation_samples_path=None,
        selection_history=selection_history,
        elapsed_seconds=elapsed_seconds,
        asarray=jnp.asarray,
        stack=jnp.stack,
    )


def _potential_signature(potential) -> dict:
    return structure_signature(potential, include_array_values=True)


def _build_run_config(
    *, source, target, spec, batch_size, train_steps, lr, ladder,
    mc_dt, mc_steps, mc_adjust, chunks, checkpoint,
    initialize_from_identity, seed, u_clip, g_clip, coeff_lambda,
    coeff_alpha, coeff_beta, melt, opt_dt, opt_steps, bg_param, t_list,
):
    return {
        "source_signature": _potential_signature(source),
        "target_signature": _potential_signature(target),
        "objective": spec.objective,
        "direction": spec.direction,
        "batch_size": batch_size,
        "train_steps": train_steps,
        "lr": lr,
        "ladder": ladder,
        "mc_dt": mc_dt,
        "mc_steps": mc_steps,
        "mc_adjust": mc_adjust,
        "chunks": chunks,
        "checkpoint": checkpoint,
        "initialize_from_identity": initialize_from_identity,
        "seed": seed,
        "u_clip": u_clip,
        "g_clip": g_clip,
        "coeff_lambda": coeff_lambda,
        "coeff_alpha": coeff_alpha,
        "coeff_beta": coeff_beta,
        "melt": melt,
        "opt_dt": opt_dt,
        "opt_steps": opt_steps,
        "bg_param": bg_param,
        "t_list": t_list,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }


def _run_boltzmann(
    x_valid,
    source,
    target,
    flow,
    *,
    spec: ObjectiveSpec,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    initialize_from_identity,
    mc_adjust,
    monitor,
    chunks,
    checkpoint,
    seed,
    run_dir,
    resume,
    problem_id,
    bg_param=None,
    t_list=None,
    u_clip=float("inf"),
    g_clip=float("inf"),
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    melt=0.0,
    opt_dt=1.0,
    opt_steps=1,
):
    initialize_from_identity = _boolean(
        "initialize_from_identity", initialize_from_identity
    )
    resume = _boolean("resume", resume)
    mc_adjust = _boolean("mc_adjust", mc_adjust)
    checkpoint = _boolean("checkpoint", checkpoint)
    batch_size = _positive_integer("batch_size", batch_size)
    train_steps = _positive_integer("train_steps", train_steps)
    ladder = _positive_integer("ladder", ladder)
    chunks = _positive_integer("chunks", chunks)
    mc_steps = _nonnegative_integer("mc_steps", mc_steps)
    mc_dt = _real_control("mc_dt", mc_dt, strictly_positive=True)
    lr = _real_control("lr", lr, minimum=0.0)
    if monitor is not None and not isinstance(monitor, Monitor):
        raise TypeError("monitor must be a Monitor or None")
    if spec.direction == "G":
        u_clip = _u_clip(u_clip)
        g_clip = _real_control(
            "g_clip", g_clip, minimum=0.0, allow_positive_infinity=True
        )
    if spec.objective in ("forward_klx", "forward_klxx"):
        coeff_lambda = _real_control("coeff_lambda", coeff_lambda, minimum=0.0)
    if spec.objective == "forward_klxx":
        melt = _real_control("melt", melt, minimum=0.0)
        opt_dt = _real_control("opt_dt", opt_dt, strictly_positive=True)
        opt_steps = _nonnegative_integer("opt_steps", opt_steps)
        coeff_alpha = _real_control("coeff_alpha", coeff_alpha, minimum=0.0)
        coeff_beta = _real_control("coeff_beta", coeff_beta, minimum=0.0)
    seed = _seed("seed", seed)
    base_key = jax.random.key(seed)
    if not resume and x_valid is None:
        raise ValueError("x_valid is required for a new run")
    if x_valid is not None:
        try:
            sample_array = jnp.asarray(x_valid)
            sample_shape = tuple(sample_array.shape)
        except (TypeError, ValueError) as exc:
            raise TypeError("x_valid must be array-like") from exc
        if len(sample_shape) != 2 or sample_shape[0] < 1:
            raise ValueError(f"x_valid must have shape [N, d], got {sample_shape}")
        if not jnp.issubdtype(sample_array.dtype, jnp.floating):
            raise TypeError("x_valid must contain real floating-point values")
        if not bool(jnp.all(jnp.isfinite(sample_array))):
            raise ValueError("x_valid must contain only finite coordinates")
        if batch_size > sample_shape[0]:
            raise ValueError("batch_size cannot exceed N_VALID")
        if chunks > sample_shape[0]:
            raise ValueError("chunks cannot exceed N_VALID")
        probe = sample_array[:1]
        del sample_array
        try:
            (
                source_shape, source_grad_shape,
                target_shape, target_grad_shape,
                forward_pair, inverse_pair,
            ) = jax.eval_shape(
                lambda values: (
                    source(values), source.grad(values),
                    target(values), target.grad(values),
                    flow.call_and_ladj(values), flow.inv_and_ladj(values),
                ),
                probe,
            )
        except Exception as exc:
            raise ValueError(
                "x_valid, source, target, and flow are structurally incompatible"
            ) from exc
        forward_shape, forward_ladj_shape = forward_pair
        inverse_shape, inverse_ladj_shape = inverse_pair
        structures_ok = (
            source_shape.shape != (1,) or target_shape.shape != (1,)
            or source_grad_shape.shape != probe.shape
            or target_grad_shape.shape != probe.shape
            or forward_shape.shape != probe.shape
            or inverse_shape.shape != probe.shape
            or forward_ladj_shape.shape != (1,)
            or inverse_ladj_shape.shape != (1,)
        )
        shapes = (
            source_shape, source_grad_shape, target_shape, target_grad_shape,
            forward_shape, forward_ladj_shape, inverse_shape, inverse_ladj_shape,
        )
        dtypes_ok = all(
            jnp.issubdtype(item.dtype, jnp.floating) for item in shapes
        )
        if structures_ok or not dtypes_ok:
            raise ValueError(
                "potentials must return real floating [N] values and [N,d] "
                "gradients; flow maps must return real floating [N,d] values "
                "and [N] log-Jacobians"
            )
    adaptive = t_list is None
    generator = spec.generator_name(fixed=not adaptive)
    parameters = (
        normalize_adaptive_policy(generator, bg_param) if adaptive else None
    )
    schedule = (
        None if adaptive else normalize_fixed_schedule(generator, t_list)
    )
    config = _build_run_config(
        source=source, target=target, spec=spec,
        batch_size=batch_size,
        train_steps=train_steps, lr=lr, ladder=ladder, mc_dt=mc_dt,
        mc_steps=mc_steps, mc_adjust=mc_adjust, chunks=chunks,
        checkpoint=checkpoint,
        initialize_from_identity=initialize_from_identity, seed=seed,
        u_clip=u_clip, g_clip=g_clip, coeff_lambda=coeff_lambda,
        coeff_alpha=coeff_alpha, coeff_beta=coeff_beta, melt=melt,
        opt_dt=opt_dt, opt_steps=opt_steps,
        bg_param=parameters.as_dict() if parameters is not None else None,
        t_list=schedule,
    )
    store = RunStore(
        run_dir,
        generator=generator,
        mode="adaptive" if adaptive else "fixed",
        problem_id=problem_id,
        config=config,
        flow=flow,
        x_valid=x_valid,
        resume=resume,
        requested_t_list=schedule,
    )
    emit_status = monitor.printer if monitor is not None else print
    lifecycle = RunLifecycle.IN_PROGRESS
    stage = None
    stage_started = None
    try:
        if store.enabled:
            assert store.manifest is not None
            saved_lifecycle = RunLifecycle(store.manifest["lifecycle"])
            lifecycle = saved_lifecycle
            if resume and saved_lifecycle is RunLifecycle.EXHAUSTED:
                raise RuntimeError(
                    "the adaptive run exhausted its immutable policy; fork it "
                    "to change retry or safety controls"
                )
            y_valid = jnp.asarray(store.load_last_samples())
            entry_flow = store.load_continuation_flow(flow)
            stages = store.restore_stage_records(flow)
            if saved_lifecycle is RunLifecycle.COMPLETE:
                store.close("clean_exit", lifecycle=RunLifecycle.COMPLETE)
                return y_valid, stages
            lifecycle = RunLifecycle.IN_PROGRESS
            current = store.load_current(flow)
        else:
            if x_valid is None:
                raise ValueError("x_valid is required for a new run")
            y_valid = jnp.asarray(x_valid)
            entry_flow = flow
            stages = []
            current = None
        if y_valid.ndim != 2 or y_valid.shape[0] < 1:
            raise ValueError(f"x_valid must have shape [N, d], got {y_valid.shape}")
        if batch_size > y_valid.shape[0]:
            raise ValueError("batch_size cannot exceed N_VALID")
        if chunks > y_valid.shape[0]:
            raise ValueError("chunks cannot exceed N_VALID")

        while True:
            t_start = float(stages[-1]["t"]) if stages else 0.0
            if adaptive:
                if t_start >= 1.0:
                    lifecycle = RunLifecycle.COMPLETE
                    break
                assert parameters is not None
                if len(stages) >= parameters["max_stages"]:
                    lifecycle = RunLifecycle.EXHAUSTED
                    break
            else:
                assert schedule is not None
                if len(stages) >= len(schedule):
                    lifecycle = RunLifecycle.COMPLETE
                    break

            stage = len(stages) + 1
            resumed_current = current is not None
            if current is not None:
                record = current["record"]
                stage = int(record["stage"])
                t_start = float(record["t_start"])
                t_end = float(record.get("next_t") or record["t_end"])
                entry_flow = current["entry_flow"]
                identity_flow = current["identity_flow"]
                training_identity_flow = current["training_identity_flow"]
                selection_history = list(record.get("selection_history", []))
                phase = StagePhase(record["phase"])
                attempt = int(record.get("current_attempt") or 0)
            else:
                if adaptive:
                    accepted = [0.0] + [float(item["t"]) for item in stages]
                    t_end = (
                        parameters["t_safe"] if not stages
                        else min(
                            accepted[-1]
                            + parameters["enlarge_factor"]
                            * (accepted[-1] - accepted[-2]),
                            1.0,
                        )
                    )
                    if 1.0 - t_end < parameters["t_tol"]:
                        t_end = 1.0
                else:
                    t_end = schedule[len(stages)]
                if not t_start < t_end <= 1.0:
                    store.set_exhaustion_reason("temperature_resolution")
                    lifecycle = RunLifecycle.EXHAUSTED
                    break
                identity_flow = entry_flow.zeros()
                training_identity_flow = _training_identity(entry_flow)
                store.prepare_stage(
                    stage, t_start, t_end, entry_flow, identity_flow,
                    training_identity_flow,
                )
                selection_history = []
                phase = StagePhase.PREPARED
                attempt = 0

            stage_started = time.perf_counter()
            if adaptive and phase is StagePhase.PREPARED:
                def persist_selection(event, next_t, exhaustion_reason):
                    store.append_selection(
                        stage, event, next_t,
                        exhaustion_reason=exhaustion_reason,
                    )

                t_end, selection_history, ok, exhaustion_reason = _select_adaptive_endpoint(
                    base_key=base_key,
                    stage=stage,
                    y_valid=y_valid,
                    source=source,
                    target=target,
                    t_start=t_start,
                    t_end=t_end,
                    parameters=parameters,
                    ladder=ladder,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_adjust=mc_adjust,
                    chunks=chunks,
                    emit_status=emit_status,
                    history=selection_history,
                    persist=persist_selection,
                )
                store.set_selection(
                    stage, t_end, exhaustion_reason=exhaustion_reason
                )
                phase = StagePhase.SELECTION_READY
                if not ok:
                    store.record_stage_session(
                        stage, time.perf_counter() - stage_started, "exhausted"
                    )
                    stage_started = None
                    lifecycle = RunLifecycle.EXHAUSTED
                    break
            elif not adaptive and phase is StagePhase.PREPARED:
                if not selection_history:
                    event = {
                        "index": 0, "t_start": t_start, "t_end": t_end,
                        "status": "skipped", "reason": "fixed_schedule",
                        "validation_sample_count": int(y_valid.shape[0]),
                    }
                    selection_history = [event]
                    store.append_selection(stage, event, t_end)
                store.set_selection(stage, t_end)
                phase = StagePhase.SELECTION_READY

            if phase is StagePhase.ADVANCE_SAVED:
                assert store.enabled and store.root is not None
                selected_flow = current["selected_flow"]
                continuation_flow = current["continuation_flow"]
                y_valid = jnp.asarray(np.load(
                    store.root / record["validation_samples_path"],
                    allow_pickle=False,
                ))
                transition_elapsed = float(
                    record.get("transition_elapsed_seconds", 0.0)
                )
                store.record_stage_session(
                    stage, time.perf_counter() - stage_started, "committed"
                )
                store.commit_stage(stage, transition_elapsed)
                stage_started = None
                stages = store.restore_stage_records(flow)
                entry_flow = continuation_flow
                current = None
                continue

            attempt_host = []
            accepted_stage = False
            max_attempts = parameters["max_retry"] if adaptive else 1
            while attempt < max_attempts or phase in (
                StagePhase.TRAINING,
                StagePhase.CANDIDATE_SAVED,
                StagePhase.EVALUATED,
            ):
                if phase in (StagePhase.SELECTION_READY, StagePhase.REJECTED):
                    attempt += 1
                    optimizer_initial = (
                        training_identity_flow
                        if initialize_from_identity else entry_flow
                    )
                    training_key = _operation_key(
                        base_key, spec.key_namespace, stage, attempt
                    )
                    store.start_attempt(
                        stage, attempt, t_end, optimizer_initial,
                        jax.random.key_data(training_key),
                    )
                    phase = StagePhase.TRAINING
                    resumed_current = False
                elif resumed_current and phase is StagePhase.TRAINING:
                    optimizer_initial = current["optimizer_initial_flow"]
                    training_key = _operation_key(
                        base_key, spec.key_namespace, stage, attempt
                    )
                    store.restart_attempt(stage)
                    resumed_current = False

                u_start = linear_combination(
                    [target, source], [t_start, 1.0 - t_start]
                )
                u_end = linear_combination(
                    [target, source], [t_end, 1.0 - t_end]
                )
                if phase is StagePhase.TRAINING:
                    emit_status(
                        f"[stage {stage:03d} attempt {attempt:02d} | "
                        f"t: {t_start:.6f} -> {t_end:.6f}] training"
                    )
                    trainer_seed = jax.random.key_data(training_key)[0]
                    train_started = time.perf_counter()
                    try:
                        candidate, batch_ess, loss_history = _train_attempt(
                            spec, y_valid, u_start, u_end,
                            optimizer_initial,
                            batch_size=batch_size,
                            train_steps=train_steps,
                            lr=lr,
                            ladder=ladder,
                            mc_dt=mc_dt,
                            mc_steps=mc_steps,
                            mc_adjust=mc_adjust,
                            monitor=monitor,
                            seed=trainer_seed,
                            checkpoint=checkpoint,
                            u_clip=u_clip,
                            g_clip=g_clip,
                            coeff_lambda=coeff_lambda,
                            coeff_alpha=coeff_alpha,
                            coeff_beta=coeff_beta,
                            melt=melt,
                            opt_dt=opt_dt,
                            opt_steps=opt_steps,
                            t_start=t_start,
                            t_end=t_end,
                            stage=stage,
                            attempt=attempt,
                        )
                        candidate, batch_ess, loss_history = jax.block_until_ready(
                            (candidate, batch_ess, loss_history)
                        )
                        jax.effects_barrier()
                    except BaseException as exc:
                        elapsed = time.perf_counter() - train_started
                        store.interrupt_attempt(
                            stage, elapsed,
                            "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt)
                            else "exception",
                            error={
                                "type": type(exc).__name__,
                                "message": str(exc)[:2000],
                            },
                        )
                        raise
                    training_elapsed = time.perf_counter() - train_started
                    try:
                        store.finish_training_execution(stage, training_elapsed)
                    except BaseException:
                        try:
                            store.mark_timing_lower_bound()
                        except BaseException:
                            pass
                        raise
                    store.save_candidate(
                        stage, candidate, loss_history, batch_ess,
                        training_elapsed,
                    )
                    phase = StagePhase.CANDIDATE_SAVED
                elif phase is StagePhase.CANDIDATE_SAVED:
                    candidate = current["candidate"]
                    batch_ess = jnp.asarray(current["batch_ess_history"])
                    loss_history = jnp.asarray(current["loss_history"])

                if phase is StagePhase.CANDIDATE_SAVED:
                    validation_started = time.perf_counter()
                    log_w_trained = _chunked_log_importance_weights(
                        y_valid, u_start, u_end, candidate, spec.direction,
                        chunks=chunks,
                    )
                    log_w_identity = _identity_log_importance_weights(
                        y_valid, u_start, u_end, chunks=chunks,
                    )
                    trained_ess = _bounded_ess(compute_ESS_log(log_w_trained))
                    identity_ess = _bounded_ess(compute_ESS_log(log_w_identity))
                    # The identity candidate is the safety fallback. A tie,
                    # especially a zero-ESS tie, is not evidence that the
                    # newly trained map is safe enough to replace it.
                    if trained_ess > identity_ess:
                        selected_flow = candidate
                        continuation_flow = candidate
                        selected = "trained"
                        selected_ess = trained_ess
                    else:
                        selected_flow = identity_flow
                        continuation_flow = training_identity_flow
                        selected = "identity"
                        selected_ess = identity_ess
                    decision = (
                        "accepted" if not adaptive
                        or selected_ess >= parameters["tau_ess"]
                        else "rejected"
                    )
                    validation_elapsed = time.perf_counter() - validation_started
                    store.save_evaluation(
                        stage,
                        valid_trained_ess=trained_ess,
                        valid_identity_ess=identity_ess,
                        valid_selected_ess=selected_ess,
                        selected=selected,
                        decision=decision,
                        selected_flow=selected_flow,
                        continuation_flow=continuation_flow,
                        validation_elapsed_seconds=validation_elapsed,
                        validation_sample_count=int(y_valid.shape[0]),
                    )
                    phase = StagePhase.EVALUATED
                elif phase is StagePhase.EVALUATED:
                    attempt_record = current["attempt"]
                    trained_ess = float(attempt_record["valid_trained_ess"])
                    identity_ess = float(attempt_record["valid_identity_ess"])
                    selected_ess = float(attempt_record["valid_selected_ess"])
                    selected = attempt_record["selected"]
                    decision = attempt_record["status"]
                    selected_flow = current["selected_flow"]
                    continuation_flow = current["continuation_flow"]
                    candidate = current["candidate"]
                    batch_ess = jnp.asarray(current["batch_ess_history"])
                    loss_history = jnp.asarray(current["loss_history"])

                emit_status(
                    f"[stage {stage:03d} attempt {attempt:02d} | "
                    f"t: {t_start:.6f} -> {t_end:.6f}] validation "
                    f"ESS={selected_ess:.4f} "
                    f"(trained={trained_ess:.4f}, identity={identity_ess:.4f}) "
                    f"{decision.upper()}"
                )
                attempt_host.append({
                    "t": t_end,
                    "loss": loss_history,
                    "batch_ess": batch_ess,
                    "valid_trained_ess": trained_ess,
                    "valid_identity_ess": identity_ess,
                    "valid_selected_ess": selected_ess,
                    "status": decision,
                })
                if decision == "rejected":
                    next_t = t_start + parameters["shrink_factor"] * (t_end - t_start)
                    if not t_start < next_t < t_end:
                        store.mark_rejected(
                            stage, None,
                            exhaustion_reason="temperature_resolution",
                        )
                        phase = StagePhase.REJECTED
                        current = None
                        break
                    store.mark_rejected(stage, next_t)
                    t_end = next_t
                    phase = StagePhase.REJECTED
                    current = None
                    continue
                accepted_stage = True
                break

            if not accepted_stage:
                store.record_stage_session(
                    stage, time.perf_counter() - stage_started, "exhausted"
                )
                stage_started = None
                lifecycle = RunLifecycle.EXHAUSTED
                break

            advance_key = _operation_key(base_key, 301, stage)
            key_resample, key_mcmc = jax.random.split(advance_key)
            log_w = (
                _chunked_log_importance_weights(
                    y_valid, u_start, u_end, selected_flow, spec.direction,
                    chunks=chunks,
                ) if selected == "trained"
                else _identity_log_importance_weights(
                    y_valid, u_start, u_end, chunks=chunks
                )
            )
            advance_started = time.perf_counter()
            y_valid = _advance_stage_samples(
                key_resample, key_mcmc, y_valid, log_w, selected_flow,
                spec.direction, u_end, mc_dt, mc_steps, mc_adjust, chunks,
            )
            y_valid = jax.block_until_ready(y_valid)
            if not bool(jnp.all(jnp.isfinite(y_valid))):
                raise ValueError(
                    "post-stage validation samples contain nonfinite coordinates"
                )
            advance_elapsed = time.perf_counter() - advance_started
            store.save_advance(stage, y_valid, advance_elapsed)
            transition_elapsed = time.perf_counter() - stage_started
            store.record_stage_session(stage, transition_elapsed, "committed")
            store.commit_stage(stage, transition_elapsed)
            stage_started = None
            if store.enabled:
                stages = store.restore_stage_records(flow)
            else:
                stages.append(_ephemeral_stage_record(
                    t_start=t_start,
                    t_end=t_end,
                    selected=selected,
                    selected_flow=selected_flow,
                    continuation_flow=continuation_flow,
                    validation_count=y_valid.shape[0],
                    attempts=attempt_host,
                    selection_history=selection_history,
                    elapsed_seconds=transition_elapsed,
                ))
            entry_flow = continuation_flow
            current = None
            emit_status(
                f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
                f"ACCEPTED in {transition_elapsed:.3f}s"
            )

        store.close("clean_exit", lifecycle=lifecycle)
        return y_valid, stages
    except KeyboardInterrupt:
        if stage is not None and stage_started is not None:
            try:
                store.record_stage_session(
                    stage, time.perf_counter() - stage_started,
                    "keyboard_interrupt",
                )
            except BaseException:
                pass
        store.close(
            "keyboard_interrupt", lifecycle=RunLifecycle.IN_PROGRESS
        )
        raise
    except BaseException:
        if stage is not None and stage_started is not None:
            try:
                store.record_stage_session(
                    stage, time.perf_counter() - stage_started, "exception"
                )
            except BaseException:
                pass
        store.close("exception", lifecycle=lifecycle)
        raise


def boltzmann_reverse_KL_F(
    x_valid: Array | None,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    *,
    initialize_from_identity: bool = True,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run adaptive reverse-KL stages; return particles and accepted stages.

    Set ``run_dir`` to persist a recoverable run and ``resume=True`` to
    continue its last incomplete stage.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=REVERSE_KL_F, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, bg_param=bg_param,
        chunks=chunks, checkpoint=checkpoint, seed=seed, run_dir=run_dir,
        resume=resume, problem_id=problem_id,
    )


def boltzmann_forward_KL_G(
    x_valid: Array | None,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    *,
    initialize_from_identity: bool = True,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    u_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run adaptive forward-KL stages; return particles and accepted stages.

    Set ``run_dir`` to persist a recoverable run and ``resume=True`` to
    continue its last incomplete stage.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=FORWARD_KL_G, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, bg_param=bg_param,
        chunks=chunks, checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip,
        seed=seed, run_dir=run_dir, resume=resume, problem_id=problem_id,
    )


def boltzmann_forward_KLX_G(
    x_valid: Array | None,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    mc_dt: float,
    mc_steps: int,
    *,
    initialize_from_identity: bool = True,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    u_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run adaptive forward-KLX stages; return particles and accepted stages.

    Set ``run_dir`` to persist a recoverable run and ``resume=True`` to
    continue its last incomplete stage.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=FORWARD_KLX_G, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, mc_adjust=mc_adjust, monitor=monitor,
        bg_param=bg_param, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed, run_dir=run_dir,
        resume=resume, problem_id=problem_id,
    )


def boltzmann_forward_KLXX_G(
    x_valid: Array | None,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_dt: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps: int,
    *,
    initialize_from_identity: bool = True,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    bg_param: dict | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    u_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run adaptive forward-KLXX stages; return particles and accepted stages.

    Set ``run_dir`` to persist a recoverable run and ``resume=True`` to
    continue its last incomplete stage.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=FORWARD_KLXX_G, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, melt=melt, opt_dt=opt_dt,
        opt_steps=opt_steps, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta, mc_adjust=mc_adjust, monitor=monitor,
        bg_param=bg_param, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed, run_dir=run_dir,
        resume=resume, problem_id=problem_id,
    )


def boltzmann_reverse_KL_F_fixed(
    x_valid: Array | None,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    mc_dt: float,
    mc_steps: int,
    t_list,
    *,
    initialize_from_identity: bool = True,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run reverse-KL stages on ``t_list``; return particles and stage records.

    Fixed schedules advance once per prescribed transition. Set ``run_dir``
    to persist a recoverable run and ``resume=True`` to continue it.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=REVERSE_KL_F, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=1, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, t_list=t_list, chunks=chunks,
        checkpoint=checkpoint, seed=seed, run_dir=run_dir, resume=resume,
        problem_id=problem_id,
    )


def boltzmann_forward_KL_G_fixed(
    x_valid: Array | None,
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
    *,
    initialize_from_identity: bool = True,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    u_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run forward-KL stages on ``t_list``; return particles and stage records.

    Fixed schedules advance once per prescribed transition. Set ``run_dir``
    to persist a recoverable run and ``resume=True`` to continue it.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=FORWARD_KL_G, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, t_list=t_list, chunks=chunks,
        checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip, seed=seed,
        run_dir=run_dir, resume=resume, problem_id=problem_id,
    )


def boltzmann_forward_KLX_G_fixed(
    x_valid: Array | None,
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
    *,
    initialize_from_identity: bool = True,
    coeff_lambda: float = 1.0,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    u_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run forward-KLX stages on ``t_list``; return particles and stage records.

    Fixed schedules advance once per prescribed transition. Set ``run_dir``
    to persist a recoverable run and ``resume=True`` to continue it.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=FORWARD_KLX_G, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, mc_adjust=mc_adjust, monitor=monitor,
        t_list=t_list, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed, run_dir=run_dir,
        resume=resume, problem_id=problem_id,
    )


def boltzmann_forward_KLXX_G_fixed(
    x_valid: Array | None,
    source: Potential,
    target: Potential,
    flow: Flow,
    batch_size: int,
    train_steps: int,
    lr: float,
    ladder: int,
    melt: float,
    opt_dt: float,
    opt_steps: int,
    mc_dt: float,
    mc_steps: int,
    t_list,
    *,
    initialize_from_identity: bool = True,
    coeff_lambda: float = 1.0,
    coeff_alpha: float = 0.5,
    coeff_beta: float = 0.5,
    mc_adjust: bool = True,
    monitor: Monitor | None = None,
    chunks: int = 1,
    checkpoint: bool = False,
    u_clip: float = float("inf"),
    g_clip: float = float("inf"),
    seed: int = 0,
    run_dir: str | Path | None = None,
    resume: bool = False,
    problem_id: str | None = None,
) -> tuple[Array, list[dict]]:
    """Run forward-KLXX stages on ``t_list``; return particles and records.

    Fixed schedules advance once per prescribed transition. Set ``run_dir``
    to persist a recoverable run and ``resume=True`` to continue it.
    """
    return _run_boltzmann(
        x_valid, source, target, flow,
        spec=FORWARD_KLXX_G, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, melt=melt, opt_dt=opt_dt,
        opt_steps=opt_steps, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta, mc_adjust=mc_adjust, monitor=monitor,
        t_list=t_list, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed, run_dir=run_dir,
        resume=resume, problem_id=problem_id,
    )

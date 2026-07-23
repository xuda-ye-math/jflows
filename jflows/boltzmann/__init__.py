"""Pure adaptive-staging and fixed-schedule Boltzmann computations.

The controller calls only the public direct trainers.  It contains no file,
manifest, resume, or artifact logic; ``jflows.boltzmann.write`` and
``jflows.boltzmann.load`` handle complete-stage persistence separately.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..flow import Flow
from ..potential import Potential, linear_combination
from ..utils.anneal import sequential_monte_carlo
from ..utils.metrics import (
    compute_ESS_log,
    importance_weights_log,
    linear_weights_from_log,
    resample,
)
from ..utils.rejuvenation import langevin
from ..train import (
    Monitor,
    _training_identity,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
    train_reverse_KL_F,
)

__all__ = [
    "boltzmann_identity",
    "boltzmann_forward_KL_G",
    "boltzmann_forward_KL_G_fixed",
    "boltzmann_forward_KLX_G",
    "boltzmann_forward_KLX_G_fixed",
    "boltzmann_forward_KLXX_G",
    "boltzmann_forward_KLXX_G_fixed",
    "boltzmann_reverse_KL_F",
    "boltzmann_reverse_KL_F_fixed",
]

_OBJECTIVES = {
    "reverse_kl": ("F", 101),
    "forward_kl": ("G", 102),
    "forward_klx": ("G", 103),
    "forward_klxx": ("G", 104),
}
_POLICY = {
    "t_safe": 0.2,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_smc": 0.0,
    "tau_ess": 0.6,
    "t_tol": 0.01,
    "max_stages": 30,
    "max_retry": 6,
}
_SELECTION_PROPOSAL_LIMIT = 60


def _adaptive_policy(values):
    policy = dict(_POLICY)
    if values:
        policy.update(values)
    return policy
_importance_kernel = eqx.filter_jit(importance_weights_log)
_identity_kernel = eqx.filter_jit(lambda y, source, target: source(y) - target(y))


def _chunked_log_weights(samples, source, target, flow, direction, chunks):
    """Evaluate trained-flow log weights in sequential device chunks."""
    result = jnp.concatenate([
        _importance_kernel(part, source, target, flow, direction)
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    return jax.block_until_ready(result)


def _identity_log_weights(samples, source, target, chunks):
    """Evaluate exact-identity log weights in sequential device chunks."""
    result = jnp.concatenate([
        _identity_kernel(part, source, target)
        for part in jnp.array_split(samples, chunks, axis=0)
    ])
    return jax.block_until_ready(result)


def _advance_stage_samples(
    key_resample,
    key_mc,
    samples,
    log_weights,
    flow,
    direction,
    target,
    mc_dt,
    mc_steps,
    mc_adjust,
    chunks,
):
    """Push, resample, and rejuvenate one accepted stage population."""
    push = flow.__call__ if direction == "F" else flow.inv
    proposal = jnp.concatenate([
        push(part) for part in jnp.array_split(samples, chunks, axis=0)
    ])
    advanced = resample(
        key_resample,
        proposal,
        linear_weights_from_log(log_weights),
        N=samples.shape[0],
    )
    return langevin(
        key_mc,
        advanced,
        target,
        dt=mc_dt,
        steps=mc_steps,
        adjust=mc_adjust,
        chunks=chunks,
    )


def _operation_key(base_key, namespace, stage, attempt=0, index=0):
    """Derive one deterministic key for a stage operation."""
    key = jax.random.fold_in(base_key, namespace)
    key = jax.random.fold_in(key, stage)
    key = jax.random.fold_in(key, attempt)
    return jax.random.fold_in(key, index)


def _select_adaptive_endpoint(
    *,
    base_key,
    stage,
    samples,
    source,
    target,
    t_start,
    t_end,
    policy,
    ladder,
    mc_dt,
    mc_steps,
    mc_adjust,
    chunks,
    emit,
):
    """Apply the optional SMC gate to one proposed stage endpoint."""
    if policy["tau_smc"] == 0.0:
        return t_end, [{
            "index": 0,
            "t_start": t_start,
            "t_end": t_end,
            "status": "skipped",
            "reason": "tau_smc_zero",
            "validation_sample_count": int(samples.shape[0]),
        }], True
    history = []
    source_bridge = linear_combination(
        [target, source], [t_start, 1.0 - t_start]
    )
    for index in range(_SELECTION_PROPOSAL_LIMIT):
        target_bridge = linear_combination(
            [target, source], [t_end, 1.0 - t_end]
        )
        key = _operation_key(base_key, 201, stage, index=index)
        _, per_level = sequential_monte_carlo(
            key,
            samples,
            source_bridge,
            target_bridge,
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps,
            adjust=mc_adjust,
            chunks=chunks,
        )
        per_level = jax.block_until_ready(per_level)
        minimum = float(jnp.min(per_level))
        accepted = minimum >= policy["tau_smc"]
        history.append({
            "index": index + 1,
            "t_start": t_start,
            "t_end": t_end,
            "smc_ess": np.asarray(per_level).tolist(),
            "minimum_smc_ess": minimum,
            "decision": "accepted" if accepted else "shrink",
            "validation_sample_count": int(samples.shape[0]),
        })
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"selection min SMC ESS={minimum:.4f} "
            f"({'accept' if accepted else 'shrink'})"
        )
        if accepted:
            return t_end, history, True
        next_t = t_start + policy["shrink_factor"] * (t_end - t_start)
        if not t_start < next_t < t_end:
            return t_end, history, False
        t_end = next_t
    return t_end, history, False


def _stage_monitor(monitor, stage, attempt):
    """Add stage context to an existing monitor without changing its sink."""
    if monitor is None:
        return None
    return Monitor(
        monitor.every,
        f"{monitor.prefix}[stage {stage:03d} attempt {attempt:02d}] ",
        monitor.printer,
    )


def _train_attempt(
    objective,
    samples,
    source,
    target,
    flow,
    *,
    pool_size,
    batch_size,
    train_steps,
    lr,
    ladder,
    mc_dt,
    mc_steps,
    mc_adjust,
    chunks,
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
):
    """Dispatch one objective to its public direct trainer."""
    common = dict(
        initialize_from_identity=False,
        t_start=t_start,
        t_end=t_end,
    )
    if objective == "reverse_kl":
        return train_reverse_KL_F(
            samples, source, target, flow, batch_size, train_steps, lr,
            mc_dt, mc_steps, mc_adjust, monitor, seed, checkpoint, **common,
        )
    if objective == "forward_kl":
        return train_forward_KL_G(
            samples, source, target, flow, batch_size, train_steps, lr,
            ladder, mc_dt, mc_steps, mc_adjust, monitor, seed, checkpoint,
            u_clip, g_clip, **common,
        )
    if objective == "forward_klx":
        return train_forward_KLX_G(
            samples, source, target, flow, batch_size, train_steps, lr,
            ladder, mc_dt, mc_steps, coeff_lambda, mc_adjust, monitor, seed,
            checkpoint, u_clip, g_clip, **common,
        )
    if objective == "forward_klxx":
        return train_forward_KLXX_G(
            samples, source, target, flow, pool_size, batch_size, train_steps,
            lr, ladder, melt, opt_dt, opt_steps, mc_dt, mc_steps,
            coeff_lambda, coeff_alpha, coeff_beta, mc_adjust, monitor, seed,
            checkpoint, u_clip, g_clip, chunks=chunks, **common,
        )
def _stage_record(
    *,
    t_start,
    t_end,
    selected,
    selected_flow,
    continuation_flow,
    sample_count,
    t_history,
    batch_ess_history,
    trained_ess_history,
    identity_ess_history,
    status_history,
    selection_history,
    elapsed_seconds,
):
    """Build one accepted in-memory stage record."""
    return {
        "t": float(t_end),
        "t_start": float(t_start),
        "valid_selected_ess": float(
            max(trained_ess_history[-1], identity_ess_history[-1])
        ),
        "valid_trained_ess": float(trained_ess_history[-1]),
        "valid_identity_ess": float(identity_ess_history[-1]),
        "valid_sample_count": int(sample_count),
        "selected": selected,
        "flow": selected_flow,
        "continuation_flow": continuation_flow,
        "t_hist": jnp.asarray(t_history),
        "batch_ess_hist": jnp.stack(batch_ess_history),
        "valid_trained_ess_hist": jnp.asarray(trained_ess_history),
        "valid_identity_ess_hist": jnp.asarray(identity_ess_history),
        "attempt_status_hist": tuple(status_history),
        "selection_history": tuple(selection_history),
        "elapsed_seconds": float(elapsed_seconds),
        "selected_flow_path": None,
        "continuation_flow_path": None,
        "validation_samples_path": None,
    }


def _identity_stage_record(
    *,
    t_start,
    t_end,
    sample_count,
    t_history,
    identity_ess_history,
    status_history,
    selection_history,
    elapsed_seconds,
):
    """Build one accepted identity-only stage record."""
    return {
        "t": float(t_end),
        "t_start": float(t_start),
        "valid_selected_ess": float(identity_ess_history[-1]),
        "valid_identity_ess": float(identity_ess_history[-1]),
        "valid_sample_count": int(sample_count),
        "selected": "identity",
        "t_hist": jnp.asarray(t_history),
        "valid_identity_ess_hist": jnp.asarray(identity_ess_history),
        "attempt_status_hist": tuple(status_history),
        "selection_history": tuple(selection_history),
        "elapsed_seconds": float(elapsed_seconds),
        "validation_samples_path": None,
    }


def iterate_identity(
    samples,
    source,
    target,
    *,
    ladder,
    mc_dt,
    mc_steps,
    mc_adjust,
    monitor,
    chunks,
    seed,
    bg_param=None,
    accepted_t=(0.0,),
    start_stage=1,
):
    """Yield adaptive-staging identity-only Boltzmann stages."""
    samples = jnp.asarray(samples)
    policy = _adaptive_policy(bg_param)
    accepted = [float(value) for value in accepted_t]
    base_key = jax.random.key(seed)
    emit = monitor.printer if monitor is not None else print
    stage = start_stage

    while accepted[-1] < 1.0:
        if stage > policy["max_stages"]:
            return
        t_start = accepted[-1]
        if len(accepted) == 1:
            t_end = policy["t_safe"]
        else:
            t_end = min(
                accepted[-1]
                + policy["enlarge_factor"]
                * (accepted[-1] - accepted[-2]),
                1.0,
            )
        if 1.0 - t_end < policy["t_tol"]:
            t_end = 1.0
        t_end, selection_history, selected_endpoint = _select_adaptive_endpoint(
            base_key=base_key,
            stage=stage,
            samples=samples,
            source=source,
            target=target,
            t_start=t_start,
            t_end=t_end,
            policy=policy,
            ladder=ladder,
            mc_dt=mc_dt,
            mc_steps=mc_steps,
            mc_adjust=mc_adjust,
            chunks=chunks,
            emit=emit,
        )
        if not selected_endpoint:
            return

        stage_started = time.perf_counter()
        t_history = []
        identity_history = []
        status_history = []
        accepted_stage = False
        for attempt in range(1, policy["max_retry"] + 1):
            source_bridge = linear_combination(
                [target, source], [t_start, 1.0 - t_start]
            )
            target_bridge = linear_combination(
                [target, source], [t_end, 1.0 - t_end]
            )
            log_weights = _identity_log_weights(
                samples, source_bridge, target_bridge, chunks
            )
            identity_ess = float(compute_ESS_log(log_weights))
            accepted_attempt = identity_ess >= policy["tau_ess"]
            status = "accepted" if accepted_attempt else "rejected"
            t_history.append(t_end)
            identity_history.append(identity_ess)
            status_history.append(status)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] identity "
                f"ESS={identity_ess:.4f} {status.upper()}"
            )
            if accepted_attempt:
                accepted_stage = True
                break
            next_t = t_start + policy["shrink_factor"] * (t_end - t_start)
            if not t_start < next_t < t_end:
                break
            t_end = next_t

        if not accepted_stage:
            return
        advance_key = _operation_key(base_key, 301, stage)
        key_resample, key_mc = jax.random.split(advance_key)
        samples = resample(
            key_resample,
            samples,
            linear_weights_from_log(log_weights),
            N=samples.shape[0],
        )
        samples = langevin(
            key_mc,
            samples,
            target_bridge,
            dt=mc_dt,
            steps=mc_steps,
            adjust=mc_adjust,
            chunks=chunks,
        )
        samples = jax.block_until_ready(samples)
        if not bool(jnp.all(jnp.isfinite(samples))):
            raise FloatingPointError(
                "post-stage samples contain nonfinite coordinates"
            )
        record = _identity_stage_record(
            t_start=t_start,
            t_end=t_end,
            sample_count=samples.shape[0],
            t_history=t_history,
            identity_ess_history=identity_history,
            status_history=status_history,
            selection_history=selection_history,
            elapsed_seconds=time.perf_counter() - stage_started,
        )
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"ACCEPTED in {record['elapsed_seconds']:.3f}s"
        )
        yield samples, record, None
        accepted.append(t_end)
        stage += 1


def iterate_boltzmann(
    samples,
    source,
    target,
    flow,
    *,
    objective,
    pool_size,
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
    accepted_t=(0.0,),
    start_stage=1,
):
    """Yield each newly completed Boltzmann stage and particle population."""
    samples = jnp.asarray(samples)

    adaptive = t_list is None
    policy = (
        _adaptive_policy(bg_param)
        if adaptive else None
    )
    schedule = (
        None if adaptive else list(t_list)
    )
    accepted = [float(value) for value in accepted_t]
    direction, namespace = _OBJECTIVES[objective]
    entry_flow = flow
    base_key = jax.random.key(seed)
    emit = monitor.printer if monitor is not None else print
    stage = start_stage

    while accepted[-1] < 1.0:
        t_start = accepted[-1]
        if adaptive:
            if stage > policy["max_stages"]:
                return
            if len(accepted) == 1:
                t_end = policy["t_safe"]
            else:
                t_end = min(
                    accepted[-1]
                    + policy["enlarge_factor"]
                    * (accepted[-1] - accepted[-2]),
                    1.0,
                )
            if 1.0 - t_end < policy["t_tol"]:
                t_end = 1.0
            t_end, selection_history, selected_endpoint = (
                _select_adaptive_endpoint(
                    base_key=base_key,
                    stage=stage,
                    samples=samples,
                    source=source,
                    target=target,
                    t_start=t_start,
                    t_end=t_end,
                    policy=policy,
                    ladder=ladder,
                    mc_dt=mc_dt,
                    mc_steps=mc_steps,
                    mc_adjust=mc_adjust,
                    chunks=chunks,
                    emit=emit,
                )
            )
            if not selected_endpoint:
                return
            max_attempts = policy["max_retry"]
        else:
            remaining = [endpoint for endpoint in schedule if endpoint > t_start]
            if not remaining:
                return
            t_end = remaining[0]
            selection_history = ({
                "index": 0,
                "t_start": t_start,
                "t_end": t_end,
                "status": "skipped",
                "reason": "fixed_schedule",
                "validation_sample_count": int(samples.shape[0]),
            },)
            max_attempts = 1

        stage_started = time.perf_counter()
        identity_flow = entry_flow.zeros()
        training_identity = _training_identity(entry_flow)
        t_history = []
        batch_histories = []
        trained_history = []
        identity_history = []
        status_history = []
        accepted_stage = False
        for attempt in range(1, max_attempts + 1):
            optimizer_initial = (
                training_identity if initialize_from_identity else entry_flow
            )
            source_bridge = linear_combination(
                [target, source], [t_start, 1.0 - t_start]
            )
            target_bridge = linear_combination(
                [target, source], [t_end, 1.0 - t_end]
            )
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] training"
            )
            training_key = _operation_key(
                base_key, namespace, stage, attempt
            )
            trainer_seed = jax.random.key_data(training_key)[0]
            candidate, batch_ess = _train_attempt(
                objective,
                samples,
                source_bridge,
                target_bridge,
                optimizer_initial,
                pool_size=pool_size,
                batch_size=batch_size,
                train_steps=train_steps,
                lr=lr,
                ladder=ladder,
                mc_dt=mc_dt,
                mc_steps=mc_steps,
                mc_adjust=mc_adjust,
                chunks=chunks,
                monitor=_stage_monitor(monitor, stage, attempt),
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
            )
            candidate, batch_ess = jax.block_until_ready(
                (candidate, batch_ess)
            )
            trained_log_weights = _chunked_log_weights(
                samples,
                source_bridge,
                target_bridge,
                candidate,
                direction,
                chunks,
            )
            identity_log_weights = _identity_log_weights(
                samples, source_bridge, target_bridge, chunks
            )
            trained_ess = float(compute_ESS_log(trained_log_weights))
            identity_ess = float(compute_ESS_log(identity_log_weights))
            if trained_ess > identity_ess:
                selected_flow = candidate
                continuation_flow = candidate
                selected = "trained"
                selected_ess = trained_ess
                selected_log_weights = trained_log_weights
            else:
                selected_flow = identity_flow
                continuation_flow = training_identity
                selected = "identity"
                selected_ess = identity_ess
                selected_log_weights = identity_log_weights
            accepted_attempt = not adaptive or selected_ess >= policy["tau_ess"]
            status = "accepted" if accepted_attempt else "rejected"
            t_history.append(t_end)
            batch_histories.append(batch_ess)
            trained_history.append(trained_ess)
            identity_history.append(identity_ess)
            status_history.append(status)
            emit(
                f"[stage {stage:03d} attempt {attempt:02d} | "
                f"t: {t_start:.6f} -> {t_end:.6f}] validation "
                f"ESS={selected_ess:.4f} "
                f"(trained={trained_ess:.4f}, identity={identity_ess:.4f}) "
                f"{status.upper()}"
            )
            if accepted_attempt:
                accepted_stage = True
                break
            next_t = t_start + policy["shrink_factor"] * (t_end - t_start)
            if not t_start < next_t < t_end:
                break
            t_end = next_t

        if not accepted_stage:
            return
        advance_key = _operation_key(base_key, 301, stage)
        key_resample, key_mc = jax.random.split(advance_key)
        samples = _advance_stage_samples(
            key_resample,
            key_mc,
            samples,
            selected_log_weights,
            selected_flow,
            direction,
            target_bridge,
            mc_dt,
            mc_steps,
            mc_adjust,
            chunks,
        )
        samples = jax.block_until_ready(samples)
        if not bool(jnp.all(jnp.isfinite(samples))):
            raise FloatingPointError(
                "post-stage samples contain nonfinite coordinates"
            )
        record = _stage_record(
            t_start=t_start,
            t_end=t_end,
            selected=selected,
            selected_flow=selected_flow,
            continuation_flow=continuation_flow,
            sample_count=samples.shape[0],
            t_history=t_history,
            batch_ess_history=batch_histories,
            trained_ess_history=trained_history,
            identity_ess_history=identity_history,
            status_history=status_history,
            selection_history=selection_history,
            elapsed_seconds=time.perf_counter() - stage_started,
        )
        emit(
            f"[stage {stage:03d} | t: {t_start:.6f} -> {t_end:.6f}] "
            f"ACCEPTED in {record['elapsed_seconds']:.3f}s"
        )
        yield samples, record, continuation_flow
        accepted.append(t_end)
        entry_flow = continuation_flow
        stage += 1


def run_boltzmann(samples, source, target, flow, **controls):
    """Collect the pure stage iterator into the public return contract."""
    stages = []
    current = jnp.asarray(samples)
    for current, record, _ in iterate_boltzmann(
        current, source, target, flow, **controls
    ):
        stages.append(record)
    return current, stages


def boltzmann_identity(
    x_valid,
    source,
    target,
    ladder,
    mc_dt,
    mc_steps,
    *,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    seed=0,
):
    """Run adaptive-staging identity-only Boltzmann stages without flow training."""
    stages = []
    current = jnp.asarray(x_valid)
    for current, record, _ in iterate_identity(
        current,
        source,
        target,
        ladder=ladder,
        mc_dt=mc_dt,
        mc_steps=mc_steps,
        mc_adjust=mc_adjust,
        monitor=monitor,
        bg_param=bg_param,
        chunks=chunks,
        seed=seed,
    ):
        stages.append(record)
    return current, stages


def boltzmann_reverse_KL_F(
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
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    seed=0,
):
    """Run adaptive-staging reverse-KL Boltzmann stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="reverse_kl", pool_size=0,
        batch_size=batch_size, train_steps=train_steps, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, bg_param=bg_param,
        chunks=chunks, checkpoint=checkpoint, seed=seed,
    )


def boltzmann_forward_KL_G(
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
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    seed=0,
):
    """Run adaptive-staging forward-KL Boltzmann stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_kl", pool_size=0,
        batch_size=batch_size, train_steps=train_steps, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, bg_param=bg_param,
        chunks=chunks, checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip,
        seed=seed,
    )


def boltzmann_forward_KLX_G(
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
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    seed=0,
):
    """Run adaptive-staging forward-KLX Boltzmann stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klx", pool_size=0,
        batch_size=batch_size, train_steps=train_steps, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, mc_adjust=mc_adjust, monitor=monitor,
        bg_param=bg_param, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed,
    )


def boltzmann_forward_KLXX_G(
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
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    seed=0,
):
    """Run adaptive-staging KLXX stages with full or separately sampled QT pools."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klxx",
        pool_size=pool_size, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, melt=melt, opt_dt=opt_dt,
        opt_steps=opt_steps, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta, mc_adjust=mc_adjust, monitor=monitor,
        bg_param=bg_param, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed,
    )


def boltzmann_reverse_KL_F_fixed(
    x_valid,
    source,
    target,
    flow,
    batch_size,
    train_steps,
    lr,
    mc_dt,
    mc_steps,
    t_list,
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    chunks=1,
    checkpoint=False,
    seed=0,
):
    """Run fixed-schedule reverse-KL Boltzmann stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="reverse_kl", pool_size=0,
        batch_size=batch_size, train_steps=train_steps, lr=lr, ladder=1,
        mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, t_list=t_list, chunks=chunks,
        checkpoint=checkpoint, seed=seed,
    )


def boltzmann_forward_KL_G_fixed(
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
    t_list,
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    chunks=1,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    seed=0,
):
    """Run fixed-schedule forward-KL Boltzmann stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_kl", pool_size=0,
        batch_size=batch_size, train_steps=train_steps, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        mc_adjust=mc_adjust, monitor=monitor, t_list=t_list, chunks=chunks,
        checkpoint=checkpoint, u_clip=u_clip, g_clip=g_clip, seed=seed,
    )


def boltzmann_forward_KLX_G_fixed(
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
    t_list,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    chunks=1,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    seed=0,
):
    """Run fixed-schedule forward-KLX Boltzmann stages."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klx", pool_size=0,
        batch_size=batch_size, train_steps=train_steps, lr=lr, ladder=ladder,
        mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, mc_adjust=mc_adjust, monitor=monitor,
        t_list=t_list, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed,
    )


def boltzmann_forward_KLXX_G_fixed(
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
    t_list,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_alpha=0.5,
    coeff_beta=0.5,
    mc_adjust=True,
    monitor=None,
    chunks=1,
    checkpoint=False,
    u_clip=float("inf"),
    g_clip=float("inf"),
    seed=0,
):
    """Run fixed-schedule KLXX stages with configurable QT pool sizing."""
    return run_boltzmann(
        x_valid, source, target, flow, objective="forward_klxx",
        pool_size=pool_size, batch_size=batch_size, train_steps=train_steps,
        lr=lr, ladder=ladder, melt=melt, opt_dt=opt_dt,
        opt_steps=opt_steps, mc_dt=mc_dt, mc_steps=mc_steps,
        initialize_from_identity=initialize_from_identity,
        coeff_lambda=coeff_lambda, coeff_alpha=coeff_alpha,
        coeff_beta=coeff_beta, mc_adjust=mc_adjust, monitor=monitor,
        t_list=t_list, chunks=chunks, checkpoint=checkpoint,
        u_clip=u_clip, g_clip=g_clip, seed=seed,
    )

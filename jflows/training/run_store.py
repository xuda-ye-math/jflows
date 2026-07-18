"""Private transactional run store for recoverable jflows Boltzmann runs.

The numerical controllers use :class:`RunStore`; users should use the
read-only helpers in :mod:`jflows.artifacts`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import equinox as eqx
import jax
import numpy as np

from .records import StageAttemptResult, build_stage_result
from .signatures import structure_signature
from .state import RunLifecycle, StagePhase
from .spec import SELECTION_PROPOSAL_LIMIT, normalize_adaptive_policy
from ..version import __version__


ARTIFACT_FORMAT = "jflows-boltzmann-run"
SCHEMA_VERSION = 1


def _transactional(method):
    """Hold the short metadata transaction lock for one store mutation."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._transaction():
            return method(self, *args, **kwargs)
    return wrapped


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(repr(array.shape).encode())
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), 1024 * 1024):
        digest.update(raw[start:start + 1024 * 1024])
    return digest.hexdigest()


def _json_value(value):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    return repr(value)


def _canonical_json(value) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _seal_record(value: dict) -> dict:
    sealed = _json_value(value)
    sealed.pop("record_digest", None)
    sealed["record_digest"] = hashlib.sha256(
        _canonical_json(sealed).encode()
    ).hexdigest()
    return sealed


def _atomic_record(path: Path, value: dict) -> dict:
    sealed = _seal_record(value)
    _atomic_json(path, sealed)
    value.clear()
    value.update(sealed)
    return sealed


def _read_record(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing {label}: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("record_digest"), str):
        raise ValueError(f"{label} has no integrity digest: {path}")
    if _seal_record(value)["record_digest"] != value["record_digest"]:
        raise ValueError(f"{label} integrity digest mismatch: {path}")
    return value


def _mark_running_attempt_unclean(root: Path, manifest: dict) -> bool:
    """Close an attempt ledger whose owning run session has ended."""
    current = manifest.get("current_stage")
    if not isinstance(current, dict) or current.get("phase") != StagePhase.TRAINING:
        return False
    stage = current.get("stage")
    if not isinstance(stage, int):
        return False
    stage_record = _read_record(
        root / "stages" / f"stage_{stage:06d}" / "stage.json",
        "stage metadata",
    )
    attempts = stage_record.get("attempts")
    if not attempts:
        return False
    attempt_path = _contained_path(root, attempts[-1], "attempt path")
    attempt = _read_record(attempt_path, "attempt metadata")
    executions = attempt.get("executions")
    if not executions or executions[-1].get("outcome") != "running":
        return False
    executions[-1]["outcome"] = "unclean_termination"
    executions[-1]["elapsed_seconds"] = 0.0
    executions[-1].pop("ended_at", None)
    _atomic_record(attempt_path, attempt)
    manifest["timing_lower_bound"] = True
    return True


def _config_digest(config: dict) -> str:
    return hashlib.sha256(_canonical_json(config).encode()).hexdigest()


def _flow_signature(flow) -> dict:
    return structure_signature(flow, include_array_values=False)


def _sample_signature(samples) -> dict:
    value = np.asarray(samples)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _array_hash(value),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(value), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_flow(path: Path, flow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        eqx.tree_serialise_leaves(tmp, flow)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_npy(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, np.asarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with tmp.open("wb") as handle:
            np.savez(handle, **{key: np.asarray(value) for key, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes on POSIX filesystems."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_manifest(root) -> dict:
    path = Path(root).expanduser().resolve() / "run.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"not a jflows run directory: {path.parent}") from exc
    if value.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError(
            f"unsupported artifact format {value.get('artifact_format')!r}; "
            f"expected {ARTIFACT_FORMAT!r}"
        )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact schema {value.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    try:
        RunLifecycle(value.get("lifecycle"))
    except ValueError as exc:
        raise ValueError(
            f"unsupported run lifecycle {value.get('lifecycle')!r}"
        ) from exc
    return value


def _contained_path(root: Path, relative, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{label} escapes the run directory: {relative}")
    return path


def _require_artifact(
    root: Path, artifacts: dict, relative, label: str,
) -> Path:
    path = _contained_path(root, relative, label)
    if relative not in artifacts:
        raise ValueError(f"{label} is not registered: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {relative}")
    return path


def _artifact_payload_receipt(artifacts: dict, relative: str) -> tuple[str, int]:
    """Return the content identity already verified by ``validate_manifest``."""
    metadata = artifacts[relative]
    return metadata["sha256"], metadata["bytes"]


def _finite_nonnegative(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _stage_elapsed_seconds(root: Path, record: dict) -> float:
    total = sum(
        float(item.get("elapsed_seconds") or 0.0)
        for item in record.get("selection_history", [])
    ) + float(record.get("timing", {}).get("advance_seconds") or 0.0)
    for relative in record.get("attempts", []):
        attempt = _read_record(root / relative, "attempt metadata")
        total += sum(
            float(item.get("elapsed_seconds") or 0.0)
            for item in attempt.get("executions", [])
        )
        total += float(attempt.get("validation_elapsed_seconds") or 0.0)
    return total


def _validate_training_payload(
    path: Path, label: str, *, expected_steps: int,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"step", "loss", "batch_ess"}:
                raise ValueError(f"{label} has unexpected arrays")
            step = np.asarray(data["step"])
            loss = np.asarray(data["loss"])
            batch_ess = np.asarray(data["batch_ess"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(label):
            raise
        raise ValueError(f"{label} is not a valid NPZ payload") from exc
    if (
        step.ndim != 1 or loss.ndim != 1 or batch_ess.ndim != 1
        or len(step) != expected_steps
        or len(step) != len(loss) or len(step) != len(batch_ess)
        or not np.array_equal(step, np.arange(1, len(step) + 1))
    ):
        raise ValueError(f"{label} has inconsistent per-step histories")
    if not np.all(np.isfinite(batch_ess)) or np.any(
        (batch_ess < 0.0) | (batch_ess > 1.0)
    ):
        raise ValueError(f"{label} has invalid batch ESS values")


def _validate_validation_payload(
    path: Path,
    *,
    stage: int,
    attempt: int,
    t_start: float,
    t_end: float,
    sample_count: int,
) -> tuple[float, float, float]:
    expected_fields = {
        "stage", "attempt", "evaluation_index", "t_start", "t_end",
        "role", "value", "sample_count",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != expected_fields:
                raise ValueError("validation history has unexpected arrays")
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("validation history"):
            raise
        raise ValueError("validation history is not a valid NPZ payload") from exc
    if any(value.shape != (3,) for value in arrays.values()):
        raise ValueError("validation history arrays must all have shape (3,)")
    if arrays["role"].tolist() != ["trained", "identity", "selected"]:
        raise ValueError("validation history roles are not canonical")
    if not np.array_equal(arrays["evaluation_index"], [1, 2, 3]):
        raise ValueError("validation history indices are not canonical")
    if not np.all(arrays["stage"] == stage) or not np.all(
        arrays["attempt"] == attempt
    ):
        raise ValueError("validation history stage/attempt binding mismatch")
    if not np.all(arrays["t_start"] == t_start) or not np.all(
        arrays["t_end"] == t_end
    ):
        raise ValueError("validation history temperature binding mismatch")
    if not np.all(arrays["sample_count"] == sample_count):
        raise ValueError("validation history sample count mismatch")
    values = np.asarray(arrays["value"], dtype=float)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("validation history ESS values must lie in [0, 1]")
    trained, identity, selected = map(float, values)
    if not math.isclose(selected, max(trained, identity), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("validation selected ESS is not the better candidate")
    return trained, identity, selected


def _validate_sample_payload(path: Path, signature: dict, label: str) -> None:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid NPY payload") from exc
    expected_shape = signature.get("shape")
    expected_dtype = signature.get("dtype")
    if list(value.shape) != expected_shape or str(value.dtype) != expected_dtype:
        raise ValueError(f"{label} shape/dtype differs from the initial population")
    for start in range(0, value.shape[0], 65_536):
        if not np.all(np.isfinite(value[start:start + 65_536])):
            raise ValueError(f"{label} contains nonfinite coordinates")


def _validate_attempt(
    root: Path,
    stage_root: Path,
    relative,
    *,
    stage: int,
    expected_attempt: int,
    artifacts: dict,
    mode: str,
    tau_ess: float | None,
    sample_count: int,
    expected_train_steps: int,
    session_count: int,
    active_session: int | None,
    validate_session_refs: bool,
) -> dict:
    path = _contained_path(root, relative, "attempt path")
    expected_path = (
        stage_root / "attempts" / f"attempt_{expected_attempt:06d}"
        / "attempt.json"
    ).resolve()
    if path != expected_path:
        raise ValueError(f"attempt path is not canonical: {relative}")
    attempt = _read_record(path, "attempt metadata")
    if attempt.get("stage") != stage or attempt.get("attempt") != expected_attempt:
        raise ValueError(f"attempt index mismatch: {relative}")
    try:
        phase = StagePhase(attempt.get("phase"))
    except ValueError as exc:
        raise ValueError(f"invalid attempt phase: {relative}") from exc
    if phase not in (
        StagePhase.TRAINING, StagePhase.CANDIDATE_SAVED, StagePhase.EVALUATED,
    ):
        raise ValueError(f"unsupported attempt phase {phase.value!r}: {relative}")
    attempt_root = path.parent
    optimizer_path = _require_artifact(
        root, artifacts, attempt.get("optimizer_initial_flow_path"),
        "optimizer-initial flow",
    )
    if optimizer_path != (attempt_root / "optimizer_initial_flow.eqx").resolve():
        raise ValueError(f"attempt has a noncanonical optimizer flow: {relative}")
    executions = attempt.get("executions")
    if not isinstance(executions, list) or not executions:
        raise ValueError(f"attempt has no execution history: {relative}")
    for index, execution in enumerate(executions, start=1):
        if execution.get("index") != index:
            raise ValueError(f"attempt execution history is not contiguous: {relative}")
        outcome = execution.get("outcome")
        if outcome not in (
            "running", "result_pending", "returned", "keyboard_interrupt",
            "exception", "unclean_termination",
        ):
            raise ValueError(f"invalid attempt execution outcome: {relative}")
        if not isinstance(execution.get("started_at"), str):
            raise ValueError(f"attempt execution has no start timestamp: {relative}")
        session = execution.get("session")
        if validate_session_refs and (
            not isinstance(session, int) or not 1 <= session <= session_count
        ):
            raise ValueError(f"attempt execution has an invalid session: {relative}")
        _finite_nonnegative(
            execution.get("elapsed_seconds"), "attempt execution duration"
        )
        if outcome == "running":
            if index != len(executions) or execution.get("ended_at") is not None:
                raise ValueError(
                    f"only the latest attempt execution may be running: {relative}"
                )
            if session != active_session:
                raise ValueError(
                    f"running attempt execution has no active run session: {relative}"
                )
        elif outcome == "unclean_termination":
            if execution.get("ended_at") is not None:
                raise ValueError(
                    f"unclean attempt execution has an end timestamp: {relative}"
                )
        elif not isinstance(execution.get("ended_at"), str):
            raise ValueError(f"closed attempt execution has no end timestamp: {relative}")
        if outcome == "exception":
            error = execution.get("error")
            if (
                not isinstance(error, dict)
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)
            ):
                raise ValueError(f"attempt exception has no diagnostic record: {relative}")
        if index < len(executions) and outcome not in (
            "result_pending", "keyboard_interrupt", "exception",
            "unclean_termination",
        ):
            raise ValueError(f"attempt execution history contains a replay after return: {relative}")
    latest_outcome = executions[-1].get("outcome")
    if phase is StagePhase.TRAINING and latest_outcome == "returned":
        raise ValueError(f"training-phase attempt already returned: {relative}")
    if phase in (StagePhase.CANDIDATE_SAVED, StagePhase.EVALUATED):
        trained_path = _require_artifact(
            root, artifacts, attempt.get("trained_flow_path"), "trained flow"
        )
        history_path = _require_artifact(
            root, artifacts, attempt.get("training_history_path"),
            "training history",
        )
        if trained_path != (attempt_root / "trained_flow.eqx").resolve():
            raise ValueError(f"attempt has a noncanonical trained flow: {relative}")
        if history_path != (attempt_root / "training_history.npz").resolve():
            raise ValueError(f"attempt has a noncanonical training history: {relative}")
        _validate_training_payload(
            history_path, "training history", expected_steps=expected_train_steps
        )
        if latest_outcome != "returned":
            raise ValueError(f"completed attempt execution did not return: {relative}")
        training_elapsed = _finite_nonnegative(
            attempt.get("training_elapsed_seconds"), "training duration"
        )
        if not math.isclose(
            training_elapsed, float(executions[-1]["elapsed_seconds"]),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"attempt training duration disagrees with execution: {relative}")
    if phase is StagePhase.EVALUATED:
        validation_path = _require_artifact(
            root, artifacts, attempt.get("validation_history_path"),
            "validation history",
        )
        selected_path = _require_artifact(
            root, artifacts, attempt.get("selected_flow_path"), "selected flow"
        )
        continuation_path = _require_artifact(
            root, artifacts, attempt.get("continuation_flow_path"),
            "continuation flow",
        )
        if validation_path != (attempt_root / "validation_history.npz").resolve():
            raise ValueError(f"attempt has a noncanonical validation history: {relative}")
        if selected_path != (attempt_root / "selected_flow.eqx").resolve():
            raise ValueError(f"attempt has a noncanonical selected flow: {relative}")
        if continuation_path != (attempt_root / "continuation_flow.eqx").resolve():
            raise ValueError(f"attempt has a noncanonical continuation flow: {relative}")
        if attempt.get("status") not in ("accepted", "rejected"):
            raise ValueError(f"evaluated attempt has invalid status: {relative}")
        if attempt.get("selected") not in ("trained", "identity"):
            raise ValueError(f"evaluated attempt has invalid selection: {relative}")
        trained, identity, selected = _validate_validation_payload(
            validation_path, stage=stage, attempt=expected_attempt,
            t_start=float(attempt["t_start"]), t_end=float(attempt["t_end"]),
            sample_count=sample_count,
        )
        expected_selected = "trained" if trained > identity else "identity"
        if attempt.get("selected") != expected_selected:
            raise ValueError(f"evaluated attempt selected the worse flow: {relative}")
        selected_relative = attempt["selected_flow_path"]
        trained_relative = attempt["trained_flow_path"]
        identity_relative = (
            stage_root / "identity_flow.eqx"
        ).relative_to(root).as_posix()
        expected_relative = (
            trained_relative if expected_selected == "trained"
            else identity_relative
        )
        if _artifact_payload_receipt(
            artifacts, selected_relative
        ) != _artifact_payload_receipt(artifacts, expected_relative):
            raise ValueError(
                f"evaluated attempt selected-flow payload disagrees with "
                f"its {expected_selected} role: {relative}"
            )
        training_identity_relative = (
            stage_root / "training_identity_flow.eqx"
        ).relative_to(root).as_posix()
        expected_continuation_relative = (
            trained_relative if expected_selected == "trained"
            else training_identity_relative
        )
        if _artifact_payload_receipt(
            artifacts, attempt["continuation_flow_path"]
        ) != _artifact_payload_receipt(
            artifacts, expected_continuation_relative
        ):
            raise ValueError(
                f"evaluated attempt continuation-flow payload disagrees with "
                f"its {expected_selected} role: {relative}"
            )
        for field, expected in (
            ("valid_trained_ess", trained),
            ("valid_identity_ess", identity),
            ("valid_selected_ess", selected),
        ):
            actual = attempt.get(field)
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"evaluated attempt {field} disagrees with history")
        expected_status = (
            "accepted" if mode == "fixed"
            or selected >= float(tau_ess) else "rejected"
        )
        if attempt.get("status") != expected_status:
            raise ValueError(f"evaluated attempt decision disagrees with ESS policy")
        _finite_nonnegative(
            attempt.get("validation_elapsed_seconds"), "validation duration"
        )
    return attempt


def _validate_stage_record(
    root: Path,
    record: dict,
    *,
    stage: int,
    artifacts: dict,
    mode: str,
    config: dict,
    sample_count: int,
    sample_signature: dict,
    expected_train_steps: int,
    session_count: int,
    active_session: int | None,
    allow_historical_policy: bool = False,
) -> tuple[StagePhase, list[dict]]:
    if record.get("stage") != stage:
        raise ValueError(f"stage record index mismatch at {stage}")
    expected_parent = stage - 1 if stage > 1 else None
    if record.get("parent_stage") != expected_parent:
        raise ValueError(f"stage {stage} parent binding mismatch")
    stage_config_digest = record.get("config_digest")
    if not isinstance(stage_config_digest, str):
        raise ValueError(f"stage {stage} has no configuration receipt")
    if not allow_historical_policy and stage_config_digest != _config_digest(config):
        raise ValueError(f"stage {stage} configuration differs from the run")
    try:
        phase = StagePhase(record.get("phase"))
    except ValueError as exc:
        raise ValueError(f"stage {stage} has an invalid phase") from exc
    try:
        t_start = float(record["t_start"])
        t_end = float(record["t_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"stage {stage} has invalid temperatures") from exc
    if not (0.0 <= t_start < t_end <= 1.0):
        raise ValueError(f"stage {stage} has a non-increasing transition")
    exhaustion_reason = record.get("exhaustion_reason")
    if exhaustion_reason not in (
        None, "temperature_resolution", "selection_limit",
    ):
        raise ValueError(f"stage {stage} has an invalid exhaustion reason")
    if exhaustion_reason is not None and phase not in (
        StagePhase.SELECTION_READY, StagePhase.REJECTED,
    ):
        raise ValueError(f"stage {stage} has a misplaced exhaustion reason")
    policy = record.get("acceptance_policy")
    if (
        not isinstance(policy, dict)
        or set(policy) != {
            "mode", "adaptive_controls", "ladder",
            "selection_proposal_limit",
        }
        or policy.get("mode") != mode
    ):
        raise ValueError(f"stage {stage} has no valid acceptance-policy receipt")
    ladder = policy.get("ladder")
    if isinstance(ladder, bool) or not isinstance(ladder, int) or ladder < 1:
        raise ValueError(f"stage {stage} has an invalid saved SMC ladder")
    selection_limit = policy.get("selection_proposal_limit")
    if isinstance(selection_limit, bool) or not isinstance(selection_limit, int):
        raise ValueError(f"stage {stage} has an invalid selection limit")
    if mode == "fixed":
        if policy.get("adaptive_controls") is not None or selection_limit != 0:
            raise ValueError(f"fixed stage {stage} has adaptive policy controls")
        controls = None
        tau_smc = tau_ess = shrink_factor = None
        max_retry = 1
    else:
        if selection_limit != SELECTION_PROPOSAL_LIMIT:
            raise ValueError(f"adaptive stage {stage} has an invalid selection limit")
        try:
            controls = normalize_adaptive_policy(
                "saved stage policy", policy.get("adaptive_controls")
            ).as_dict()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"adaptive stage {stage} has invalid saved controls"
            ) from exc
        if controls != policy.get("adaptive_controls"):
            raise ValueError(f"adaptive stage {stage} controls are not canonical")
        tau_smc = float(controls["tau_smc"])
        tau_ess = float(controls["tau_ess"])
        shrink_factor = float(controls["shrink_factor"])
        max_retry = int(controls["max_retry"])
    expected_policy = {
        "mode": mode,
        "adaptive_controls": (
            None if mode == "fixed" else config.get("bg_param")
        ),
        "ladder": int(config["ladder"]),
        "selection_proposal_limit": (
            0 if mode == "fixed" else SELECTION_PROPOSAL_LIMIT
        ),
    }
    if not allow_historical_policy and policy != expected_policy:
        raise ValueError(f"stage {stage} acceptance policy differs from the run")
    stage_root = root / "stages" / f"stage_{stage:06d}"
    entry_path = _require_artifact(
        root, artifacts, record.get("entry_flow_path"), "entry flow"
    )
    identity_path = _require_artifact(
        root, artifacts, record.get("identity_flow_path"), "identity flow"
    )
    training_identity_path = _require_artifact(
        root, artifacts, record.get("training_identity_flow_path"),
        "training-identity flow",
    )
    if entry_path != (stage_root / "entry_flow.eqx").resolve():
        raise ValueError(f"stage {stage} has a noncanonical entry flow")
    if identity_path != (stage_root / "identity_flow.eqx").resolve():
        raise ValueError(f"stage {stage} has a noncanonical identity flow")
    if training_identity_path != (
        stage_root / "training_identity_flow.eqx"
    ).resolve():
        raise ValueError(f"stage {stage} has a noncanonical training-identity flow")
    selection = record.get("selection_history")
    if not isinstance(selection, list):
        raise ValueError(f"stage {stage} selection history is invalid")
    if mode == "adaptive" and len(selection) > selection_limit:
        raise ValueError(f"stage {stage} exceeds its selection proposal limit")
    previous_selection_t = None
    previous_selection_decision = None
    for index, item in enumerate(selection):
        if not isinstance(item, dict):
            raise ValueError(f"stage {stage} selection event is invalid")
        try:
            item_start = float(item["t_start"])
            item_end = float(item["t_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"stage {stage} selection temperatures are invalid") from exc
        if item_start != t_start or not t_start < item_end <= 1.0:
            raise ValueError(f"stage {stage} selection temperatures are invalid")
        if previous_selection_t is not None and not item_end < previous_selection_t:
            raise ValueError(f"stage {stage} selection proposals do not shrink")
        if item.get("validation_sample_count") != sample_count:
            raise ValueError(f"stage {stage} selection sample count is invalid")
        if item.get("status") == "skipped":
            if index != 0 or item.get("index") != 0 or len(selection) != 1:
                raise ValueError(f"stage {stage} skipped selection is not canonical")
            expected_reason = (
                "fixed_schedule" if mode == "fixed" else "tau_smc_zero"
            )
            if (
                item.get("reason") != expected_reason
                or mode == "adaptive" and tau_smc != 0.0
            ):
                raise ValueError(f"stage {stage} skipped selection violates policy")
            previous_selection_decision = "accepted"
        else:
            if mode != "adaptive" or tau_smc == 0.0:
                raise ValueError(f"stage {stage} unexpectedly ran selection SMC")
            if item.get("index") != index + 1:
                raise ValueError(f"stage {stage} selection decision is invalid")
            values = np.asarray(item.get("smc_ess"), dtype=float)
            minimum = item.get("minimum_smc_ess")
            if (
                values.shape != (ladder,)
                or not np.all(np.isfinite(values))
                or np.any((values < 0.0) | (values > 1.0))
                or not isinstance(minimum, (int, float))
                or not math.isclose(
                    float(minimum), float(values.min()),
                    rel_tol=0.0, abs_tol=1e-12,
                )
            ):
                raise ValueError(f"stage {stage} selection ESS history is invalid")
            _finite_nonnegative(
                item.get("elapsed_seconds"), "selection duration"
            )
            expected_decision = (
                "accepted" if float(minimum) >= tau_smc else "shrink"
            )
            if item.get("decision") != expected_decision:
                raise ValueError(f"stage {stage} selection decision violates policy")
            if index > 0:
                if previous_selection_decision != "shrink":
                    raise ValueError(f"stage {stage} continued completed selection")
                expected_t = t_start + shrink_factor * (
                    previous_selection_t - t_start
                )
                if item_end != expected_t:
                    raise ValueError(
                        f"stage {stage} selection shrink violates policy"
                    )
            previous_selection_decision = expected_decision
        previous_selection_t = item_end
    relatives = record.get("attempts")
    if not isinstance(relatives, list):
        raise ValueError(f"stage {stage} attempts must be a list")
    attempts = [
        _validate_attempt(
            root, stage_root, relative, stage=stage,
            expected_attempt=index, artifacts=artifacts, mode=mode,
            tau_ess=tau_ess, sample_count=sample_count,
            expected_train_steps=expected_train_steps,
            session_count=session_count,
            active_session=active_session,
            validate_session_refs=not allow_historical_policy,
        )
        for index, relative in enumerate(relatives, start=1)
    ]
    previous_t_end = None
    for attempt in attempts:
        attempt_start = float(attempt.get("t_start"))
        attempt_end = float(attempt.get("t_end"))
        if attempt_start != t_start or not t_start < attempt_end <= 1.0:
            raise ValueError(f"stage {stage} attempt temperatures are invalid")
        if previous_t_end is not None:
            expected_t = t_start + shrink_factor * (previous_t_end - t_start)
            if attempt_end != expected_t:
                raise ValueError(f"stage {stage} retry endpoint violates policy")
        previous_t_end = attempt_end
    if len(attempts) > max_retry:
        raise ValueError(f"stage {stage} exceeds its retry policy")
    if attempts and float(attempts[-1]["t_end"]) != t_end:
        raise ValueError(f"stage {stage} endpoint disagrees with its current attempt")
    if phase is StagePhase.PREPARED and attempts:
        raise ValueError(f"prepared stage {stage} already contains attempts")
    if phase is StagePhase.SELECTION_READY and attempts:
        raise ValueError(f"selection-ready stage {stage} already contains attempts")
    if phase in (StagePhase.PREPARED, StagePhase.SELECTION_READY) and record.get(
        "current_attempt"
    ) is not None:
        raise ValueError(f"stage {stage} has a premature current attempt")
    if not selection and phase is not StagePhase.PREPARED:
        raise ValueError(f"stage {stage} has no durable selection event")
    if selection:
        last_selection = selection[-1]
        selected_endpoint = float(last_selection["t_end"])
        selection_done = (
            last_selection.get("status") == "skipped"
            or last_selection.get("decision") == "accepted"
        )
        if selection_done:
            expected_selected_endpoint = (
                float(attempts[0]["t_end"]) if attempts else t_end
            )
            if expected_selected_endpoint != selected_endpoint:
                raise ValueError(f"stage {stage} endpoint disagrees with selection")
        exhaustion_reason = record.get("exhaustion_reason")
        if not selection_done:
            resolution_stop = (
                exhaustion_reason == "temperature_resolution"
                and phase is StagePhase.SELECTION_READY
                and t_end == selected_endpoint
            )
            expected_next = t_start + shrink_factor * (
                selected_endpoint - t_start
            )
            if not resolution_stop and t_end != expected_next:
                raise ValueError(
                    f"stage {stage} next selection endpoint violates policy"
                )
        if phase not in (StagePhase.PREPARED, StagePhase.SELECTION_READY) and not selection_done:
            raise ValueError(f"stage {stage} trained before selection completed")
    if phase in (
        StagePhase.TRAINING, StagePhase.CANDIDATE_SAVED, StagePhase.EVALUATED,
        StagePhase.REJECTED, StagePhase.ADVANCE_SAVED, StagePhase.COMMITTED,
    ):
        if not attempts or record.get("current_attempt") != len(attempts):
            raise ValueError(f"stage {stage} has invalid current-attempt metadata")
    for earlier in attempts[:-1]:
        if (
            earlier.get("phase") != StagePhase.EVALUATED
            or earlier.get("status") != "rejected"
        ):
            raise ValueError(f"stage {stage} has a non-rejected earlier attempt")
    if phase is StagePhase.TRAINING and attempts[-1]["phase"] != phase:
        raise ValueError(f"stage {stage} training phase disagrees with its attempt")
    if phase is StagePhase.CANDIDATE_SAVED and attempts[-1]["phase"] != phase:
        raise ValueError(f"stage {stage} candidate phase disagrees with its attempt")
    if phase in (
        StagePhase.EVALUATED, StagePhase.REJECTED,
        StagePhase.ADVANCE_SAVED, StagePhase.COMMITTED,
    ):
        attempt = attempts[-1]
        if attempt["phase"] != StagePhase.EVALUATED:
            raise ValueError(f"stage {stage} has no evaluated current attempt")
        for field, label in (
            ("selected_flow_path", "selected flow"),
            ("continuation_flow_path", "continuation flow"),
        ):
            _require_artifact(root, artifacts, record.get(field), label)
            if record.get(field) != attempt.get(field):
                raise ValueError(f"stage {stage} {field} disagrees with its attempt")
        if record.get("decision") != attempt.get("status"):
            raise ValueError(f"stage {stage} decision disagrees with its attempt")
        if record.get("selected") != attempt.get("selected"):
            raise ValueError(f"stage {stage} selection disagrees with its attempt")
        for field in (
            "valid_trained_ess", "valid_identity_ess", "valid_selected_ess",
        ):
            if record.get(field) != attempt.get(field):
                raise ValueError(f"stage {stage} {field} disagrees with its attempt")
    if phase is StagePhase.REJECTED:
        if record.get("decision") != "rejected" or "next_t" not in record:
            raise ValueError(f"rejected stage {stage} has no retry decision")
        if record.get("exhaustion_reason") == "temperature_resolution":
            if record["next_t"] is not None:
                raise ValueError(f"rejected stage {stage} has a spurious retry endpoint")
        else:
            next_t = float(record["next_t"])
            expected_next = t_start + shrink_factor * (t_end - t_start)
            if next_t != expected_next:
                raise ValueError(
                    f"rejected stage {stage} retry endpoint violates policy"
                )
    if phase in (StagePhase.ADVANCE_SAVED, StagePhase.COMMITTED):
        if record.get("decision") != "accepted":
            raise ValueError(f"advanced stage {stage} was not accepted")
        sample_path = _require_artifact(
            root, artifacts, record.get("validation_samples_path"),
            "validation samples",
        )
        if sample_path != (stage_root / "validation_samples.npy").resolve():
            raise ValueError(
                f"stage {stage} has a noncanonical validation checkpoint"
            )
        _validate_sample_payload(
            sample_path, sample_signature, "validation samples"
        )
    if phase is StagePhase.COMMITTED:
        if record.get("status") != "accepted" or not isinstance(
            record.get("transition_elapsed_seconds"), (int, float)
        ):
            raise ValueError(f"committed stage {stage} has invalid status or timing")
        _finite_nonnegative(
            record.get("transition_elapsed_seconds"), "stage transition duration"
        )
        if attempts[-1].get("status") != "accepted":
            raise ValueError(f"committed stage {stage} has no accepted final attempt")
        expected_accepted_time = sum(
            float(item["elapsed_seconds"])
            for item in attempts[-1].get("executions", [])
        ) + float(attempts[-1].get("validation_elapsed_seconds") or 0.0)
        accepted_time = _finite_nonnegative(
            record.get("accepted_attempt_elapsed_seconds"),
            "accepted attempt duration",
        )
        if not math.isclose(
            accepted_time, expected_accepted_time, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"committed stage {stage} accepted-attempt timing mismatch")
    for item in selection:
        if "elapsed_seconds" in item:
            _finite_nonnegative(item["elapsed_seconds"], "selection duration")
    timing = record.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"stage {stage} timing metadata is invalid")
    accumulated = _finite_nonnegative(
        timing.get("accumulated_seconds"), "stage accumulated duration"
    )
    expected_accumulated = _stage_elapsed_seconds(root, record)
    if not math.isclose(
        accumulated, expected_accumulated, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"stage {stage} accumulated timing mismatch")
    segments = timing.get("session_segments")
    if not isinstance(segments, list):
        raise ValueError(f"stage {stage} session timing is invalid")
    previous_session = 0
    for segment in segments:
        session = segment.get("session") if isinstance(segment, dict) else None
        if not isinstance(session, int) or session <= previous_session:
            raise ValueError(f"stage {stage} session timing is not ordered")
        if not allow_historical_policy and not 1 <= session <= session_count:
            raise ValueError(f"stage {stage} timing has an invalid session")
        if segment.get("outcome") not in (
            "committed", "exhausted", "keyboard_interrupt", "exception",
        ):
            raise ValueError(f"stage {stage} session timing has an invalid outcome")
        _finite_nonnegative(
            segment.get("elapsed_seconds"), "stage session duration"
        )
        previous_session = session
    if phase is StagePhase.COMMITTED:
        segment_total = sum(float(item["elapsed_seconds"]) for item in segments)
        expected_transition = max(accumulated, segment_total)
        if not math.isclose(
            float(record["transition_elapsed_seconds"]), expected_transition,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"committed stage {stage} transition timing mismatch")
    return phase, attempts


def validate_manifest(root, manifest: dict | None = None) -> dict:
    """Validate schema, indexed payloads, and the committed stage prefix."""
    root = Path(root).expanduser().resolve()
    value = read_manifest(root) if manifest is None else manifest
    if value.get("config_digest") != _config_digest(value.get("config", {})):
        raise ValueError("run config digest mismatch")
    mode = value.get("mode")
    if mode not in ("adaptive", "fixed"):
        raise ValueError(f"invalid run mode {mode!r}")
    config = value.get("config")
    if not isinstance(config, dict):
        raise ValueError("run config is missing or invalid")
    sessions = value.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("run session history must be a list")
    session_count = len(sessions)
    active_session = (
        session_count
        if sessions and isinstance(sessions[-1], dict)
        and sessions[-1].get("outcome") == "running"
        else None
    )
    try:
        expected_train_steps = int(config["train_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("run train_steps configuration is missing or invalid") from exc
    if expected_train_steps < 1:
        raise ValueError("run train_steps configuration must be positive")
    try:
        sample_signature = value["sample_signature"]
        sample_count = int(sample_signature["shape"][0])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("run sample signature is missing or invalid") from exc
    if sample_count < 1:
        raise ValueError("run sample count must be positive")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("run artifact registry is missing or invalid")
    for relative, metadata in artifacts.items():
        path = _contained_path(root, relative, "artifact path")
        if not path.is_file():
            raise FileNotFoundError(f"missing run artifact: {relative}")
        if path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"artifact size mismatch: {relative}")
        if _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"artifact checksum mismatch: {relative}")
        if "shape" in metadata or "dtype" in metadata:
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"invalid array artifact: {relative}") from exc
            if list(array.shape) != metadata.get("shape"):
                raise ValueError(f"artifact shape mismatch: {relative}")
            if str(array.dtype) != metadata.get("dtype"):
                raise ValueError(f"artifact dtype mismatch: {relative}")
    _require_artifact(
        root, artifacts, value.get("initial_flow_path"), "initial flow"
    )
    initial_samples = _require_artifact(
        root, artifacts, value.get("initial_samples_path"), "initial samples"
    )
    _validate_sample_payload(initial_samples, sample_signature, "initial samples")
    if _sample_signature(
        np.load(initial_samples, mmap_mode="r", allow_pickle=False)
    ) != sample_signature:
        raise ValueError("initial sample payload does not match its run signature")

    completed = value.get("completed_stages")
    if not isinstance(completed, list):
        raise ValueError("completed_stages must be a list")
    accepted = [0.0]
    saw_unclean_attempt = False
    inherited_stage = int(
        (value.get("fork_parent") or {}).get("accepted_stage", 0)
    )
    for expected_stage, summary in enumerate(completed, start=1):
        if summary.get("stage") != expected_stage:
            raise ValueError("completed stages are not a contiguous prefix")
        stage_relative = summary.get("stage_path")
        if not isinstance(stage_relative, str):
            raise ValueError(f"completed stage {expected_stage} has no stage path")
        stage_path = (root / stage_relative).resolve()
        if root not in stage_path.parents:
            raise ValueError(f"completed stage path escapes run directory: {stage_relative}")
        expected_stage_path = (
            root / "stages" / f"stage_{expected_stage:06d}" / "stage.json"
        ).resolve()
        if stage_path != expected_stage_path:
            raise ValueError(
                f"completed stage {expected_stage} has a noncanonical stage path"
            )
        try:
            record = _read_record(stage_path, "stage metadata")
            commit = _read_record(
                stage_path.parent / "commit.json", "stage commit"
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"completed stage {expected_stage} is missing metadata"
            ) from exc
        phase, attempts = _validate_stage_record(
            root, record, stage=expected_stage, artifacts=artifacts,
            mode=mode, config=config, sample_count=sample_count,
            sample_signature=sample_signature,
            expected_train_steps=expected_train_steps,
            session_count=session_count,
            active_session=active_session,
            allow_historical_policy=expected_stage <= inherited_stage,
        )
        saw_unclean_attempt |= any(
            execution.get("outcome") == "unclean_termination"
            for attempt in attempts
            for execution in attempt.get("executions", [])
        )
        if phase is not StagePhase.COMMITTED:
            raise ValueError(f"completed stage {expected_stage} is not committed")
        expected_parent = expected_stage - 1 if expected_stage > 1 else None
        bindings = {
            "run_id": value.get("run_id"),
            "stage": expected_stage,
            "parent_stage": expected_parent,
            "expected_completed_prefix": expected_stage - 1,
            "config_digest": record.get("config_digest"),
            "t_start": record.get("t_start"),
            "t_end": record.get("t_end"),
        }
        for key, expected in bindings.items():
            if commit.get(key) != expected:
                raise ValueError(
                    f"completed stage {expected_stage} commit binding mismatch: {key}"
                )
        if commit.get("stage_record_digest") != record.get("record_digest"):
            raise ValueError(
                f"completed stage {expected_stage} metadata receipt mismatch"
            )
        attempt_receipts = {
            relative: attempt["record_digest"]
            for relative, attempt in zip(
                record.get("attempts", []), attempts, strict=True
            )
        }
        if commit.get("attempt_record_digests") != attempt_receipts:
            raise ValueError(
                f"completed stage {expected_stage} attempt receipt mismatch"
            )
        stage_prefix = f"stages/stage_{expected_stage:06d}/"
        expected_artifacts = {
            relative: metadata for relative, metadata in artifacts.items()
            if relative.startswith(stage_prefix)
        }
        if commit.get("stage_artifacts") != expected_artifacts:
            raise ValueError(
                f"completed stage {expected_stage} artifact receipt mismatch"
            )
        for field in (
            "selected_flow_path", "continuation_flow_path",
            "validation_samples_path",
        ):
            relative = record.get(field)
            if not relative or summary.get(field) != relative:
                raise ValueError(
                    f"completed stage {expected_stage} summary mismatch: {field}"
                )
            if commit.get("artifacts", {}).get(field) != artifacts.get(relative):
                raise ValueError(
                    f"completed stage {expected_stage} artifact binding mismatch: {field}"
                )
        for field in (
            "t_start", "t_end", "transition_elapsed_seconds",
            "accepted_attempt_elapsed_seconds",
        ):
            if summary.get(field) != record.get(field):
                raise ValueError(
                    f"completed stage {expected_stage} summary mismatch: {field}"
                )
        accepted.append(float(record["t_end"]))
        if record["t_start"] != accepted[-2]:
            raise ValueError(f"completed stage {expected_stage} has the wrong start")
    if value.get("accepted_t_list") != accepted:
        raise ValueError("accepted temperature history does not match completed stages")
    if value.get("mode") == "adaptive" and value.get("t_list") != accepted:
        raise ValueError("adaptive temperature history does not match completed stages")

    current = value.get("current_stage")
    current_phase = None
    current_record = None
    current_attempts = []
    if current is not None:
        stage = current.get("stage")
        phase = current.get("phase")
        if not isinstance(stage, int) or not isinstance(phase, str):
            raise ValueError("current_stage is invalid")
        try:
            phase = StagePhase(phase)
        except ValueError as exc:
            raise ValueError("current_stage phase is invalid") from exc
        if phase is StagePhase.COMMITTED:
            if not completed or stage != completed[-1]["stage"]:
                raise ValueError("committed current_stage is outside completed prefix")
        elif stage != len(completed) + 1:
            raise ValueError("incomplete current_stage does not follow completed prefix")
        stage_record = _read_record(
            root / "stages" / f"stage_{stage:06d}" / "stage.json",
            "stage metadata",
        )
        record_phase, current_attempts = _validate_stage_record(
            root, stage_record, stage=stage, artifacts=artifacts,
            mode=mode, config=config, sample_count=sample_count,
            sample_signature=sample_signature,
            expected_train_steps=expected_train_steps,
            session_count=session_count,
            active_session=active_session,
            allow_historical_policy=stage <= inherited_stage,
        )
        saw_unclean_attempt |= any(
            execution.get("outcome") == "unclean_termination"
            for attempt in current_attempts
            for execution in attempt.get("executions", [])
        )
        current_phase = record_phase
        current_record = stage_record
        if record_phase is not phase:
            raise ValueError("current_stage phase does not match its stage record")
        if phase is not StagePhase.COMMITTED and stage_record["t_start"] != accepted[-1]:
            raise ValueError("current stage does not start at the accepted endpoint")

    requested = value.get("requested_t_list")
    if mode == "fixed":
        if not isinstance(requested, list) or not requested:
            raise ValueError("fixed run has no requested schedule")
        expected_t_list = [0.0, *[float(item) for item in requested]]
        if value.get("t_list") != expected_t_list:
            raise ValueError("fixed temperature history does not match its schedule")
        if accepted != expected_t_list[:len(accepted)]:
            raise ValueError("fixed accepted stages do not follow the schedule")
    elif requested is not None:
        raise ValueError("adaptive run unexpectedly stores a fixed schedule")
    lifecycle = RunLifecycle(value.get("lifecycle"))
    run_exhaustion_reason = value.get("exhaustion_reason")
    if run_exhaustion_reason not in (None, "temperature_resolution"):
        raise ValueError("run has an invalid exhaustion reason")
    if run_exhaustion_reason is not None and lifecycle is not RunLifecycle.EXHAUSTED:
        raise ValueError("a non-exhausted run retains an exhaustion reason")
    reached_target = bool(accepted and accepted[-1] == 1.0)
    has_incomplete = bool(
        current is not None and current.get("phase") != StagePhase.COMMITTED
    )
    if lifecycle is RunLifecycle.COMPLETE:
        fixed_done = mode == "fixed" and len(completed) == len(requested)
        adaptive_done = mode == "adaptive" and reached_target
        if not (fixed_done or adaptive_done) or has_incomplete:
            raise ValueError("complete lifecycle disagrees with the saved stage state")
    if lifecycle is RunLifecycle.EXHAUSTED:
        if mode != "adaptive" or reached_target:
            raise ValueError("exhausted lifecycle is invalid for this run state")
        policy = config.get("bg_param") or {}
        max_stages = int(policy.get("max_stages", 0))
        max_retry = int(policy.get("max_retry", 0))
        stage_limit = max_stages > 0 and len(completed) >= max_stages
        retry_limit = (
            current_phase is StagePhase.REJECTED
            and max_retry > 0 and len(current_attempts) >= max_retry
        )
        selection_limit = (
            current_phase is StagePhase.SELECTION_READY
            and current_record is not None
            and bool(current_record.get("selection_history"))
            and current_record["selection_history"][-1].get("decision") == "shrink"
        )
        resolution_limit = (
            value.get("exhaustion_reason") == "temperature_resolution"
            or current_record is not None
            and current_record.get("exhaustion_reason") == "temperature_resolution"
        )
        if not (stage_limit or retry_limit or selection_limit or resolution_limit):
            raise ValueError("exhausted lifecycle has no exhausted policy limit")
    if lifecycle is RunLifecycle.FORK_PENDING:
        if (
            mode != "adaptive" or reached_target
            or not isinstance(value.get("fork_parent"), dict)
        ):
            raise ValueError("fork-pending lifecycle has no valid adaptive parent")

    active_elapsed = 0.0
    saw_unclean = False
    for index, session in enumerate(sessions, start=1):
        if not isinstance(session, dict) or session.get("index") != index:
            raise ValueError("run session history is not contiguous")
        if not isinstance(session.get("started_at"), str):
            raise ValueError("run session has no start timestamp")
        outcome = session.get("outcome")
        if outcome not in (
            "running", "clean_exit", "keyboard_interrupt", "exception",
            "unclean_termination",
        ):
            raise ValueError("run session has an invalid outcome")
        duration = _finite_nonnegative(
            session.get("active_seconds"), "session active duration"
        )
        active_elapsed += duration
        if outcome == "running":
            if index != len(sessions) or session.get("ended_at") is not None:
                raise ValueError("only the latest run session may be active")
            if lifecycle is not RunLifecycle.IN_PROGRESS:
                raise ValueError("a terminal run cannot have an active session")
        elif outcome == "unclean_termination":
            saw_unclean = True
        elif not isinstance(session.get("ended_at"), str):
            raise ValueError("closed run session has no end timestamp")
    saved_active = _finite_nonnegative(
        value.get("active_elapsed_seconds"), "total active duration"
    )
    if not math.isclose(
        saved_active, active_elapsed, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("total active duration disagrees with session history")
    if not isinstance(value.get("timing_lower_bound"), bool):
        raise ValueError("timing_lower_bound must be Boolean")
    if (saw_unclean or saw_unclean_attempt) and not value["timing_lower_bound"]:
        raise ValueError("unclean termination requires a timing lower bound")
    fork_parent = value.get("fork_parent")
    inherited_active = value.get("inherited_active_elapsed_seconds", 0.0)
    inherited_training = value.get(
        "inherited_training_elapsed_seconds_total", 0.0
    )
    _finite_nonnegative(inherited_active, "inherited active duration")
    _finite_nonnegative(inherited_training, "inherited training duration")
    if fork_parent is None:
        if float(inherited_active) != 0.0 or float(inherited_training) != 0.0:
            raise ValueError("non-forked run has inherited timing")
    else:
        if not isinstance(fork_parent, dict) or not 0 <= inherited_stage <= len(completed):
            raise ValueError("fork provenance has an invalid inherited prefix")

    selection_proposals = []
    attempted = []
    training_elapsed = 0.0
    stage_count = len(completed) + int(
        current_phase is not None and current_phase is not StagePhase.COMMITTED
    )
    for stage_index in range(1, stage_count + 1):
        stage_record = _read_record(
            root / "stages" / f"stage_{stage_index:06d}" / "stage.json",
            "stage metadata",
        )
        attempt_paths = stage_record.get("attempts", [])
        selection_proposals.extend(
            float(item["t_end"])
            for item in stage_record.get("selection_history", [])
        )
        if attempt_paths:
            for relative in attempt_paths:
                attempt = _read_record(root / relative, "attempt metadata")
                attempted.append(float(attempt["t_end"]))
                if stage_index > inherited_stage:
                    training_elapsed += sum(
                        float(item["elapsed_seconds"])
                        for item in attempt.get("executions", [])
                    )
    if value.get("selection_t_list") != selection_proposals:
        raise ValueError("selection temperature history disagrees with stage records")
    if value.get("attempted_t_list") != attempted:
        raise ValueError("attempted temperature history disagrees with stage records")
    saved_training_elapsed = _finite_nonnegative(
        value.get("training_elapsed_seconds_total"), "total training duration"
    )
    if not math.isclose(
        saved_training_elapsed, training_elapsed, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("total training duration disagrees with attempt records")
    return value


def _artifact_metadata(path: Path, *, array: bool = False) -> dict:
    metadata = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    if array:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        metadata.update({"shape": list(value.shape), "dtype": str(value.dtype)})
    return metadata


def _register_recovered(
    root: Path, manifest: dict, relative, *, array: bool = False,
) -> None:
    if not isinstance(relative, str) or relative in manifest["artifacts"]:
        return
    path = _contained_path(root, relative, "recovered artifact")
    if path.is_file():
        manifest["artifacts"][relative] = _artifact_metadata(path, array=array)


def _promote_commit(root: Path, manifest: dict, stage: int, record: dict) -> bool:
    stage_path = root / "stages" / f"stage_{stage:06d}" / "stage.json"
    commit_path = stage_path.parent / "commit.json"
    if not commit_path.is_file():
        record["phase"] = StagePhase.ADVANCE_SAVED.value
        record["status"] = "in_progress"
        _atomic_record(stage_path, record)
        manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.ADVANCE_SAVED.value,
        }
        return False
    commit = _read_record(commit_path, "stage commit")
    expected = {
        "run_id": manifest.get("run_id"),
        "stage": stage,
        "parent_stage": stage - 1 if stage > 1 else None,
        "expected_completed_prefix": stage - 1,
        "config_digest": record.get("config_digest"),
        "t_start": record.get("t_start"),
        "t_end": record.get("t_end"),
    }
    for key, expected_value in expected.items():
        if commit.get(key) != expected_value:
            raise ValueError(f"stage {stage} commit binding mismatch: {key}")
    if commit.get("stage_record_digest") != record.get("record_digest"):
        raise ValueError(f"stage {stage} metadata receipt mismatch")
    attempt_receipts = {
        relative: _read_record(root / relative, "attempt metadata")["record_digest"]
        for relative in record.get("attempts", [])
    }
    if commit.get("attempt_record_digests") != attempt_receipts:
        raise ValueError(f"stage {stage} attempt receipt mismatch")
    for relative, metadata in commit.get("stage_artifacts", {}).items():
        _contained_path(root, relative, "committed artifact")
        manifest["artifacts"].setdefault(relative, metadata)
    for field, relative in commit.get("artifacts", {}).items():
        path = record.get(field)
        if path:
            manifest["artifacts"].setdefault(path, relative)
    summary = {
        "stage": stage,
        "t_start": record["t_start"],
        "t_end": record["t_end"],
        "stage_path": stage_path.relative_to(root).as_posix(),
        "selected_flow_path": record["selected_flow_path"],
        "validation_samples_path": record["validation_samples_path"],
        "continuation_flow_path": record["continuation_flow_path"],
        "transition_elapsed_seconds": float(record["transition_elapsed_seconds"]),
        "accepted_attempt_elapsed_seconds": float(
            record["accepted_attempt_elapsed_seconds"]
        ),
    }
    manifest["completed_stages"].append(summary)
    manifest["current_stage"] = {
        "stage": stage,
        "phase": StagePhase.COMMITTED.value,
    }
    return True


def _reconcile_stage_payloads(
    root: Path, manifest: dict, stage: int, record: dict,
) -> tuple[dict, bool]:
    """Promote complete payload transactions whose manifest write was lost."""
    changed = False
    stage_root = root / "stages" / f"stage_{stage:06d}"
    disk_attempts = sorted(stage_root.glob("attempts/attempt_*/attempt.json"))
    recorded = list(record.get("attempts", []))
    if len(disk_attempts) < len(recorded):
        raise FileNotFoundError(f"stage {stage} is missing attempt metadata")
    for index, path in enumerate(disk_attempts, start=1):
        relative = path.relative_to(root).as_posix()
        expected_relative = (
            Path("stages") / f"stage_{stage:06d}" / "attempts"
            / f"attempt_{index:06d}" / "attempt.json"
        ).as_posix()
        if relative != expected_relative:
            raise ValueError(f"stage {stage} has a noncanonical attempt path")
        if index <= len(recorded):
            if recorded[index - 1] != relative:
                raise ValueError(f"stage {stage} attempt paths are not canonical")
            continue
        if index != len(recorded) + 1:
            raise ValueError(f"stage {stage} attempt history is not contiguous")
        attempt = _read_record(path, "attempt metadata")
        if attempt.get("stage") != stage or attempt.get("attempt") != index:
            raise ValueError(f"stage {stage} has invalid recovered attempt metadata")
        recorded.append(relative)
        record["attempts"] = recorded
        record["current_attempt"] = index
        record["phase"] = StagePhase.TRAINING.value
        changed = True

    if recorded:
        attempt_path = root / recorded[-1]
        attempt = _read_record(attempt_path, "attempt metadata")
        attempt_root = attempt_path.parent
        candidate_rel = (attempt_root / "trained_flow.eqx").relative_to(root).as_posix()
        training_rel = (attempt_root / "training_history.npz").relative_to(root).as_posix()
        if (
            attempt.get("phase") == StagePhase.TRAINING
            and (root / candidate_rel).is_file()
            and (root / training_rel).is_file()
        ):
            _validate_training_payload(
                root / training_rel, "recovered training history",
                expected_steps=int(manifest["config"]["train_steps"]),
            )
            execution = attempt["executions"][-1]
            known_elapsed = execution.get("outcome") == "result_pending"
            elapsed = (
                float(execution.get("elapsed_seconds") or 0.0)
                if known_elapsed else 0.0
            )
            attempt.update({
                "phase": StagePhase.CANDIDATE_SAVED.value,
                "trained_flow_path": candidate_rel,
                "training_history_path": training_rel,
                "training_elapsed_seconds": elapsed,
            })
            execution.update({
                "outcome": "returned",
                "ended_at": execution.get("ended_at") if known_elapsed else _utc_now(),
                "elapsed_seconds": elapsed,
            })
            if not known_elapsed:
                manifest["timing_lower_bound"] = True
            _atomic_record(attempt_path, attempt)
            changed = True

        validation_rel = (
            attempt_root / "validation_history.npz"
        ).relative_to(root).as_posix()
        selected_rel = (
            attempt_root / "selected_flow.eqx"
        ).relative_to(root).as_posix()
        continuation_rel = (
            attempt_root / "continuation_flow.eqx"
        ).relative_to(root).as_posix()
        if (
            attempt.get("phase") == StagePhase.CANDIDATE_SAVED
            and all((root / item).is_file() for item in (
                validation_rel, selected_rel, continuation_rel,
            ))
        ):
            trained, identity, selected_ess = _validate_validation_payload(
                root / validation_rel,
                stage=stage,
                attempt=int(attempt["attempt"]),
                t_start=float(attempt["t_start"]),
                t_end=float(attempt["t_end"]),
                sample_count=int(manifest["sample_signature"]["shape"][0]),
            )
            selected = "trained" if trained > identity else "identity"
            policy = record.get("acceptance_policy") or {}
            controls = policy.get("adaptive_controls") or {}
            accepted = manifest.get("mode") == "fixed" or selected_ess >= float(
                controls.get("tau_ess", 0.0)
            )
            attempt.update({
                "phase": StagePhase.EVALUATED.value,
                "status": "accepted" if accepted else "rejected",
                "selected": selected,
                "valid_trained_ess": trained,
                "valid_identity_ess": identity,
                "valid_selected_ess": selected_ess,
                "validation_history_path": validation_rel,
                "selected_flow_path": selected_rel,
                "continuation_flow_path": continuation_rel,
                "validation_elapsed_seconds": 0.0,
            })
            manifest["timing_lower_bound"] = True
            _atomic_record(attempt_path, attempt)
            changed = True

        attempt_phase = StagePhase(attempt["phase"])
        record_phase = StagePhase(record["phase"])
        if attempt_phase is StagePhase.CANDIDATE_SAVED and record_phase in (
            StagePhase.SELECTION_READY, StagePhase.TRAINING,
        ):
            record["phase"] = StagePhase.CANDIDATE_SAVED.value
            changed = True
        if attempt_phase is StagePhase.EVALUATED and record_phase in (
            StagePhase.SELECTION_READY, StagePhase.TRAINING,
            StagePhase.CANDIDATE_SAVED,
        ):
            record.update({
                "phase": StagePhase.EVALUATED.value,
                "decision": attempt["status"],
                "selected": attempt["selected"],
                "valid_trained_ess": attempt["valid_trained_ess"],
                "valid_identity_ess": attempt["valid_identity_ess"],
                "valid_selected_ess": attempt["valid_selected_ess"],
                "selected_flow_path": attempt["selected_flow_path"],
                "continuation_flow_path": attempt["continuation_flow_path"],
            })
            changed = True

    advance_rel = (
        stage_root / "validation_samples.npy"
    ).relative_to(root).as_posix()
    if (
        record.get("phase") == StagePhase.EVALUATED
        and record.get("decision") == "accepted"
        and (root / advance_rel).is_file()
    ):
        record["validation_samples_path"] = advance_rel
        record["phase"] = StagePhase.ADVANCE_SAVED.value
        record.setdefault("timing", {})["advance_seconds"] = 0.0
        manifest["timing_lower_bound"] = True
        changed = True
    if changed:
        record.setdefault("timing", {})["accumulated_seconds"] = (
            _stage_elapsed_seconds(root, record)
        )
        _atomic_record(stage_root / "stage.json", record)
    return record, changed


def _reconcile_run(root: Path, manifest: dict) -> dict:
    """Recover every payload-first/manifest-last transaction boundary."""
    inherited_stage = int(
        (manifest.get("fork_parent") or {}).get("accepted_stage", 0)
    )
    stable_prefix = json.loads(json.dumps(manifest))
    completed_prefix = stable_prefix.get("completed_stages", [])
    stable_prefix["current_stage"] = (
        {
            "stage": completed_prefix[-1]["stage"],
            "phase": StagePhase.COMMITTED.value,
        }
        if completed_prefix else None
    )
    stable_prefix["lifecycle"] = RunLifecycle.IN_PROGRESS.value
    stable_prefix.pop("exhaustion_reason", None)
    prefix_selection = []
    prefix_attempted = []
    prefix_training_elapsed = 0.0
    for index in range(1, len(completed_prefix) + 1):
        prefix_record = _read_record(
            root / "stages" / f"stage_{index:06d}" / "stage.json",
            "stage metadata",
        )
        prefix_selection.extend(
            float(item["t_end"])
            for item in prefix_record.get("selection_history", [])
        )
        for relative in prefix_record.get("attempts", []):
            prefix_attempt = _read_record(root / relative, "attempt metadata")
            prefix_attempted.append(float(prefix_attempt["t_end"]))
            if index > inherited_stage:
                prefix_training_elapsed += sum(
                    float(item["elapsed_seconds"])
                    for item in prefix_attempt.get("executions", [])
                )
    stable_prefix["selection_t_list"] = prefix_selection
    stable_prefix["attempted_t_list"] = prefix_attempted
    stable_prefix["training_elapsed_seconds_total"] = prefix_training_elapsed
    validate_manifest(root, stable_prefix)

    completed = manifest.get("completed_stages", [])
    stage = len(completed) + 1
    stage_path = root / "stages" / f"stage_{stage:06d}" / "stage.json"
    changed = False
    if stage_path.is_file():
        record = _read_record(stage_path, "stage metadata")
        record, recovered = _reconcile_stage_payloads(
            root, manifest, stage, record
        )
        changed |= recovered
        for field in (
            "entry_flow_path", "identity_flow_path",
            "training_identity_flow_path", "selected_flow_path",
            "continuation_flow_path", "validation_samples_path",
        ):
            _register_recovered(
                root, manifest, record.get(field),
                array=field == "validation_samples_path",
            )
        for relative in record.get("attempts", []):
            attempt = _read_record(root / relative, "attempt metadata")
            for field in (
                "optimizer_initial_flow_path", "trained_flow_path",
                "training_history_path", "validation_history_path",
                "selected_flow_path", "continuation_flow_path",
            ):
                _register_recovered(root, manifest, attempt.get(field))
        phase = StagePhase(record["phase"])
        manifest["current_stage"] = {"stage": stage, "phase": phase.value}
        changed = True
        if (
            record.get("exhaustion_reason") in (
                "temperature_resolution", "selection_limit",
            )
            and phase in (StagePhase.SELECTION_READY, StagePhase.REJECTED)
        ):
            manifest["lifecycle"] = RunLifecycle.EXHAUSTED.value
        if phase is StagePhase.COMMITTED:
            changed |= _promote_commit(root, manifest, stage, record)
    else:
        current = manifest.get("current_stage")
        if current and current.get("phase") != StagePhase.COMMITTED:
            raise FileNotFoundError("current stage metadata is missing")

    accepted = [0.0]
    selection_proposals = []
    attempted = []
    training_seconds = 0.0
    stage_count = len(manifest.get("completed_stages", []))
    current = manifest.get("current_stage")
    if current and current.get("phase") != StagePhase.COMMITTED:
        stage_count += 1
    for index in range(1, stage_count + 1):
        path = root / "stages" / f"stage_{index:06d}" / "stage.json"
        record = _read_record(path, "stage metadata")
        if index <= len(manifest.get("completed_stages", [])):
            accepted.append(float(record["t_end"]))
        attempt_paths = record.get("attempts", [])
        selection_proposals.extend(
            float(item["t_end"])
            for item in record.get("selection_history", [])
        )
        if attempt_paths:
            for relative in attempt_paths:
                attempt = _read_record(root / relative, "attempt metadata")
                attempted.append(float(attempt["t_end"]))
                if index > inherited_stage:
                    training_seconds += sum(
                        float(item.get("elapsed_seconds") or 0.0)
                        for item in attempt.get("executions", [])
                    )
    manifest["accepted_t_list"] = accepted
    manifest["selection_t_list"] = selection_proposals
    manifest["attempted_t_list"] = attempted
    manifest["training_elapsed_seconds_total"] = training_seconds
    if manifest.get("mode") == "adaptive":
        manifest["t_list"] = list(accepted)
    if changed:
        _atomic_json(root / "run.json", manifest)
    return manifest


def fork_run_directory(source, destination, *, problem_id: str | None = None) -> dict:
    """Copy an exhausted adaptive run's accepted prefix into a fork."""
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if source == destination:
        raise ValueError("fork destination must differ from the parent run")
    lock_paths = sorted({
        source.parent / f".{source.name}.jflows.lock",
        destination.parent / f".{destination.name}.jflows.lock",
    })
    handles = []
    staging = destination.with_name(
        f".{destination.name}.forking-{uuid4().hex}"
    )
    try:
        for lock_path in lock_paths:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(f"run directory is locked: {lock_path}") from exc
            handles.append(handle)
        parent = validate_manifest(source)
        if parent.get("mode") != "adaptive":
            raise ValueError("only adaptive runs can be forked")
        if parent.get("lifecycle") != RunLifecycle.EXHAUSTED:
            raise ValueError("only an exhausted adaptive run can be forked")
        if problem_id is None:
            problem_id = parent.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("fork problem_id must be a nonempty string")
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"fork destination must be empty: {destination}"
            )
        if destination.exists():
            destination.rmdir()
        for orphan in destination.parent.glob(f".{destination.name}.forking-*"):
            if orphan != staging:
                shutil.rmtree(orphan, ignore_errors=True)
        staging.mkdir(parents=True)
        shutil.copytree(source / "initial", staging / "initial", dirs_exist_ok=True)
        for summary in parent["completed_stages"]:
            stage = int(summary["stage"])
            relative = Path("stages") / f"stage_{stage:06d}"
            shutil.copytree(source / relative, staging / relative)

        child = json.loads(json.dumps(parent))
        parent_run_sha256 = _sha256(source / "run.json")
        child["run_id"] = uuid4().hex
        child["problem_id"] = problem_id
        child["created_at"] = _utc_now()
        child["updated_at"] = child["created_at"]
        child["lifecycle"] = RunLifecycle.FORK_PENDING.value
        child.pop("exhaustion_reason", None)
        child["sessions"] = []
        child["active_elapsed_seconds"] = 0.0
        child["training_elapsed_seconds_total"] = 0.0
        child["timing_lower_bound"] = bool(parent.get("timing_lower_bound"))
        child["inherited_active_elapsed_seconds"] = float(
            parent.get("inherited_active_elapsed_seconds") or 0.0
        ) + float(
            parent.get("active_elapsed_seconds") or 0.0
        )
        child["inherited_training_elapsed_seconds_total"] = float(
            parent.get("inherited_training_elapsed_seconds_total") or 0.0
        ) + float(
            parent.get("training_elapsed_seconds_total") or 0.0
        )
        child["fork_parent"] = {
            "run_id": parent["run_id"],
            "run_dir": source.as_posix(),
            "run_manifest_sha256": parent_run_sha256,
            "lifecycle": parent["lifecycle"],
            "accepted_stage": len(parent["completed_stages"]),
            "accepted_t": float(parent["accepted_t_list"][-1]),
            "created_at": child["created_at"],
        }
        if child["completed_stages"]:
            last = child["completed_stages"][-1]["stage"]
            child["current_stage"] = {
                "stage": last,
                "phase": StagePhase.COMMITTED.value,
            }
        else:
            child["current_stage"] = None
        keep_prefixes = ["initial/"] + [
            f"stages/stage_{int(item['stage']):06d}/"
            for item in child["completed_stages"]
        ]
        child["artifacts"] = {
            relative: metadata
            for relative, metadata in child["artifacts"].items()
            if any(relative.startswith(prefix) for prefix in keep_prefixes)
        }
        selection_proposals = []
        attempted = []
        for summary in child["completed_stages"]:
            record = _read_record(
                staging / summary["stage_path"], "stage metadata"
            )
            selection_proposals.extend(
                float(item["t_end"])
                for item in record.get("selection_history", [])
            )
            attempted.extend(
                float(_read_record(staging / path, "attempt metadata")["t_end"])
                for path in record.get("attempts", [])
            )
            commit_path = (
                staging / summary["stage_path"]
            ).parent / "commit.json"
            commit = _read_record(commit_path, "stage commit")
            commit["run_id"] = child["run_id"]
            _atomic_record(commit_path, commit)
        child["selection_t_list"] = selection_proposals
        child["attempted_t_list"] = attempted
        _atomic_json(staging / "run.json", child)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return validate_manifest(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def read_flow(root, relative_path: str, template):
    return eqx.tree_deserialise_leaves(
        Path(root).expanduser().resolve() / relative_path, template
    )


class RunStore:
    """Eager run-state transaction manager used outside JIT."""

    def __init__(
        self,
        root,
        *,
        generator: str,
        mode: str,
        problem_id: str | None,
        config: dict,
        flow,
        x_valid,
        resume: bool,
        requested_t_list: list[float] | None = None,
    ):
        self.root = None if root is None else Path(root).expanduser().resolve()
        self._transaction_lock_path = (
            None if self.root is None else self.root.parent
            / f".{self.root.name}.jflows.transaction.lock"
        )
        self.manifest: dict[str, Any] | None = None
        self._lock_handle = None
        self._session_started: float | None = None
        self._closed = False
        if self.root is None:
            if resume:
                raise ValueError("resume=True requires run_dir")
            return
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("problem_id must be a nonempty string when run_dir is used")
        self._acquire_lock()
        try:
            if resume:
                with self._transaction():
                    self._open_existing(
                        generator, mode, problem_id, config, flow, x_valid,
                        requested_t_list,
                    )
            else:
                self._create(
                    generator, mode, problem_id, config, flow, x_valid,
                    requested_t_list,
                )
        except BaseException:
            self.release()
            raise

    @property
    def enabled(self) -> bool:
        return self.root is not None

    @contextmanager
    def _transaction(self):
        if self._transaction_lock_path is None:
            yield
            return
        self._transaction_lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._transaction_lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _acquire_lock(self) -> None:
        assert self.root is not None
        self.root.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.root.parent / f".{self.root.name}.jflows.lock"
        self._lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(
                self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError(f"run directory is locked by another writer: {self.root}") from exc

    def _create(
        self, generator, mode, problem_id, config, flow, x_valid,
        requested_t_list,
    ) -> None:
        assert self.root is not None
        if x_valid is None:
            raise ValueError("x_valid is required for a new run")
        final_root = self.root
        if final_root.exists() and any(final_root.iterdir()):
            raise FileExistsError(f"run_dir must be empty for a new run: {final_root}")
        for orphan in final_root.parent.glob(f".{final_root.name}.creating-*"):
            shutil.rmtree(orphan, ignore_errors=True)
        staging = final_root.with_name(
            f".{final_root.name}.creating-{uuid4().hex}"
        )
        if final_root.exists():
            final_root.rmdir()
        self.root = staging
        staging.mkdir(parents=True)
        try:
            self._create_staged(
                generator, mode, problem_id, config, flow, x_valid,
                requested_t_list,
            )
            _fsync_directory(staging)
            os.replace(staging, final_root)
            _fsync_directory(final_root.parent)
            self.root = final_root
        except BaseException:
            self.root = final_root
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _create_staged(
        self, generator, mode, problem_id, config, flow, x_valid,
        requested_t_list,
    ) -> None:
        assert self.root is not None
        normalized_config = _json_value(config)
        initial_flow = "initial/flow.eqx"
        initial_samples = "initial/validation_samples.npy"
        _atomic_flow(self.root / initial_flow, flow)
        _atomic_npy(self.root / initial_samples, x_valid)
        now = _utc_now()
        t_list = [0.0]
        if mode == "fixed":
            t_list.extend(float(value) for value in (requested_t_list or []))
        self.manifest = {
            "artifact_format": ARTIFACT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "package_version": __version__,
            "run_id": uuid4().hex,
            "generator": generator,
            "mode": mode,
            "problem_id": problem_id,
            "created_at": now,
            "updated_at": now,
            "lifecycle": RunLifecycle.IN_PROGRESS.value,
            "config": normalized_config,
            "config_digest": _config_digest(normalized_config),
            "flow_signature": _flow_signature(flow),
            "sample_signature": _sample_signature(x_valid),
            "initial_flow_path": initial_flow,
            "initial_samples_path": initial_samples,
            "requested_t_list": _json_value(requested_t_list),
            "t_list": t_list,
            "accepted_t_list": [0.0],
            "selection_t_list": [],
            "attempted_t_list": [],
            "completed_stages": [],
            "current_stage": None,
            "artifacts": {},
            "sessions": [],
            "active_elapsed_seconds": 0.0,
            "training_elapsed_seconds_total": 0.0,
            "timing_lower_bound": False,
        }
        self._register(initial_flow)
        self._register(initial_samples, array=np.asarray(x_valid))
        self._start_session()
        self._write_manifest()

    def _open_existing(
        self, generator, mode, problem_id, config, flow, x_valid,
        requested_t_list,
    ) -> None:
        assert self.root is not None
        self.manifest = read_manifest(self.root)
        self.manifest = _reconcile_run(self.root, self.manifest)
        if self.manifest.get("lifecycle") == RunLifecycle.FORK_PENDING:
            self._activate_fork(
                generator, mode, problem_id, config, flow, x_valid,
                requested_t_list,
            )
        sessions = self.manifest.setdefault("sessions", [])
        abandoned_session = bool(
            sessions and sessions[-1].get("outcome") == "running"
        )
        if abandoned_session:
            sessions[-1]["outcome"] = "unclean_termination"
            self.manifest["timing_lower_bound"] = True
            _mark_running_attempt_unclean(self.root, self.manifest)
        self.manifest = validate_manifest(self.root, self.manifest)
        expected = {
            "generator": generator,
            "mode": mode,
            "problem_id": problem_id,
            "config_digest": _config_digest(_json_value(config)),
        }
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(
                    f"resume mismatch for {key}: saved {self.manifest.get(key)!r}, "
                    f"requested {value!r}"
                )
        if self.manifest.get("flow_signature") != _flow_signature(flow):
            raise ValueError("resume flow treedef/static/leaf signature mismatch")
        saved_schedule = self.manifest.get("requested_t_list")
        if saved_schedule != _json_value(requested_t_list):
            raise ValueError("resume fixed t_list does not match the saved schedule")
        if x_valid is not None:
            signature = _sample_signature(x_valid)
            if signature != self.manifest.get("sample_signature"):
                raise ValueError("resume x_valid does not match saved initial samples")
        lifecycle = RunLifecycle(self.manifest["lifecycle"])
        if lifecycle is RunLifecycle.IN_PROGRESS:
            self._start_session()
            self._write_manifest()
        elif abandoned_session:
            self._write_manifest()

    def _activate_fork(
        self, generator, mode, problem_id, config, flow, x_valid,
        requested_t_list,
    ) -> None:
        assert self.root is not None and self.manifest is not None
        if mode != "adaptive" or requested_t_list is not None:
            raise ValueError("a forked adaptive run must resume in adaptive mode")
        for key, requested in (
            ("generator", generator), ("mode", mode),
            ("problem_id", problem_id),
        ):
            if self.manifest.get(key) != requested:
                raise ValueError(
                    f"fork activation mismatch for {key}: "
                    f"saved {self.manifest.get(key)!r}, requested {requested!r}"
                )
        if self.manifest.get("flow_signature") != _flow_signature(flow):
            raise ValueError("fork flow treedef/static/leaf signature mismatch")
        if x_valid is not None and _sample_signature(x_valid) != self.manifest.get(
            "sample_signature"
        ):
            raise ValueError("fork x_valid does not match saved initial samples")
        requested_config = _json_value(config)
        saved_config = self.manifest.get("config", {})
        for key in sorted(set(saved_config) | set(requested_config)):
            if key == "bg_param":
                continue
            if saved_config.get(key) != requested_config.get(key):
                raise ValueError(
                    "fork activation may change only adaptive bg_param controls; "
                    f"mismatch at {key}"
                )
        self.manifest["config"] = requested_config
        self.manifest["config_digest"] = _config_digest(requested_config)
        self.manifest["lifecycle"] = RunLifecycle.IN_PROGRESS.value
        self.manifest.pop("exhaustion_reason", None)
        self.manifest["fork_parent"]["activated_at"] = _utc_now()
        self.manifest["fork_parent"]["changed_bg_param"] = (
            saved_config.get("bg_param") != requested_config.get("bg_param")
        )
        self._write_manifest()

    def _start_session(self) -> None:
        assert self.manifest is not None
        self._session_started = time.perf_counter()
        self.manifest["sessions"].append({
            "index": len(self.manifest["sessions"]) + 1,
            "started_at": _utc_now(),
            "ended_at": None,
            "active_seconds": 0.0,
            "outcome": "running",
        })

    def _write_manifest(self) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        sessions = self.manifest.get("sessions", [])
        if (
            self._session_started is not None and sessions
            and sessions[-1].get("outcome") == "running"
        ):
            sessions[-1]["active_seconds"] = float(
                time.perf_counter() - self._session_started
            )
        self.manifest["active_elapsed_seconds"] = sum(
            float(item.get("active_seconds") or 0.0) for item in sessions
        )
        self.manifest["updated_at"] = _utc_now()
        _atomic_json(self.root / "run.json", self.manifest)

    def _register(self, relative: str, *, array=None) -> dict:
        assert self.root is not None and self.manifest is not None
        path = self.root / relative
        metadata = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        if array is not None:
            value = np.asarray(array)
            metadata.update({"shape": list(value.shape), "dtype": str(value.dtype)})
        self.manifest["artifacts"][relative] = metadata
        return metadata

    def _stage_rel(self, stage: int) -> Path:
        return Path("stages") / f"stage_{stage:06d}"

    def _stage_path(self, stage: int) -> Path:
        assert self.root is not None
        return self.root / self._stage_rel(stage)

    def _read_stage(self, stage: int) -> dict:
        return _read_record(
            self._stage_path(stage) / "stage.json", "stage metadata"
        )

    def _write_stage(self, stage: int, value: dict) -> None:
        assert self.root is not None
        value.setdefault("timing", {})["accumulated_seconds"] = (
            _stage_elapsed_seconds(self.root, value)
        )
        _atomic_record(self._stage_path(stage) / "stage.json", value)
        self._write_manifest()

    @_transactional
    def prepare_stage(
        self, stage: int, t_start: float, t_end: float,
        entry_flow, identity_flow, training_identity_flow,
    ) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        rel = self._stage_rel(stage)
        entry_rel = (rel / "entry_flow.eqx").as_posix()
        identity_rel = (rel / "identity_flow.eqx").as_posix()
        training_identity_rel = (rel / "training_identity_flow.eqx").as_posix()
        _atomic_flow(self.root / entry_rel, entry_flow)
        _atomic_flow(self.root / identity_rel, identity_flow)
        _atomic_flow(self.root / training_identity_rel, training_identity_flow)
        self._register(entry_rel)
        self._register(identity_rel)
        self._register(training_identity_rel)
        record = {
            "stage": stage,
            "parent_stage": stage - 1 if stage > 1 else None,
            "config_digest": self.manifest["config_digest"],
            "t_start": float(t_start),
            "t_end": float(t_end),
            "phase": StagePhase.PREPARED.value,
            "status": "in_progress",
            "acceptance_policy": {
                "mode": self.manifest["mode"],
                "adaptive_controls": (
                    None if self.manifest["mode"] == "fixed"
                    else self.manifest["config"].get("bg_param")
                ),
                "ladder": int(self.manifest["config"]["ladder"]),
                "selection_proposal_limit": (
                    0 if self.manifest["mode"] == "fixed"
                    else SELECTION_PROPOSAL_LIMIT
                ),
            },
            "entry_flow_path": entry_rel,
            "identity_flow_path": identity_rel,
            "training_identity_flow_path": training_identity_rel,
            "selection_history": [],
            "attempts": [],
            "current_attempt": None,
            "selected_flow_path": None,
            "continuation_flow_path": None,
            "validation_samples_path": None,
            "timing": {
                "accumulated_seconds": 0.0,
                "session_segments": [],
            },
        }
        _atomic_record(self.root / rel / "stage.json", record)
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.PREPARED.value,
        }
        self._write_manifest()

    @_transactional
    def append_selection(
        self, stage: int, event: dict, next_t: float,
        *, exhaustion_reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        assert self.manifest is not None
        record = self._read_stage(stage)
        record["selection_history"].append(_json_value(event))
        record["t_end"] = float(next_t)
        if exhaustion_reason is not None:
            record["phase"] = StagePhase.SELECTION_READY.value
            record["exhaustion_reason"] = exhaustion_reason
            self.manifest["current_stage"] = {
                "stage": stage, "phase": StagePhase.SELECTION_READY.value,
            }
            self.manifest["lifecycle"] = RunLifecycle.EXHAUSTED.value
        self.manifest["selection_t_list"].append(float(event["t_end"]))
        self._write_stage(stage, record)

    @_transactional
    def set_selection(
        self, stage: int, t_end: float, *, exhaustion_reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        record = self._read_stage(stage)
        record["t_end"] = float(t_end)
        record["phase"] = StagePhase.SELECTION_READY.value
        if exhaustion_reason is not None:
            record["exhaustion_reason"] = exhaustion_reason
            self.manifest["lifecycle"] = RunLifecycle.EXHAUSTED.value
        assert self.manifest is not None
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.SELECTION_READY.value,
        }
        self._write_stage(stage, record)

    @_transactional
    def start_attempt(
        self, stage: int, attempt: int, t_end: float,
        optimizer_initial_flow, raw_key,
    ) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        record = self._read_stage(stage)
        rel = self._stage_rel(stage) / "attempts" / f"attempt_{attempt:06d}"
        flow_rel = (rel / "optimizer_initial_flow.eqx").as_posix()
        _atomic_flow(self.root / flow_rel, optimizer_initial_flow)
        self._register(flow_rel)
        attempt_record = {
            "stage": stage,
            "attempt": attempt,
            "t_start": record["t_start"],
            "t_end": float(t_end),
            "phase": StagePhase.TRAINING.value,
            "status": "in_progress",
            "optimizer_initial_flow_path": flow_rel,
            "trained_flow_path": None,
            "training_history_path": None,
            "validation_history_path": None,
            "raw_key": _json_value(np.asarray(raw_key)),
            "executions": [{
                "index": 1, "session": len(self.manifest["sessions"]),
                "started_at": _utc_now(), "outcome": "running",
                "elapsed_seconds": 0.0,
            }],
        }
        attempt_json = (rel / "attempt.json").as_posix()
        _atomic_record(self.root / attempt_json, attempt_record)
        record["attempts"].append(attempt_json)
        record["current_attempt"] = attempt
        record["t_end"] = float(t_end)
        record["phase"] = StagePhase.TRAINING.value
        self.manifest["attempted_t_list"].append(float(t_end))
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.TRAINING.value,
        }
        self._write_stage(stage, record)

    @_transactional
    def interrupt_attempt(
        self, stage: int, elapsed_seconds: float, outcome: str, *, error=None,
    ) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        stage_record = self._read_stage(stage)
        if not stage_record.get("attempts"):
            return
        attempt_path = self.root / stage_record["attempts"][-1]
        attempt = _read_record(attempt_path, "attempt metadata")
        execution = attempt["executions"][-1]
        if execution.get("outcome") != "running":
            raise RuntimeError("only a running attempt execution can be interrupted")
        execution.update({
            "outcome": outcome, "ended_at": _utc_now(),
            "elapsed_seconds": float(elapsed_seconds),
        })
        if error is not None:
            execution["error"] = _json_value(error)
        _atomic_record(attempt_path, attempt)
        self.manifest["training_elapsed_seconds_total"] += float(elapsed_seconds)
        self._write_stage(stage, stage_record)

    @_transactional
    def finish_training_execution(self, stage: int, elapsed_seconds: float) -> None:
        """Durably account for a returned kernel before saving its payload."""
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        stage_record = self._read_stage(stage)
        attempt_path = self.root / stage_record["attempts"][-1]
        attempt = _read_record(attempt_path, "attempt metadata")
        execution = attempt["executions"][-1]
        if execution.get("outcome") != "running":
            raise RuntimeError("training execution is not running")
        execution.update({
            "outcome": "result_pending", "ended_at": _utc_now(),
            "elapsed_seconds": float(elapsed_seconds),
        })
        _atomic_record(attempt_path, attempt)
        self.manifest["training_elapsed_seconds_total"] += float(elapsed_seconds)
        self._write_stage(stage, stage_record)

    @_transactional
    def mark_timing_lower_bound(self) -> None:
        if not self.enabled:
            return
        assert self.manifest is not None
        self.manifest["timing_lower_bound"] = True
        self._write_manifest()

    @_transactional
    def set_exhaustion_reason(self, reason: str) -> None:
        if not self.enabled:
            return
        if reason != "temperature_resolution":
            raise ValueError(f"unsupported exhaustion reason {reason!r}")
        assert self.manifest is not None
        self.manifest["exhaustion_reason"] = reason
        self.manifest["lifecycle"] = RunLifecycle.EXHAUSTED.value
        self._write_manifest()

    @_transactional
    def record_stage_session(
        self, stage: int, elapsed_seconds: float, outcome: str,
    ) -> None:
        if not self.enabled:
            return
        if outcome not in (
            "committed", "exhausted", "keyboard_interrupt", "exception",
        ):
            raise ValueError(f"invalid stage-session outcome {outcome!r}")
        assert self.manifest is not None
        record = self._read_stage(stage)
        # A durable stage commit is the terminal stage-session outcome. An
        # exception after that transaction (for example while promoting the
        # manifest) belongs to run reconciliation, not to the completed stage.
        if StagePhase(record["phase"]) is StagePhase.COMMITTED:
            return
        session = len(self.manifest["sessions"])
        segments = record.setdefault("timing", {}).setdefault(
            "session_segments", []
        )
        segment = {
            "session": session,
            "elapsed_seconds": float(elapsed_seconds),
            "outcome": outcome,
        }
        if segments and segments[-1].get("session") == session:
            if float(elapsed_seconds) >= float(
                segments[-1].get("elapsed_seconds") or 0.0
            ):
                segments[-1] = segment
        else:
            segments.append(segment)
        self._write_stage(stage, record)

    @_transactional
    def restart_attempt(self, stage: int) -> dict:
        if not self.enabled:
            return {}
        assert self.root is not None and self.manifest is not None
        stage_record = self._read_stage(stage)
        path = self.root / stage_record["attempts"][-1]
        attempt = _read_record(path, "attempt metadata")
        latest = attempt["executions"][-1]["outcome"]
        if latest == "returned":
            raise RuntimeError("a returned attempt execution cannot be replayed")
        if latest == "running":
            attempt["executions"][-1]["outcome"] = "unclean_termination"
            self.manifest["timing_lower_bound"] = True
        attempt["executions"].append({
            "index": len(attempt["executions"]) + 1,
            "session": len(self.manifest["sessions"]),
            "started_at": _utc_now(), "outcome": "running",
            "elapsed_seconds": 0.0,
        })
        _atomic_record(path, attempt)
        return attempt

    @_transactional
    def save_candidate(
        self, stage: int, candidate, loss_history, batch_ess_history,
        elapsed_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        stage_record = self._read_stage(stage)
        attempt_path = self.root / stage_record["attempts"][-1]
        attempt = _read_record(attempt_path, "attempt metadata")
        rel_dir = attempt_path.parent.relative_to(self.root)
        flow_rel = (rel_dir / "trained_flow.eqx").as_posix()
        history_rel = (rel_dir / "training_history.npz").as_posix()
        _atomic_flow(self.root / flow_rel, candidate)
        steps = np.arange(1, len(np.asarray(batch_ess_history)) + 1)
        _atomic_npz(
            self.root / history_rel,
            step=steps, loss=loss_history, batch_ess=batch_ess_history,
        )
        self._register(flow_rel)
        self._register(history_rel)
        attempt.update({
            "phase": StagePhase.CANDIDATE_SAVED.value,
            "trained_flow_path": flow_rel,
            "training_history_path": history_rel,
            "training_elapsed_seconds": float(elapsed_seconds),
        })
        execution = attempt["executions"][-1]
        if execution.get("outcome") not in ("running", "result_pending"):
            raise RuntimeError("candidate payload has no replayable execution")
        execution.update({
            "outcome": "returned", "ended_at": _utc_now(),
            "elapsed_seconds": float(elapsed_seconds),
        })
        _atomic_record(attempt_path, attempt)
        stage_record["phase"] = StagePhase.CANDIDATE_SAVED.value
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.CANDIDATE_SAVED.value,
        }
        self._write_stage(stage, stage_record)

    @_transactional
    def save_evaluation(
        self,
        stage: int,
        *,
        valid_trained_ess: float,
        valid_identity_ess: float,
        valid_selected_ess: float,
        selected: str,
        decision: str,
        selected_flow,
        continuation_flow,
        validation_elapsed_seconds: float,
        validation_sample_count: int,
    ) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        stage_record = self._read_stage(stage)
        attempt_path = self.root / stage_record["attempts"][-1]
        attempt = _read_record(attempt_path, "attempt metadata")
        rel_dir = attempt_path.parent.relative_to(self.root)
        history_rel = (rel_dir / "validation_history.npz").as_posix()
        roles = np.asarray(["trained", "identity", "selected"])
        values = np.asarray([
            valid_trained_ess, valid_identity_ess, valid_selected_ess,
        ])
        _atomic_npz(
            self.root / history_rel,
            stage=np.full(3, stage, dtype=np.int64),
            attempt=np.full(3, int(attempt["attempt"]), dtype=np.int64),
            evaluation_index=np.arange(1, 4, dtype=np.int64),
            t_start=np.full(3, float(attempt["t_start"])),
            t_end=np.full(3, float(attempt["t_end"])),
            role=roles,
            value=values,
            sample_count=np.full(3, validation_sample_count, dtype=np.int64),
        )
        self._register(history_rel)
        selected_rel = (rel_dir / "selected_flow.eqx").as_posix()
        continuation_rel = (rel_dir / "continuation_flow.eqx").as_posix()
        _atomic_flow(self.root / selected_rel, selected_flow)
        _atomic_flow(self.root / continuation_rel, continuation_flow)
        self._register(selected_rel)
        self._register(continuation_rel)
        attempt.update({
            "phase": StagePhase.EVALUATED.value,
            "status": decision,
            "selected": selected,
            "valid_trained_ess": float(valid_trained_ess),
            "valid_identity_ess": float(valid_identity_ess),
            "valid_selected_ess": float(valid_selected_ess),
            "validation_history_path": history_rel,
            "selected_flow_path": selected_rel,
            "continuation_flow_path": continuation_rel,
            "validation_elapsed_seconds": float(validation_elapsed_seconds),
        })
        _atomic_record(attempt_path, attempt)
        stage_record.update({
            "phase": StagePhase.EVALUATED.value,
            "decision": decision,
            "selected": selected,
            "valid_trained_ess": float(valid_trained_ess),
            "valid_identity_ess": float(valid_identity_ess),
            "valid_selected_ess": float(valid_selected_ess),
            "selected_flow_path": selected_rel,
            "continuation_flow_path": continuation_rel,
        })
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.EVALUATED.value,
        }
        self._write_stage(stage, stage_record)

    @_transactional
    def mark_rejected(
        self, stage: int, next_t: float | None,
        *, exhaustion_reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        assert self.manifest is not None
        record = self._read_stage(stage)
        record["phase"] = StagePhase.REJECTED.value
        record["next_t"] = _json_value(next_t)
        if exhaustion_reason is not None:
            record["exhaustion_reason"] = exhaustion_reason
            self.manifest["lifecycle"] = RunLifecycle.EXHAUSTED.value
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.REJECTED.value,
        }
        self._write_stage(stage, record)

    @_transactional
    def save_advance(self, stage: int, samples, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        assert self.root is not None and self.manifest is not None
        record = self._read_stage(stage)
        sample_array = np.asarray(samples)
        if not np.all(np.isfinite(sample_array)):
            raise ValueError("post-stage validation samples contain nonfinite coordinates")
        rel = (self._stage_rel(stage) / "validation_samples.npy").as_posix()
        _atomic_npy(self.root / rel, sample_array)
        self._register(rel, array=sample_array)
        record["validation_samples_path"] = rel
        record["phase"] = StagePhase.ADVANCE_SAVED.value
        record.setdefault("timing", {})["advance_seconds"] = float(elapsed_seconds)
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.ADVANCE_SAVED.value,
        }
        self._write_stage(stage, record)

    @_transactional
    def commit_stage(self, stage: int, transition_elapsed_seconds: float) -> dict:
        if not self.enabled:
            return {}
        assert self.root is not None and self.manifest is not None
        record = self._read_stage(stage)
        required = (
            "selected_flow_path", "continuation_flow_path",
            "validation_samples_path",
        )
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise RuntimeError(f"cannot commit stage {stage}; missing {missing}")
        measured = sum(
            float(item.get("elapsed_seconds") or 0.0)
            for item in record.get("selection_history", [])
        ) + float(record.get("timing", {}).get("advance_seconds") or 0.0)
        for attempt_relative in record.get("attempts", []):
            attempt_record = _read_record(
                self.root / attempt_relative, "attempt metadata"
            )
            measured += sum(
                float(item.get("elapsed_seconds") or 0.0)
                for item in attempt_record.get("executions", [])
            )
            measured += float(
                attempt_record.get("validation_elapsed_seconds") or 0.0
            )
        transition_elapsed_seconds = max(
            float(transition_elapsed_seconds), measured,
            sum(
                float(item.get("elapsed_seconds") or 0.0)
                for item in record.get("timing", {}).get("session_segments", [])
            ),
        )
        accepted_attempt = _read_record(
            self.root / record["attempts"][-1], "attempt metadata"
        )
        accepted_attempt_elapsed = sum(
            float(item.get("elapsed_seconds") or 0.0)
            for item in accepted_attempt.get("executions", [])
        ) + float(accepted_attempt.get("validation_elapsed_seconds") or 0.0)
        record.update({
            "phase": StagePhase.COMMITTED.value, "status": "accepted",
            "transition_elapsed_seconds": float(transition_elapsed_seconds),
            "accepted_attempt_elapsed_seconds": accepted_attempt_elapsed,
        })
        record.setdefault("timing", {})["accumulated_seconds"] = measured
        _atomic_record(self._stage_path(stage) / "stage.json", record)
        commit = {
            "run_id": self.manifest["run_id"],
            "stage": stage,
            "parent_stage": record["parent_stage"],
            "expected_completed_prefix": stage - 1,
            "t_start": record["t_start"],
            "t_end": record["t_end"],
            "config_digest": record["config_digest"],
            "stage_record_digest": record["record_digest"],
            "attempt_record_digests": {
                relative: _read_record(
                    self.root / relative, "attempt metadata"
                )["record_digest"]
                for relative in record.get("attempts", [])
            },
            "stage_artifacts": {
                relative: metadata
                for relative, metadata in self.manifest["artifacts"].items()
                if relative.startswith(f"stages/stage_{stage:06d}/")
            },
            "artifacts": {
                key: self.manifest["artifacts"][record[key]]
                for key in required
            },
        }
        _atomic_record(self._stage_path(stage) / "commit.json", commit)
        summary = {
            "stage": stage,
            "t_start": record["t_start"],
            "t_end": record["t_end"],
            "stage_path": (self._stage_rel(stage) / "stage.json").as_posix(),
            "selected_flow_path": record["selected_flow_path"],
            "validation_samples_path": record["validation_samples_path"],
            "continuation_flow_path": record["continuation_flow_path"],
            "transition_elapsed_seconds": float(transition_elapsed_seconds),
            "accepted_attempt_elapsed_seconds": accepted_attempt_elapsed,
        }
        completed = self.manifest["completed_stages"]
        if len(completed) < stage:
            completed.append(summary)
            self.manifest["accepted_t_list"].append(float(record["t_end"]))
            if self.manifest["mode"] == "adaptive":
                self.manifest["t_list"] = list(self.manifest["accepted_t_list"])
        self.manifest["current_stage"] = {
            "stage": stage,
            "phase": StagePhase.COMMITTED.value,
        }
        self._write_manifest()
        return record

    def load_last_samples(self):
        assert self.root is not None and self.manifest is not None
        completed = self.manifest["completed_stages"]
        relative = (
            completed[-1]["validation_samples_path"] if completed
            else self.manifest["initial_samples_path"]
        )
        return np.load(self.root / relative, allow_pickle=False)

    def load_continuation_flow(self, template):
        assert self.root is not None and self.manifest is not None
        completed = self.manifest["completed_stages"]
        relative = (
            completed[-1]["continuation_flow_path"] if completed
            else self.manifest["initial_flow_path"]
        )
        return read_flow(self.root, relative, template)

    def load_current(self, template) -> dict | None:
        if not self.enabled:
            return None
        assert self.root is not None and self.manifest is not None
        current = self.manifest.get("current_stage")
        if not current or current.get("phase") == StagePhase.COMMITTED:
            return None
        record = self._read_stage(int(current["stage"]))
        result = {"record": record}
        for role in (
            "entry_flow", "identity_flow", "training_identity_flow",
            "selected_flow", "continuation_flow",
        ):
            path = record.get(f"{role}_path")
            if path:
                result[role] = read_flow(self.root, path, template)
        if record.get("attempts"):
            attempt_path = self.root / record["attempts"][-1]
            attempt = _read_record(attempt_path, "attempt metadata")
            result["attempt"] = attempt
            if attempt.get("optimizer_initial_flow_path"):
                result["optimizer_initial_flow"] = read_flow(
                    self.root, attempt["optimizer_initial_flow_path"], template
                )
            if attempt.get("trained_flow_path"):
                result["candidate"] = read_flow(
                    self.root, attempt["trained_flow_path"], template
                )
            if attempt.get("training_history_path"):
                with np.load(self.root / attempt["training_history_path"]) as data:
                    result["loss_history"] = data["loss"]
                    result["batch_ess_history"] = data["batch_ess"]
        return result

    def restore_stage_records(self, template) -> list[dict]:
        if not self.enabled:
            return []
        assert self.root is not None and self.manifest is not None
        result = []
        for summary in self.manifest["completed_stages"]:
            record = _read_record(
                self.root / summary["stage_path"], "stage metadata"
            )
            attempts = [
                _read_record(self.root / path, "attempt metadata")
                for path in record["attempts"]
            ]
            normalized_attempts = []
            for attempt in attempts:
                with np.load(self.root / attempt["training_history_path"]) as data:
                    normalized_attempts.append(StageAttemptResult(
                        t_end=attempt["t_end"],
                        loss_history=data["loss"].copy(),
                        batch_ess_history=data["batch_ess"].copy(),
                        valid_trained_ess=attempt["valid_trained_ess"],
                        valid_identity_ess=attempt["valid_identity_ess"],
                        valid_selected_ess=attempt["valid_selected_ess"],
                        status=attempt["status"],
                        trained_flow_path=attempt["trained_flow_path"],
                    ))
            validation_sample_count = int(np.load(
                self.root / record["validation_samples_path"], mmap_mode="r",
                allow_pickle=False,
            ).shape[0])
            result.append(build_stage_result(
                t_start=record["t_start"],
                t_end=record["t_end"],
                selected=record["selected"],
                selected_flow=read_flow(
                    self.root, record["selected_flow_path"], template
                ),
                continuation_flow=read_flow(
                    self.root, record["continuation_flow_path"], template
                ),
                validation_sample_count=validation_sample_count,
                attempts=normalized_attempts,
                selected_flow_path=record["selected_flow_path"],
                continuation_flow_path=record["continuation_flow_path"],
                validation_samples_path=record["validation_samples_path"],
                selection_history=record.get("selection_history", []),
                elapsed_seconds=record["transition_elapsed_seconds"],
                asarray=np.asarray,
                stack=np.stack,
            ))
        return result

    def close(
        self,
        outcome: str,
        *,
        lifecycle: str | RunLifecycle | None = None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.enabled:
                assert self.root is not None and self.manifest is not None
                with self._transaction():
                    if self._session_started is not None:
                        _mark_running_attempt_unclean(self.root, self.manifest)
                        elapsed = time.perf_counter() - self._session_started
                        session = self.manifest["sessions"][-1]
                        session.update({
                            "ended_at": _utc_now(),
                            "active_seconds": float(elapsed),
                            "outcome": outcome,
                        })
                    if lifecycle is not None:
                        requested = RunLifecycle(lifecycle)
                        durable = RunLifecycle(self.manifest["lifecycle"])
                        if not (
                            durable is RunLifecycle.EXHAUSTED
                            and requested is RunLifecycle.IN_PROGRESS
                        ):
                            self.manifest["lifecycle"] = requested.value
                    if self._session_started is not None:
                        self._write_manifest()
        finally:
            self.release()

    def release(self) -> None:
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None

    def __del__(self):
        self.release()

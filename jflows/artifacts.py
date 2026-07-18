"""Inspection and explicit forking helpers for jflows run directories."""

from __future__ import annotations

import copy
import fcntl
import os
import shutil
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from uuid import uuid4

import numpy as np

from .training.run_store import (
    _read_record,
    fork_run_directory,
    read_flow,
    read_manifest,
    validate_manifest,
)


__all__ = [
    "inspect_run",
    "validate_run",
    "load_stage_flow",
    "load_validation_samples",
    "load_training_history",
    "fork_run",
]


def _root(run_dir) -> Path:
    return Path(run_dir).expanduser().resolve()


@contextmanager
def _read_snapshot(run_dir):
    root = _root(run_dir)
    lock_path = root.parent / f".{root.name}.jflows.transaction.lock"
    snapshot = root.parent / f".{root.name}.reader-{uuid4().hex}"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    locked = False
    try:
        handle = lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        locked = True
        if not root.is_dir():
            raise FileNotFoundError(f"not a jflows run directory: {root}")
        snapshot.mkdir()
        for directory, names, files in os.walk(root):
            relative = Path(directory).relative_to(root)
            snapshot_directory = snapshot / relative
            snapshot_directory.mkdir(parents=True, exist_ok=True)
            for name in names:
                (snapshot_directory / name).mkdir(exist_ok=True)
            for name in files:
                os.link(Path(directory) / name, snapshot_directory / name)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        locked = False
        handle.close()
        handle = None
        yield snapshot
    finally:
        if locked and handle is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if handle is not None:
            handle.close()
        shutil.rmtree(snapshot, ignore_errors=True)


def _consistent_read(function):
    @wraps(function)
    def wrapped(run_dir, *args, **kwargs):
        with _read_snapshot(run_dir) as snapshot:
            return function(snapshot, *args, **kwargs)
    return wrapped


@_consistent_read
def inspect_run(run_dir) -> dict:
    """Return a detached copy of the human-readable run manifest."""
    return copy.deepcopy(read_manifest(_root(run_dir)))


@_consistent_read
def validate_run(run_dir) -> dict:
    """Validate payloads and committed-stage bindings; return the manifest."""
    root = _root(run_dir)
    return copy.deepcopy(validate_manifest(root))


def fork_run(run_dir, destination, *, problem_id: str | None = None) -> dict:
    """Fork an exhausted adaptive run from its validated accepted prefix.

    Resume the returned child with the same generator and numerical controls.
    On that first resume, only ``bg_param`` may differ from the parent run.
    """
    return copy.deepcopy(
        fork_run_directory(run_dir, destination, problem_id=problem_id)
    )


def _artifact_path(root: Path, manifest: dict, relative, label: str) -> Path:
    if not isinstance(relative, str) or relative not in manifest["artifacts"]:
        raise FileNotFoundError(f"{label} is not a registered run artifact")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"{label} escapes the run directory")
    return path


def _stage(run_dir, stage: int) -> tuple[Path, dict, dict]:
    root = _root(run_dir)
    if not isinstance(stage, int) or stage < 1:
        raise ValueError("stage must be a positive integer")
    manifest = validate_manifest(root)
    indexed = {
        int(item["stage"]) for item in manifest["completed_stages"]
    }
    current = manifest.get("current_stage")
    if current is not None:
        indexed.add(int(current["stage"]))
    if stage not in indexed:
        raise FileNotFoundError(f"stage {stage} is not indexed by the run manifest")
    path = root / "stages" / f"stage_{stage:06d}" / "stage.json"
    try:
        record = _read_record(path, "stage metadata")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"stage {stage} is not present in {root}") from exc
    return root, record, manifest


@_consistent_read
def load_stage_flow(
    run_dir,
    stage: int,
    role: str,
    template,
    *,
    attempt: int | None = None,
):
    """Load one saved flow role using a compatible caller template."""
    root, record, manifest = _stage(run_dir, stage)
    role_fields = {
        "entry": "entry_flow_path",
        "identity": "identity_flow_path",
        "selected": "selected_flow_path",
        "continuation": "continuation_flow_path",
    }
    if role in ("optimizer_initial", "trained"):
        attempts = record.get("attempts", [])
        index = len(attempts) if attempt is None else attempt
        if not (1 <= index <= len(attempts)):
            raise ValueError(f"attempt {index} is not present in stage {stage}")
        attempt_path = (root / attempts[index - 1]).resolve()
        if root not in attempt_path.parents:
            raise ValueError("attempt metadata escapes the run directory")
        attempt_record = _read_record(attempt_path, "attempt metadata")
        field = (
            "optimizer_initial_flow_path" if role == "optimizer_initial"
            else "trained_flow_path"
        )
        relative = attempt_record.get(field)
    else:
        try:
            field = role_fields[role]
        except KeyError as exc:
            raise ValueError(
                "role must be entry, identity, optimizer_initial, trained, "
                "selected, or continuation"
            ) from exc
        relative = record.get(field)
    if not relative:
        raise FileNotFoundError(
            f"flow role {role!r} is not durable for stage {stage}"
        )
    _artifact_path(root, manifest, relative, f"flow role {role!r}")
    return read_flow(root, relative, template)


@_consistent_read
def load_validation_samples(run_dir, stage: int | None = None, *, mmap_mode=None):
    """Load initial samples (`stage=None`) or one accepted post-stage set.

    ``mmap_mode`` may be ``None``, read-only ``"r"``, or copy-on-write
    ``"c"``. Writable memory maps are rejected because snapshot files share
    immutable checkpoint inodes with the run directory.
    """
    if mmap_mode not in (None, "r", "c"):
        raise ValueError("mmap_mode must be None, 'r', or copy-on-write 'c'")
    root = _root(run_dir)
    manifest = validate_manifest(root)
    if stage is None:
        relative = manifest["initial_samples_path"]
    else:
        _, record, manifest = _stage(root, stage)
        relative = record.get("validation_samples_path")
        if not relative:
            raise FileNotFoundError(f"stage {stage} has no accepted sample checkpoint")
    path = _artifact_path(root, manifest, relative, "validation samples")
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


@_consistent_read
def load_training_history(run_dir, stage: int, attempt: int = 1) -> dict:
    """Load full per-step loss/batch ESS and indexed validation ESS."""
    root, record, manifest = _stage(run_dir, stage)
    attempts = record.get("attempts", [])
    if not (1 <= attempt <= len(attempts)):
        raise ValueError(f"attempt {attempt} is not present in stage {stage}")
    attempt_path = (root / attempts[attempt - 1]).resolve()
    if root not in attempt_path.parents:
        raise ValueError("attempt metadata escapes the run directory")
    attempt_record = _read_record(attempt_path, "attempt metadata")
    result = {"attempt": copy.deepcopy(attempt_record)}
    training = attempt_record.get("training_history_path")
    if training:
        path = _artifact_path(root, manifest, training, "training history")
        with np.load(path, allow_pickle=False) as data:
            result.update({key: data[key].copy() for key in data.files})
    validation = attempt_record.get("validation_history_path")
    if validation:
        path = _artifact_path(root, manifest, validation, "validation history")
        with np.load(path, allow_pickle=False) as data:
            result["validation"] = {
                key: data[key].copy() for key in data.files
            }
    return result

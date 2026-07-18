"""Write complete Boltzmann stages to disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import equinox as eqx
import numpy as np

__all__ = ["create", "finish", "stage"]


def _value(value):
    """Convert array values for JSON output."""
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def jsonfile(path: Path, value) -> None:
    """Write one JSON file atomically."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _array(path: Path, value) -> None:
    """Write one array atomically."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def _history(path: Path, record: dict) -> None:
    """Write one stage history atomically."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            t_hist=record["t_hist"],
            batch_ess_hist=record["batch_ess_hist"],
            valid_trained_ess_hist=record["valid_trained_ess_hist"],
            valid_identity_ess_hist=record["valid_identity_ess_hist"],
        )
    os.replace(temporary, path)


def _flow(path: Path, flow) -> None:
    """Write one Equinox flow atomically."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    eqx.tree_serialise_leaves(temporary, flow)
    os.replace(temporary, path)


def create(run_dir, problem_id, config, samples, flow) -> dict:
    """Create a run containing its initial samples and flow."""
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "stages").mkdir(exist_ok=True)
    _array(root / "initial_samples.npy", samples)
    _flow(root / "initial_flow.eqx", flow)
    run = {
        "format": "jflows-stage-resume-1",
        "problem_id": problem_id,
        "status": "running",
        "config": _value(config),
        "initial_samples_path": "initial_samples.npy",
        "initial_flow_path": "initial_flow.eqx",
        "stages": [],
    }
    jsonfile(root / "run.json", run)
    return run


def stage(run_dir, run: dict, record: dict, samples) -> dict:
    """Write and publish one complete accepted stage."""
    root = Path(run_dir).expanduser().resolve()
    number = len(run["stages"]) + 1
    relative = Path("stages") / f"stage_{number:06d}"
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    selected = relative / "selected.eqx"
    continuation = relative / "continuation.eqx"
    population = relative / "samples.npy"
    history = relative / "history.npz"
    _flow(root / selected, record["flow"])
    _flow(root / continuation, record["continuation_flow"])
    _array(root / population, samples)
    _history(root / history, record)
    metadata = {
        "stage": number,
        "t": record["t"],
        "t_start": record["t_start"],
        "valid_selected_ess": record["valid_selected_ess"],
        "valid_trained_ess": record["valid_trained_ess"],
        "valid_identity_ess": record["valid_identity_ess"],
        "valid_sample_count": record["valid_sample_count"],
        "selected": record["selected"],
        "attempt_status_hist": record["attempt_status_hist"],
        "selection_history": record["selection_history"],
        "elapsed_seconds": record["elapsed_seconds"],
        "selected_flow_path": str(selected),
        "continuation_flow_path": str(continuation),
        "validation_samples_path": str(population),
        "history_path": str(history),
    }
    jsonfile(directory / "stage.json", metadata)
    run["stages"].append({"stage": number, "path": str(relative)})
    jsonfile(root / "run.json", run)
    saved = dict(record)
    saved.update({
        "selected_flow_path": str(selected),
        "continuation_flow_path": str(continuation),
        "validation_samples_path": str(population),
    })
    return saved


def finish(run_dir, run: dict, status: str) -> None:
    """Write the run status after the last complete stage."""
    run["status"] = status
    jsonfile(Path(run_dir).expanduser().resolve() / "run.json", run)

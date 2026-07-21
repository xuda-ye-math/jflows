"""Load complete Boltzmann stages from disk."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import equinox as eqx
import numpy as np

from .write import create, finish, stage

__all__ = [
    "fork",
    "fork_run",
    "inspect_run",
    "load",
    "load_stage_flow",
    "load_training_history",
    "load_validation_samples",
    "manifest",
    "run",
    "validate",
    "validate_run",
]


def manifest(run_dir) -> dict:
    """Read a run manifest."""
    path = Path(run_dir).expanduser().resolve() / "run.json"
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _stage(root: Path, item: dict) -> dict:
    """Read one committed stage record."""
    path = root / item["path"] / "stage.json"
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate(run_dir) -> dict:
    """Check the files required to continue a stage-level run."""
    root = Path(run_dir).expanduser().resolve()
    record = manifest(root)
    required = [record["initial_samples_path"]]
    if "initial_flow_path" in record:
        required.append(record["initial_flow_path"])
    for item in record["stages"]:
        stage = _stage(root, item)
        required.extend([stage["validation_samples_path"], stage["history_path"]])
        if "selected_flow_path" in stage:
            required.extend([
                stage["selected_flow_path"],
                stage["continuation_flow_path"],
            ])
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    return record


def load(run_dir, template=None):
    """Load the last population, optional flow, and stage records."""
    root = Path(run_dir).expanduser().resolve()
    run = validate(root)
    samples = np.load(root / run["initial_samples_path"], allow_pickle=False)
    continuation = None
    if "initial_flow_path" in run:
        continuation = eqx.tree_deserialise_leaves(
            root / run["initial_flow_path"], template
        )
    stages = []
    for item in run["stages"]:
        saved = _stage(root, item)
        with np.load(root / saved["history_path"], allow_pickle=False) as data:
            record = {
                "t": float(saved["t"]),
                "t_start": float(saved["t_start"]),
                "valid_selected_ess": float(saved["valid_selected_ess"]),
                "valid_identity_ess": float(saved["valid_identity_ess"]),
                "valid_sample_count": int(saved["valid_sample_count"]),
                "selected": saved["selected"],
                "attempt_status_hist": tuple(saved["attempt_status_hist"]),
                "selection_history": tuple(saved["selection_history"]),
                "elapsed_seconds": float(saved["elapsed_seconds"]),
                "validation_samples_path": saved["validation_samples_path"],
            }
            record.update({key: data[key].copy() for key in data.files})
        if "selected_flow_path" in saved:
            record.update({
                "valid_trained_ess": float(saved["valid_trained_ess"]),
                "selected_flow_path": saved["selected_flow_path"],
                "continuation_flow_path": saved["continuation_flow_path"],
            })
            record["flow"] = eqx.tree_deserialise_leaves(
                root / saved["selected_flow_path"], template
            )
            continuation = eqx.tree_deserialise_leaves(
                root / saved["continuation_flow_path"], template
            )
            record["continuation_flow"] = continuation
        stages.append(record)
        samples = np.load(
            root / saved["validation_samples_path"], allow_pickle=False
        )
    return samples, continuation, stages


def fork(run_dir, destination, problem_id=None) -> dict:
    """Copy a complete-stage prefix into a new run directory."""
    source = Path(run_dir).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    record = validate(source)
    shutil.copytree(source, destination)
    record["problem_id"] = problem_id or record["problem_id"]
    record["status"] = "running"
    from .write import jsonfile

    jsonfile(destination / "run.json", record)
    return record


inspect_run = manifest
validate_run = validate
fork_run = fork


def load_stage_flow(run_dir, stage: int, role: str, template):
    """Load the selected or continuation flow from one stored stage."""
    root = Path(run_dir).expanduser().resolve()
    saved = _stage(root, validate(root)["stages"][stage - 1])
    return eqx.tree_deserialise_leaves(
        root / saved[f"{role}_flow_path"], template
    )


def load_validation_samples(run_dir, stage=None, *, mmap_mode=None):
    """Load the initial or post-stage validation population."""
    root = Path(run_dir).expanduser().resolve()
    record = validate(root)
    if stage is None:
        relative = record["initial_samples_path"]
    else:
        relative = _stage(
            root, record["stages"][stage - 1]
        )["validation_samples_path"]
    return np.load(root / relative, mmap_mode=mmap_mode, allow_pickle=False)


def load_training_history(run_dir, stage: int) -> dict:
    """Load batch and validation ESS histories from one stored stage."""
    root = Path(run_dir).expanduser().resolve()
    saved = _stage(root, validate(root)["stages"][stage - 1])
    with np.load(root / saved["history_path"], allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    result["attempt_status_hist"] = tuple(saved["attempt_status_hist"])
    result["selection_history"] = tuple(saved["selection_history"])
    return result


def run(
    run_dir,
    problem_id,
    config,
    samples,
    flow,
    iterate,
    *,
    resume=False,
):
    """Save each complete stage and continue from the last one on resume."""
    if resume:
        samples, flow, records = load(run_dir, flow)
        saved = manifest(run_dir)
    else:
        records = []
        saved = create(run_dir, problem_id, config, samples, flow)
    accepted = (0.0, *(record["t"] for record in records))
    for samples, record, flow in iterate(
        samples, flow, accepted, len(records) + 1
    ):
        records.append(stage(run_dir, saved, record, samples))
        accepted = (*accepted, record["t"])
    status = "complete" if accepted[-1] == 1.0 else "exhausted"
    finish(run_dir, saved, status)
    return samples, records

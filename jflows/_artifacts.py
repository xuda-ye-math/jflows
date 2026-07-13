"""Private, eager persistence for Boltzmann-generator flow attempts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import equinox as eqx
import numpy as np


class FlowArtifactWriter:
    """Persist BG attempts without entering, or changing, compiled code.

    Paths stored in the manifest are relative to ``root`` so a complete run
    directory remains movable.  Every file is written to a sibling temporary
    path and atomically replaced.
    """

    def __init__(self, root, generator: str):
        self.root = None if root is None else Path(root).expanduser().resolve()
        self.manifest: dict[str, Any] | None = None
        if self.root is not None:
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError(
                    f"flow_dir must be empty for a new run: {self.root}"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": 1,
                "generator": generator,
                "status": "running",
                "last_t": 0.0,
                "accepted_stages": 0,
                "stages": [],
            }
            self._write_manifest()

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def start_stage(self, stage: int, t_prev: float) -> None:
        if not self.enabled:
            return
        assert self.manifest is not None
        entry = {
            "stage": stage,
            "t_prev": float(t_prev),
            "status": "running",
            "attempts": [],
            "selected_flow_path": None,
        }
        stages = self.manifest["stages"]
        if len(stages) >= stage:
            stages[stage - 1] = entry
            del stages[stage:]
        else:
            stages.append(entry)
        self._write_manifest()

    def save_attempt(
        self,
        stage: int,
        attempt: int,
        t: float,
        flow,
        *,
        status: str,
        selected: str,
        valid_selected_ess: float,
        valid_trained_ess: float,
        valid_identity_ess: float,
        batch_ess,
    ) -> str | None:
        if not self.enabled:
            return None
        assert self.root is not None and self.manifest is not None
        rel_dir = Path(f"stage_{stage:04d}") / f"attempt_{attempt:04d}"
        abs_dir = self.root / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        flow_rel = rel_dir / "trained_flow.eqx"
        self._write_flow(self.root / flow_rel, flow)
        monitor_rel = rel_dir / "monitor.npz"
        self._write_npz(
            self.root / monitor_rel,
            batch_ess_hist=np.asarray(batch_ess),
        )
        metadata = {
            "stage": stage,
            "attempt": attempt,
            "t": float(t),
            "status": status,
            "selected": selected,
            "valid_selected_ess": float(valid_selected_ess),
            "valid_trained_ess": float(valid_trained_ess),
            "valid_identity_ess": float(valid_identity_ess),
            "trained_flow_path": flow_rel.as_posix(),
            "batch_ess_path": monitor_rel.as_posix(),
        }
        self._write_json(abs_dir / "metadata.json", metadata)
        stage_entry = self.manifest["stages"][stage - 1]
        attempts = stage_entry["attempts"]
        if len(attempts) >= attempt:
            attempts[attempt - 1] = metadata
            del attempts[attempt:]
        else:
            attempts.append(metadata)
        self._write_manifest()
        return flow_rel.as_posix()

    def accept_stage(
        self,
        stage: int,
        t: float,
        selected_flow,
        *,
        selected: str,
        valid_selected_ess: float,
    ) -> str | None:
        if not self.enabled:
            return None
        assert self.root is not None and self.manifest is not None
        rel = Path(f"stage_{stage:04d}") / "selected_flow.eqx"
        self._write_flow(self.root / rel, selected_flow)
        entry = self.manifest["stages"][stage - 1]
        entry.update({
            "t": float(t),
            "status": "accepted",
            "selected": selected,
            "valid_selected_ess": float(valid_selected_ess),
            "selected_flow_path": rel.as_posix(),
        })
        self.manifest["last_t"] = float(t)
        self.manifest["accepted_stages"] = stage
        self._write_manifest()
        return rel.as_posix()

    def fail_stage(self, stage: int, status: str, **metadata) -> None:
        if not self.enabled:
            return
        assert self.manifest is not None
        entry = self.manifest["stages"][stage - 1]
        entry["status"] = status
        for key, value in metadata.items():
            entry[key] = _json_value(value)
        self._write_manifest()

    def finish(self, *, complete: bool, last_t: float, accepted_stages: int) -> None:
        if not self.enabled:
            return
        assert self.manifest is not None
        self.manifest.update({
            "status": "complete" if complete else "incomplete",
            "last_t": float(last_t),
            "accepted_stages": int(accepted_stages),
        })
        self._write_manifest()

    def _write_manifest(self) -> None:
        assert self.root is not None and self.manifest is not None
        self._write_json(self.root / "run_manifest.json", self.manifest)

    @staticmethod
    def _write_flow(path: Path, flow) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        try:
            eqx.tree_serialise_leaves(tmp, flow)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(value, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _write_npz(path: Path, **arrays) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
        try:
            with tmp.open("wb") as fh:
                np.savez(fh, **arrays)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)

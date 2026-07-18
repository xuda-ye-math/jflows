"""Smoke tests for transactional Boltzmann artifacts and resume."""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jflows import __version__

import jflows.training.boltzmann as boltzmann_module
import jflows.training.run_store as run_store_module
import jflows.artifacts as artifacts_module
from jflows.training.run_store import RunStore, _seal_record
from jflows.artifacts import (
    fork_run,
    inspect_run,
    load_stage_flow,
    load_training_history,
    load_validation_samples,
    validate_run,
)
from jflows.train import (
    boltzmann_forward_KL_G,
    boltzmann_forward_KL_G_fixed,
)
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, Potential


def _problem():
    potential = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
    samples = potential.samples(jax.random.key(1), 16)
    flow = NSF(
        jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
        bins=4, transforms=1, hidden_features=(8,),
    ).zeros()
    return potential, samples, flow


def _refresh_artifact_receipt(directory, path: Path) -> None:
    manifest_path = Path(directory) / "run.json"
    manifest = json.loads(manifest_path.read_text())
    relative = path.relative_to(directory).as_posix()
    manifest["artifacts"][relative] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _refresh_committed_artifact_receipt(directory, path: Path) -> None:
    """Reseal a completed stage after an intentionally adversarial rewrite."""
    _refresh_artifact_receipt(directory, path)
    root = Path(directory)
    manifest = json.loads((root / "run.json").read_text())
    relative = path.relative_to(root).as_posix()
    receipt = manifest["artifacts"][relative]
    stage_root = next(
        parent for parent in path.parents if parent.name.startswith("stage_")
    )
    stage = json.loads((stage_root / "stage.json").read_text())
    commit_path = stage_root / "commit.json"
    commit = json.loads(commit_path.read_text())
    commit["stage_artifacts"][relative] = receipt
    for field in (
        "selected_flow_path", "continuation_flow_path",
        "validation_samples_path",
    ):
        if stage.get(field) == relative:
            commit["artifacts"][field] = receipt
    commit_path.write_text(
        json.dumps(_seal_record(commit), indent=2, sort_keys=True) + "\n"
    )


def _rewrite_stage_and_commit(directory, stage_index: int, mutate) -> None:
    stage_root = Path(directory) / "stages" / f"stage_{stage_index:06d}"
    stage_path = stage_root / "stage.json"
    stage = json.loads(stage_path.read_text())
    mutate(stage)
    stage = _seal_record(stage)
    stage_path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")
    commit_path = stage_root / "commit.json"
    commit = json.loads(commit_path.read_text())
    commit["stage_record_digest"] = stage["record_digest"]
    commit_path.write_text(
        json.dumps(_seal_record(commit), indent=2, sort_keys=True) + "\n"
    )


def _fixed(samples, potential, flow, run_dir, *, resume=False):
    return boltzmann_forward_KL_G_fixed(
        None if resume else samples,
        potential,
        potential,
        flow,
        batch_size=8,
        train_steps=1,
        lr=0.0,
        ladder=1,
        mc_dt=1e-3,
        mc_steps=0,
        t_list=[0.5, 1.0],
        run_dir=run_dir,
        resume=resume,
        problem_id="fixed-resume",
    )


def _one_fixed(samples, potential, flow, run_dir, *, resume=False):
    return boltzmann_forward_KL_G_fixed(
        None if resume else samples,
        potential,
        potential,
        flow,
        batch_size=8,
        train_steps=1,
        lr=0.0,
        ladder=1,
        mc_dt=1e-3,
        mc_steps=0,
        t_list=[1.0],
        run_dir=run_dir,
        resume=resume,
        problem_id="one-stage-resume",
    )


def _interrupt_stage_two(call):
    original = boltzmann_module._train_attempt
    state = {"raised": False}

    def fail_once(*args, **kwargs):
        if kwargs["stage"] == 2 and not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    boltzmann_module._train_attempt = fail_once
    try:
        try:
            call()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the injected interruption did not occur")
    finally:
        boltzmann_module._train_attempt = original


def _crash_after_store_transaction(method_name: str, call) -> None:
    """Emulate SIGKILL after payload writes but before manifest promotion."""
    original_method = getattr(RunStore, method_name)
    original_close = RunStore.close
    state = {"raised": False}

    def crash(self, *args, **kwargs):
        manifest_path = self.root / "run.json"
        stale_manifest = manifest_path.read_text(encoding="utf-8")
        stage = int(args[0])
        stage_path = self._stage_path(stage) / "stage.json"
        stale_stage = (
            stage_path.read_text(encoding="utf-8") if stage_path.is_file() else None
        )
        stale_attempt_path = None
        stale_attempt = None
        if stale_stage is not None:
            stage_record = json.loads(stale_stage)
            if stage_record.get("attempts"):
                stale_attempt_path = self.root / stage_record["attempts"][-1]
                stale_attempt = stale_attempt_path.read_text(encoding="utf-8")
        result = original_method(self, *args, **kwargs)
        if not state["raised"]:
            state["raised"] = True
            manifest_path.write_text(stale_manifest, encoding="utf-8")
            if method_name in (
                "append_selection", "start_attempt", "save_candidate", "save_evaluation",
                "save_advance",
            ) and stale_stage is not None:
                stage_path.write_text(stale_stage, encoding="utf-8")
            if method_name in ("save_candidate", "save_evaluation"):
                assert stale_attempt_path is not None and stale_attempt is not None
                stale_attempt_path.write_text(stale_attempt, encoding="utf-8")
            raise KeyboardInterrupt
        return result

    def release_without_manifest(self, *args, **kwargs):
        self._closed = True
        self.release()

    setattr(RunStore, method_name, crash)
    RunStore.close = release_without_manifest
    try:
        try:
            call()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError(f"{method_name} crash injection did not run")
    finally:
        setattr(RunStore, method_name, original_method)
        RunStore.close = original_close


def test_complete_artifacts() -> None:
    potential, samples, flow = _problem()
    _, ephemeral_stages = _fixed(samples, potential, flow, None)
    with tempfile.TemporaryDirectory() as directory:
        y, stages = _fixed(samples, potential, flow, directory)
        manifest = validate_run(directory)
        assert manifest["artifact_format"] == "jflows-boltzmann-run"
        assert manifest["schema_version"] == 1
        assert manifest["package_version"] == __version__
        assert manifest["lifecycle"] == "complete"
        assert manifest["t_list"] == [0.0, 0.5, 1.0]
        assert manifest["accepted_t_list"] == [0.0, 0.5, 1.0]
        assert len(manifest["completed_stages"]) == 2
        assert manifest["current_stage"] == {"stage": 2, "phase": "committed"}
        assert manifest["training_elapsed_seconds_total"] >= 0.0
        assert manifest["active_elapsed_seconds"] >= 0.0
        assert len(stages) == 2 and stages[-1]["valid_sample_count"] == 16
        assert set(stages[-1]) == set(ephemeral_stages[-1])
        assert stages[-1]["batch_ess_hist"].shape == (1, 1)
        assert stages[-1]["loss_hist"].shape == (1, 1)
        assert ephemeral_stages[-1]["batch_ess_hist"].shape == (1, 1)
        assert ephemeral_stages[-1]["selected_flow_path"] is None
        assert stages[-1]["selected_flow_path"] is not None
        assert np.array_equal(load_validation_samples(directory), np.asarray(samples))
        assert load_validation_samples(directory, 2).shape == (16, 2)
        original_samples = np.asarray(load_validation_samples(directory)).copy()
        for writable_mode in ("r+", "w+"):
            try:
                load_validation_samples(directory, mmap_mode=writable_mode)
            except ValueError as exc:
                assert "mmap_mode" in str(exc)
            else:
                raise AssertionError("writable checkpoint memory map was accepted")
        assert np.array_equal(
            load_validation_samples(directory), original_samples
        )
        validate_run(directory)
        history = load_training_history(directory, 2)
        assert history["step"].tolist() == [1]
        assert history["batch_ess"].shape == (1,)
        assert history["loss"].shape == (1,)
        assert history["validation"]["role"].tolist() == [
            "trained", "identity", "selected"
        ]
        for role in (
            "entry", "identity", "optimizer_initial", "trained",
            "selected", "continuation",
        ):
            loaded = load_stage_flow(directory, 2, role, flow)
            assert type(loaded) is type(flow)
        y_resumed, stages_resumed = _fixed(
            samples, potential, flow, directory, resume=True
        )
        assert jnp.array_equal(y, y_resumed)
        assert len(stages_resumed) == len(stages)
        # A terminal no-op resume is read-only and opens no crashable session.
        assert len(inspect_run(directory)["sessions"]) == 1


def test_fixed_interruption_resume() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _interrupt_stage_two(
            lambda: _fixed(samples, potential, flow, directory)
        )
        manifest = inspect_run(directory)
        assert manifest["lifecycle"] == "in_progress"
        assert len(manifest["completed_stages"]) == 1
        assert manifest["current_stage"] == {"stage": 2, "phase": "training"}
        load_stage_flow(directory, 2, "optimizer_initial", flow)
        y, stages = _fixed(samples, potential, flow, directory, resume=True)
        manifest = validate_run(directory)
        assert manifest["lifecycle"] == "complete"
        assert len(stages) == 2 and y.shape == samples.shape
        stage_record = json.loads(
            (Path(directory) / "stages/stage_000002/stage.json").read_text()
        )
        attempt = json.loads(
            (Path(directory) / stage_record["attempts"][0]).read_text()
        )
        assert [item["outcome"] for item in attempt["executions"]] == [
            "keyboard_interrupt", "returned"
        ]
        execution_seconds = sum(
            item["elapsed_seconds"] for item in attempt["executions"]
        )
        measured_stage_seconds = (
            execution_seconds
            + attempt["validation_elapsed_seconds"]
            + stage_record["timing"]["advance_seconds"]
            + sum(
                item.get("elapsed_seconds", 0.0)
                for item in stage_record["selection_history"]
            )
        )
        assert stage_record["transition_elapsed_seconds"] >= measured_stage_seconds
        assert manifest["training_elapsed_seconds_total"] >= execution_seconds


def test_candidate_saved_resume_does_not_retrain() -> None:
    potential, samples, flow = _problem()
    original_save = RunStore.save_candidate
    state = {"raised": False}

    def interrupt_after_save(self, *args, **kwargs):
        original_save(self, *args, **kwargs)
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt

    with tempfile.TemporaryDirectory() as directory:
        RunStore.save_candidate = interrupt_after_save
        try:
            try:
                boltzmann_forward_KL_G_fixed(
                    samples, potential, potential, flow,
                    batch_size=8, train_steps=1, lr=0.0, ladder=1,
                    mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                    run_dir=directory, problem_id="candidate-resume",
                )
            except KeyboardInterrupt:
                pass
        finally:
            RunStore.save_candidate = original_save
        assert inspect_run(directory)["current_stage"]["phase"] == "candidate_saved"
        original_train = boltzmann_module._train_attempt

        def must_not_train(*args, **kwargs):
            raise AssertionError("candidate_saved resume retrained the candidate")

        boltzmann_module._train_attempt = must_not_train
        try:
            y, stages = boltzmann_forward_KL_G_fixed(
                None, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                run_dir=directory, resume=True,
                problem_id="candidate-resume",
            )
        finally:
            boltzmann_module._train_attempt = original_train
        assert y.shape == samples.shape and len(stages) == 1


def test_advance_saved_resume_commits_without_retraining() -> None:
    potential, samples, flow = _problem()
    original_save = RunStore.save_advance
    state = {"raised": False}

    def interrupt_after_save(self, *args, **kwargs):
        original_save(self, *args, **kwargs)
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt

    with tempfile.TemporaryDirectory() as directory:
        RunStore.save_advance = interrupt_after_save
        try:
            try:
                boltzmann_forward_KL_G_fixed(
                    samples, potential, potential, flow,
                    batch_size=8, train_steps=1, lr=0.0, ladder=1,
                    mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                    run_dir=directory, problem_id="advance-resume",
                )
            except KeyboardInterrupt:
                pass
        finally:
            RunStore.save_advance = original_save
        assert inspect_run(directory)["current_stage"]["phase"] == "advance_saved"
        original_train = boltzmann_module._train_attempt

        def must_not_train(*args, **kwargs):
            raise AssertionError("advance_saved resume retrained the stage")

        boltzmann_module._train_attempt = must_not_train
        try:
            y, stages = boltzmann_forward_KL_G_fixed(
                None, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                run_dir=directory, resume=True, problem_id="advance-resume",
            )
        finally:
            boltzmann_module._train_attempt = original_train
        assert y.shape == samples.shape and len(stages) == 1
        assert inspect_run(directory)["lifecycle"] == "complete"


def test_adaptive_interruption_resume() -> None:
    potential, samples, flow = _problem()
    kwargs = dict(
        batch_size=8, train_steps=1, lr=0.0, ladder=1,
        mc_dt=1e-3, mc_steps=0,
        bg_param={"t_safe": 0.5, "tau_ess": 0.0, "max_stages": 2},
        problem_id="adaptive-resume",
    )
    with tempfile.TemporaryDirectory() as directory:
        _interrupt_stage_two(lambda: boltzmann_forward_KL_G(
            samples, potential, potential, flow, run_dir=directory, **kwargs
        ))
        manifest = inspect_run(directory)
        assert len(manifest["completed_stages"]) == 1
        assert manifest["current_stage"] == {"stage": 2, "phase": "training"}
        y, stages = boltzmann_forward_KL_G(
            None, potential, potential, flow,
            run_dir=directory, resume=True, **kwargs,
        )
        assert y.shape == samples.shape and stages[-1]["t"] == 1.0
        assert inspect_run(directory)["lifecycle"] == "complete"


def test_exhausted_stage_is_durable_but_not_completed() -> None:
    source, samples, flow = _problem()
    target = Nlog_Gaussian([3.0, -2.0], [1.0, 1.0])
    kwargs = dict(
        batch_size=8,
        train_steps=1,
        lr=0.0,
        ladder=1,
        mc_dt=1e-3,
        mc_steps=0,
        bg_param={
            "t_safe": 1.0,
            "tau_smc": 0.0,
            "tau_ess": 1.0,
            "max_stages": 1,
            "max_retry": 1,
        },
        problem_id="adaptive-exhausted",
    )
    with tempfile.TemporaryDirectory() as directory:
        y, stages = boltzmann_forward_KL_G(
            samples, source, target, flow, run_dir=directory, **kwargs
        )
        manifest = validate_run(directory)
        assert y.shape == samples.shape and stages == []
        assert manifest["lifecycle"] == "exhausted"
        assert manifest["completed_stages"] == []
        assert manifest["accepted_t_list"] == [0.0]
        assert manifest["current_stage"] == {"stage": 1, "phase": "rejected"}
        load_stage_flow(directory, 1, "trained", flow)
        history = load_training_history(directory, 1)
        assert history["attempt"]["status"] == "rejected"
        assert history["batch_ess"].shape == (1,)
        try:
            boltzmann_forward_KL_G(
                None, source, target, flow,
                run_dir=directory, resume=True, **kwargs,
            )
        except RuntimeError as exc:
            assert "exhausted" in str(exc)
        else:
            raise AssertionError("exhausted adaptive run resumed silently")
        with tempfile.TemporaryDirectory() as fork_root:
            child = Path(fork_root) / "continued"
            original_copytree = run_store_module.shutil.copytree
            fault = {"raised": False}

            def fail_fork_copy(*args, **kwargs):
                if not fault["raised"]:
                    fault["raised"] = True
                    raise OSError("injected fork copy failure")
                return original_copytree(*args, **kwargs)

            run_store_module.shutil.copytree = fail_fork_copy
            try:
                try:
                    fork_run(directory, child)
                except OSError:
                    pass
                else:
                    raise AssertionError("fork creation fault did not fire")
            finally:
                run_store_module.shutil.copytree = original_copytree
            assert not child.exists()
            fork_manifest = fork_run(directory, child)
            assert fork_manifest["lifecycle"] == "fork_pending"
            assert fork_manifest["fork_parent"]["run_id"] == manifest["run_id"]
            changed = dict(kwargs)
            changed["bg_param"] = {
                "t_safe": 0.5,
                "tau_smc": 0.0,
                "tau_ess": 0.0,
                "max_stages": 3,
                "max_retry": 1,
            }
            original_json = run_store_module._atomic_json
            activation_fault = {"raised": False}

            def fail_activation(path, value):
                if (
                    not activation_fault["raised"]
                    and Path(path) == child / "run.json"
                    and value.get("lifecycle") == "in_progress"
                ):
                    activation_fault["raised"] = True
                    raise OSError("injected fork activation failure")
                return original_json(path, value)

            run_store_module._atomic_json = fail_activation
            try:
                try:
                    boltzmann_forward_KL_G(
                        None, source, target, flow,
                        run_dir=child, resume=True, **changed,
                    )
                except OSError:
                    pass
                else:
                    raise AssertionError("fork activation fault did not fire")
            finally:
                run_store_module._atomic_json = original_json
            assert validate_run(child)["lifecycle"] == "fork_pending"
            fork_y, fork_stages = boltzmann_forward_KL_G(
                None, source, target, flow,
                run_dir=child, resume=True, **changed,
            )
            fork_manifest = validate_run(child)
            assert fork_y.shape == samples.shape and fork_stages[-1]["t"] == 1.0
            assert fork_manifest["lifecycle"] == "complete"
            assert fork_manifest["fork_parent"]["changed_bg_param"] is True


def test_fork_preserves_inherited_acceptance_policy() -> None:
    source, samples, flow = _problem()
    target = Nlog_Gaussian([3.0, -2.0], [1.0, 1.0])
    parent_kwargs = dict(
        batch_size=8, train_steps=1, lr=0.0, ladder=1,
        mc_dt=1e-3, mc_steps=0,
        bg_param={
            "t_safe": 0.5, "tau_smc": 0.0, "tau_ess": 0.3,
            "max_stages": 1, "max_retry": 1,
        },
        problem_id="fork-policy",
    )
    with tempfile.TemporaryDirectory() as parent_directory:
        _, parent_stages = boltzmann_forward_KL_G(
            samples, source, target, flow,
            run_dir=parent_directory, **parent_kwargs,
        )
        parent = validate_run(parent_directory)
        assert parent["lifecycle"] == "exhausted" and len(parent_stages) == 1
        assert parent_stages[0]["valid_selected_ess"] < 0.9
        parent_path = Path(parent_directory) / "run.json"
        parent_raw = json.loads(parent_path.read_text())
        parent_raw["exhaustion_reason"] = "temperature_resolution"
        parent_raw["timing_lower_bound"] = True
        parent_raw["sessions"][-1]["outcome"] = "unclean_termination"
        parent_raw["sessions"][-1]["ended_at"] = None
        parent_path.write_text(
            json.dumps(parent_raw, indent=2, sort_keys=True) + "\n"
        )
        validate_run(parent_directory)
        with tempfile.TemporaryDirectory() as fork_root:
            child = Path(fork_root) / "child"
            pending_child = fork_run(parent_directory, child)
            assert "exhaustion_reason" not in pending_child
            assert pending_child["timing_lower_bound"] is True
            child_kwargs = dict(parent_kwargs)
            child_kwargs["bg_param"] = {
                "t_safe": 0.5, "tau_smc": 0.0, "tau_ess": 0.9,
                "max_stages": 3, "max_retry": 1,
            }
            boltzmann_forward_KL_G(
                None, source, target, flow,
                run_dir=child, resume=True, **child_kwargs,
            )
            child_manifest = validate_run(child)
            assert "exhaustion_reason" not in child_manifest
            assert child_manifest["timing_lower_bound"] is True
            inherited = json.loads(
                (child / "stages/stage_000001/stage.json").read_text()
            )
            assert (
                inherited["acceptance_policy"]["adaptive_controls"]["tau_ess"]
                == 0.3
            )
            assert child_manifest["config"]["bg_param"]["tau_ess"] == 0.9
            with tempfile.TemporaryDirectory() as grand_root:
                grandchild = Path(grand_root) / "grandchild"
                grand = fork_run(child, grandchild)
                expected_training = (
                    child_manifest["inherited_training_elapsed_seconds_total"]
                    + child_manifest["training_elapsed_seconds_total"]
                )
                expected_active = (
                    child_manifest["inherited_active_elapsed_seconds"]
                    + child_manifest["active_elapsed_seconds"]
                )
                assert grand["inherited_training_elapsed_seconds_total"] == expected_training
                assert grand["inherited_active_elapsed_seconds"] == expected_active


def test_clean_breaking_api_and_validation() -> None:
    for function in (
        boltzmann_forward_KL_G, boltzmann_forward_KL_G_fixed,
    ):
        signature = inspect.signature(function)
        assert "run_dir" in signature.parameters
        assert "resume" in signature.parameters
        assert "initialize_from_identity" in signature.parameters
        assert "u_clip" in signature.parameters
        assert "pool" + "_size" not in signature.parameters
        assert "flow" + "_dir" not in signature.parameters
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _fixed(samples, potential, flow, directory)
        wrong = samples.at[0, 0].add(1.0)
        try:
            boltzmann_forward_KL_G_fixed(
                wrong, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[0.5, 1.0],
                run_dir=directory, resume=True, problem_id="fixed-resume",
            )
        except ValueError as exc:
            assert "x_valid" in str(exc)
        else:
            raise AssertionError("resume accepted different initial samples")
        changed_potential = Nlog_Gaussian([0.5, 0.0], [1.0, 1.0])
        try:
            boltzmann_forward_KL_G_fixed(
                None, changed_potential, changed_potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[0.5, 1.0],
                run_dir=directory, resume=True, problem_id="fixed-resume",
            )
        except ValueError as exc:
            assert "config_digest" in str(exc)
        else:
            raise AssertionError("resume accepted changed potential parameters")


def test_commit_boundary_reconciliation() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        manifest_path = Path(directory) / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["completed_stages"] = []
        manifest["accepted_t_list"] = [0.0]
        manifest["current_stage"] = {"stage": 1, "phase": "advance_saved"}
        manifest["lifecycle"] = "in_progress"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        original_train = boltzmann_module._train_attempt

        def must_not_train(*args, **kwargs):
            raise AssertionError("commit-boundary resume retrained the stage")

        boltzmann_module._train_attempt = must_not_train
        try:
            y, stages = _one_fixed(
                samples, potential, flow, directory, resume=True
            )
        finally:
            boltzmann_module._train_attempt = original_train
        manifest = validate_run(directory)
        assert y.shape == samples.shape and len(stages) == 1
        assert manifest["current_stage"] == {"stage": 1, "phase": "committed"}
        assert len(manifest["completed_stages"]) == 1


def test_every_transaction_boundary_recovers() -> None:
    potential, samples, flow = _problem()
    for method_name in (
        "prepare_stage", "set_selection", "start_attempt", "save_candidate",
        "append_selection", "save_evaluation", "save_advance", "commit_stage",
    ):
        with tempfile.TemporaryDirectory() as directory:
            _crash_after_store_transaction(
                method_name,
                lambda: _one_fixed(samples, potential, flow, directory),
            )
            y, stages = _one_fixed(
                samples, potential, flow, directory, resume=True
            )
            manifest = validate_run(directory)
            assert y.shape == samples.shape and len(stages) == 1
            assert manifest["lifecycle"] == "complete"
            assert manifest["accepted_t_list"] == [0.0, 1.0]


def test_rejected_transaction_boundary_recovers() -> None:
    source, samples, flow = _problem()
    target = Nlog_Gaussian([3.0, -2.0], [1.0, 1.0])
    kwargs = dict(
        batch_size=8, train_steps=1, lr=0.0, ladder=1,
        mc_dt=1e-3, mc_steps=0,
        bg_param={
            "t_safe": 1.0, "tau_smc": 0.0, "tau_ess": 1.0,
            "max_stages": 1, "max_retry": 2,
        },
        problem_id="rejected-boundary",
    )
    with tempfile.TemporaryDirectory() as directory:
        _crash_after_store_transaction(
            "mark_rejected",
            lambda: boltzmann_forward_KL_G(
                samples, source, target, flow, run_dir=directory, **kwargs
            ),
        )
        y, stages = boltzmann_forward_KL_G(
            None, source, target, flow,
            run_dir=directory, resume=True, **kwargs,
        )
        manifest = validate_run(directory)
        assert y.shape == samples.shape and stages == []
        assert manifest["lifecycle"] == "exhausted"
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        assert len(stage["attempts"]) == 2


def test_cross_process_resume_signatures() -> None:
    code = textwrap.dedent("""
        import sys
        import jax
        import jax.numpy as jnp
        from jflows.flow import NSF
        from jflows.potential import potential_from
        from jflows.training.signatures import structure_signature
        from jflows.train import boltzmann_forward_KL_G_fixed

        def energy(x):
            return 0.5 * jnp.sum(x * x, axis=-1)

        potential = potential_from(energy)
        flow = NSF(
            jax.random.key(2), [-4.0, -4.0], [4.0, 4.0],
            bins=4, transforms=1, hidden_features=(8,),
        ).zeros()
        samples = jax.random.normal(jax.random.key(1), (16, 2))
        resume = sys.argv[2] == "resume"

        class Token:
            __slots__ = ()

        def recursive(value):
            return recursive(value) if False else value

        mapping = {}
        for value in (("b", "a") if resume else ("a", "b")):
            mapping[Token()] = value
        print("STRUCTURE_DIGEST=" + structure_signature(
            {"mapping": mapping, "recursive": recursive},
            include_array_values=True,
        )["digest"])
        boltzmann_forward_KL_G_fixed(
            None if resume else samples,
            potential, potential, flow,
            batch_size=8, train_steps=1, lr=0.0, ladder=1,
            mc_dt=1e-3, mc_steps=0, t_list=[1.0],
            run_dir=sys.argv[1], resume=resume,
            problem_id="cross-process",
        )
    """)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    with tempfile.TemporaryDirectory() as directory:
        digests = []
        for mode in ("create", "resume"):
            completed = subprocess.run(
                [sys.executable, "-c", code, directory, mode],
                env=environment, text=True, capture_output=True,
            )
            if completed.returncode:
                raise AssertionError(
                    f"cross-process {mode} failed:\n{completed.stdout}"
                    f"\n{completed.stderr}"
                )
            digests.append(next(
                line.split("=", 1)[1]
                for line in completed.stdout.splitlines()
                if line.startswith("STRUCTURE_DIGEST=")
            ))
        assert digests[0] == digests[1]
        assert validate_run(directory)["lifecycle"] == "complete"


def test_lifecycle_and_attempt_schema_corruption() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _interrupt_stage_two(lambda: _fixed(samples, potential, flow, directory))
        manifest_path = Path(directory) / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["lifecycle"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "complete lifecycle" in str(exc)
        else:
            raise AssertionError("validation accepted a false complete lifecycle")

    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        stage_path = Path(directory) / "stages/stage_000001/stage.json"
        stage = json.loads(stage_path.read_text())
        stage["attempts"][0] = "../../outside/attempt.json"
        stage = _seal_record(stage)
        stage_path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")
        try:
            validate_run(directory)
        except ValueError as exc:
            assert (
                "integrity digest" in str(exc) or "escapes" in str(exc)
                or "not canonical" in str(exc)
            )
        else:
            raise AssertionError("validation accepted an escaping attempt path")


def test_malformed_recovery_payload_is_not_normalized() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _crash_after_store_transaction(
            "save_evaluation",
            lambda: _one_fixed(samples, potential, flow, directory),
        )
        history_path = (
            Path(directory) / "stages/stage_000001/attempts/attempt_000001"
            / "validation_history.npz"
        )
        with np.load(history_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        arrays["value"][-1] = 0.5
        np.savez(history_path, **arrays)
        # Keep the byte-level receipt consistent so this specifically probes
        # semantic recovery validation rather than the earlier checksum gate.
        _refresh_artifact_receipt(directory, history_path)
        try:
            _one_fixed(samples, potential, flow, directory, resume=True)
        except ValueError as exc:
            assert "selected ESS" in str(exc)
        else:
            raise AssertionError("resume normalized a malformed validation payload")


def test_metadata_receipts_and_public_loaders_reject_tampering() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        stage_path = Path(directory) / "stages/stage_000001/stage.json"
        stage = json.loads(stage_path.read_text())
        orphan = Path(directory) / "stages/stage_000999/stage.json"
        orphan.parent.mkdir(parents=True)
        orphan.write_text(stage_path.read_text())
        try:
            load_stage_flow(directory, 999, "selected", flow)
        except FileNotFoundError as exc:
            assert "not indexed" in str(exc)
        else:
            raise AssertionError("public loader accepted an orphan stage")
        attempt_path = Path(directory) / stage["attempts"][0]
        attempt = json.loads(attempt_path.read_text())
        attempt["valid_selected_ess"] = 0.5
        stage["valid_selected_ess"] = 0.5
        attempt_path.write_text(
            json.dumps(_seal_record(attempt), indent=2, sort_keys=True) + "\n"
        )
        stage_path.write_text(
            json.dumps(_seal_record(stage), indent=2, sort_keys=True) + "\n"
        )
        for operation in (
            lambda: validate_run(directory),
            lambda: load_training_history(directory, 1),
            lambda: load_stage_flow(directory, 1, "selected", flow),
        ):
            try:
                operation()
            except ValueError as exc:
                assert "disagrees" in str(exc) or "receipt" in str(exc)
            else:
                raise AssertionError("tampered metadata reached a public loader")


def test_session_timing_tampering_is_rejected() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        manifest_path = Path(directory) / "run.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["sessions"][-1]["outcome"] == "clean_exit"
        manifest["active_elapsed_seconds"] += 1.0
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "active duration" in str(exc)
        else:
            raise AssertionError("validation accepted a forged active-time total")


def test_terminal_resume_repairs_abandoned_session_without_opening_one() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        manifest_path = Path(directory) / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sessions"].append({
            "index": 2,
            "started_at": "2026-07-17T00:00:00Z",
            "ended_at": None,
            "active_seconds": 0.0,
            "outcome": "running",
        })
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        y, stages = _one_fixed(
            samples, potential, flow, directory, resume=True
        )
        repaired = validate_run(directory)
        assert y.shape == samples.shape and len(stages) == 1
        assert repaired["lifecycle"] == "complete"
        assert repaired["sessions"][-1]["outcome"] == "unclean_termination"
        assert repaired["timing_lower_bound"] is True
        session_count = len(repaired["sessions"])
        _one_fixed(samples, potential, flow, directory, resume=True)
        assert len(inspect_run(directory)["sessions"]) == session_count


def test_preflight_failures_do_not_materialize_run_directory() -> None:
    potential, samples, flow = _problem()
    cases = (
        {"batch_size": 32},
        {"chunks": 32},
        {"g_clip": -1.0},
        {"mc_dt": 0.0},
        {"seed": 2**32},
    )
    with tempfile.TemporaryDirectory() as root:
        for index, override in enumerate(cases):
            destination = Path(root) / f"invalid-{index}"
            kwargs = dict(
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                run_dir=destination, problem_id="preflight",
            )
            kwargs.update(override)
            try:
                boltzmann_forward_KL_G_fixed(
                    samples, potential, potential, flow, **kwargs
                )
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"preflight accepted {override}")
            assert not destination.exists()
        for label, invalid_samples in (
            ("dtype", np.asarray([["bad", "data"]] * 4)),
            ("integer", np.zeros((4, 2), dtype=np.int32)),
            ("dimension", jnp.zeros((4, 3))),
            ("nonfinite", samples.at[0, 0].set(jnp.nan)),
        ):
            destination = Path(root) / f"invalid-{label}"
            try:
                boltzmann_forward_KL_G_fixed(
                    invalid_samples, potential, potential, flow,
                    batch_size=4, train_steps=1, lr=0.0, ladder=1,
                    mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                    run_dir=destination, problem_id="preflight",
                )
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"preflight accepted invalid {label}")
            assert not destination.exists()

        for index, invalid_policy in enumerate(([], (), False, 0, "")):
            destination = Path(root) / f"invalid-policy-{index}"
            try:
                boltzmann_forward_KL_G(
                    samples, potential, potential, flow,
                    batch_size=4, train_steps=1, lr=0.0, ladder=1,
                    mc_dt=1e-3, mc_steps=0, bg_param=invalid_policy,
                    run_dir=destination, problem_id="preflight",
                )
            except TypeError:
                pass
            else:
                raise AssertionError("preflight accepted a non-mapping policy")
            assert not destination.exists()

        class Integer_Potential(Potential):
            def __call__(self, values):
                return jnp.zeros(values.shape[0], dtype=jnp.int32)

        destination = Path(root) / "invalid-potential"
        invalid_potential = Integer_Potential()
        try:
            boltzmann_forward_KL_G_fixed(
                samples, invalid_potential, invalid_potential, flow,
                batch_size=4, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                run_dir=destination, problem_id="preflight",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("preflight accepted an integer-valued potential")
        assert not destination.exists()


def test_initial_creation_failure_is_retryable() -> None:
    potential, samples, flow = _problem()
    original = run_store_module._atomic_npy
    state = {"raised": False}

    def fail_initial_samples(path, value):
        if not state["raised"] and path.name == "validation_samples.npy":
            state["raised"] = True
            raise OSError("injected initial sample write failure")
        return original(path, value)

    with tempfile.TemporaryDirectory() as root:
        destination = Path(root) / "retryable-create"
        run_store_module._atomic_npy = fail_initial_samples
        try:
            try:
                _one_fixed(samples, potential, flow, destination)
            except OSError:
                pass
            else:
                raise AssertionError("initial creation fault did not fire")
        finally:
            run_store_module._atomic_npy = original
        assert not destination.exists()
        y, stages = _one_fixed(samples, potential, flow, destination)
        assert y.shape == samples.shape and len(stages) == 1
        assert validate_run(destination)["lifecycle"] == "complete"


def test_selection_proposals_survive_interruption() -> None:
    potential, samples, flow = _problem()
    original_smc = boltzmann_module._smc_jit
    original_append = RunStore.append_selection
    calls = {"count": 0}
    interrupted = {"done": False}

    def fake_smc(key, values, source, target, **kwargs):
        del key, source, target, kwargs
        calls["count"] += 1
        ess = 0.1 if calls["count"] == 1 else 1.0
        return values, jnp.asarray([ess])

    def interrupt_after_proposal(self, *args, **kwargs):
        result = original_append(self, *args, **kwargs)
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt
        return result

    kwargs = dict(
        batch_size=8, train_steps=1, lr=0.0, ladder=1,
        mc_dt=1e-3, mc_steps=0,
        bg_param={
            "t_safe": 1.0, "shrink_factor": 0.5,
            "tau_smc": 0.5, "tau_ess": 0.0, "max_stages": 2,
        },
        problem_id="selection-resume",
    )
    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._smc_jit = fake_smc
        RunStore.append_selection = interrupt_after_proposal
        try:
            try:
                boltzmann_forward_KL_G(
                    samples, potential, potential, flow,
                    run_dir=directory, **kwargs,
                )
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("selection interruption did not fire")
        finally:
            RunStore.append_selection = original_append
        interrupted_manifest = inspect_run(directory)
        assert interrupted_manifest["selection_t_list"] == [1.0]
        assert interrupted_manifest["attempted_t_list"] == []
        try:
            y, stages = boltzmann_forward_KL_G(
                None, potential, potential, flow,
                run_dir=directory, resume=True, **kwargs,
            )
        finally:
            boltzmann_module._smc_jit = original_smc
        manifest = validate_run(directory)
        assert y.shape == samples.shape and stages[-1]["t"] == 1.0
        assert manifest["selection_t_list"][:2] == [1.0, 0.5]
        assert calls["count"] >= 2


def test_execution_ledger_and_history_length_integrity() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _interrupt_stage_two(lambda: _fixed(samples, potential, flow, directory))
        stage_path = Path(directory) / "stages/stage_000002/stage.json"
        stage = json.loads(stage_path.read_text())
        attempt_path = Path(directory) / stage["attempts"][0]
        attempt = json.loads(attempt_path.read_text())
        attempt["executions"] = [
            {
                "index": 1, "session": 1,
                "started_at": "2026-07-17T00:00:00Z",
                "ended_at": "2026-07-17T00:00:01Z",
                "outcome": "returned", "elapsed_seconds": 1.0,
            },
            {
                "index": 2, "session": 999,
                "started_at": "2026-07-17T00:00:02Z",
                "ended_at": None,
                "outcome": "running", "elapsed_seconds": 0.0,
            },
        ]
        attempt_path.write_text(
            json.dumps(_seal_record(attempt), indent=2, sort_keys=True) + "\n"
        )
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "session" in str(exc) or "replay after return" in str(exc)
        else:
            raise AssertionError("impossible attempt execution history was accepted")

    with tempfile.TemporaryDirectory() as directory:
        _crash_after_store_transaction(
            "save_candidate",
            lambda: _one_fixed(samples, potential, flow, directory),
        )
        history_path = (
            Path(directory) / "stages/stage_000001/attempts/attempt_000001"
            / "training_history.npz"
        )
        np.savez(
            history_path,
            step=np.asarray([1, 2]),
            loss=np.asarray([0.0, 0.0]),
            batch_ess=np.asarray([1.0, 1.0]),
        )
        _refresh_artifact_receipt(directory, history_path)
        try:
            _one_fixed(samples, potential, flow, directory, resume=True)
        except ValueError as exc:
            assert "per-step histories" in str(exc)
        else:
            raise AssertionError("recovery accepted the wrong training-history length")


def test_kernel_time_is_durable_before_candidate_payload() -> None:
    potential, samples, flow = _problem()
    original = RunStore.save_candidate
    state = {"raised": False}

    def fail_candidate_once(self, *args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise OSError("injected candidate persistence failure")
        return original(self, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        RunStore.save_candidate = fail_candidate_once
        try:
            try:
                _one_fixed(samples, potential, flow, directory)
            except OSError:
                pass
            else:
                raise AssertionError("candidate persistence fault did not fire")
        finally:
            RunStore.save_candidate = original
        manifest = inspect_run(directory)
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((Path(directory) / stage["attempts"][0]).read_text())
        assert attempt["executions"][-1]["outcome"] == "result_pending"
        assert attempt["executions"][-1]["elapsed_seconds"] >= 0.0
        assert manifest["training_elapsed_seconds_total"] >= 0.0
        y, stages = _one_fixed(samples, potential, flow, directory, resume=True)
        assert y.shape == samples.shape and len(stages) == 1
        final_stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        final_attempt = json.loads(
            (Path(directory) / final_stage["attempts"][0]).read_text()
        )
        assert [item["outcome"] for item in final_attempt["executions"]] == [
            "result_pending", "returned",
        ]

    with tempfile.TemporaryDirectory() as directory:
        _crash_after_store_transaction(
            "save_candidate",
            lambda: _one_fixed(samples, potential, flow, directory),
        )
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        attempt_path = Path(directory) / stage["attempts"][0]
        pending = json.loads(attempt_path.read_text())
        assert pending["executions"][-1]["outcome"] == "result_pending"
        elapsed = pending["executions"][-1]["elapsed_seconds"]
        _one_fixed(samples, potential, flow, directory, resume=True)
        recovered = json.loads(attempt_path.read_text())
        assert recovered["executions"][-1]["outcome"] == "returned"
        assert recovered["executions"][-1]["elapsed_seconds"] == elapsed


def test_public_reader_waits_for_short_writer_transaction() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        lock_path = Path(directory).parent / (
            f".{Path(directory).name}.jflows.transaction.lock"
        )
        handle = lock_path.open("a+")
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        result = {"done": False, "error": None}

        def reader():
            try:
                validate_run(directory)
                result["done"] = True
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=reader)
        thread.start()
        time.sleep(0.05)
        assert not result["done"] and result["error"] is None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        thread.join(timeout=5.0)
        assert result["done"] and result["error"] is None


def test_reader_construction_failure_cleans_snapshot() -> None:
    potential, samples, flow = _problem()
    original = artifacts_module.os.link
    state = {"raised": False}

    def fail_first_link(*args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise OSError("injected hard-link failure")
        return original(*args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        artifacts_module.os.link = fail_first_link
        try:
            try:
                validate_run(directory)
            except OSError as exc:
                assert "hard-link" in str(exc)
            else:
                raise AssertionError("reader snapshot failure did not fire")
        finally:
            artifacts_module.os.link = original
        leftovers = list(
            Path(directory).parent.glob(
                f".{Path(directory).name}.reader-*"
            )
        )
        assert leftovers == []
        validate_run(directory)


def test_sample_validation_is_memory_bounded() -> None:
    potential, samples, flow = _problem()
    original = run_store_module.np.load

    def require_mmap_for_samples(path, *args, **kwargs):
        if Path(path).name == "validation_samples.npy":
            assert kwargs.get("mmap_mode") == "r"
        return original(path, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        run_store_module.np.load = require_mmap_for_samples
        try:
            validate_run(directory)
        finally:
            run_store_module.np.load = original


def test_adaptive_policy_receipt_binds_selection_decision() -> None:
    potential, samples, flow = _problem()
    original = boltzmann_module._smc_jit

    def accept_smc(key, values, source, target, **kwargs):
        del key, source, target, kwargs
        return values, jnp.asarray([1.0])

    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._smc_jit = accept_smc
        try:
            boltzmann_forward_KL_G(
                samples, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0,
                bg_param={
                    "t_safe": 1.0, "tau_smc": 0.5, "tau_ess": 0.0,
                    "max_stages": 2,
                },
                run_dir=directory, problem_id="policy-receipt",
            )
        finally:
            boltzmann_module._smc_jit = original

        def contradict_policy(stage):
            event = stage["selection_history"][0]
            event["smc_ess"] = [0.1]
            event["minimum_smc_ess"] = 0.1
            event["decision"] = "accepted"

        _rewrite_stage_and_commit(directory, 1, contradict_policy)
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "selection decision violates policy" in str(exc)
        else:
            raise AssertionError("semantic validation accepted an impossible decision")


def test_selected_flow_payloads_are_bound_to_their_roles() -> None:
    potential, samples, _ = _problem()

    def run_with_ess(directory, values):
        flow = NSF(
            jax.random.key(91), [-4.0, -4.0], [4.0, 4.0],
            bins=4, transforms=1, hidden_features=(8,),
        )
        original = boltzmann_module._bounded_ess
        prescribed = iter(values)
        boltzmann_module._bounded_ess = lambda value: next(prescribed)
        try:
            boltzmann_forward_KL_G_fixed(
                samples, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0, t_list=[1.0],
                initialize_from_identity=False,
                run_dir=directory, problem_id="flow-role-binding",
            )
        finally:
            boltzmann_module._bounded_ess = original

    with tempfile.TemporaryDirectory() as directory:
        run_with_ess(directory, (0.1, 0.9))
        root = Path(directory)
        stage = json.loads(
            (root / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((root / stage["attempts"][0]).read_text())
        assert attempt["selected"] == "identity"
        selected_path = root / attempt["selected_flow_path"]
        trained_path = root / attempt["trained_flow_path"]
        assert selected_path.read_bytes() != trained_path.read_bytes()
        selected_path.write_bytes(trained_path.read_bytes())
        _refresh_committed_artifact_receipt(directory, selected_path)
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "selected-flow payload" in str(exc)
        else:
            raise AssertionError("validation accepted the wrong selected flow role")

    with tempfile.TemporaryDirectory() as directory:
        run_with_ess(directory, (0.9, 0.1))
        root = Path(directory)
        stage = json.loads(
            (root / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((root / stage["attempts"][0]).read_text())
        assert attempt["selected"] == "trained"
        continuation_path = root / attempt["continuation_flow_path"]
        identity_path = root / stage["identity_flow_path"]
        assert continuation_path.read_bytes() != identity_path.read_bytes()
        continuation_path.write_bytes(identity_path.read_bytes())
        _refresh_committed_artifact_receipt(directory, continuation_path)
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "continuation-flow payload" in str(exc)
        else:
            raise AssertionError("validation accepted the wrong continuation role")

    with tempfile.TemporaryDirectory() as directory:
        run_with_ess(directory, (0.1, 0.9))
        root = Path(directory)
        stage = json.loads(
            (root / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((root / stage["attempts"][0]).read_text())
        assert attempt["selected"] == "identity"
        continuation_path = root / attempt["continuation_flow_path"]
        trained_path = root / attempt["trained_flow_path"]
        assert continuation_path.read_bytes() != trained_path.read_bytes()
        continuation_path.write_bytes(trained_path.read_bytes())
        _refresh_committed_artifact_receipt(directory, continuation_path)
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "continuation-flow payload" in str(exc)
        else:
            raise AssertionError(
                "validation accepted a trained continuation after identity selection"
            )


def test_fixed_selection_append_is_idempotent() -> None:
    potential, samples, flow = _problem()
    original = RunStore.set_selection
    state = {"raised": False}

    def interrupt_before_promotion(self, *args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        RunStore.set_selection = interrupt_before_promotion
        try:
            try:
                _one_fixed(samples, potential, flow, directory)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("fixed selection interruption did not fire")
        finally:
            RunStore.set_selection = original
        stage_path = Path(directory) / "stages/stage_000001/stage.json"
        interrupted = json.loads(stage_path.read_text())
        assert interrupted["phase"] == "prepared"
        assert len(interrupted["selection_history"]) == 1
        _one_fixed(samples, potential, flow, directory, resume=True)
        resumed = json.loads(stage_path.read_text())
        assert len(resumed["selection_history"]) == 1
        validate_run(directory)


def test_zero_ess_tie_selects_identity() -> None:
    potential, samples, flow = _problem()
    original = boltzmann_module._bounded_ess
    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._bounded_ess = lambda value: 0.0
        try:
            _one_fixed(samples, potential, flow, directory)
        finally:
            boltzmann_module._bounded_ess = original
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((Path(directory) / stage["attempts"][0]).read_text())
        assert attempt["valid_trained_ess"] == 0.0
        assert attempt["valid_identity_ess"] == 0.0
        assert attempt["selected"] == "identity"
        validate_run(directory)


def test_postcommit_interrupt_keeps_stage_immutable() -> None:
    potential, samples, flow = _problem()
    original = RunStore.restore_stage_records
    state = {"raised": False}

    def interrupt_after_commit(self, *args, **kwargs):
        if (
            not state["raised"] and self.manifest is not None
            and self.manifest.get("completed_stages")
        ):
            state["raised"] = True
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        RunStore.restore_stage_records = interrupt_after_commit
        try:
            try:
                _one_fixed(samples, potential, flow, directory)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("post-commit interruption did not fire")
        finally:
            RunStore.restore_stage_records = original
        validate_run(directory)
        y, stages = _one_fixed(samples, potential, flow, directory, resume=True)
        assert y.shape == samples.shape and len(stages) == 1
        validate_run(directory)


def test_unclean_attempt_marks_timing_lower_bound() -> None:
    potential, samples, flow = _problem()
    original = RunStore.finish_training_execution
    state = {"raised": False}

    def interrupt_before_receipt(self, *args, **kwargs):
        if not state["raised"]:
            state["raised"] = True
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        RunStore.finish_training_execution = interrupt_before_receipt
        try:
            try:
                _one_fixed(samples, potential, flow, directory)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("training receipt interruption did not fire")
        finally:
            RunStore.finish_training_execution = original
        interrupted = validate_run(directory)
        assert interrupted["timing_lower_bound"] is True
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((Path(directory) / stage["attempts"][0]).read_text())
        assert attempt["executions"][-1]["outcome"] == "unclean_termination"
        assert interrupted["sessions"][-1]["outcome"] == "keyboard_interrupt"
        y, stages = _one_fixed(samples, potential, flow, directory, resume=True)
        final = validate_run(directory)
        assert y.shape == samples.shape and len(stages) == 1
        assert final["timing_lower_bound"] is True


def test_failure_diagnostics_and_temperature_resolution_are_durable() -> None:
    potential, samples, flow = _problem()
    original_train = boltzmann_module._train_attempt

    def fail_training(*args, **kwargs):
        raise ValueError("diagnostic sentinel")

    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._train_attempt = fail_training
        try:
            try:
                _one_fixed(samples, potential, flow, directory)
            except ValueError as exc:
                assert "diagnostic sentinel" in str(exc)
            else:
                raise AssertionError("training failure did not fire")
        finally:
            boltzmann_module._train_attempt = original_train
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        attempt = json.loads((Path(directory) / stage["attempts"][0]).read_text())
        failure = attempt["executions"][-1]
        assert failure["outcome"] == "exception"
        assert failure["error"] == {
            "type": "ValueError", "message": "diagnostic sentinel",
        }
        validate_run(directory)

    original_smc = boltzmann_module._smc_jit

    def reject_smc(key, values, source, target, **kwargs):
        del key, source, target, kwargs
        return values, jnp.asarray([0.0])

    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._smc_jit = reject_smc
        try:
            y, stages = boltzmann_forward_KL_G(
                samples, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0,
                bg_param={
                    "t_safe": 1.0, "shrink_factor": 1e-100,
                    "tau_smc": 0.5, "tau_ess": 0.0,
                    "max_stages": 2, "max_retry": 8,
                },
                run_dir=directory, problem_id="selection-resolution",
            )
        finally:
            boltzmann_module._smc_jit = original_smc
        manifest = validate_run(directory)
        assert y.shape == samples.shape and stages == []
        assert manifest["lifecycle"] == "exhausted"
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        assert stage["exhaustion_reason"] == "temperature_resolution"
        assert stage["t_end"] > stage["t_start"]

    original_bounded = boltzmann_module._bounded_ess
    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._bounded_ess = lambda value: 0.0
        try:
            y, stages = boltzmann_forward_KL_G(
                samples, potential, potential, flow,
                batch_size=8, train_steps=1, lr=0.0, ladder=1,
                mc_dt=1e-3, mc_steps=0,
                bg_param={
                    "t_safe": 1.0, "shrink_factor": 1e-100,
                    "tau_smc": 0.0, "tau_ess": 1.0,
                    "max_stages": 2, "max_retry": 8,
                },
                run_dir=directory, problem_id="retry-resolution",
            )
        finally:
            boltzmann_module._bounded_ess = original_bounded
        manifest = validate_run(directory)
        assert y.shape == samples.shape and stages == []
        assert manifest["lifecycle"] == "exhausted"
        stage = json.loads(
            (Path(directory) / "stages/stage_000001/stage.json").read_text()
        )
        assert stage["exhaustion_reason"] == "temperature_resolution"
        assert stage["next_t"] is None


def test_nonfinite_advance_is_not_committed() -> None:
    potential, samples, flow = _problem()
    original = boltzmann_module._advance_stage_samples
    state = {"raised": False}

    def invalid_advance(*args, **kwargs):
        result = original(*args, **kwargs)
        if not state["raised"]:
            state["raised"] = True
            return result.at[0, 0].set(jnp.nan)
        return result

    with tempfile.TemporaryDirectory() as directory:
        boltzmann_module._advance_stage_samples = invalid_advance
        try:
            try:
                _one_fixed(samples, potential, flow, directory)
            except ValueError as exc:
                assert "nonfinite" in str(exc)
            else:
                raise AssertionError("nonfinite advance was committed")
        finally:
            boltzmann_module._advance_stage_samples = original
        manifest = validate_run(directory)
        assert manifest["completed_stages"] == []
        assert manifest["current_stage"]["phase"] == "evaluated"
        y, stages = _one_fixed(samples, potential, flow, directory, resume=True)
        assert y.shape == samples.shape and len(stages) == 1

    state["raised"] = False
    boltzmann_module._advance_stage_samples = invalid_advance
    try:
        try:
            _one_fixed(samples, potential, flow, None)
        except ValueError as exc:
            assert "nonfinite" in str(exc)
        else:
            raise AssertionError("ephemeral nonfinite advance was accepted")
    finally:
        boltzmann_module._advance_stage_samples = original


def test_corruption_blocks_validation_and_resume() -> None:
    potential, samples, flow = _problem()
    with tempfile.TemporaryDirectory() as directory:
        _one_fixed(samples, potential, flow, directory)
        manifest = inspect_run(directory)
        corrupt = Path(directory) / manifest["initial_samples_path"]
        with corrupt.open("ab") as handle:
            handle.write(b"corruption")
        try:
            validate_run(directory)
        except ValueError as exc:
            assert "artifact size mismatch" in str(exc)
        else:
            raise AssertionError("validate_run accepted a corrupted payload")
        try:
            _one_fixed(samples, potential, flow, directory, resume=True)
        except ValueError as exc:
            assert "artifact size mismatch" in str(exc)
        else:
            raise AssertionError("resume accepted a corrupted payload")


def main() -> None:
    test_complete_artifacts()
    test_fixed_interruption_resume()
    test_candidate_saved_resume_does_not_retrain()
    test_advance_saved_resume_commits_without_retraining()
    test_adaptive_interruption_resume()
    test_exhausted_stage_is_durable_but_not_completed()
    test_fork_preserves_inherited_acceptance_policy()
    test_clean_breaking_api_and_validation()
    test_commit_boundary_reconciliation()
    test_every_transaction_boundary_recovers()
    test_rejected_transaction_boundary_recovers()
    test_cross_process_resume_signatures()
    test_lifecycle_and_attempt_schema_corruption()
    test_malformed_recovery_payload_is_not_normalized()
    test_metadata_receipts_and_public_loaders_reject_tampering()
    test_session_timing_tampering_is_rejected()
    test_terminal_resume_repairs_abandoned_session_without_opening_one()
    test_preflight_failures_do_not_materialize_run_directory()
    test_initial_creation_failure_is_retryable()
    test_selection_proposals_survive_interruption()
    test_execution_ledger_and_history_length_integrity()
    test_kernel_time_is_durable_before_candidate_payload()
    test_public_reader_waits_for_short_writer_transaction()
    test_reader_construction_failure_cleans_snapshot()
    test_sample_validation_is_memory_bounded()
    test_adaptive_policy_receipt_binds_selection_decision()
    test_selected_flow_payloads_are_bound_to_their_roles()
    test_fixed_selection_append_is_idempotent()
    test_zero_ess_tie_selects_identity()
    test_postcommit_interrupt_keeps_stage_immutable()
    test_unclean_attempt_marks_timing_lower_bound()
    test_failure_diagnostics_and_temperature_resolution_are_durable()
    test_nonfinite_advance_is_not_committed()
    test_corruption_blocks_validation_and_resume()
    print("test_boltzmann_artifacts: OK")


if __name__ == "__main__":
    main()

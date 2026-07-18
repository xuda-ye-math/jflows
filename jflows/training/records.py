"""Structured results shared by training, orchestration, and recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Sequence

from jax import Array

from ..flow import Flow


class TrainingResult(NamedTuple):
    flow: Flow
    batch_ess_history: Array
    loss_history: Array


@dataclass(frozen=True)
class StageAttemptResult:
    """Normalized attempt data used by ephemeral and restored runs."""

    t_end: float
    loss_history: Any
    batch_ess_history: Any
    valid_trained_ess: float
    valid_identity_ess: float
    valid_selected_ess: float
    status: str
    trained_flow_path: str | None = None


def build_stage_result(
    *,
    t_start: float,
    t_end: float,
    selected: str,
    selected_flow,
    continuation_flow,
    validation_sample_count: int,
    attempts: Sequence[StageAttemptResult],
    selected_flow_path: str | None,
    continuation_flow_path: str | None,
    validation_samples_path: str | None,
    selection_history,
    elapsed_seconds: float,
    asarray: Callable,
    stack: Callable,
) -> dict:
    """Build the single stage-result contract returned by every controller."""
    if not attempts:
        raise ValueError("an accepted stage must contain at least one attempt")
    last = attempts[-1]
    return {
        "t": float(t_end),
        "t_start": float(t_start),
        "valid_selected_ess": float(last.valid_selected_ess),
        "valid_trained_ess": float(last.valid_trained_ess),
        "valid_identity_ess": float(last.valid_identity_ess),
        "valid_sample_count": int(validation_sample_count),
        "selected": selected,
        "flow": selected_flow,
        "continuation_flow": continuation_flow,
        "t_hist": asarray([attempt.t_end for attempt in attempts]),
        "loss_hist": stack([
            attempt.loss_history for attempt in attempts
        ]),
        "batch_ess_hist": stack([
            attempt.batch_ess_history for attempt in attempts
        ]),
        "valid_trained_ess_hist": asarray([
            attempt.valid_trained_ess for attempt in attempts
        ]),
        "valid_identity_ess_hist": asarray([
            attempt.valid_identity_ess for attempt in attempts
        ]),
        "attempt_status_hist": tuple(attempt.status for attempt in attempts),
        "trained_flow_path_hist": tuple(
            attempt.trained_flow_path for attempt in attempts
        ),
        "selected_flow_path": selected_flow_path,
        "continuation_flow_path": continuation_flow_path,
        "validation_samples_path": validation_samples_path,
        "selection_history": tuple(selection_history),
        "elapsed_seconds": float(elapsed_seconds),
    }

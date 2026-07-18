"""Lifecycle vocabulary shared by the Boltzmann controller and run store."""

from enum import StrEnum


class RunLifecycle(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    EXHAUSTED = "exhausted"
    FORK_PENDING = "fork_pending"


class StagePhase(StrEnum):
    PREPARED = "prepared"
    SELECTION_READY = "selection_ready"
    TRAINING = "training"
    CANDIDATE_SAVED = "candidate_saved"
    EVALUATED = "evaluated"
    REJECTED = "rejected"
    ADVANCE_SAVED = "advance_saved"
    COMMITTED = "committed"

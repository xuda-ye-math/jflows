"""Typed objective and schedule specifications for Boltzmann training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

from .validation import require_positive_integer, require_real_control


SELECTION_PROPOSAL_LIMIT = 60


@dataclass(frozen=True)
class ObjectiveSpec:
    objective: Literal["reverse_kl", "forward_kl", "forward_klx", "forward_klxx"]
    direction: Literal["F", "G"]
    key_namespace: int
    adaptive_generator: str
    fixed_generator: str

    def __post_init__(self) -> None:
        expected_direction = {
            "reverse_kl": "F",
            "forward_kl": "G",
            "forward_klx": "G",
            "forward_klxx": "G",
        }.get(self.objective)
        if expected_direction is None:
            raise ValueError(f"unsupported objective {self.objective!r}")
        if self.direction != expected_direction:
            raise ValueError(
                f"objective {self.objective!r} requires direction "
                f"{expected_direction!r}, got {self.direction!r}"
            )
        if self.key_namespace < 0:
            raise ValueError("key_namespace must be nonnegative")
        if self.fixed_generator != f"{self.adaptive_generator}_fixed":
            raise ValueError(
                "fixed_generator must be the adaptive generator name with "
                "the '_fixed' suffix"
            )

    def generator_name(self, *, fixed: bool) -> str:
        return self.fixed_generator if fixed else self.adaptive_generator


REVERSE_KL_F = ObjectiveSpec(
    "reverse_kl", "F", 101,
    "boltzmann_reverse_KL_F", "boltzmann_reverse_KL_F_fixed",
)
FORWARD_KL_G = ObjectiveSpec(
    "forward_kl", "G", 102,
    "boltzmann_forward_KL_G", "boltzmann_forward_KL_G_fixed",
)
FORWARD_KLX_G = ObjectiveSpec(
    "forward_klx", "G", 103,
    "boltzmann_forward_KLX_G", "boltzmann_forward_KLX_G_fixed",
)
FORWARD_KLXX_G = ObjectiveSpec(
    "forward_klxx", "G", 104,
    "boltzmann_forward_KLXX_G", "boltzmann_forward_KLXX_G_fixed",
)


@dataclass(frozen=True)
class AdaptivePolicy:
    t_safe: float = 0.2
    shrink_factor: float = 0.7
    enlarge_factor: float = 1.5
    tau_smc: float = 0.0
    tau_ess: float = 0.6
    t_tol: float = 0.01
    max_stages: int = 30
    max_retry: int = 6

    def __getitem__(self, key: str):
        return getattr(self, key)

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_adaptive_policy(name: str, values: dict | None) -> AdaptivePolicy:
    defaults = AdaptivePolicy().as_dict()
    if values is not None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name}: bg_param must be a mapping or None")
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"{name}: unknown bg_param keys {sorted(unknown)}")
        defaults.update(values)
    for key in (
        "t_safe", "shrink_factor", "enlarge_factor", "tau_smc",
        "tau_ess", "t_tol",
    ):
        try:
            defaults[key] = require_real_control(key, defaults[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {key} must be real") from exc
    if not (
        0.0 < defaults["t_safe"] <= 1.0
        and 0.0 < defaults["shrink_factor"] < 1.0
        and defaults["enlarge_factor"] > 0.0
        and 0.0 <= defaults["tau_smc"] <= 1.0
        and 0.0 <= defaults["tau_ess"] <= 1.0
        and 0.0 <= defaults["t_tol"] <= 1.0
    ):
        raise ValueError(f"{name}: invalid bg_param {defaults!r}")
    if defaults["t_safe"] < 1.0 and min(
        defaults["t_safe"] * (1.0 + defaults["enlarge_factor"]), 1.0
    ) <= defaults["t_safe"]:
        raise ValueError(
            f"{name}: enlarge_factor={defaults['enlarge_factor']!r} is too small "
            "to advance the ladder in floating-point arithmetic"
        )
    defaults["max_stages"] = require_positive_integer(
        "max_stages", defaults["max_stages"]
    )
    defaults["max_retry"] = require_positive_integer(
        "max_retry", defaults["max_retry"]
    )
    return AdaptivePolicy(**defaults)


def normalize_fixed_schedule(name: str, t_list) -> list[float]:
    try:
        schedule = [
            require_real_control("t_list endpoint", value) for value in t_list
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: t_list must contain real endpoints") from exc
    if not schedule:
        raise ValueError(f"{name}: t_list is empty")
    if not all(0.0 < value <= 1.0 for value in schedule):
        raise ValueError(f"{name}: t_list endpoints must lie in (0, 1]")
    if any(left >= right for left, right in zip(schedule, schedule[1:])):
        raise ValueError(f"{name}: t_list must be strictly increasing")
    return schedule

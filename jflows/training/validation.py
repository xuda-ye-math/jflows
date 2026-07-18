"""Shared validation for public orchestration inputs."""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np


def require_boolean(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}")
    return value


def require_positive_integer(name: str, value) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def require_nonnegative_integer(name: str, value) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def require_real_control(
    name, value, *, minimum=None, strictly_positive=False,
    allow_positive_infinity=False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    finite = math.isfinite(result)
    if not finite and not (allow_positive_infinity and result == math.inf):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and not result > 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def require_seed(name: str, value, *, preserve_array: bool = False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer scalar")
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer scalar")
    result = int(array)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit in an unsigned 32-bit JAX key")
    if preserve_array and hasattr(value, "shape"):
        # Keep explicit scalar arrays dynamic across ``eqx.filter_jit`` calls.
        # Boltzmann orchestration intentionally uses the default Python-int
        # result because its seed is part of the durable run configuration.
        return value
    return result

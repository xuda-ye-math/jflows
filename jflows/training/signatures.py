"""Process-stable structural signatures for recoverable runs."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import marshal
import math
from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import numpy as np


def _qualified(value) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _array_descriptor(value, *, include_values: bool, arrays: list) -> dict:
    array = np.ascontiguousarray(np.asarray(value))
    summary = {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    if include_values:
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
        summary["sha256"] = digest.hexdigest()
    index = len(arrays)
    arrays.append(summary)
    return {"kind": "array", "index": index, **summary}


def _describe(
    value,
    *,
    include_array_values: bool,
    arrays: list,
    seen: dict[int, int],
):
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "float", "value": "nan"}
        if math.isinf(value):
            return {"kind": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return _describe(
            value.item(), include_array_values=include_array_values,
            arrays=arrays, seen=seen,
        )
    if eqx.is_array(value):
        return _array_descriptor(
            value, include_values=include_array_values, arrays=arrays
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "sha256": _digest_bytes(bytes(value))}
    if isinstance(value, Path):
        return {"kind": "path", "value": value.as_posix()}
    if isinstance(value, type):
        return {"kind": "type", "name": _qualified(value)}
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _describe(value.start, include_array_values=include_array_values,
                               arrays=arrays, seen=seen),
            "stop": _describe(value.stop, include_array_values=include_array_values,
                              arrays=arrays, seen=seen),
            "step": _describe(value.step, include_array_values=include_array_values,
                              arrays=arrays, seen=seen),
        }

    track_identity = callable(value) or dataclasses.is_dataclass(value) or isinstance(
        value, (tuple, list, Mapping, set, frozenset)
    )
    if track_identity:
        marker = id(value)
        if marker in seen:
            return {"kind": "reference", "index": seen[marker]}
        seen[marker] = len(seen)

    if isinstance(value, functools.partial):
        return {
            "kind": "partial",
            "function": _describe(
                value.func, include_array_values=include_array_values,
                arrays=arrays, seen=seen,
            ),
            "args": _describe(
                value.args, include_array_values=include_array_values,
                arrays=arrays, seen=seen,
            ),
            "keywords": _describe(
                value.keywords or {}, include_array_values=include_array_values,
                arrays=arrays, seen=seen,
            ),
        }
    if inspect.ismethod(value):
        return {
            "kind": "bound_method",
            "function": _describe(
                value.__func__, include_array_values=include_array_values,
                arrays=arrays, seen=seen,
            ),
            "owner": _describe(
                value.__self__, include_array_values=include_array_values,
                arrays=arrays, seen=seen,
            ),
        }
    if callable(value) and (
        inspect.isfunction(value)
        or inspect.isbuiltin(value)
        or getattr(value, "__module__", None) is not None
        and getattr(value, "__qualname__", None) is not None
    ):
        descriptor = {
            "kind": "callable",
            "module": getattr(value, "__module__", None),
            "qualname": getattr(
                value, "__qualname__", getattr(value, "__name__", None)
            ),
        }
        code = getattr(value, "__code__", None)
        if code is not None:
            descriptor["code_sha256"] = _digest_bytes(marshal.dumps(code))
            descriptor["defaults"] = _describe(
                getattr(value, "__defaults__", None),
                include_array_values=include_array_values,
                arrays=arrays,
                seen=seen,
            )
            descriptor["kwdefaults"] = _describe(
                getattr(value, "__kwdefaults__", None),
                include_array_values=include_array_values,
                arrays=arrays,
                seen=seen,
            )
            closure = getattr(value, "__closure__", None)
            descriptor["closure"] = _describe(
                tuple(cell.cell_contents for cell in closure) if closure else (),
                include_array_values=include_array_values,
                arrays=arrays,
                seen=seen,
            )
        return descriptor

    if dataclasses.is_dataclass(value):
        return {
            "kind": "dataclass",
            "type": _qualified(value),
            "fields": [
                [
                    field.name,
                    _describe(
                        getattr(value, field.name),
                        include_array_values=include_array_values,
                        arrays=arrays,
                        seen=seen,
                    ),
                ]
                for field in dataclasses.fields(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _describe(item, include_array_values=include_array_values,
                          arrays=arrays, seen=seen)
                for item in value
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _describe(item, include_array_values=include_array_values,
                          arrays=arrays, seen=seen)
                for item in value
            ],
        }
    if isinstance(value, Mapping):
        items = [
            [
                _describe(key, include_array_values=include_array_values,
                          arrays=arrays, seen=seen),
                _describe(item, include_array_values=include_array_values,
                          arrays=arrays, seen=seen),
            ]
            for key, item in sorted(
                value.items(),
                key=lambda pair: _canonical([
                    _describe(
                        pair[0], include_array_values=include_array_values,
                        arrays=[], seen={},
                    ),
                    _describe(
                        pair[1], include_array_values=include_array_values,
                        arrays=[], seen={},
                    ),
                ]),
            )
        ]
        return {"kind": "mapping", "type": _qualified(value), "items": items}
    if isinstance(value, (set, frozenset)):
        items = [
            _describe(item, include_array_values=include_array_values,
                      arrays=arrays, seen=seen)
            for item in sorted(
                value,
                key=lambda item: _canonical(_describe(
                    item, include_array_values=include_array_values,
                    arrays=[], seen={},
                )),
            )
        ]
        return {"kind": "set", "type": _qualified(value), "items": items}
    if hasattr(value, "__dict__"):
        return {
            "kind": "object",
            "type": _qualified(value),
            "attributes": _describe(
                vars(value), include_array_values=include_array_values,
                arrays=arrays, seen=seen,
            ),
        }
    return {"kind": "opaque", "type": _qualified(value)}


def structure_signature(value, *, include_array_values: bool) -> dict:
    """Return a JSON-safe signature that is stable across Python processes."""
    arrays: list[dict] = []
    structure = _describe(
        value,
        include_array_values=include_array_values,
        arrays=arrays,
        seen={},
    )
    return {
        "type": _qualified(value),
        "digest": _digest_bytes(_canonical(structure).encode()),
        "arrays": arrays,
    }

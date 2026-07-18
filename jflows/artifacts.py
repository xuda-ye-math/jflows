"""Save and load medium-level training outputs."""

from pathlib import Path

import equinox as eqx
import numpy as np

__all__ = [
    "load_flow",
    "load_history",
    "load_samples",
    "save_flow",
    "save_history",
    "save_samples",
]


def save_flow(path, flow) -> None:
    """Save one flow's array leaves."""
    eqx.tree_serialise_leaves(Path(path), flow)


def load_flow(path, template):
    """Load flow leaves into a matching template."""
    return eqx.tree_deserialise_leaves(Path(path), template)


def save_samples(path, samples) -> None:
    """Save one sample array."""
    np.save(Path(path), np.asarray(samples), allow_pickle=False)


def load_samples(path):
    """Load one sample array."""
    return np.load(Path(path), allow_pickle=False)


def save_history(path, **history) -> None:
    """Save named training-history arrays."""
    np.savez(Path(path), **history)


def load_history(path) -> dict:
    """Load named training-history arrays."""
    with np.load(Path(path), allow_pickle=False) as history:
        return {name: history[name].copy() for name in history.files}

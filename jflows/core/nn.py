"""Neural-network building blocks used by jflows flows.

Adapted from zuko's MLP machinery:
    - Linear / MLP                    — used by RealNVP coupling MLP and CNF ODE MLP
    - MaskedLinear / MaskedMLP        — used by MAF conditioner

Weight initialisation matches the torch conventions the port inherits:
uniform on [-1/sqrt(fan_in), 1/sqrt(fan_in)] for weights and biases.

JAX notes:
    - every constructor takes a PRNG `key` as its first argument;
    - `MLP` / `MaskedMLP` support `mlp[i]` indexing, returning the i-th
      *linear* layer (activations are not indexed) — `mlp[-1]` is the
      output layer, as in the torch Sequential;
    - `activation` is a plain callable (e.g. `jax.nn.relu`), not a class.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


__all__ = ["Linear", "MaskedLinear", "MaskedMLP", "MLP"]


# ──────────────────────────────────────────────────────────────────────
# Linear — basic dense layer with optional stack dim
# ──────────────────────────────────────────────────────────────────────

def linear(x: Array, W: Array, b: Array | None = None) -> Array:
    """Forward of W x + b that handles both stacked and non-stacked W."""
    if W.ndim == 2:
        y = x @ W.T
    else:
        y = jnp.einsum("...ij,...j->...i", W, x)
    return y if b is None else y + b


class Linear(eqx.Module):
    """y = x W^T + b. With `stack=S`, builds S independent operators."""

    weight: Array
    bias: Array | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        bias: bool = True,
        stack: int | None = None,
    ) -> None:
        shape = () if stack is None else (stack,)
        bound = 1 / in_features**0.5
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.uniform(
            wkey, (*shape, out_features, in_features), minval=-bound, maxval=bound
        )
        self.bias = (
            jax.random.uniform(bkey, (*shape, out_features), minval=-bound, maxval=bound)
            if bias
            else None
        )
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: Array) -> Array:
        return linear(x, self.weight, self.bias)


# ──────────────────────────────────────────────────────────────────────
# MLP — multi-layer perceptron (RealNVP coupling, CNF ODE drift)
# ──────────────────────────────────────────────────────────────────────

class MLP(eqx.Module):
    """Multi-layer perceptron with configurable hidden widths/activation."""

    linears: tuple[Linear, ...]
    activation: Callable[[Array], Array] = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[Array], Array] | None = None,
        **kwargs,
    ) -> None:
        if activation is None:
            activation = jax.nn.relu
        keys = jax.random.split(key, len(hidden_features) + 1)
        self.linears = tuple(
            Linear(k, before, after, **kwargs)
            for k, before, after in zip(
                keys,
                (in_features, *hidden_features),
                (*hidden_features, out_features),
                strict=True,
            )
        )
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features

    def __getitem__(self, index: int) -> Linear:
        return self.linears[index]

    def __call__(self, x: Array) -> Array:
        for lin in self.linears[:-1]:
            x = self.activation(lin(x))
        return self.linears[-1](x)


# ──────────────────────────────────────────────────────────────────────
# MaskedLinear / MaskedMLP — autoregressive conditioner backbone (MAF)
# ──────────────────────────────────────────────────────────────────────

class MaskedLinear(eqx.Module):
    """Linear with a fixed boolean adjacency mask on the weight matrix."""

    weight: Array
    bias: Array
    mask: Array

    def __init__(self, key: Array, adjacency: np.ndarray) -> None:
        adjacency = np.asarray(adjacency, dtype=bool)
        out_features, in_features = adjacency.shape
        bound = 1 / in_features**0.5
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.uniform(
            wkey, (out_features, in_features), minval=-bound, maxval=bound
        )
        self.bias = jax.random.uniform(
            bkey, (out_features,), minval=-bound, maxval=bound
        )
        self.mask = jnp.asarray(adjacency)

    def __call__(self, x: Array) -> Array:
        return linear(x, self.mask * self.weight, self.bias)


class MaskedMLP(eqx.Module):
    """Masked MLP whose Jacobian entries dy_i/dx_j are zero where A_ij = 0."""

    linears: tuple[MaskedLinear, ...]
    activation: Callable[[Array], Array] = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        adjacency: np.ndarray,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[Array], Array] | None = None,
    ) -> None:
        adjacency = np.asarray(adjacency, dtype=bool)
        out_features, in_features = adjacency.shape
        if activation is None:
            activation = jax.nn.relu

        # Merge outputs with identical dependency sets so the masked
        # linear layers can be smaller.
        adjacency, inverse = np.unique(adjacency, axis=0, return_inverse=True)
        inverse = inverse.ravel()

        # P_ij = 1 iff A_ik = 1 ∀k with A_jk = 1 ("i depends on at most
        # what j depends on" — used to extend masks through hidden layers).
        precedence = (
            adjacency.astype(np.float64) @ adjacency.astype(np.float64).T
            == adjacency.sum(axis=-1)
        )

        keys = jax.random.split(key, len(hidden_features) + 1)
        linears: list[MaskedLinear] = []
        indices: np.ndarray | None = None
        for i, features in enumerate((*hidden_features, out_features)):
            if i > 0:
                assert indices is not None
                mask = precedence[:, indices]
            else:
                mask = adjacency

            if (~mask).all():
                raise ValueError("The adjacency matrix leads to a null Jacobian.")

            if i < len(hidden_features):
                reachable = np.nonzero(mask.sum(axis=-1))[0]
                indices = reachable[np.arange(features) % len(reachable)]
                mask = mask[indices]
            else:
                mask = mask[inverse]

            linears.append(MaskedLinear(keys[i], mask))

        self.linears = tuple(linears)
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features

    def __getitem__(self, index: int) -> MaskedLinear:
        return self.linears[index]

    def __call__(self, x: Array) -> Array:
        for lin in self.linears[:-1]:
            x = self.activation(lin(x))
        return self.linears[-1](x)

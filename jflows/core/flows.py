"""Unconditional lazy transformations used by jflows's public flows.

Adapted from zuko's flow machinery, with all `context` / `c=None` plumbing
removed. Each lazy transform is an eqx.Module whose `__call__()`
(no argument) returns a concrete `Transform` from `.transforms`.

Public classes:
    - MaskedAutoregressiveTransform — backbone of NSF / NCSF
    - GeneralCouplingTransform      — backbone of RealNVP
    - FFJTransform                  — backbone of CNF
    - OTFlowLazy                    — backbone of OTFlow
    - LinearMixingTransform         — Glow-style 1x1 mixing for RealNVP

And one factory helper:
    - CircularRQSTransform(*phi, bound, slope) → Transform

JAX design notes:
    - `univariate` is a plain callable (e.g. `MonotonicRQSTransform`);
      the per-coordinate `bound` array and `slope` float are explicit
      fields forwarded to it — pytrees cannot hold parent back-references,
      so there is no bound-method closure over the flow.
    - `_BoundMethod` wraps a (module, method-name) pair as a pytree
      callable, so a Transform holding `meta` / `f` still exposes the
      conditioner's parameters to JAX transformations.
    - `zeros()` returns a NEW module (modules are immutable).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import pi, prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .nn import MLP, MaskedMLP
from .numerics import broadcast, unpack
from .otflow import OTFlowTransform, OTPhi
from .transforms import (
    AutoregressiveTransform,
    CircularShiftTransform,
    ComposedTransform,
    CouplingTransform,
    DependentTransform,
    FreeFormJacobianTransform,
    LULinearTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
    RotationTransform,
    Transform,
)


__all__ = [
    "CircularRQSTransform",
    "FFJTransform",
    "GeneralCouplingTransform",
    "LinearMixingTransform",
    "MaskedAutoregressiveTransform",
    "OTFlowLazy",
]


class _BoundMethod(eqx.Module):
    """Pytree-safe bound method: `(module, name)` called as `module.name(*args)`.

    Keeps the module's parameters visible to JAX transformations (a plain
    Python bound method is an opaque pytree leaf).
    """

    module: eqx.Module
    name: str = eqx.field(static=True)

    def __call__(self, *args):
        return getattr(self.module, self.name)(*args)


# ──────────────────────────────────────────────────────────────────────
# Masked autoregressive
# ──────────────────────────────────────────────────────────────────────

class MaskedAutoregressiveTransform(eqx.Module):
    """Lazy unconditional masked autoregressive transformation.

    Fully autoregressive (passes == features): each output i sees only
    inputs with strictly smaller order index. The `univariate` argument
    is the constructor for the per-coordinate bijection used inside the
    autoregressive scheme; `bound` / `slope`, when given, are forwarded
    to it as keyword arguments.

    Arguments:
        key:             PRNG key for the conditioner initialisation.
        features:        number of features d.
        univariate:      Callable(*phi, **kwargs) -> Transform.
                         Default MonotonicAffineTransform.
        shapes:          per-parameter shape (excluding feature dim).
                         E.g. for RQS: [(K,), (K,), (K - 1,)].
        order:           feature ordering, shape (d,). Defaults to arange(d).
        hidden_features: MLP hidden widths.
        activation:      activation callable (e.g. jax.nn.relu).
        bound:           optional per-coord bound array forwarded to
                         `univariate` (NSF / NCSF spline bound).
        slope:           optional slope float forwarded to `univariate`.
        circular:        periodic conditioner (the NCSF case). The MLP is
                         fed the circle embedding (cos, sin) of every
                         conditioning coordinate, period-matched to
                         2 * bound per coordinate, so the conditioner —
                         and hence the modeled density — is continuous
                         across the seam theta = +-bound and invariant
                         under full-period shifts of the input (a genuine
                         density on the torus). Requires `bound`.
    """

    hyper: MaskedMLP
    order: Array
    univariate: Callable[..., Transform] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    total: int = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    bound: Array | None
    slope: float | None = eqx.field(static=True)
    circular: bool = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        features: int,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[tuple[int, ...]] = ((), ()),
        order: Array | np.ndarray | None = None,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[Array], Array] | None = None,
        bound: Array | None = None,
        slope: float | None = None,
        circular: bool = False,
    ) -> None:
        self.univariate = univariate
        self.shapes = tuple(tuple(s) for s in shapes)
        self.total = sum(prod(s) for s in self.shapes)
        self.bound = bound
        self.slope = slope
        self.circular = circular
        assert not circular or bound is not None, "circular conditioner needs `bound`"

        if order is None:
            order = np.arange(features)
        else:
            order = np.asarray(order, dtype=np.int64)
        assert order.ndim == 1 and order.shape[0] == features

        self.order = jnp.asarray(order)
        self.passes = features

        adjacency = order[:, None] > order
        adjacency = np.repeat(adjacency, repeats=self.total, axis=0)
        if circular:
            # (cos, sin) embedding: both features of coordinate j inherit
            # coordinate j's autoregressive order (columns d -> 2d).
            adjacency = np.repeat(adjacency, repeats=2, axis=1)

        self.hyper = MaskedMLP(
            key,
            adjacency,
            hidden_features=hidden_features,
            activation=activation,
        )

    def _kwargs(self) -> dict:
        kwargs = {}
        if self.bound is not None:
            # buffer semantics: the spline bound is not trainable
            kwargs["bound"] = jax.lax.stop_gradient(self.bound)
        if self.slope is not None:
            kwargs["slope"] = self.slope
        return kwargs

    def meta(self, x: Array) -> Transform:
        h = x
        if self.circular:
            # circle embedding of the conditioning angles, interleaved
            # [cos t_0, sin t_0, cos t_1, ...] to match the duplicated
            # mask columns; period-matched to 2 * bound per coordinate.
            w = jnp.pi * x / jax.lax.stop_gradient(self.bound)
            h = jnp.stack([jnp.cos(w), jnp.sin(w)], axis=-1)
            h = h.reshape(*x.shape[:-1], -1)
        phi = self.hyper(h)
        phi = phi.reshape(*phi.shape[:-1], -1, self.total)
        phi = unpack(phi, self.shapes)
        # DependentTransform reinterprets the last dim as event-dim so
        # log|det J| sums over coordinates automatically.
        return DependentTransform(self.univariate(*phi, **self._kwargs()), 1)

    def __call__(self) -> Transform:
        return AutoregressiveTransform(_BoundMethod(self, "meta"), self.passes)


# ──────────────────────────────────────────────────────────────────────
# Coupling
# ──────────────────────────────────────────────────────────────────────

class GeneralCouplingTransform(eqx.Module):
    """Lazy unconditional general coupling transformation.

    The MLP input dim is `features_a` (the number of "kept" coordinates).

    Arguments:
        key:             PRNG key for the conditioner initialisation.
        features:        number of features d.
        mask:            boolean vector; True = kept, False = transformed.
        univariate:      Callable(*phi, **kwargs) -> Transform.
        shapes:          per-parameter shape (excluding feature dim).
        hidden_features: MLP hidden widths.
        activation:      activation callable.
        bound / slope:   optional kwargs forwarded to `univariate`.
    """

    hyper: MLP
    mask: tuple[bool, ...] = eqx.field(static=True)
    univariate: Callable[..., Transform] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    total: int = eqx.field(static=True)
    bound: Array | None
    slope: float | None = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        features: int,
        mask: np.ndarray,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: Sequence[tuple[int, ...]] = ((), ()),
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[Array], Array] | None = None,
        bound: Array | None = None,
        slope: float | None = None,
    ) -> None:
        self.univariate = univariate
        self.shapes = tuple(tuple(s) for s in shapes)
        self.total = sum(prod(s) for s in self.shapes)
        self.bound = bound
        self.slope = slope

        mask = np.asarray(mask, dtype=bool)
        assert mask.ndim == 1 and mask.shape[0] == features

        features_a = int(mask.sum())
        features_b = features - features_a
        assert features_a > 0 and features_b > 0

        self.mask = tuple(bool(v) for v in mask)
        self.hyper = MLP(
            key,
            features_a,
            features_b * self.total,
            hidden_features=hidden_features,
            activation=activation,
        )

    def _kwargs(self) -> dict:
        kwargs = {}
        if self.bound is not None:
            # buffer semantics: the spline bound is not trainable
            kwargs["bound"] = jax.lax.stop_gradient(self.bound)
        if self.slope is not None:
            kwargs["slope"] = self.slope
        return kwargs

    def meta(self, x: Array) -> Transform:
        phi = self.hyper(x)
        phi = phi.reshape(*phi.shape[:-1], -1, self.total)
        phi = unpack(phi, self.shapes)
        return DependentTransform(self.univariate(*phi, **self._kwargs()), 1)

    def __call__(self) -> Transform:
        return CouplingTransform(_BoundMethod(self, "meta"), np.asarray(self.mask))

    def zeros(self) -> "GeneralCouplingTransform":
        """Reset to identity by zeroing the last conditioner-MLP layer's
        weight and bias. With phi = 0 the univariate transform reduces
        to the identity on the masked-out coordinates. Returns a new
        module (modules are immutable)."""
        return eqx.tree_at(
            lambda m: (m.hyper.linears[-1].weight, m.hyper.linears[-1].bias),
            self,
            replace_fn=jnp.zeros_like,
        )


# ──────────────────────────────────────────────────────────────────────
# FFJORD continuous flow
# ──────────────────────────────────────────────────────────────────────

class FFJTransform(eqx.Module):
    """Lazy unconditional free-form-Jacobian transformation (FFJORD).

    Time is embedded via 2 * frequency cos/sin components, concatenated
    with x as the ODE-MLP input.

    Arguments:
        key: PRNG key (conditioner init + Hutchinson probe).
        dimension: number of features d.
        frequency: number of time-embedding frequencies in the drift.
        nt: number of fixed RK4 steps the transform integrates.
        exact: exact O(d) trace if True, Hutchinson estimate if False.
        hidden_features: ODE-MLP layer widths.
        activation: ODE-MLP activation callable.
    """

    ode: MLP
    freqs: Array
    key: Array
    nt: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        dimension: int,
        frequency: int = 3,
        nt: int = 8,
        exact: bool = True,
        hidden_features: Sequence[int] = (64, 64),
        activation: Callable[[Array], Array] | None = None,
    ) -> None:
        if activation is None:
            activation = jax.nn.elu
        okey, hkey = jax.random.split(key)
        self.ode = MLP(
            okey,
            dimension + 2 * frequency,
            dimension,
            hidden_features=hidden_features,
            activation=activation,
        )
        self.freqs = jnp.arange(1, frequency + 1) * pi
        self.key = hkey
        self.nt = nt
        self.exact = exact

    def f(self, t: Array, x: Array) -> Array:
        # buffer semantics: the time-embedding frequencies are not trainable
        t = jax.lax.stop_gradient(self.freqs) * t[..., None]
        t = jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)
        x = jnp.concatenate(broadcast(t, x, ignore=1), axis=-1)
        return self.ode(x)

    def __call__(self) -> Transform:
        # The fixed-step RK4 integrator runs under lax.scan, so parameter
        # gradients flow through the trajectory natively; `f` is wrapped as
        # a pytree callable so the ODE-MLP parameters stay visible to JAX.
        return FreeFormJacobianTransform(
            f=_BoundMethod(self, "f"),
            t0=0.0,
            t1=1.0,
            nt=self.nt,
            exact=self.exact,
            key=self.key,
        )


# ──────────────────────────────────────────────────────────────────────
# OT-Flow continuous flow
# ──────────────────────────────────────────────────────────────────────

class OTFlowLazy(eqx.Module):
    """Lazy unconditional OT-Flow transformation.

    The continuous-time counterpart of `FFJTransform`, but with the ODE
    velocity field parameterised as the negative gradient of a scalar
    potential `Φ_theta` (an `OTPhi`). `__call__()` returns a fresh
    `OTFlowTransform` built from the potential's current parameters.
    Unlike FFJORD, the divergence is closed-form, so no `exact` /
    Hutchinson flag is needed.

    Arguments:
        key:        PRNG key for Φ's initialisation.
        dimension:  spatial dimension d.
        hidden:     hidden width of Φ's ResNet.
        layer:      number of ResNet layers inside Φ (>= 2).
        rank:       rank of Φ's quadratic term (clamped to <= dimension + 1).
        nt:         number of fixed RK4 steps the transform integrates.
        time_bound: (t0, t1) trajectory time bounds.
    """

    phi: OTPhi
    dimension: int = eqx.field(static=True)
    nt: int = eqx.field(static=True)
    time_bound: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        dimension: int,
        hidden: int = 64,
        layer: int = 3,
        rank: int = 10,
        nt: int = 8,
        time_bound: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.dimension = dimension
        self.nt = nt
        self.time_bound = time_bound
        self.phi = OTPhi(key, dimension=dimension, hidden=hidden, layer=layer, rank=rank)

    def __call__(self) -> Transform:
        t0, t1 = self.time_bound
        return OTFlowTransform(
            phi=self.phi,
            dimension=self.dimension,
            t0=t0,
            t1=t1,
            nt=self.nt,
        )


# ──────────────────────────────────────────────────────────────────────
# CircularRQSTransform helper
# ──────────────────────────────────────────────────────────────────────

def CircularRQSTransform(
    *phi: Array,
    bound: Array | float = pi,
    slope: float = 1e-3,
) -> Transform:
    """Circular RQS bijection on per-coord [-bound, bound].

    Composes the modular wrap with a circular monotonic RQS sharing the
    same `bound`. The spline takes K (= bins) unconstrained derivatives
    and wrap-shares the first onto the last knot (d_0 = d_K, one
    learnable seam slope), so each coordinate map is a C¹ circle
    diffeomorphism with a trainable seam density. With `bound` a (d,)
    array, each coordinate gets its own period 2 * bound_i.

    The wrap is applied on BOTH sides of the spline: the outer wrap is
    the identity on forward outputs (the spline maps into the box) but
    wraps the INVERSE's input before the spline inverse, so both
    directions — and their ladj — are invariant under full-period shifts
    of the input (any angle representative evaluates identically).
    """
    return ComposedTransform(
        CircularShiftTransform(bound=bound),
        MonotonicRQSTransform(*phi, bound=bound, slope=slope, circular=True),
        CircularShiftTransform(bound=bound),
    )


# ──────────────────────────────────────────────────────────────────────
# LinearMixingTransform — lazy 1x1 invertible "conv" for RealNVP / Glow
# ──────────────────────────────────────────────────────────────────────

class LinearMixingTransform(eqx.Module):
    """Lazy unconditional linear mixing on R^d.

    The Glow-style 1x1 invertible "convolution" used to interleave
    cross-coordinate mixing between coupling layers. Two parametric
    families are supported:

      - kind="rotation": orthogonal map R = exp(A - A^T) of the
        skew-symmetric part of a free d x d matrix; log|det| ≡ 0.
      - kind="lu":       PLU-style map L @ U with L lower-triangular
        (diagonal carries the log|det|) and U strict-upper-triangular
        plus identity (unit diagonal).

    Initialised at identity (rotation: A = 0 -> R = exp(0) = I;
    lu: LU = I -> L = I, U = I) so that the layer contributes nothing
    at construction time (no PRNG key needed). Training moves the matrix
    away from identity through normal gradients; `zeros()` returns a new
    module reset to identity.
    """

    weight: Array
    kind: str = eqx.field(static=True)
    features: int = eqx.field(static=True)

    def __init__(self, features: int, kind: str = "rotation") -> None:
        assert kind in ("rotation", "lu"), f"unknown mixing kind {kind!r}"
        self.kind = kind
        self.features = features
        if kind == "rotation":
            self.weight = jnp.zeros((features, features))
        else:  # "lu"
            self.weight = jnp.eye(features)

    def __call__(self) -> Transform:
        if self.kind == "rotation":
            return RotationTransform(self.weight)
        return LULinearTransform(self.weight)

    def zeros(self) -> "LinearMixingTransform":
        """Reset to identity (A = 0 or LU = I). Returns a new module."""
        replacement = (
            jnp.zeros_like(self.weight)
            if self.kind == "rotation"
            else jnp.eye(self.features, dtype=self.weight.dtype)
        )
        return eqx.tree_at(lambda m: m.weight, self, replacement)

"""Bijective transformations used by jflows flows.

Adapted from zuko's transform machinery, restricted to the subset that any
jflows flow actually uses, plus two behavioural tweaks:

  - `MonotonicRQSTransform.bound` may be a per-coordinate `(d,)` array
    (or anything broadcastable to `widths.shape[:-1]`), not just a
    scalar. NSF / NCSF use this so spline knots span
    `[-halfwidth_i, halfwidth_i]` per coordinate without needing an
    affine scaling sandwich.
  - `CircularShiftTransform.bound` accepts the same per-coord array.
  - `MonotonicRQSTransform(..., circular=True)` wrap-shares the first
    derivative onto the last knot (d_0 = d_K, one learnable seam slope)
    for C¹ circular splines with a trainable seam density (used by NCSF;
    Rezende et al., 2020).

JAX design notes:

  - A `Transform` is an `eqx.Module` (an immutable pytree). It is built
    fresh by `flow.t()` from the flow's current parameters — inside a
    traced function this is free, and there is no stale-parameter hazard.
  - Instead of torch's constraint objects, each transform carries integer
    `domain_dim` / `codomain_dim` event dims (0 = scalar-wise,
    1 = vector); `ComposedTransform` uses them to sum log|det J| over
    exactly the right trailing dims.
  - `t.inv` returns an `Inverse` view (or a structural inverse for
    `ComposedTransform` / `FreeFormJacobianTransform`); `t.inv(y)` and
    `t.inv.call_and_ladj(y)` work as in zuko.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from textwrap import indent
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from .numerics import broadcast, rk4_fixed


__all__ = [
    "AdditiveTransform",
    "AutoregressiveTransform",
    "CircularShiftTransform",
    "ComposedTransform",
    "CouplingTransform",
    "DependentTransform",
    "FreeFormJacobianTransform",
    "IdentityTransform",
    "Inverse",
    "LULinearTransform",
    "MonotonicAffineTransform",
    "MonotonicRQSTransform",
    "RotationTransform",
    "Transform",
]


def _sum_rightmost(x: Array, n: int) -> Array:
    """Sum over the last `n` dims (no-op for n = 0)."""
    if n == 0:
        return x
    return x.sum(axis=tuple(range(-n, 0)))


# ──────────────────────────────────────────────────────────────────────
# Transform base + Inverse view
# ──────────────────────────────────────────────────────────────────────

class Transform(eqx.Module):
    """Base bijection: `y = t(x)` with `t.inv`, `log_abs_det_jacobian`,
    and a fused `call_and_ladj(x) -> (y, log|det J|)`.

    Subclasses set `domain_dim` / `codomain_dim` (event dims: 0 for
    scalar-wise transforms, 1 for vector transforms).
    """

    domain_dim: ClassVar[int] = 0
    codomain_dim: ClassVar[int] = 0
    bijective: ClassVar[bool] = True

    def __call__(self, x: Array) -> Array:
        raise NotImplementedError

    def _inverse(self, y: Array) -> Array:
        raise NotImplementedError

    @property
    def inv(self) -> "Transform":
        return Inverse(self)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        y = self(x)
        return y, self.log_abs_det_jacobian(x, y)


class Inverse(Transform):
    """Inverse view of a base transform: `Inverse(t)(y) == t._inverse(y)`."""

    base: Transform
    domain_dim: int = eqx.field(static=True)
    codomain_dim: int = eqx.field(static=True)

    def __init__(self, base: Transform) -> None:
        self.base = base
        self.domain_dim = base.codomain_dim
        self.codomain_dim = base.domain_dim

    def __repr__(self) -> str:
        return f"Inverse({self.base})"

    def __call__(self, y: Array) -> Array:
        return self.base._inverse(y)

    def _inverse(self, x: Array) -> Array:
        return self.base(x)

    @property
    def inv(self) -> Transform:
        return self.base

    def log_abs_det_jacobian(self, y: Array, x: Array) -> Array:
        return -self.base.log_abs_det_jacobian(x, y)

    def call_and_ladj(self, y: Array) -> tuple[Array, Array]:
        x = self.base._inverse(y)
        return x, -self.base.log_abs_det_jacobian(x, y)


# ──────────────────────────────────────────────────────────────────────
# Composed / Dependent / Identity / Additive  — small structural pieces
# ──────────────────────────────────────────────────────────────────────

class ComposedTransform(Transform):
    """Composition f_n ∘ ... ∘ f_0 with fused call_and_ladj."""

    transforms: tuple[Transform, ...]
    domain_dim: int = eqx.field(static=True)
    codomain_dim: int = eqx.field(static=True)

    def __init__(self, *transforms: Transform) -> None:
        assert transforms, "'transforms' cannot be empty"

        event_dim = 0
        for t in reversed(transforms):
            event_dim = t.domain_dim + max(event_dim - t.codomain_dim, 0)
        self.domain_dim = event_dim

        for t in transforms:
            event_dim += t.codomain_dim - t.domain_dim
        self.codomain_dim = event_dim
        self.transforms = tuple(transforms)

    def __repr__(self) -> str:
        lines = [f"({i}): {t}" for i, t in enumerate(self.transforms)]
        body = indent("\n".join(lines), "  ")
        return f"{self.__class__.__name__}(\n{body}\n)"

    def __call__(self, x: Array) -> Array:
        for t in self.transforms:
            x = t(x)
        return x

    @property
    def inv(self) -> Transform:
        return ComposedTransform(*[t.inv for t in reversed(self.transforms)])

    def _inverse(self, y: Array) -> Array:
        for t in reversed(self.transforms):
            y = t.inv(y)
        return y

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        event_dim = self.domain_dim
        acc: Array | float = 0.0
        for t in self.transforms:
            x, ladj = t.call_and_ladj(x)
            acc = acc + _sum_rightmost(ladj, event_dim - t.domain_dim)
            event_dim += t.codomain_dim - t.domain_dim
        return x, acc


class DependentTransform(Transform):
    """Treat the last `reinterpreted` dims of base as dependent
    (log|det J| sums over them)."""

    base: Transform
    reinterpreted: int = eqx.field(static=True)
    domain_dim: int = eqx.field(static=True)
    codomain_dim: int = eqx.field(static=True)

    def __init__(self, base: Transform, reinterpreted: int) -> None:
        self.base = base
        self.reinterpreted = reinterpreted
        self.domain_dim = base.domain_dim + reinterpreted
        self.codomain_dim = base.codomain_dim + reinterpreted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.base}, {self.reinterpreted})"

    def __call__(self, x: Array) -> Array:
        return self.base(x)

    @property
    def inv(self) -> Transform:
        return DependentTransform(self.base.inv, self.reinterpreted)

    def _inverse(self, y: Array) -> Array:
        return self.base.inv(y)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        ladj = self.base.log_abs_det_jacobian(x, y)
        return _sum_rightmost(ladj, self.reinterpreted)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        y, ladj = self.base.call_and_ladj(x)
        return y, _sum_rightmost(ladj, self.reinterpreted)


class IdentityTransform(Transform):
    """f(x) = x."""

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, IdentityTransform)

    __hash__ = object.__hash__

    def __call__(self, x: Array) -> Array:
        return x

    def _inverse(self, y: Array) -> Array:
        return y

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return jnp.zeros_like(x)


class AdditiveTransform(Transform):
    """f(x) = x + shift.

    Used by NSF/NCSF to translate the box [a, b]^d to the per-coord
    centred box [-half, half]^d (and back). shift is typically a (d,)
    array; broadcasting with `x` of shape (..., d) is automatic.
    """

    shift: Array

    def __init__(self, shift: Array) -> None:
        self.shift = shift

    def __call__(self, x: Array) -> Array:
        # buffer semantics: the shift (NSF/NCSF box center) is not trainable,
        # also when the transform itself is the differentiated pytree
        return x + lax.stop_gradient(self.shift)

    def _inverse(self, y: Array) -> Array:
        return y - lax.stop_gradient(self.shift)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return jnp.zeros_like(x)


# ──────────────────────────────────────────────────────────────────────
# MonotonicAffineTransform  — used as default univariate in MAF
# ──────────────────────────────────────────────────────────────────────

class MonotonicAffineTransform(Transform):
    """f(x) = exp(a) * x + b with a clamped to [log(slope), -log(slope)].

    Default univariate transform inside autoregressive / coupling flows.
    """

    shift: Array
    log_scale: Array
    scale: Array

    def __init__(self, shift: Array, scale: Array, slope: float = 1e-3) -> None:
        self.shift = shift
        self.log_scale = scale / (1 + jnp.abs(scale / math.log(slope)))
        self.scale = jnp.exp(self.log_scale)

    def __call__(self, x: Array) -> Array:
        return x * self.scale + self.shift

    def _inverse(self, y: Array) -> Array:
        return (y - self.shift) / self.scale

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return jnp.broadcast_to(self.log_scale, x.shape)


# ──────────────────────────────────────────────────────────────────────
# MonotonicRQSTransform  — knots on per-coord [-bound, bound]
# ──────────────────────────────────────────────────────────────────────

class MonotonicRQSTransform(Transform):
    """Monotonic rational-quadratic spline on per-coord [-bound, bound].

    Reference:
        Neural Spline Flows (Durkan et al., 2019) — https://arxiv.org/abs/1906.04032

    `bound` may be a scalar or an array broadcastable to
    `widths.shape[:-1]`. With a per-coord (d,) array, each coordinate's
    knots span [-bound_i, bound_i] independently — this is what NSF /
    NCSF use to avoid the affine scaling sandwich.

    With `circular=False` (default), the K - 1 unconstrained derivatives
    are padded so both boundary knot slopes equal 1, matching the
    identity tails outside [-bound, bound] (the NSF case). With
    `circular=True`, the K unconstrained derivatives are wrap-shared —
    the first is copied onto the last knot, so d_0 = d_K is a single
    learnable value — giving a C¹ circle diffeomorphism with a trainable
    seam slope (Rezende et al., "Normalizing Flows on Tori and Spheres",
    2020; the NCSF case).

    Arguments:
        widths:      unconstrained bin widths,      shape (..., K)
        heights:     unconstrained bin heights,     shape (..., K)
        derivatives: unconstrained knot slopes,     shape (..., K - 1),
                     or (..., K) when `circular=True`.
        bound:       (co)domain bound; scalar or shape (..., 1) / (..., d).
        slope:       lower-bound on every segment's slope (numeric stability).
        circular:    derivative boundary handling, see above.
    """

    horizontal: Array
    vertical: Array
    derivatives: Array

    def __init__(
        self,
        widths: Array,
        heights: Array,
        derivatives: Array,
        bound: Array | float = 1.0,
        slope: float = 1e-3,
        circular: bool = False,
    ) -> None:
        widths = widths / (1 + jnp.abs(2 * widths / math.log(slope)))
        heights = heights / (1 + jnp.abs(2 * heights / math.log(slope)))
        derivatives = derivatives / (1 + jnp.abs(derivatives / math.log(slope)))

        pad_last = [(0, 0)] * (widths.ndim - 1)
        widths = jnp.pad(jax.nn.softmax(widths, axis=-1), [*pad_last, (1, 0)])
        heights = jnp.pad(jax.nn.softmax(heights, axis=-1), [*pad_last, (1, 0)])
        if circular:
            # d_0 = d_K, one learnable seam slope (C¹ across the seam).
            derivatives = jnp.concatenate([derivatives, derivatives[..., :1]], axis=-1)
        else:
            # d_0 = d_K = 1, matching the identity tails outside the box.
            derivatives = jnp.pad(derivatives, [*pad_last, (1, 1)])

        # Per-coord bound broadcast: append a knot-index dim to `bound`
        # so it lines up with widths/heights' last dim.
        B = bound[..., None] if isinstance(bound, (Array, np.ndarray)) else bound

        self.horizontal = B * (2 * jnp.cumsum(widths, axis=-1) - 1)
        self.vertical = B * (2 * jnp.cumsum(heights, axis=-1) - 1)
        self.derivatives = jnp.exp(derivatives)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bins={self.bins})"

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def bin(self, k: Array) -> tuple[Array, ...]:
        mask = jnp.logical_and(0 <= k, k < self.bins)

        k = k % self.bins
        k0_k1 = jnp.stack((k, k + 1))

        k0_k1, hs, vs, ds = broadcast(
            k0_k1[..., None],
            self.horizontal,
            self.vertical,
            self.derivatives,
            ignore=1,
        )

        x0, x1 = jnp.take_along_axis(hs, k0_k1, axis=-1).squeeze(-1)
        y0, y1 = jnp.take_along_axis(vs, k0_k1, axis=-1).squeeze(-1)
        d0, d1 = jnp.take_along_axis(ds, k0_k1, axis=-1).squeeze(-1)

        s = (y1 - y0) / (x1 - x0)
        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def searchsorted(seq: Array, value: Array) -> Array:
        return jnp.sum(seq < value[..., None], axis=-1)

    def __call__(self, x: Array) -> Array:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (
            s + (d0 + d1 - 2 * s) * z * (1 - z)
        )
        return jnp.where(mask, y, x)

    def _inverse(self, y: Array) -> Array:
        k = self.searchsorted(self.vertical, y) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        y_ = mask * (y - y0)
        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_

        z = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
        x = x0 + z * (x1 - x0)
        return jnp.where(mask, x, y)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        k = self.searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (
            s + (d0 + d1 - 2 * s) * z * (1 - z)
        )

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z)) ** 2
        )
        return jnp.where(mask, y, x), mask * jnp.log(jacobian)


# ──────────────────────────────────────────────────────────────────────
# CircularShiftTransform  — wrap-around on per-coord [-bound, bound]
# ──────────────────────────────────────────────────────────────────────

class CircularShiftTransform(Transform):
    """Circular shift bijection on per-coord [-bound, bound].

    f(x) = ((x + bound) mod 2*bound) - bound

    `bound` may be a scalar or an array broadcastable with the last dim
    of x; with `bound: (d,)`, each coordinate wraps on its own period
    2*bound_i. The bijection is identity-modulo-period on each coordinate.
    """

    bound: Array | float

    def __init__(self, bound: Array | float = 1.0) -> None:
        self.bound = bound

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bound={self.bound})"

    def __call__(self, x: Array) -> Array:
        return jnp.remainder(x + self.bound, 2 * self.bound) - self.bound

    def _inverse(self, y: Array) -> Array:
        return jnp.remainder(y + self.bound, 2 * self.bound) - self.bound

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return jnp.zeros_like(x)


# ──────────────────────────────────────────────────────────────────────
# AutoregressiveTransform / CouplingTransform / FreeFormJacobianTransform
# ──────────────────────────────────────────────────────────────────────

class AutoregressiveTransform(Transform):
    """y_i = f(x_i | x_<i) — autoregressive scheme.

    `meta(x)` returns a univariate Transform whose parameters depend
    autoregressively on x via a masked MLP. The inverse runs `passes`
    fixed-point sweeps under a `lax.scan`.
    """

    meta: Callable[[Array], Transform]
    passes: int = eqx.field(static=True)

    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __init__(self, meta: Callable[[Array], Transform], passes: int) -> None:
        self.meta = meta
        self.passes = passes

    def __call__(self, x: Array) -> Array:
        return self.meta(x)(x)

    def _inverse(self, y: Array) -> Array:
        def body(x, _):
            return self.meta(x).inv(y), None

        x, _ = lax.scan(body, jnp.zeros_like(y), None, length=self.passes)
        return x

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return self.meta(x).log_abs_det_jacobian(x, y)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        return self.meta(x).call_and_ladj(x)


class CouplingTransform(Transform):
    """y_a = x_a, y_b = f(x_b | x_a) — coupling scheme.

    mask: boolean vector; True = "kept" (x_a), False = "transformed" (x_b).
    The mask must be a concrete (numpy) array — it fixes the transform's
    structure, so it is stored as static index tuples.
    """

    meta: Callable[[Array], Transform]
    idx_a: tuple[int, ...] = eqx.field(static=True)
    idx_b: tuple[int, ...] = eqx.field(static=True)

    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __init__(self, meta: Callable[[Array], Transform], mask) -> None:
        mask = np.asarray(mask, dtype=bool)
        self.meta = meta
        self.idx_a = tuple(np.nonzero(mask)[0].tolist())
        self.idx_b = tuple(np.nonzero(~mask)[0].tolist())

    def split(self, x: Array) -> tuple[Array, Array]:
        idx_a = jnp.asarray(self.idx_a, dtype=jnp.int32)
        idx_b = jnp.asarray(self.idx_b, dtype=jnp.int32)
        return x[..., idx_a], x[..., idx_b]

    def merge(self, x_a: Array, x_b: Array, shape: tuple[int, ...]) -> Array:
        x = jnp.zeros(shape, dtype=x_a.dtype)
        x = x.at[..., jnp.asarray(self.idx_a, dtype=jnp.int32)].set(x_a)
        x = x.at[..., jnp.asarray(self.idx_b, dtype=jnp.int32)].set(x_b)
        return x

    def __call__(self, x: Array) -> Array:
        x_a, x_b = self.split(x)
        y_b = self.meta(x_a)(x_b)
        return self.merge(x_a, y_b, x.shape)

    def _inverse(self, y: Array) -> Array:
        y_a, y_b = self.split(y)
        x_b = self.meta(y_a).inv(y_b)
        return self.merge(y_a, x_b, y.shape)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        x_a, x_b = self.split(x)
        _, y_b = self.split(y)
        return self.meta(x_a).log_abs_det_jacobian(x_b, y_b)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        x_a, x_b = self.split(x)
        y_b, ladj = self.meta(x_a).call_and_ladj(x_b)
        y = self.merge(x_a, y_b, x.shape)
        return y, ladj


class FreeFormJacobianTransform(Transform):
    """FFJORD continuous-time bijection: dx/dt = f_phi(t, x).

    `exact=True`  → exact log|det J| via an O(d) JVP sweep per drift eval.
    `exact=False` → Hutchinson trace estimator (stochastic); requires a
        PRNG `key` — the probe noise is drawn once per `call_and_ladj`
        from that key (rebuild the transform with a fresh key for a new
        probe).

    Integration is fixed-step RK4 (`rk4_fixed`) under `lax.scan`;
    gradients to the drift's parameters flow through the scan natively.

    Arguments:
        f:     drift `f(t, x)` returning `dx/dt` with the shape of `x`.
        t0:    trajectory start time.
        t1:    trajectory end time.
        nt:    number of fixed RK4 steps.
        exact: if True, the log-det uses the exact O(d) Jacobian-trace
            sweep; if False, a Hutchinson stochastic estimate.
        key:   PRNG key for the Hutchinson probe (exact=False only).
    """

    f: Callable[[Array, Array], Array]
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    nt: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    key: Array | None

    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __init__(
        self,
        f: Callable[[Array, Array], Array],
        t0: float = 0.0,
        t1: float = 1.0,
        nt: int = 8,
        exact: bool = True,
        key: Array | None = None,
    ) -> None:
        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.nt = nt
        self.exact = exact
        self.key = key

    def __call__(self, x: Array) -> Array:
        return rk4_fixed(self.f, x, self.t0, self.t1, self.nt)

    @property
    def inv(self) -> Transform:
        return FreeFormJacobianTransform(
            f=self.f,
            t0=self.t1,
            t1=self.t0,
            nt=self.nt,
            exact=self.exact,
            key=self.key,
        )

    def _inverse(self, y: Array) -> Array:
        return rk4_fixed(self.f, y, self.t1, self.t0, self.nt)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        d = x.shape[-1]

        if self.exact:
            eye = jnp.eye(d, dtype=x.dtype)
        else:
            if self.key is None:
                raise ValueError(
                    "FreeFormJacobianTransform(exact=False) needs a PRNG `key` "
                    "for the Hutchinson probe."
                )
            # re-randomize the probe per call (content-hashed key, grad-severed
            # by the int cast) so the Hutchinson trace is unbiased across
            # training batches instead of frozen to the construction-time key
            seed = (jnp.abs(x).sum() * 1e3).astype(jnp.int32)
            eps = jax.random.normal(
                jax.random.fold_in(self.key, seed), x.shape, dtype=x.dtype
            )

        def f_aug(t: Array, z: Array) -> Array:
            xs = z[..., :d]
            if self.exact:
                dx = self.f(t, xs)
                tangents = jax.vmap(
                    lambda e: jax.jvp(
                        lambda u: self.f(t, u),
                        (xs,),
                        (jnp.broadcast_to(e, xs.shape),),
                    )[1]
                )(eye)
                trace = jnp.einsum("i...i->...", tangents)
            else:
                dx, vjp_fn = jax.vjp(lambda u: self.f(t, u), xs)
                epsjp = vjp_fn(eps)[0]
                trace = (epsjp * eps).sum(axis=-1)
            return jnp.concatenate([dx, trace[..., None]], axis=-1)

        z0 = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], 1), dtype=x.dtype)], axis=-1)
        zT = rk4_fixed(f_aug, z0, self.t0, self.t1, self.nt)
        return zT[..., :d], zT[..., d]


# ──────────────────────────────────────────────────────────────────────
# Linear mixing transforms on R^d (Glow-style 1x1 invertible "conv")
# ──────────────────────────────────────────────────────────────────────

class RotationTransform(Transform):
    r"""Rotation `f(x) = R x` with `R = exp(A - A^T)` orthogonal.

    Because `A - A^T` is skew-symmetric, the matrix exponential is
    orthogonal, so the transform is volume-preserving and its
    log-abs-determinant is identically zero.

    Arguments:
        A: square matrix `A`, with shape `(*, D, D)`.
    """

    A: Array

    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __init__(self, A: Array) -> None:
        self.A = A

    @property
    def R(self) -> Array:
        return jax.scipy.linalg.expm(self.A - self.A.mT)

    def __call__(self, x: Array) -> Array:
        return jnp.einsum("...ij,...j->...i", self.R, x)

    def _inverse(self, y: Array) -> Array:
        return jnp.einsum("...ij,...i->...j", self.R, y)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        return jnp.zeros_like(x[..., 0])

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        # Compute R once and share between y and ladj — this is the path
        # ComposedTransform.call_and_ladj takes during training, so the
        # matrix exponential runs exactly once per forward step.
        R = jax.scipy.linalg.expm(self.A - self.A.mT)
        y = jnp.einsum("...ij,...j->...i", R, x)
        return y, jnp.zeros_like(x[..., 0])


class LULinearTransform(Transform):
    r"""Linear map `f(x) = L U x` with LU decomposition.

    `L` is the lower-triangular part of the input matrix (diagonal
    included, providing `log|det|`); `U` is the strict upper-triangular
    part plus the identity (unit diagonal), so the forward is a single
    `L @ U @ x` and the log-abs-determinant is `sum(log|diag(L)|)`.

    Numerical safety: `log|det| = sum log|diag(L)|` diverges to `-inf`
    if any diagonal entry crosses zero during training (nothing in the
    parameterisation keeps `diag(L)` away from the origin). The log is
    therefore computed on `|diag|` clamped to a small floor `_LADJ_EPS`,
    so a near-singular `L` produces a large-but-finite negative ladj
    with a saturated zero gradient on the affected entries — preventing
    the loss from exploding to NaN.

    Arguments:
        LU: matrix whose lower / upper triangular parts hold the non-zero
            elements of `L` and `U`, with shape `(D, D)`.
    """

    LU: Array

    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    # Floor for |diag(L)| inside the log to avoid -inf / NaN ladj when the
    # learnable LU diagonal crosses zero. log(1e-12) ≈ -27.6, so the
    # per-entry penalty stays large but finite; gradient w.r.t. the
    # diagonal saturates to 0 in the clamped region.
    _LADJ_EPS: ClassVar[float] = 1e-12

    def __init__(self, LU: Array) -> None:
        self.LU = LU

    @property
    def L(self) -> Array:
        return jnp.tril(self.LU)

    @property
    def U(self) -> Array:
        I = jnp.eye(self.LU.shape[-1], dtype=self.LU.dtype)
        return jnp.triu(self.LU, k=1) + I

    def __call__(self, x: Array) -> Array:
        return jnp.einsum("...ij,...j->...i", self.L @ self.U, x)

    def _inverse(self, y: Array) -> Array:
        d = self.LU.shape[-1]
        flat = y.reshape(-1, d).T
        z = jax.scipy.linalg.solve_triangular(self.L, flat, lower=True)
        x = jax.scipy.linalg.solve_triangular(self.U, z, lower=False, unit_diagonal=True)
        return x.T.reshape(y.shape)

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        diag = jnp.diagonal(self.L, axis1=-2, axis2=-1)
        ladj = jnp.log(jnp.maximum(jnp.abs(diag), self._LADJ_EPS)).sum(axis=-1)
        return jnp.broadcast_to(ladj, x[..., 0].shape)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        # Compute L and U once and share — ComposedTransform.call_and_ladj
        # routes through here on the training hot path.
        I = jnp.eye(self.LU.shape[-1], dtype=self.LU.dtype)
        L = jnp.tril(self.LU)
        U = jnp.triu(self.LU, k=1) + I
        y = jnp.einsum("...ij,...j->...i", L @ U, x)
        diag = jnp.diagonal(L, axis1=-2, axis2=-1)
        ladj = jnp.log(jnp.maximum(jnp.abs(diag), self._LADJ_EPS)).sum(axis=-1)
        return y, jnp.broadcast_to(ladj, x[..., 0].shape)

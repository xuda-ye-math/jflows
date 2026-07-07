"""Pure array utilities used by jflows's transforms and flows.

Adapted from `zuko/utils.py`:
    - Partial: eqx.Module wrapper of functools.partial
    - bisection: implicit-grad bisection root finder (lax.custom_root)
    - broadcast: jnp.broadcast_to over the leading dims
    - gauss_legendre: n-point quadrature on [a, b]
    - rk4_fixed: fixed-step RK4 ODE integrator (lax.scan)
    - unpack: split a packed array along its last dim by shapes

This module is independent of distributions, flows, conditioning, etc.

JAX note: the `phi` argument of `bisection` / `gauss_legendre` is accepted
for signature parity but is unnecessary — JAX differentiates through the
closed-over parameters of `f` automatically (via `lax.custom_root`'s
closure conversion, and native autodiff respectively).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from itertools import accumulate
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax


__all__ = [
    "Partial",
    "bisection",
    "broadcast",
    "gauss_legendre",
    "rk4_fixed",
    "unpack",
]


# ──────────────────────────────────────────────────────────────────────
# Partial — eqx.Module-aware functools.partial
# ──────────────────────────────────────────────────────────────────────

class Partial(eqx.Module):
    """An eqx.Module-aware version of functools.partial.

    Array args stay pytree leaves, so they are traced/differentiated like
    any other module field; submodules in `f` ride along as sub-pytrees.
    Calling returns f(*args, *extra, **kwargs, **extra).

    The `buffer` flag is accepted for signature parity and ignored — JAX
    draws no parameter/buffer distinction (use `jax.lax.stop_gradient`
    at the call site to freeze an argument).
    """

    f: Callable
    args: tuple
    kwargs: dict

    def __init__(self, f: Callable, /, *args, buffer: bool = False, **kwargs) -> None:
        del buffer  # signature parity only
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs) -> Any:
        return self.f(*self.args, *args, **self.kwargs, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Bisection — implicit-grad bisection root finder
# ──────────────────────────────────────────────────────────────────────

def bisection(
    f: Callable[[Array], Array],
    y: Array,
    a: float | Array,
    b: float | Array,
    n: int = 16,
    phi: Iterable[Array] = (),
) -> Array:
    """Bisection root finder for `f(x) = y` with implicit-grad backward.

    `f` must be elementwise monotonically increasing on [a, b] (the search
    keeps the sub-interval where `f(c) < y`). Gradients w.r.t. `y` and any
    parameters closed over by `f` come from the implicit function theorem
    (`lax.custom_root`), not from differentiating the `n` halving steps.
    """
    del phi  # signature parity only; closures are differentiated natively

    a = jnp.asarray(a, dtype=y.dtype)
    b = jnp.asarray(b, dtype=y.dtype)

    def g(x: Array) -> Array:
        return f(x) - y

    def solve(g_: Callable[[Array], Array], x0: Array) -> Array:
        lo = jnp.broadcast_to(a, x0.shape)
        hi = jnp.broadcast_to(b, x0.shape)

        def body(_, lo_hi):
            lo, hi = lo_hi
            c = (lo + hi) / 2
            mask = g_(c) < 0
            return jnp.where(mask, c, lo), jnp.where(mask, hi, c)

        lo, hi = lax.fori_loop(0, n, body, (lo, hi))
        return (lo + hi) / 2

    def tangent_solve(g_lin: Callable[[Array], Array], t: Array) -> Array:
        # g is elementwise, so its linearization is diagonal: J = g_lin(1).
        return t / g_lin(jnp.ones_like(t))

    x0 = jnp.broadcast_to((a + b) / 2, y.shape)
    return lax.custom_root(g, x0, solve, tangent_solve)


# ──────────────────────────────────────────────────────────────────────
# broadcast — jnp.broadcast_to over the leading dims
# ──────────────────────────────────────────────────────────────────────

def broadcast(*arrays: Array, ignore: int | Sequence[int] = 0) -> list[Array]:
    """Broadcast arrays over leading dims; keep the last `ignore` dims intact."""

    if isinstance(ignore, int):
        ignore = [ignore] * len(arrays)

    dims = [a.ndim - i for a, i in zip(arrays, ignore, strict=True)]
    common = jnp.broadcast_shapes(
        *(a.shape[:i] for a, i in zip(arrays, dims, strict=True))
    )

    return [
        jnp.broadcast_to(a, common + a.shape[i:])
        for a, i in zip(arrays, dims, strict=True)
    ]


# ──────────────────────────────────────────────────────────────────────
# Gauss–Legendre — n-point quadrature on [a, b]
# ──────────────────────────────────────────────────────────────────────

@cache
def _leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def gauss_legendre(
    f: Callable[[Array], Array],
    a: Array,
    b: Array,
    n: int = 3,
    phi: Iterable[Array] = (),
) -> Array:
    """n-point Gauss-Legendre quadrature of f over [a, b].

    Gradients (w.r.t. `a`, `b`, and anything `f` closes over) flow through
    the quadrature rule by native autodiff — exact whenever the rule itself
    is exact (polynomials of degree <= 2n - 1).
    """
    del phi  # signature parity only

    nodes, weights = _leggauss(n)
    nodes = jnp.asarray(nodes, dtype=a.dtype)
    weights = jnp.asarray(weights, dtype=a.dtype)

    xs = a[..., None] + (b - a)[..., None] * nodes
    xs = jnp.moveaxis(xs, -1, 0)
    return (b - a) * jnp.tensordot(weights, f(xs), axes=1)


# ──────────────────────────────────────────────────────────────────────
# ODE solver — fixed-step RK4 (lax.scan)
# ──────────────────────────────────────────────────────────────────────

def rk4_fixed(
    f: Callable[[Array, Array], Array],
    x: Array,
    t0: float | Array,
    t1: float | Array,
    nt: int,
) -> Array:
    """Integrate dx/dt = f(t, x) from t0 to t1 with `nt` fixed RK4 steps.

    The only integrator in jflows. It takes equal steps under a `lax.scan`,
    so gradients through the trajectory come from scan's native reverse
    mode (equivalent to unrolling). Intended for short trajectories
    (nt ~ 4-24): a deterministic flop budget, one trace regardless of `nt`.
    Continuous flows (`CNF`/FFJORD, `OTFlow`) pack their augmented
    `(x, ladj, ...)` state into a single array and integrate it here.

    The sign of `(t1 - t0)` sets the direction: backward integration
    (t1 < t0) just makes the step `h` negative.
    """
    t0 = jnp.asarray(t0, dtype=x.dtype)
    t1 = jnp.asarray(t1, dtype=x.dtype)
    h = (t1 - t0) / nt

    def step(carry, _):
        t, x = carry
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        return (t + h, x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)), None

    (_, x), _ = lax.scan(step, (t0, x), None, length=nt)
    return x


# ──────────────────────────────────────────────────────────────────────
# unpack — split a packed array along its last dim by shapes
# ──────────────────────────────────────────────────────────────────────

def unpack(x: Array, shapes: Sequence[tuple[int, ...]]) -> Sequence[Array]:
    """Inverse of `jnp.concatenate([a.reshape(-1) for a in arrays])` given shapes."""
    sizes = [math.prod(s) for s in shapes]
    splits = list(accumulate(sizes))[:-1]
    parts = jnp.split(x, splits, axis=-1)
    return tuple(
        p.reshape(p.shape[:-1] + tuple(s)) if s else p.reshape(p.shape[:-1])
        for p, s in zip(parts, shapes, strict=True)
    )

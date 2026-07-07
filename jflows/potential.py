"""Energy potentials and base distributions for jflows.

Every potential `U` is the energy of the unnormalized density
`exp(-U)`; the concrete classes carry the `Nlog_` prefix (negative log)
to mark the translation from distribution language to potential
language:

    Potential             — abstract base; __call__(x) -> U(x), .grad(x)
    potential_from        — wrap a callable as a Potential instance
    linear_combination    — flat linear combination sum_k c_k * U_k
    Nlog_Uniform          — U = -log of a uniform box density (constant)
    Nlog_Gaussian         — U = -log of a diagonal Gaussian density
    Nlog_Gaussian_Mixture — U = -log of a diagonal Gaussian mixture density

Potentials form a vector space: `0.5 * u1 + 0.5 * u2`, `u1 - u2`, `-u`,
`u / 2`, and `sum([v1, v2])` all build flat, identity-merged linear
combinations (see `linear_combination`).

All energies are defined up to an additive constant. There are no
temperature arguments: temperature scaling, where needed, is expressed
through the potential itself.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


__all__ = [
    "Nlog_Gaussian",
    "Nlog_Gaussian_Mixture",
    "Nlog_Uniform",
    "Potential",
    "linear_combination",
    "potential_from",
]


# ──────────────────────────────────────────────────────────────────────
# Potential — abstract base with batched gradient
# ──────────────────────────────────────────────────────────────────────

class Potential(eqx.Module):
    """
    Generic Potential class. __call__() computes the potential function.

    .grad(x) returns the batched gradient dU/dx via vmap(grad(.)); it is
    always available (no enabling step) and jit-compatible — wrap the hot
    loop (MALA accept/reject, Langevin steps, importance-sampling
    reweighting) in jax.jit at the call site and XLA fuses the whole body:

        u = Nlog_Gaussian(mean, variance)
        v = u(x)        # [N]
        g = u.grad(x)   # [N, d]
    """

    def __call__(self, x: Array) -> Array:
        """
        Input:
            x: Array [N, d]
        Output:
            U(x): Array [N]
        """
        raise NotImplementedError

    def grad(self, x: Array) -> Array:
        """
        Input:
            x: Array [N, d]
        Output:
            grad U(x): Array [N, d]
        """
        single = lambda xi: self(xi[None])[0]  # [d] -> scalar
        return jax.vmap(jax.grad(single))(x)

    # ── vector-space algebra: c * U, U + V, U - V, -U, U / c, sum([...]) ──
    # Every expression funnels into `linear_combination`, which keeps the
    # term list flat and merges repeated instances (see its docstring).

    def __add__(self, other) -> "Potential":
        if isinstance(other, Potential):
            return linear_combination([self, other], [1.0, 1.0])
        return NotImplemented

    def __radd__(self, other) -> "Potential":
        if isinstance(other, (int, float)) and other == 0:  # builtin sum()
            return self
        return NotImplemented

    def __sub__(self, other) -> "Potential":
        if isinstance(other, Potential):
            return linear_combination([self, other], [1.0, -1.0])
        return NotImplemented

    def __mul__(self, c) -> "Potential":
        if isinstance(c, Potential):
            return NotImplemented
        return linear_combination([self], [c])

    __rmul__ = __mul__

    def __neg__(self) -> "Potential":
        return linear_combination([self], [-1.0])

    def __truediv__(self, c) -> "Potential":
        if isinstance(c, Potential):
            return NotImplemented
        return linear_combination([self], [1.0 / c])


# ──────────────────────────────────────────────────────────────────────
# Functional wrappers — turn a plain callable into a Potential instance
# ──────────────────────────────────────────────────────────────────────

class _Function_Potential(Potential):
    fn: Callable[[Array], Array] = eqx.field(static=True)

    def __call__(self, x: Array) -> Array:
        return self.fn(x)


def potential_from(fn: Callable[[Array], Array]) -> Potential:
    """Wrap a stateless callable `(x: Array) -> Array` as a ready-to-use
    `Potential` *instance* — like writing the subclass by hand and
    instantiating it, in one line.

    Returns the instance directly (lowercase-factory convention; the
    name is lowercase because the return is an instance, not a class).
    The instance supports the full toolchain (`.grad(x)`, jit, vmap) —
    there just won't be any state because `fn` is a plain function.

    Example:
        def myforward(x):
            # x: [N, d] -> [N]   (batched for efficiency)
            return 0.5 * (x ** 2).sum(-1) + 2 * jnp.cos(x[:, 0])

        u = potential_from(myforward)   # instance, ready to use

    For potentials that carry state (physical constants, arrays, …),
    subclass `Potential` directly instead.
    """
    return _Function_Potential(fn)


# ──────────────────────────────────────────────────────────────────────
# Compositional — linear combinations of potentials (annealing bridges)
# ──────────────────────────────────────────────────────────────────────

class _Linear_Combination(Potential):
    """
    Flat linear combination of potentials:
        U(x) = sum_k coeffs[k] * terms[k](x).

    Built by `linear_combination` (or the arithmetic operators on
    `Potential`) — never construct directly. `terms` is always flat and
    identity-merged; `coeffs` is a `(n,)` array leaf, so retuning the
    coefficients preserves the pytree structure and jitted consumers do
    not recompile.
    """

    terms: tuple[Potential, ...]
    coeffs: Array

    def __init__(self, terms: tuple[Potential, ...], coeffs: Array) -> None:
        self.terms = tuple(terms)
        self.coeffs = coeffs

    def __call__(self, x: Array) -> Array:
        """
        Input:
            x: Array [N, d]
        Output:
            U(x): Array [N]
        """
        out = self.coeffs[0] * self.terms[0](x)
        for k in range(1, len(self.terms)):
            out = out + self.coeffs[k] * self.terms[k](x)
        return out

    def grad(self, x: Array) -> Array:
        """
        Linearity: grad U(x) = sum_k coeffs[k] * grad U_k(x), reusing each
        child's own `grad`.

        Input:
            x: Array [N, d]
        Output:
            grad U(x): Array [N, d]
        """
        out = self.coeffs[0] * self.terms[0].grad(x)
        for k in range(1, len(self.terms)):
            out = out + self.coeffs[k] * self.terms[k].grad(x)
        return out


def linear_combination(
    potentials: list[Potential] | tuple[Potential, ...],
    coeffs: list[float] | tuple[float, ...] | Array | None = None,
) -> Potential:
    """Build the flat linear combination `U(x) = sum_k c_k * U_k(x)`.

    Lowercase-factory convention: returns a `Potential` instance. The
    same algebra is available through operators on any `Potential`:

        0.5 * u1 + 0.5 * u2      u1 - u2      -u      u / 2      sum([v1, v2])

    Potentials form a vector space over their INSTANCES: nested
    combinations are absorbed (never wrapped), and repeated instances —
    the same Python object reached through any number of combinations —
    are merged with summed coefficients:

        V1 = 0.5 * U1 + 0.5 * U2
        V2 = 0.5 * U2 + 0.5 * U3
        W  = V1 + V2      # terms (U1, U2, U3), coeffs (0.5, 1.0, 0.5)

    Merging is by object identity (`is`), not by parameter equality —
    two structurally identical but distinct instances stay separate
    terms. Build combinations eagerly (outside jit) when merging
    matters: pytree copies made by JAX transformations break identity,
    so combining two combinations that already crossed a jit/grad
    boundary and share a term will not merge — the result is still
    numerically correct, it just carries redundant terms.

    Input:
        potentials: list/tuple of N Potential instances (N >= 1).
        coeffs:     list/tuple of N floats or a 1-d array of shape [N]
                    holding the matching coefficients; entries may be
                    traced (building a bridge `(1 - c) * U0 + c * U1`
                    inside a jitted step is fine). If None (default),
                    defaults to a uniform 1/N on each potential, i.e.
                    the plain average U(x) = (1/N) * sum_k U_k(x).

    Annealing bridges: retune by rebuilding
    `linear_combination([u0, u1], [1 - c, c])` per rung, or replace the
    `.coeffs` leaf via `eqx.tree_at` with an array of the same dtype
    (plain `jnp.asarray([...])` matches — coefficients are stored with
    the default float dtype). Either way the pytree leaves keep the same
    shape/dtype, so jitted functions of the combination do not recompile.

    Gradients and jit: for a fixed target, keep the combination out of
    the differentiated argument (standard eqx filtering) — its `coeffs`
    and child parameters are ordinary array leaves and would receive
    gradients if included. Combinations are unhashable (array leaves):
    pass them as pytree arguments or closure constants, never as jit
    static arguments.
    """
    potentials = list(potentials)
    assert len(potentials) >= 1, "linear_combination needs at least one term"
    if coeffs is None:
        coeffs = [1.0 / len(potentials)] * len(potentials)
    else:
        coeffs = list(coeffs)
    assert len(potentials) == len(coeffs), (
        f"potentials ({len(potentials)}) and coeffs ({len(coeffs)}) "
        f"must have the same length"
    )

    order: list[int] = []                # first-appearance order of ids
    acc: dict[int, list] = {}            # id -> [potential, coefficient]

    def absorb(U: Potential, c) -> None:
        if isinstance(U, _Linear_Combination):
            for k, child in enumerate(U.terms):
                absorb(child, c * U.coeffs[k])
        else:
            key = id(U)
            if key in acc:
                acc[key][1] = acc[key][1] + c
            else:
                acc[key] = [U, c]
                order.append(key)

    for U, c in zip(potentials, coeffs, strict=True):
        absorb(U, c)

    terms = tuple(acc[key][0] for key in order)
    # Canonical coefficient aval (strong, default float dtype): the same
    # combination rebuilt from floats, arrays, or tracers always carries
    # identical pytree leaves, so jitted consumers never retrace on a retune.
    cs = jnp.stack([jnp.asarray(acc[key][1]) for key in order])
    cs = cs.astype(jnp.result_type(float))
    assert cs.ndim == 1, f"coefficients must be scalars, got coeffs shape {cs.shape}"
    return _Linear_Combination(terms, cs)


# ──────────────────────────────────────────────────────────────────────
# Concrete potentials — Nlog_Uniform, Nlog_Gaussian, Nlog_Gaussian_Mixture
# ──────────────────────────────────────────────────────────────────────

class Nlog_Uniform(Potential):
    """
    Negative log of the uniform density on the box [a, b]: a constant
    potential (zero, up to the additive normalisation constant).
    """

    a: Array
    b: Array
    d: int = eqx.field(static=True)

    def __init__(
        self,
        a: Array | list[float],
        b: Array | list[float],
    ) -> None:
        """
        Input:
            a: Array [d] or list[float]   lower bounds of the rectangle
            b: Array [d] or list[float]   upper bounds of the rectangle
        """
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        assert a.shape == b.shape
        self.a = a
        self.b = b
        self.d = a.shape[0]

    def __call__(self, x: Array) -> Array:
        """
        Input:
            x: Array [N, d]
        Output:
            U(x): Array [N]
        """
        return jnp.zeros(x.shape[:-1], dtype=x.dtype)

    def samples(self, key: Array, N: int) -> Array:
        """
        Generate N independent samples in the rectangle region [a, b].
        Input:
            key: PRNG key
            N:   int   number of samples
        Output:
            x: Array [N, d]
        """
        u = jax.random.uniform(key, (N, self.d), dtype=self.a.dtype)
        return self.a + (self.b - self.a) * u


class Nlog_Gaussian(Potential):
    """
    Negative log of a diagonal Gaussian density (up to an additive
    constant):
        U(x) = 0.5 * sum_i (x_i - mean_i)^2 / variance_i
    """

    mean: Array
    variance: Array
    d: int = eqx.field(static=True)

    def __init__(
        self,
        mean: Array | list[float],
        variance: Array | list[float],
    ) -> None:
        """
        Input:
            mean:     Array [d] or list[float]   per-coordinate mean
            variance: Array [d] or list[float]   per-coordinate variance (positive)
        """
        mean = jnp.asarray(mean)
        variance = jnp.asarray(variance)
        assert mean.shape == variance.shape
        self.mean = mean
        self.variance = variance
        self.d = mean.shape[0]

    def __call__(self, x: Array) -> Array:
        """
        Input:
            x: Array [N, d]
        Output:
            U(x): Array [N]
        """
        return 0.5 * ((x - self.mean) ** 2 / self.variance).sum(axis=-1)

    def samples(self, key: Array, N: int) -> Array:
        """
        Generate N independent samples from the diagonal Gaussian
        exp(-U(x)).
        Input:
            key: PRNG key
            N:   int   number of samples
        Output:
            x: Array [N, d]
        """
        z = jax.random.normal(key, (N, self.d), dtype=self.mean.dtype)
        return self.mean + jnp.sqrt(self.variance) * z


class Nlog_Gaussian_Mixture(Potential):
    """
    Negative log of a diagonal Gaussian mixture density with K
    components. The unnormalized density is
        mu(x) propto sum_k w_k * N(x | mean_k, diag(variance_k)),
    and the potential U(x) = -log mu(x) (up to an additive constant).
    
    Memory: each energy/gradient evaluation materializes an
    [N, K, d] difference tensor (chunk the batch externally at large K).
    """

    log_weights: Array
    mean: Array
    variance: Array
    K: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(
        self,
        weights: Array | list[float],
        mean: Array | list[list[float]],
        variance: Array | list[list[float]],
    ) -> None:
        """
        Input:
            weights:  Array [K] or list[float]             mixture weights (non-negative, not required to be normalized)
            mean:     Array [K, d] or list[list[float]]    per-component, per-coordinate mean
            variance: Array [K, d] or list[list[float]]    per-component, per-coordinate variance (positive)
        """
        weights = jnp.asarray(weights)
        mean = jnp.asarray(mean)
        variance = jnp.asarray(variance)
        assert weights.ndim == 1 and mean.ndim == 2 and variance.ndim == 2
        assert mean.shape == variance.shape
        assert weights.shape[0] == mean.shape[0]
        log_w = jnp.log(weights)
        self.log_weights = log_w - jax.scipy.special.logsumexp(log_w, axis=0)  # normalized log-weights
        self.mean = mean
        self.variance = variance
        self.K = mean.shape[0]
        self.d = mean.shape[1]

    def __call__(self, x: Array) -> Array:
        """
        Input:
            x: Array [N, d]
        Output:
            U(x): Array [N]
        """
        diff = x[..., None, :] - self.mean  # [N, K, d]
        log_comp = (
            -0.5 * (diff**2 / self.variance).sum(axis=-1)
            - 0.5 * jnp.log(self.variance).sum(axis=-1)
        )  # [N, K]
        return -jax.scipy.special.logsumexp(self.log_weights + log_comp, axis=-1)  # [N]

    def samples(self, key: Array, N: int) -> Array:
        """
        Generate N independent samples from the diagonal Gaussian mixture.
        Input:
            key: PRNG key
            N:   int   number of samples
        Output:
            x: Array [N, d]
        """
        kidx, kz = jax.random.split(key)
        idx = jax.random.categorical(kidx, self.log_weights, shape=(N,))  # [N]
        z = jax.random.normal(kz, (N, self.d), dtype=self.mean.dtype)
        return self.mean[idx] + jnp.sqrt(self.variance[idx]) * z

"""Unconditional normalizing flows for energy-based sampling.

Public API:
    Flow              — abstract base; subclasses implement .t() -> ComposedTransform
    NSF               — Neural Spline Flow on [a, b]^d (translation-sandwiched RQS)
    NCSF              — Neural Circular Spline Flow on [a, b]^d (periodic per coord)
    CNF               — Continuous Normalizing Flow on R^d (FFJORD)
    OTFlow            — Optimal-transport continuous flow on R^d (closed-form trace)
    RealNVP           — affine-coupling flow on R^d
    ComposedTransform — re-exported from .core.transforms

All flows assume context = 0, i.e. one fixed target. NSF and NCSF
parameterise their inner spline on per-coordinate
`[-halfwidth_i, halfwidth_i]` and wrap it with an additive
translation by ±center_i (no scaling), so the box-bound geometry
[a, b]^d is honoured without distorting the conditioner's dynamic
range.

JAX conventions:
    - every constructor takes a PRNG `key` as its first argument;
    - flows are immutable pytrees — `zeros()` returns a NEW flow;
    - `activation` is a callable (e.g. `jax.nn.silu`), not a class.
"""

from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from .core.flows import (
    CircularRQSTransform,
    FFJTransform,
    GeneralCouplingTransform,
    LinearMixingTransform,
    MaskedAutoregressiveTransform,
    OTFlowLazy,
)
from .core.transforms import (
    AdditiveTransform,
    ComposedTransform,
    MonotonicRQSTransform,
)


__all__ = ["CNF", "ComposedTransform", "Flow", "NCSF", "NSF", "OTFlow", "RealNVP"]


class Flow(eqx.Module):
    """Abstract base class for every normalizing flow in jflows.

    Subclasses are eqx.Modules (immutable pytrees — train them with the
    packed drivers in `jflows.train`, or `eqx.filter_grad` / optax-style
    updates) and must implement:

        def t(self) -> ComposedTransform: ...

    High-level usage goes through the flow itself — the `type` argument
    of the losses / importance weights / AIS names the direction the
    flow's transform acts in:

        y       = flow(x)                 # forward map
        y, ladj = flow.call_and_ladj(x)   # forward map & log|det J|
        x       = flow.inv(y)             # inverse map
        x, ladj = flow.inv_and_ladj(y)    # inverse map & its log|det J|

    `t()` is the advanced composition layer: it returns the underlying
    `ComposedTransform` for chaining transforms, custom pipelines, and
    the core machinery. Building it inside a jit-traced function is
    free, and the four methods above are thin delegations to it.
    """

    @abstractmethod
    def t(self) -> ComposedTransform: ...

    def __call__(self, x: Array) -> Array:
        """Forward map y (the transform's native direction)."""
        return self.t()(x)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        """Forward map and its log|det J|: (y, ladj)."""
        return self.t().call_and_ladj(x)

    def inv(self, y: Array) -> Array:
        """Inverse map x (pre-image of y)."""
        return self.t().inv(y)

    def inv_and_ladj(self, y: Array) -> tuple[Array, Array]:
        """Inverse map and its log|det J|: (x, ladj)."""
        return self.t().inv.call_and_ladj(y)


def _make_orders(key: Array, d: int, transforms: int, randmask: bool) -> list[np.ndarray]:
    """Per-layer feature orderings: fresh permutations (randmask) or
    arange / reversed-arange alternation."""
    if randmask:
        keys = jax.random.split(key, transforms)
        return [np.asarray(jax.random.permutation(k, d)) for k in keys]
    return [
        np.arange(d) if i % 2 == 0 else np.arange(d)[::-1]
        for i in range(transforms)
    ]


# ──────────────────────────────────────────────────────────────────────
# NSF — Neural Spline Flow on [a, b]^d
# ──────────────────────────────────────────────────────────────────────

class NSF(Flow):
    """Neural Spline Flow on [a_1, b_1] x ... x [a_d, b_d].

    The MAF-RQS conditioner runs on the per-coord centred box
    [-halfwidth_i, halfwidth_i]; t() sandwiches it with two
    AdditiveTransform shifts by ±center_i. No scaling — log|det J| is
    fully contributed by the inner spline.

    Arguments:
        key: PRNG key (feature orderings + conditioner initialisation).
        a: lower corner of the box, shape (d,).
        b: upper corner of the box, shape (d,).
        bins: number of spline knots per coordinate; more bins give finer
            local detail at the cost of parameters and overfitting risk
            (recommend: 8-16 for smooth densities, up to 32 for sharper
            features).
        slope: minimum slope of each spline segment in the monotonic RQS
            transform. Acts as a floor on the derivative to keep the
            bijection strictly increasing and numerically stable
            (recommend: 1e-3 to 1e-2).
        transforms: number of stacked autoregressive layers. Too few
            underfits multimodal targets; too many hurts optimization
            (recommend: 4-6).
        randmask: per-layer feature ordering. True (default) draws a
            fresh permutation per layer — recommended at d >= 4
            because it breaks the bipartite symmetry that the alternating
            scheme imposes. False uses arange(d) / reversed alternation.
            Reproducible from the constructor `key` in either case.
        hidden_features: per-layer widths of the autoregressive conditioner
            MLP. A mild bottleneck works well (recommend: (64, 64) or
            (128, 64, 128); widen before deepening).
        activation: activation callable used inside the conditioner MLP
            (recommend: jax.nn.silu or jax.nn.gelu for smooth targets,
            jax.nn.relu only when speed matters).
    """

    a: Array
    b: Array
    center: Array
    halfwidth: Array
    slope: float = eqx.field(static=True)
    _maf: tuple[MaskedAutoregressiveTransform, ...]

    def __init__(
        self,
        key: Array,
        a: Array | list[float],
        b: Array | list[float],
        bins: int = 8,
        slope: float = 1e-3,
        transforms: int = 4,
        randmask: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: Callable[[Array], Array] = jax.nn.silu,
    ) -> None:
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        assert a.shape == b.shape and a.ndim == 1
        d = a.shape[0]

        self.a = a
        self.b = b
        self.center = (a + b) / 2
        self.halfwidth = (b - a) / 2
        self.slope = slope

        okey, mkey = jax.random.split(key)
        orders = _make_orders(okey, d, transforms, randmask)
        mkeys = jax.random.split(mkey, transforms)
        self._maf = tuple(
            MaskedAutoregressiveTransform(
                mkeys[i],
                features=d,
                univariate=MonotonicRQSTransform,
                shapes=[(bins,), (bins,), (bins - 1,)],
                order=orders[i],
                hidden_features=hidden_features,
                activation=activation,
                bound=self.halfwidth,
                slope=slope,
            )
            for i in range(transforms)
        )

    def t(self) -> ComposedTransform:
        """Bijection on [a, b]^d as a ComposedTransform.

        Supports .inv and .call_and_ladj(x) -> (y, log|det J|).
        """
        inner = ComposedTransform(*[m() for m in self._maf])
        center = lax.stop_gradient(self.center)  # buffer semantics: not trainable
        return ComposedTransform(
            AdditiveTransform(shift=-center),
            inner,
            AdditiveTransform(shift=center),
        )

    def zeros(self) -> "NSF":
        """Identity-initialised copy: the last layer of each conditioner
        MLP is zeroed. Returns a new flow (flows are immutable)."""
        return eqx.tree_at(
            lambda f: sum(
                (
                    [m.hyper.linears[-1].weight, m.hyper.linears[-1].bias]
                    for m in f._maf
                ),
                [],
            ),
            self,
            replace_fn=jnp.zeros_like,
        )


# ──────────────────────────────────────────────────────────────────────
# NCSF — Neural Circular Spline Flow on [a, b]^d
# ──────────────────────────────────────────────────────────────────────

class NCSF(Flow):
    """Neural Circular Spline Flow on [a_1, b_1] x ... x [a_d, b_d],
    each coordinate periodic with its own period b_i - a_i.

    The MAF circular-RQS conditioner runs on the per-coord centred box
    [-halfwidth_i, halfwidth_i]; t() sandwiches it with AdditiveTransform
    shifts by ±center_i (no scaling). Default a = [-pi, ..., -pi],
    b = [pi, ..., pi] gives the NCSF on the d-torus.

    The circular spline carries `bins` derivative parameters per
    coordinate, the first wrap-shared onto the last knot (d_0 = d_K, one
    learnable seam slope), so each coordinate map is a C¹ circle
    diffeomorphism with a trainable seam density (Rezende et al.,
    "Normalizing Flows on Tori and Spheres", 2020).

    The autoregressive conditioner is periodic: each conditioning
    coordinate enters the MLP as its circle embedding (cos, sin),
    period-matched to the coordinate's own period, so the modeled joint
    density is continuous across every seam theta_i = +-halfwidth_i and
    invariant under full-period shifts of the input — a genuine density
    on the torus, evaluated identically on wrapped and unwrapped angle
    representatives.

    Arguments:
        key: PRNG key (feature orderings + conditioner initialisation).
        a: lower corner of the box, shape (d,) (typically -pi).
        b: upper corner of the box, shape (d,) (typically  pi).
        bins: number of spline knots per coordinate (recommend: 8-16).
        slope: minimum slope of each spline segment (recommend: 1e-3 to 1e-2).
        transforms: number of stacked autoregressive layers (recommend: 4-6).
        randmask: per-layer feature ordering. True (default) draws a
            fresh permutation per layer — the only legal expressivity
            lever on a torus (linear mixings break periodicity). False
            uses arange(d) / reversed alternation. Reproducible from the
            constructor `key` in either case.
        hidden_features: per-layer widths of the autoregressive conditioner
            MLP (recommend: (64, 64) or (128, 64, 128)).
        activation: activation callable used inside the conditioner MLP
            (recommend: jax.nn.silu or jax.nn.gelu).
    """

    a: Array
    b: Array
    center: Array
    halfwidth: Array
    slope: float = eqx.field(static=True)
    _maf: tuple[MaskedAutoregressiveTransform, ...]

    def __init__(
        self,
        key: Array,
        a: Array | list[float],
        b: Array | list[float],
        bins: int = 8,
        slope: float = 1e-3,
        transforms: int = 4,
        randmask: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: Callable[[Array], Array] = jax.nn.silu,
    ) -> None:
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        assert a.shape == b.shape and a.ndim == 1
        d = a.shape[0]

        self.a = a
        self.b = b
        self.center = (a + b) / 2
        self.halfwidth = (b - a) / 2
        self.slope = slope

        okey, mkey = jax.random.split(key)
        orders = _make_orders(okey, d, transforms, randmask)
        mkeys = jax.random.split(mkey, transforms)
        self._maf = tuple(
            MaskedAutoregressiveTransform(
                mkeys[i],
                features=d,
                univariate=CircularRQSTransform,
                shapes=[(bins,), (bins,), (bins,)],
                order=orders[i],
                hidden_features=hidden_features,
                activation=activation,
                bound=self.halfwidth,
                slope=slope,
                circular=True,
            )
            for i in range(transforms)
        )

    def t(self) -> ComposedTransform:
        """Bijection on [a, b]^d as a ComposedTransform."""
        inner = ComposedTransform(*[m() for m in self._maf])
        center = lax.stop_gradient(self.center)  # buffer semantics: not trainable
        return ComposedTransform(
            AdditiveTransform(shift=-center),
            inner,
            AdditiveTransform(shift=center),
        )

    def zeros(self) -> "NCSF":
        """Identity-initialised copy: the last layer of each conditioner
        MLP is zeroed. Returns a new flow (flows are immutable)."""
        return eqx.tree_at(
            lambda f: sum(
                (
                    [m.hyper.linears[-1].weight, m.hyper.linears[-1].bias]
                    for m in f._maf
                ),
                [],
            ),
            self,
            replace_fn=jnp.zeros_like,
        )


# ──────────────────────────────────────────────────────────────────────
# CNF — Continuous Normalizing Flow on R^d (FFJORD)
# ──────────────────────────────────────────────────────────────────────

class CNF(Flow):
    """Continuous normalizing flow (CNF) with a free-form Jacobian (FFJORD).

    Acts as a bijection on R^d via an ODE drift learned by an MLP. The
    exact-log-det path is O(d) ODE evaluations per Jacobian; pass
    `exact=False` to switch to a Hutchinson stochastic estimate (faster,
    biased gradients during training; the probe noise is drawn from the
    constructor key).

    Integration is fixed-step RK4 (the only integrator in jflows): a
    deterministic flop budget under a single `lax.scan` trace. Accuracy is
    set by `nt` rather than solver tolerances; round-trip and log-det error
    fall ~16x per doubling of `nt`.

    Arguments:
        key: PRNG key (drift-MLP initialisation + Hutchinson probe).
        dimension: number of features d.
        frequency: number of time-embedding frequencies in the ODE drift
            (recommend: 3-6).
        nt: number of fixed RK4 steps (recommend: 8-24; raise for tighter
            inverse round-trip / log-det accuracy at linear cost).
        exact: if True, evaluate log|det J| exactly via the augmented ODE.
            If False, use the Hutchinson trace estimator.
        hidden_features: ODE-MLP layer widths (recommend: (64, 64)).
        activation: ODE-MLP activation callable (recommend: jax.nn.silu).
    """

    _ffj: FFJTransform

    def __init__(
        self,
        key: Array,
        dimension: int,
        frequency: int = 3,
        nt: int = 16,
        exact: bool = True,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: Callable[[Array], Array] = jax.nn.silu,
    ) -> None:
        self._ffj = FFJTransform(
            key,
            dimension=dimension,
            frequency=frequency,
            nt=nt,
            exact=exact,
            hidden_features=hidden_features,
            activation=activation,
        )

    def t(self) -> ComposedTransform:
        """Bijection on R^d as a length-1 ComposedTransform."""
        return ComposedTransform(self._ffj())

    def zeros(self) -> "CNF":
        """Drift = 0 → ODE flows trivially, identity bijection.
        Returns a new flow (flows are immutable)."""
        return eqx.tree_at(
            lambda f: (
                f._ffj.ode.linears[-1].weight,
                f._ffj.ode.linears[-1].bias,
            ),
            self,
            replace_fn=jnp.zeros_like,
        )


# ──────────────────────────────────────────────────────────────────────
# OTFlow — optimal-transport continuous flow on R^d
# ──────────────────────────────────────────────────────────────────────

class OTFlow(Flow):
    """Optimal-transport continuous normalizing flow (OT-Flow).

    Reference:
        Onken, Fung, Li, Ruthotto. "OT-Flow: Fast and Accurate Continuous
        Normalizing Flows via Optimal Transport." AAAI 2021.
        https://arxiv.org/abs/2006.00104

    An upgraded `CNF`: a bijection on R^d defined by the ODE
    `dx/dt = -∇_x Φ_θ(t, x)` for a learnable scalar potential `Φ_θ`. Because
    `Φ_θ` has the OT-Flow antiderivative-of-tanh ResNet plus low-rank
    quadratic structure, the divergence `tr(∇_x v) = -tr(∇²_x Φ)` is computed
    in closed form (O(d·m)) — no Hutchinson estimator and no augmented O(d)
    Jacobian ODE as in FFJORD, which is where OT-Flow's speedup comes from.
    Integration uses a fixed-step RK4 scheme (deterministic flop budget,
    one `lax.scan` trace).

    The forward map and `log|det J|` follow the standard `(y, ladj)` contract,
    so an `OTFlow` is a drop-in `Flow` for `reverse_KL_F` and the SMC utilities.

    Arguments:
        key: PRNG key for Φ's initialisation.
        dimension: number of features d.
        hidden: hidden width m of Φ's ResNet (recommend: 32-128).
        layer: number of ResNet layers nTh inside Φ; must be >= 2
            (recommend: 2-5).
        rank: rank of Φ's low-rank quadratic term, clamped to <= d + 1
            (recommend: min(10, d + 1)).
        nt: number of fixed RK4 steps. A well-regularised OT path needs few
            (recommend: 4-12); more steps tighten the inverse round-trip and
            log-det accuracy at linear cost.
        time_bound: (t0, t1) integration interval (default (0.0, 1.0)).

    Unlike the other flows, OTFlow takes no `activation` argument: the
    closed-form Hessian trace is derived specifically for the
    antiderivative-of-tanh activation, so it is not configurable.
    """

    _ot: OTFlowLazy

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
        self._ot = OTFlowLazy(
            key,
            dimension=dimension,
            hidden=hidden,
            layer=layer,
            rank=rank,
            nt=nt,
            time_bound=time_bound,
        )

    def t(self) -> ComposedTransform:
        """Bijection on R^d as a length-1 ComposedTransform."""
        return ComposedTransform(self._ot())

    def zeros(self) -> "OTFlow":
        """Φ ≡ 0 → drift ∇Φ = 0 → identity bijection with ladj = 0.

        Zeros every head of the potential: the ResNet output weight `w`, the
        quadratic factor `A` (so AᵀA = 0), and the linear head `c`.
        Returns a new flow (flows are immutable)."""
        return eqx.tree_at(
            lambda f: (
                f._ot.phi.w.weight,
                f._ot.phi.A,
                f._ot.phi.c.weight,
            ),
            self,
            replace_fn=jnp.zeros_like,
        )


# ──────────────────────────────────────────────────────────────────────
# RealNVP — affine-coupling flow on R^d
# ──────────────────────────────────────────────────────────────────────

class RealNVP(Flow):
    """Affine-coupling normalizing flow (RealNVP, Dinh et al. 2016).

    N stacked coupling transforms with checkered (or random) feature
    masks; inverse and log|det J| are closed-form O(d). Optionally
    interleaves a learnable d x d linear mixing layer between every
    pair of consecutive couplings — the same idea as Glow's
    "invertible 1x1 convolution" on R^d.

    Arguments:
        key: PRNG key (masks + conditioner initialisations).
        dimension: number of features d.
        transforms: number of stacked coupling layers (recommend: 4-8).
        randmask: if True (default), draw a fresh randomised checkered
            mask per layer (better mixing at d >= 4). If False, use the
            canonical alternating-checkered RealNVP masks. Reproducible
            from the constructor `key` in either case.
        mixing: one of None | "rotation" | "lu".
            If None (default), no mixing layer is inserted and the
            flow is a pure stack of coupling transforms. If "rotation",
            an orthogonal `R = exp(A - A^T)` map (log|det| ≡ 0) is
            inserted between every two consecutive couplings (i.e.
            `transforms - 1` mixing layers total). If "lu", a PLU map
            `L @ U` is inserted instead — its learnable diagonal of `L`
            provides a non-trivial log|det| that supplements the
            coupling layers. Mixing layers are initialised at identity.
        hidden_features: per-layer widths of the coupling-conditioner MLP
            (recommend: (64, 64) or (128, 128)).
        activation: conditioner MLP activation callable (recommend:
            jax.nn.silu).
    """

    _layers: tuple[eqx.Module, ...]

    def __init__(
        self,
        key: Array,
        dimension: int,
        transforms: int = 4,
        randmask: bool = True,
        mixing: str | None = None,
        hidden_features: tuple[int, ...] = (64, 64),
        activation: Callable[[Array], Array] = jax.nn.silu,
    ) -> None:
        mask_key, layer_key = jax.random.split(key)
        mkeys = jax.random.split(mask_key, transforms)
        lkeys = jax.random.split(layer_key, transforms)

        layers: list[eqx.Module] = []
        for i in range(transforms):
            if randmask:
                mask = np.asarray(jax.random.permutation(mkeys[i], dimension)) % 2 == i % 2
            else:
                mask = np.arange(dimension) % 2 == i % 2
            layers.append(
                GeneralCouplingTransform(
                    lkeys[i],
                    features=dimension,
                    mask=mask,
                    hidden_features=hidden_features,
                    activation=activation,
                )
            )
            # Insert a mixing layer between this coupling and the next
            # (skip after the last coupling — no successor to mix into).
            if mixing is not None and i < transforms - 1:
                layers.append(
                    LinearMixingTransform(features=dimension, kind=mixing)
                )
        self._layers = tuple(layers)

    def t(self) -> ComposedTransform:
        """Bijection on R^d as the composition of all coupling (and mixing) layers."""
        return ComposedTransform(*[layer() for layer in self._layers])

    def zeros(self) -> "RealNVP":
        """Reset every layer to identity. `GeneralCouplingTransform` and
        `LinearMixingTransform` both expose a `.zeros()` method, so the
        same iteration handles couplings and mixing layers uniformly.
        Returns a new flow (flows are immutable)."""
        return eqx.tree_at(
            lambda f: f._layers,
            self,
            tuple(layer.zeros() for layer in self._layers),
        )

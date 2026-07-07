"""Optimal-transport flow (OT-Flow) machinery for jflows.

Adapted from the reference OT-Flow implementation of

    Onken, Fung, Li, Ruthotto.
    "OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal
     Transport." AAAI 2021. https://arxiv.org/abs/2006.00104

(`OT-Flow/src/Phi.py`, `OT-Flow/src/OTFlowProblem.py`).

The defining idea: parameterise the ODE velocity field as the (negative)
gradient of a scalar potential, `v_theta(t, x) = -∇_x Φ_theta(t, x)`. Because
`Φ` has a specific antiderivative-of-tanh ResNet + low-rank quadratic
structure, the divergence `tr(∇_x v) = -tr(∇²_x Φ)` is available in *closed
form* (`OTPhi.trHess`) at `O(d·m)` cost per evaluation — no Hutchinson
estimator and no augmented O(d) Jacobian ODE as in FFJORD.

Public objects:
    - antideriv_tanh / deriv_tanh — the activation and its 2nd derivative
    - ResNN                       — the residual network N inside Φ
    - OTPhi                       — the scalar potential Φ_theta
    - OTFlowTransform             — Transform integrating dx/dt = -∇Φ via RK4
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .nn import Linear
from .numerics import rk4_fixed
from .transforms import Transform


__all__ = [
    "OTFlowTransform",
    "OTPhi",
    "ResNN",
    "antideriv_tanh",
    "deriv_tanh",
]


# ──────────────────────────────────────────────────────────────────────
# Activations — antiderivative of tanh and its second derivative
# ──────────────────────────────────────────────────────────────────────

def antideriv_tanh(x: Array) -> Array:
    """Antiderivative of tanh: ∫tanh = log cosh, written stably.

    `act(x) = |x| + log(1 + exp(-2|x|))` equals `log(cosh(x)) + log 2`,
    computed via the |x| factorisation to avoid overflow. Its first
    derivative is `tanh`, which is what makes the closed-form Hessian
    trace in `OTPhi.trHess` possible.
    """
    return jnp.abs(x) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(x)))


def deriv_tanh(x: Array) -> Array:
    """Second derivative of `antideriv_tanh`, i.e. tanh'(x) = 1 - tanh(x)²."""
    return 1 - jnp.tanh(x) ** 2


# ──────────────────────────────────────────────────────────────────────
# ResNN — residual network N(s) inside the potential Φ
# ──────────────────────────────────────────────────────────────────────

class ResNN(eqx.Module):
    """Residual network on space-time inputs `s = [x; t]` of width `dimension + 1`.

    Layout (matching the OT-Flow reference):
        u_0     = act(K_0 s + b_0)
        u_i     = u_{i-1} + h · act(K_i u_{i-1} + b_i),   i = 1 … layer-1
    with step `h = 1/(layer-1)` and `act = antideriv_tanh`. There are `layer`
    linear layers total: one opening `(dimension+1) → hidden` and `layer-1`
    square `hidden → hidden` layers.

    Arguments:
        key: PRNG key for the layer initialisations.
        dimension: spatial dimension d (network sees `dimension + 1` with
            time appended).
        hidden: hidden width of the ResNet.
        layer: number of ResNet layers (>= 2).
    """

    layers: tuple[Linear, ...]
    dimension: int = eqx.field(static=True)
    hidden: int = eqx.field(static=True)
    layer: int = eqx.field(static=True)
    h: float = eqx.field(static=True)

    def __init__(self, key: Array, dimension: int, hidden: int, layer: int = 2) -> None:
        assert layer >= 2, "layer must be an integer >= 2"
        self.dimension = dimension
        self.hidden = hidden
        self.layer = layer
        keys = jax.random.split(key, layer)
        layers = [Linear(keys[0], dimension + 1, hidden, bias=True)]  # opening layer
        for i in range(1, layer):  # residual layers
            layers.append(Linear(keys[i], hidden, hidden, bias=True))
        self.layers = tuple(layers)
        self.h = 1.0 / (layer - 1)  # ResNet step size

    def __call__(self, x: Array) -> Array:
        """Forward propagation N(s); x is `(nex, dimension+1)`, returns `(nex, hidden)`."""
        x = antideriv_tanh(self.layers[0](x))
        for i in range(1, self.layer):
            x = x + self.h * antideriv_tanh(self.layers[i](x))
        return x


# ──────────────────────────────────────────────────────────────────────
# OTPhi — scalar potential Φ_theta with closed-form gradient + Hessian trace
# ──────────────────────────────────────────────────────────────────────

class OTPhi(eqx.Module):
    r"""Scalar potential `Φ(s) = wᵀN(s) + ½ sᵀ(AᵀA) s + bᵀs + c`, `s = [x; t]`.

    The low-rank quadratic term uses `A ∈ ℝ^{r×(d+1)}` (so `AᵀA` is a
    rank-`r` symmetric PSD matrix); `b` is the weight of a single linear
    head. `trHess` returns both `∇_s Φ` and `tr(∇²_x Φ)` analytically — see
    Eq. (11) and Eq. (13) of the OT-Flow paper — using the fact that
    `act' = tanh` and `act'' = 1 - tanh²`.

    Note the reference's constant bias `c` is dropped: only `∇Φ`, `tr∇²Φ`,
    and `∂_tΦ` ever feed the flow, none of which depend on an additive
    constant, so that bias is non-identifiable and would carry a permanently
    zero gradient. Every remaining parameter is identifiable.

    Initial values: the ResNet and `A` are randomly initialised (Xavier
    uniform for `A`), while `w = 1` and `b = 0`. The flow's `zeros()`
    zeros all heads to recover the identity bijection.

    Arguments:
        key: PRNG key for the parameter initialisations.
        dimension: spatial dimension d (network sees `dimension + 1` with
            time appended).
        hidden: hidden width of the ResNet.
        layer: number of ResNet layers (>= 2).
        rank: rank of the quadratic term (clamped to `<= dimension + 1`).
    """

    A: Array
    c: Linear
    w: Linear
    N: ResNN
    dimension: int = eqx.field(static=True)
    hidden: int = eqx.field(static=True)
    layer: int = eqx.field(static=True)

    def __init__(self, key: Array, dimension: int, hidden: int, layer: int, rank: int = 10) -> None:
        self.dimension = dimension
        self.hidden = hidden
        self.layer = layer

        rank = min(rank, dimension + 1)  # rank cannot exceed the input dimension
        kA, kc, kw, kN = jax.random.split(key, 4)
        bound = (6.0 / (rank + dimension + 1)) ** 0.5  # Xavier uniform
        self.A = jax.random.uniform(kA, (rank, dimension + 1), minval=-bound, maxval=bound)

        # Match the reference initial values: w = 1, b = 0.
        self.c = eqx.tree_at(  # bᵀ[x;t]
            lambda l: l.weight, Linear(kc, dimension + 1, 1, bias=False),
            replace_fn=jnp.zeros_like,
        )
        self.w = eqx.tree_at(
            lambda l: l.weight, Linear(kw, hidden, 1, bias=False),
            replace_fn=jnp.ones_like,
        )
        self.N = ResNN(kN, dimension, hidden, layer)

    def __call__(self, x: Array) -> Array:
        """Φ(s) for `s = x` of shape `(nex, d+1)`; returns `(nex, 1)`."""
        symA = self.A.T @ self.A  # AᵀA, shape (d+1, d+1)
        quad = 0.5 * jnp.sum((x @ symA) * x, axis=1, keepdims=True)
        return self.w(self.N(x)) + quad + self.c(x)

    def trHess(self, x: Array, justGrad: bool = False):
        """Closed-form `∇_s Φ` and `tr(∇²_x Φ)` (trace over the spatial block).

        `x` is `(nex, d+1)` space-time input. Returns `∇_s Φ` of shape
        `(nex, d+1)` (the last column is `∂Φ/∂t`); when `justGrad=False`
        also returns `tr(∇²_x Φ)` of shape `(nex,)`, summing only the
        spatial `d×d` block of the Hessian. Recomputes the ResNet forward
        pass internally (it needs the per-layer pre-activations).
        """
        N = self.N
        m = N.layers[0].weight.shape[0]
        nex = x.shape[0]
        d = x.shape[1] - 1
        symA = self.A.T @ self.A

        u = []                 # u_0 … u_{layer-1} from the forward pass
        z = N.layer * [None]   # z_0 … z_{layer-1} from the gradient backward pass

        # Forward pass through the ResNet, caching pre-/post-activations.
        opening = N.layers[0](x)          # K_0 s + b_0
        u.append(antideriv_tanh(opening))  # u_0
        feat = u[0]
        for i in range(1, N.layer):
            feat = feat + N.h * antideriv_tanh(N.layers[i](feat))
            u.append(feat)

        tanhopen = jnp.tanh(opening)  # act'(K_0 s + b_0)

        # Backward pass accumulating the gradient z_i.
        for i in range(N.layer - 1, 0, -1):
            term = self.w.weight.T if i == N.layer - 1 else z[i + 1]
            z[i] = term + N.h * (
                N.layers[i].weight.T @ (jnp.tanh(N.layers[i](u[i - 1])).T * term)
            )
        z[0] = N.layers[0].weight.T @ (tanhopen.T * z[1])
        grad = z[0] + symA @ x.T + self.c.weight.T

        if justGrad:
            return grad.T

        # ── trace of the Hessian (spatial block only) ──
        # t_0: contribution of the opening layer.
        Kopen = N.layers[0].weight[:, 0:d]   # drop the time column
        temp = deriv_tanh(opening.T) * z[1]
        trH = jnp.sum(
            temp.reshape(m, -1, nex) * Kopen[:, :, None] ** 2, axis=(0, 1)
        )

        # ∇_s u_0ᵀ, propagated forward as Jac of shape (m, d, nex).
        temp = tanhopen.T  # act'(K_0 s + b_0)
        Jac = Kopen[:, :, None] * temp[:, None, :]

        # t_i: contribution of each residual layer.
        for i in range(1, N.layer):
            KJ = (N.layers[i].weight @ Jac.reshape(m, -1)).reshape(m, -1, nex)
            term = self.w.weight.T if i == N.layer - 1 else z[i + 1]
            temp = N.layers[i](u[i - 1]).T  # K_i u_{i-1} + b_i
            t_i = jnp.sum(
                (deriv_tanh(temp) * term).reshape(m, -1, nex) * KJ**2, axis=(0, 1)
            )
            trH = trH + N.h * t_i
            Jac = Jac + N.h * jnp.tanh(temp).reshape(m, -1, nex) * KJ

        return grad.T, trH + jnp.trace(symA[0:d, 0:d])


# ──────────────────────────────────────────────────────────────────────
# OTFlowTransform — bijection integrating dx/dt = -∇Φ via fixed-step RK4
# ──────────────────────────────────────────────────────────────────────

class OTFlowTransform(Transform):
    r"""Continuous bijection `x ↦ y` flowing `dx/dt = -∇_x Φ(t, x)`.

    The forward map and `log|det J|` come from a single fixed-step RK4
    integration of the augmented state; the closed-form `OTPhi.trHess`
    supplies the divergence, so no Hutchinson estimate or Jacobian ODE is
    needed.

    `_inverse` integrates the same drift backward in time. It is therefore
    RK4-approximate (not closed-form), and round-trip accuracy improves with
    `nt`.

    Arguments:
        phi: the `OTPhi` potential.
        dimension: spatial dimension d.
        t0:  trajectory start time.
        t1:  trajectory end time.
        nt:  number of fixed RK4 steps.
    """

    phi: OTPhi
    dimension: int = eqx.field(static=True)
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    nt: int = eqx.field(static=True)

    domain_dim: ClassVar[int] = 1
    codomain_dim: ClassVar[int] = 1

    def __init__(
        self,
        phi: OTPhi,
        dimension: int,
        t0: float = 0.0,
        t1: float = 1.0,
        nt: int = 8,
    ) -> None:
        self.phi = phi
        self.dimension = dimension
        self.t0 = t0
        self.t1 = t1
        self.nt = nt

    def _with_time(self, t: Array, x: Array) -> Array:
        """Append the scalar time `t` as a final column, giving `[x; t]`."""
        return jnp.concatenate([x, t * jnp.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)

    # ── pure-position drift (gradient only) for __call__ / _inverse ──
    def _drift(self, t: Array, x: Array) -> Array:
        grad = self.phi.trHess(self._with_time(t, x), justGrad=True)
        return -grad[:, : self.dimension]

    def __call__(self, x: Array) -> Array:
        x2 = x.reshape(-1, x.shape[-1])   # flatten leading batch dims to 2-D
        y = rk4_fixed(self._drift, x2, self.t0, self.t1, self.nt)
        return y.reshape(*x.shape[:-1], self.dimension)

    def _inverse(self, y: Array) -> Array:
        y2 = y.reshape(-1, y.shape[-1])
        x = rk4_fixed(self._drift, y2, self.t1, self.t0, self.nt)
        return x.reshape(*y.shape[:-1], self.dimension)

    # ── augmented dynamics: [x, ℓ] for ladj, [x, ℓ, v, r] for OT costs ──
    def _f_ladj(self, t: Array, z: Array) -> Array:
        d = self.dimension
        grad, trH = self.phi.trHess(self._with_time(t, z[:, :d]))
        dx = -grad[:, :d]
        dl = -trH[:, None]
        return jnp.concatenate([dx, dl], axis=1)

    def _f_full(self, t: Array, z: Array) -> Array:
        d = self.dimension
        grad, trH = self.phi.trHess(self._with_time(t, z[:, :d]))
        dx = -grad[:, :d]
        dl = -trH[:, None]
        dv = 0.5 * jnp.sum(dx**2, axis=1, keepdims=True)       # ½|∇Φ|²
        dr = jnp.abs(dv - grad[:, -1:])                        # |½|∇Φ|² - ∂_tΦ|
        return jnp.concatenate([dx, dl, dv, dr], axis=1)

    def call_and_ladj(self, x: Array) -> tuple[Array, Array]:
        d = self.dimension
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])   # flatten leading batch dims to 2-D
        z0 = jnp.concatenate([x2, jnp.zeros((x2.shape[0], 1), dtype=x.dtype)], axis=1)
        zT = rk4_fixed(self._f_ladj, z0, self.t0, self.t1, self.nt)
        return zT[:, :d].reshape(*lead, d), zT[:, d].reshape(lead)

    def call_full(self, x: Array) -> tuple[Array, Array, Array, Array]:
        """Forward map plus OT diagnostics in one integration.

        Returns `(y, ladj, transport_cost, hjb_residual)` where
        `transport_cost = ∫ ½|∇Φ|² dt` and
        `hjb_residual   = ∫ |½|∇Φ|² - ∂_tΦ| dt` — the per-sample integrated
        OT regularisers.
        """
        d = self.dimension
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])   # flatten leading batch dims to 2-D
        z0 = jnp.concatenate([x2, jnp.zeros((x2.shape[0], 3), dtype=x.dtype)], axis=1)
        zT = rk4_fixed(self._f_full, z0, self.t0, self.t1, self.nt)
        return (zT[:, :d].reshape(*lead, d), zT[:, d].reshape(lead),
                zT[:, d + 1].reshape(lead), zT[:, d + 2].reshape(lead))

    def log_abs_det_jacobian(self, x: Array, y: Array) -> Array:
        _, ladj = self.call_and_ladj(x)
        return ladj

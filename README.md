# jflows

JAX normalizing flows for unconditional energy-based sampling and Boltzmann generators — the JAX + [equinox](https://github.com/patrick-kidger/equinox) twin of [zflows](https://github.com/xuda-ye-math/zflows).

> **Status: experimental.** Tested only on **Linux + NVIDIA GPU** (CUDA-enabled `jax`).
>
> This project was developed with [Claude Code](https://claude.com/claude-code).

## Features

**Flexible flow classes and hyperparameters, one unified interface.** Five flow classes are supported — **NSF** (Neural Spline Flow), **NCSF** (Neural *Circular* Spline Flow, for periodic / angular domains), **CNF** (Continuous Normalizing Flow / FFJORD), **OTFlow** (optimal-transport continuous flow with a closed-form trace), and **RealNVP** (closed-form affine-coupling bijection on $\mathbb R^d$) — with the constructors

```python
import jax
from jflows.flow import NSF, NCSF, CNF, OTFlow, RealNVP

key = jax.random.key(0)

NSF(key, a=[0.0, 0.0], b=[1.0, 1.0], bins=8, slope=1e-3, transforms=4, randmask=True, hidden_features=(64, 64), activation=jax.nn.silu)
NCSF(key, a=[-1.0, -1.0], b=[1.0, 1.0], bins=8, slope=1e-3, transforms=4, randmask=True, hidden_features=(64, 64), activation=jax.nn.silu)
CNF(key, dimension=8, frequency=3, nt=16, exact=True, hidden_features=(64, 64), activation=jax.nn.silu)
OTFlow(key, dimension=8, hidden=64, layer=3, rank=10, nt=8, time_bound=(0.0, 1.0))
RealNVP(key, dimension=8, transforms=4, randmask=True, mixing="lu", hidden_features=(64, 64), activation=jax.nn.silu)
```

all subclassing the same `Flow` abstract class (an `eqx.Module`, i.e. an immutable pytree). The flow itself is the user-facing object:

```python
from jflows.flow import NSF

flow = NSF(key, ...)          # or NCSF(...), CNF(...), OTFlow(...), RealNVP(...)
flow = flow.zeros()           # identity initialisation (returns a NEW instance)

y       = flow(x)             # forward map
y, ladj = flow.call_and_ladj(x)   # forward map & log|det J|
x_back  = flow.inv(y)             # inverse map
x, ladj = flow.inv_and_ladj(y)    # inverse map & its log|det J|
```

Because modules are immutable pytrees, `flow.zeros()` (and every training step) returns a new instance rather than mutating in place, and flows jit / vmap / grad like any other JAX value. `flow.t()` remains available as the advanced composition layer (it returns the underlying `ComposedTransform` for chaining transforms and custom pipelines); the high-level API never needs it.

**Explicit PRNG keys.** There is no global seed in JAX: every random entry point — flow constructors, `Potential.samples`, `resample`, `langevin`, `hamiltonian_monte_carlo`, `sequential_monte_carlo`, `annealed_importance_sampling` — takes a `key` as its first argument.

**Unified `Potential` class with a vector-space algebra.** Every energy function subclasses one `Potential` base (potentials are energies of $\mu \propto e^{-U}$; there are no temperature arguments). Define a custom potential by subclassing `Potential` and implementing `__call__`:

```python
from jax import Array
from jflows.potential import Potential

class My_Potential(Potential):   # any user-defined energy
    def __call__(self, x: Array) -> Array:   # Array [N, d] -> Array [N]
        return ...

u = My_Potential()   # `u(x)` evaluates the energy; `u.grad(x)` its gradient
```

Naming follows one simple rule throughout the project: classes capitalize the first letter of each word (`My_Potential`, `Nlog_Gaussian`), instances lowercase it (`u`, `u0`, `u_target`). Built-ins are `Nlog_Uniform`, `Nlog_Gaussian`, `Nlog_Gaussian_Mixture` (all with key-first `.samples(key, N)`), and `potential_from(...)` wraps a plain `(x) -> Array` callable into a ready-to-use instance.

Potentials form a vector space: `c * u`, `u0 + u1`, `u0 - u1`, `-u`, `u / c`, and `sum([...])` all return potentials, with repeated instances merged by identity into one flat linear combination. `linear_combination` is the explicit constructor for annealing bridges:

```python
from jflows.potential import linear_combination

u = linear_combination([u0, u1], [1.0 - c, c])   # U = (1-c) U0 + c U1
```

Coefficients are a plain array leaf, so retuning `c` along an annealing ladder never triggers recompilation.

**Per-sample KL losses with one `type` argument.** `reverse_KL` and `forward_KL` take the flow and dispatch on `type`: `'F'` when the flow maps source → target, `'G'` when it maps target → source (the flow is differentiated in its native direction only — no inverse map in training). Every loss returns the full per-sample vector, shape `[N]`, for post-hoc reweighting / clipping; reduce with `.mean()`:

```python
from jflows.loss import reverse_KL, forward_KL, OT_loss

loss = reverse_KL(x, target, flow, type="F").mean()   # source samples x
loss = forward_KL(y, source, flow, type="G").mean()   # target samples y
loss = OT_loss(x, target, otflow, alpha_C=1.0, alpha_R=1.0).mean()   # OTFlow + OT regularizers
```

**Packed training drivers.** `jflows.train` packs a whole training stage — Adam on the flow's parameters, the full step loop under one `lax.scan` — into a single compiled call that regenerates its batch *inside every Adam step* (the X-regularization data pipeline: no frozen batch is ever reused, so a fixed sample set does not get memorized):

```python
from jflows.train import train_reverse_KL, train_forward_KL, boltzmann_reverse_KL, Monitor

# reverse KL: each step draws n_batch samples from the fixed set x_valid and
# freshens them with Langevin rejuvenation at the source
flow, ess = train_reverse_KL(x_valid, source, target, flow, type="F",
                             n_batch=2000, steps=200, lr=1e-3,
                             mc_step=1e-3, mc_iters=100)

# forward KL: each step manufactures its target batch by single-rung AIS
# through the CURRENT flow (pushforward -> reweight -> resample -> Langevin)
flow, ess = train_forward_KL(x_valid, source, target, flow, type="G",
                             n_batch=2000, steps=200, lr=1e-3,
                             ladder=1, mc_step=1e-3, mc_iters=100)
```

Both drivers are deterministic (per-step keys derive from a fixed internal seed), compile once regardless of `steps`, and return the trained flow together with the per-step batch-ESS history. An optional `Monitor` reports live from inside the compiled loop:

```python
flow, ess = train_reverse_KL(..., monitor=Monitor(every=10, prefix="[reverse KL] "))
# [reverse KL] step    10   loss = +4.7476e+00   ESS = 0.3542
# [reverse KL] step    20   loss = +3.7126e+00   ESS = 0.4879
```

On top of the stage trainers, `boltzmann_reverse_KL` runs the full annealed
Boltzmann generator on the bridge ladder $U_t = (1-t)\,U_0 + t\,U_1$ with an
ADAPTIVE coefficient: the stage flows are connected step by step — each stage
trains the warm-started flow as the incremental map $\mu_{t_{k-1}} \to \mu_{t_k}$
on the advancing particle set, accepts on the incremental importance-sampling
ESS (rejected stages shrink $t_k$ and retry with fresh randomness), and
advances the set by reweight → resample → Langevin at $U_{t_k}$:

```python
from jflows.train import boltzmann_reverse_KL

flow, y, stages = boltzmann_reverse_KL(
    x_valid, source, target, flow, type="F",
    n_batch=2000, steps=500, lr=1e-4, mc_step=1e-3, mc_iters=100,
    bg_param={"t_safe": 0.1, "shrink_factor": 0.7, "enlarge_factor": 1.5, "tau": 0.6},
)
# y      : the particle set at the target (the generator's sample output)
# stages : per-stage records {"t", "ess", "flow"} — the saved incremental maps
```

The trainer, weight evaluation, and advance are `filter_jit`-compiled once for
the whole ladder (retuned bridge coefficients are array leaves, so no
recompilation), and a `chunk` argument bounds the full-set stage operations at
large particle counts.

**SMC-style utilities with a two-level interface.** `jflows.utils` provides the *propose → reweight → resample → rejuvenate* building blocks, each at two levels: a packed loop for direct use and a per-step kernel for custom schedules:

```python
from jflows.utils import (
    importance_weights, importance_weights_log,   # flow IS weights (type='F'/'G')
    compute_ESS, compute_ESS_log,                 # effective sample size
    coverage,                                     # k-NN mode-collapse diagnostic
    resample,                                     # multinomial resampling
    langevin, langevin_step,                      # ULA / MALA / tamed
    stochastic_heun, hamiltonian_monte_carlo,     # more rejuvenation kernels
    sequential_monte_carlo, smc,                  # annealed SMC over a bridge ladder
    annealed_importance_sampling, ais,            # AIS through a trained flow
    lbfgs, adamw,                                 # batched optimizers (+ init/step kernels)
)

log_w = importance_weights_log(samples, source, target, flow, type="F", chunk=1)
ess   = compute_ESS_log(log_w)
y     = ais(key, samples, source, target, flow, type="G", ladder=1, step=1e-3, iters=100)
```

`chunk` splits batches along dim 0 to bound peak VRAM (statistically equivalent to `chunk=1`). There is no compile machinery to manage: everything is jit-friendly by construction, and `jax.jit` at the call site covers the rest.

**Package layout.**

```
jflows
├── core
│   ├── flows.py
│   ├── __init__.py
│   ├── nn.py
│   ├── numerics.py
│   ├── otflow.py
│   └── transforms.py
├── flow.py
├── __init__.py
├── loss.py
├── potential.py
├── train.py
└── utils
    ├── annealing.py
    ├── __init__.py
    ├── metrics.py
    ├── optimization.py
    └── rejuvenation.py
```

## Installation

`jflows` is pure Python; the runtime dependencies are [`jax`](https://docs.jax.dev) (install the CUDA build for GPU support, e.g. `pip install "jax[cuda13]"`), [`equinox`](https://github.com/patrick-kidger/equinox), and `numpy`.

**1. Clone the repository.**

```bash
git clone https://github.com/xuda-ye-math/jflows.git
cd jflows
```

**2. Install in editable mode.** Local edits take effect immediately:

```bash
pip install -e .
```

**3. Verify the install.**

```bash
python -c "import jflows; print(jflows.__doc__)"
```

**Importing.** Use the five submodules `flow`, `potential`, `loss`, `train`, `utils`, and call `help(foo_name)` to read the documents. For example:

```python
from jflows.flow import NSF, RealNVP
from jflows.potential import Potential, Nlog_Gaussian
from jflows.loss import reverse_KL, forward_KL
from jflows.train import train_reverse_KL, train_forward_KL, Monitor
from jflows.utils import importance_weights, compute_ESS, resample, langevin

help(NSF)
```

## Mathematical Background

<details>
<summary>click to expand; renders best in VS Code</summary>

Given a confining potential $U_1(x)$, energy-based sampling draws from the Boltzmann distribution $\mu_1 \propto \exp(-U_1)$. The normalizing-flow recipe: pick a tractable source $\mu_0 \propto \exp(-U_0)$ and learn a diffeomorphism $F$ such that $F_{\#}\mu_0 \approx \mu_1$, using the change-of-variable formula

$$
(F_{\#}\mu_0)(y) = \frac{\mu_0(x)}{|\det J_F(x)|}, \qquad y = F(x).
$$

The **reverse KL** involves only the energy $U_1$ and source samples:

$$
\mathcal L_{\mathrm{reverse}}[F] = \mathbb E_{x \sim \mu_0} \big[ U_1(F(x)) - \log |\det J_F(x)| \big],
$$

estimated per sample by `reverse_KL(x, target, flow, type)`. The **forward KL** uses target samples $y \sim \mu_1$:

$$
\mathcal L_{\mathrm{forward}}[G] = \mathbb E_{y \sim \mu_1} \big[ U_0(G(y)) - \log |\det J_G(y)| \big], \qquad G = F^{-1},
$$

estimated per sample by `forward_KL(y, source, flow, type)`. In the 100% energy-driven workflow the target samples are never a dataset: they are manufactured on the fly by annealed importance sampling through the current flow (`ais`, or internally by `train_forward_KL`). The `type` argument names the direction the trained flow acts in — `'F'` for source → target, `'G'` for target → source — and each flow is differentiated in its native direction only.

Once trained, new samples from $\mu_1$ are generated by pushing fresh source samples through the flow, with exact importance weights (`importance_weights`) and the effective sample size (`compute_ESS`) quantifying the residual mismatch.

</details>

## Examples

[`example/2D_single.py`](example/2D_single.py) trains one NSF on a three-mode 2D Gaussian mixture with each KL objective — reverse KL on Langevin-freshened source batches, forward KL on AIS-manufactured target batches — and compares them by ESS. Run from the repo root:

```bash
python -m example.2D_single
```

<p align="center"><img src="https://raw.githubusercontent.com/xuda-ye-math/jflows/main/example/2D_single.png" alt="2D single-stage training" width="1000px"></p>

[`example/4D_boltzmann.py`](example/4D_boltzmann.py) runs `boltzmann_reverse_KL` on the 4D two-charge target of the zflows reference test — two particles on a soft annulus with regularized Coulomb repulsion — where a direct flow proposal has ESS ~ 0. The adaptive ladder reaches $t = 1$ in five first-attempt stages:

```bash
python -m example.4D_boltzmann
```

<p align="center"><img src="https://raw.githubusercontent.com/xuda-ye-math/jflows/main/example/4D_boltzmann.png" alt="4D Boltzmann generator" width="1000px"></p>

Numerical results and discussion: [`example/results.md`](example/results.md).

## Acknowledgements

`jflows` is the JAX port of [zflows](https://github.com/xuda-ye-math/zflows), and both are strongly inspired by [zuko](https://github.com/probabilists/zuko): the flow, transform, and masked-MLP machinery vendored into `jflows.core` is a stripped-down port of zuko's. Credit for the underlying design — and for the clean, composable `Transform` API the public flows build on — belongs to the zuko authors.

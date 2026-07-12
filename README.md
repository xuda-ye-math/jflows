<p align="center"><img src="jflows.png" alt="jflows banner" width="800px"></p>
<p align="center"><sub><em>Banner designed by ChatGPT: the character Jax from "The Amazing Digital Circus", a nod to jflows being built on Google JAX.</em></sub></p>

# jflows

JAX normalizing flows for unconditional energy-based sampling and Boltzmann generators, built on [equinox](https://github.com/patrick-kidger/equinox).

> **Status: experimental.** Tested only on **Linux + NVIDIA GPU** (CUDA-enabled `jax`); Google TPU and AMD GPU should also work. JAX GPU is not supported on Windows — not even under WSL.
>
> This project was developed with [Claude Code](https://claude.com/claude-code).

## Features

**NSF and NCSF are the first-class models.** The **NSF** (Neural Spline Flow, on a box $[a, b]^d$) and the **NCSF** (Neural *Circular* Spline Flow, on the torus — the periodic domains of molecular angles) carry the energy-based workflows this package is built for; **CNF** (FFJORD), **OTFlow** (closed-form-trace continuous flow), and **RealNVP** (closed-form affine coupling) round out the family under one unified interface:

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

u = linear_combination([u1, u0], [t, 1.0 - t])   # U_t = (1-t) U0 + t U1
```

Coefficients are a plain array leaf, so retuning `t` along an annealing ladder never triggers recompilation.

**A strict interface hierarchy.** The API has three levels. The LOW level is the building blocks — per-sample losses and the SMC toolkit — for assembling custom pipelines. The MEDIUM level — the stage-training drivers — composes those blocks into one-call trainers. The HIGH level — the annealed Boltzmann generator — chains the stage trainers into the full adaptive-ladder pipeline. Everything below is organized in that order.

**Low level: per-sample KL losses.** Each loss fixes the flow direction its suffix names — `reverse_KL_F` applies the flow as the forward map F (source → target), `forward_KL_G` as the inverse map G (target → source) — so every loss is differentiated in its native direction and no inverse map ever enters training. Every loss returns the full per-sample vector, shape `[N]`, for post-hoc reweighting / clipping; reduce with `.mean()`:

```python
from jflows.loss import reverse_KL_F, forward_KL_G

loss = reverse_KL_F(x, target, flow).mean()   # source samples x, flow as F
loss = forward_KL_G(y, source, flow).mean()   # target samples y, flow as G
```

**Low level: the SMC toolkit.** `jflows.utils` provides the *propose → reweight → resample → rejuvenate* building blocks, with complete routines for direct use and per-step kernels for custom schedules:

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

`sequential_monte_carlo` is classical potential-space SMC and rejuvenates at
the matching intermediate bridge at each level. Flow-proposal
`annealed_importance_sampling` uses one fraction of the proposal-to-target log
weight per level but rejuvenates at the final target every time. This avoids
flow-density derivatives inside MCMC, so it is a deliberately biased,
score-free target surrogate rather than exact AIS/SMC.

`chunk` splits batches along dim 0 to bound peak VRAM (statistically equivalent
to `chunk=1`). The full-set importance-weight helper used by the Boltzmann
drivers evaluates those chunks sequentially so they do not all live in one XLA
graph. Rejuvenation and optimizer routines retain their original compiled-scan
implementations.

**Medium level: training drivers.** Every `jflows.train` stage driver is
`eqx.filter_jit`-compiled and runs all Adam steps in one `lax.scan`. Sampling,
loss/gradient evaluation, and the Adam update remain in that compiled stage.
Every step draws a new subset and regenerates its training data, so no frozen
batch is reused:

```python
from jflows.train import train_reverse_KL_F, train_forward_KL_G, Monitor

# reverse KL: each step draws n_batch samples from the fixed set x_valid and
# freshens them with Langevin rejuvenation at the source (flow fixed as F)
flow, ess = train_reverse_KL_F(x_valid, source, target, flow,
                               n_batch=2000, steps=200, lr=1e-3,
                               mc_step=1e-3, mc_iters=100)

# forward KL: each step manufactures its target batch by one-level AIS
# through the CURRENT flow (pushforward -> reweight -> resample -> Langevin;
# flow fixed as G)
flow, ess = train_forward_KL_G(x_valid, source, target, flow,
                               n_batch=2000, steps=200, lr=1e-3,
                               ladder=1, mc_step=1e-3, mc_iters=100)
```

Both drivers are deterministic (per-step keys derive from a fixed internal
seed). Each fixed configuration compiles as one stage call and returns the
trained flow together with the per-step batch-ESS history. An optional
`Monitor` reports from inside the compiled scan via `jax.debug.callback`:

```python
flow, ess = train_reverse_KL_F(..., monitor=Monitor(every=10, prefix="[reverse KL] "))
# [reverse KL] step    10   loss = +4.7476e+00   ESS = 0.3542
# [reverse KL] step    20   loss = +3.7126e+00   ESS = 0.4879
```

**High level: the annealed Boltzmann generator.** On top of the stage trainers, `boltzmann_reverse_KL_F` runs the full annealed
Boltzmann generator on the bridge ladder $U_t = (1-t)\,U_0 + t\,U_1$ with an
ADAPTIVE coefficient: the stage flows are connected step by step — each stage
trains the warm-started flow as the incremental map $\mu_{t_{k-1}} \to \mu_{t_k}$
on the advancing particle set, accepts on the incremental importance-sampling
ESS (rejected stages shrink $t_k$ and retry with fresh randomness), and
advances the set by reweight → resample → Langevin at $U_{t_k}$ (MALA by default; `mc_adjust=False` for plain ULA).
After each stage an identity check keeps whichever of the trained flow and the
identity map (pure SMC reweighting, computed with no flow inverse) has the higher
incremental ESS, so a stage is never worse than SMC:

```python
from jflows.boltzmann import boltzmann_reverse_KL_F

y_valid, stages = boltzmann_reverse_KL_F(
    x_valid, source, target, flow,
    n_pool=24000, n_batch=2000, steps=500, lr=1e-4, ladder=1, mc_step=1e-3, mc_iters=100,
    bg_param={"t_safe": 0.1, "shrink_factor": 0.7, "enlarge_factor": 1.5, "tau_ess": 0.6},
)
# y_valid : the advanced validation set at the target (the generator's sample output)
# stages  : per-stage records {"t", "ess", "flow", "ess_history", "imp_history"} —
#           the coefficient, the accepted incremental ESS (trained or identity),
#           the saved incremental map, its per-step training-ESS history, and the
#           improvement over the identity fallback (max(0, trained - identity) ESS)
```

The adaptive coefficient/retry loop stays in Python, while every accepted or
rejected training attempt invokes one compiled stage scan. Selection SMC,
stage advancement, and fixed-shape flow operations retain the committed
`filter_jit` treatment; retuned bridge coefficients are array leaves. The
full-set importance-weight helper evaluates `chunk` partitions sequentially to
bound peak memory.

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
    ├── anneal.py
    ├── __init__.py
    ├── metrics.py
    ├── optimization.py
    ├── quench.py
    └── rejuvenation.py
```

## Installation

`jflows` is pure Python and requires JAX and Equinox. Prepare an environment
containing compatible versions of those dependencies; this project does not
prescribe how the dependency stack is installed.

For readers who want a conventional editable installation from GitHub:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/xuda-ye-math/jflows.git "$HOME/src/jflows"
cd "$HOME/src/jflows"
pip install -e .
```

### Maintainer live-source runs

On the project workstation, do not perform the editable installation above.
Development and test runs use the canonical checkout directly through
`PYTHONPATH`; local edits then take effect on the next Python process:

```bash
PYTHONPATH=/mnt/projects/jflows python your_script.py
```

Verify the live-source provenance explicitly:

```bash
PYTHONPATH=/mnt/projects/jflows python -c \
  "from pathlib import Path; import jflows; print(Path(jflows.__file__).resolve())"
```

Repository examples and smoke modules use the same pattern:

```bash
PYTHONPATH=/mnt/projects/jflows python -m smoke.test_flow
```

**Importing.** Use the public submodules `flow`, `potential`, `loss`, `train`,
`boltzmann`, and `utils`, and call `help(foo_name)` to read the documents. For
example:

```python
from jflows.flow import NSF, RealNVP
from jflows.potential import Potential, Nlog_Gaussian
from jflows.loss import reverse_KL_F, forward_KL_G
from jflows.train import train_reverse_KL_F, train_forward_KL_G, Monitor
from jflows.boltzmann import boltzmann_reverse_KL_F
from jflows.utils import importance_weights, compute_ESS, resample, langevin

help(NSF)
```

## Examples

Worked examples for each model — a 2D Gaussian mixture, a 3D periodic (NCSF) target, and the 4D two-charge annealed Boltzmann generator — live in [`example/`](example). See [`example/results.md`](example/results.md) for the scripts, figures, and discussion.

## Acknowledgements

`jflows` is strongly inspired by [zuko](https://github.com/probabilists/zuko): the flow, transform, and masked-MLP machinery vendored into `jflows.core` is a stripped-down port of zuko's. Credit for the underlying design — and for the clean, composable `Transform` API the public flows build on — belongs to the zuko authors.

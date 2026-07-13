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

`RealNVP(mixing="lu")` requires float32 or float64 parameters because JAX's
GPU triangular-solve primitive does not support float16/bfloat16 inverses.

For `CNF(exact=False)`, Hutchinson trace estimation uses one call-shared probe.
The packed trainers replace its key every optimizer step, yielding stochastic
trace gradients without letting the drift adapt to one frozen projection.
Low-level loss calls accept an optional `trace_key`; omitting it retains the
constructor key for backward-compatible, reproducible evaluation.

All subclass the same `Flow` abstract class (an `eqx.Module`, i.e. an immutable pytree). The flow itself is the user-facing object:

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

`OTFlow` has one special initialization constraint: its positive-semidefinite
quadratic head is parameterized as $A^\top A$, so the exact identity
`OTFlow.zeros()` necessarily leaves that head at the absorbing point $A=0$.
Use `OTFlow(...).near_identity()` for energy training of the full model. It
keeps the same pytree/checkpoint layout and seeds $\lVert A\rVert=10^{-6}$,
making the map a numerical identity in float32 while preserving a nonzero
gradient. Keep `zeros()` when an exact identity map is required.

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

Naming follows one simple rule throughout the project: classes capitalize the first letter of each word (`My_Potential`, `Nlog_Gaussian`), instances lowercase it (`u`, `u0`, `u_target`). Built-ins are `Nlog_Uniform`, `Nlog_Gaussian`, `Nlog_Gaussian_Mixture` (all with key-first `.samples(key, N)`), and `potential_from(...)` wraps a plain `(x) -> Array` callable into a ready-to-use instance. `Nlog_Uniform(a, b)` uses `[a, b]` for sampling but intentionally evaluates to the same constant outside the box; wrap periodic coordinates or provide an explicit confining potential when a hard support boundary is required.

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
    linear_weights_from_log,                      # safe max-shifted conversion
    compute_ESS, compute_ESS_log,                 # effective sample size
    coverage,                                     # k-NN mode-collapse diagnostic
    resample,                                     # multinomial resampling
    langevin, langevin_step,                      # ULA / MALA / tamed
    stochastic_heun, hamiltonian_monte_carlo,     # more rejuvenation kernels
    sequential_monte_carlo, smc,                  # annealed SMC over a bridge ladder
    annealed_importance_sampling, ais,            # AIS through a trained flow
    lbfgs, adamw,                                 # batched optimizers (+ init/step kernels)
)

log_w = importance_weights_log(samples, source, target, flow, type="F", chunks=1)
ess   = compute_ESS_log(log_w)
y     = ais(key, samples, source, target, flow, type="G",
            ladder=1, mc_dt=1e-3, mc_steps=100)
```

`sequential_monte_carlo` is classical potential-space SMC and rejuvenates at
the matching intermediate bridge at each level. Flow-proposal
`annealed_importance_sampling` uses one fraction of the proposal-to-target log
weight per level but rejuvenates at the final target every time. This avoids
flow-density derivatives inside MCMC, so it is a deliberately biased,
score-free target surrogate rather than exact AIS/SMC.
Its first correction is evaluated directly during the original source-to-target
push (including the matching Jacobian); later levels refresh latent pre-images
after resampling/rejuvenation. This distinction matters for fixed-step CNF and
OTFlow maps, whose numerical inverse is approximate.

`chunks` splits batches along dim 0 (statistically equivalent to `chunks=1`). The
standalone importance-weight helper and the full-set weight evaluation used by
the Boltzmann drivers iterate eagerly over compiled per-chunk kernels, which
reliably bounds those operations' peak graph size. A chunk loop nested inside a
larger compiled stage (notably particle advancement/rejuvenation) is an
execution partition, but XLA may schedule buffers across chunks; it is not a
strict peak-VRAM guarantee. Rejuvenation and optimizer routines retain their
compiled-scan implementations.

**Medium level: training drivers.** Every `jflows.train` stage driver calls an
`eqx.filter_jit`-compiled kernel that runs all Adam steps in one `lax.scan`. Sampling,
loss/gradient evaluation, and the Adam update remain in that compiled stage.
Every step draws a new subset and regenerates its training data, so no frozen
batch is reused:

```python
from jflows.train import train_reverse_KL_F, train_forward_KL_G, Monitor

# reverse KL: each step draws batch_size samples from the fixed set x_valid and
# freshens them with Langevin rejuvenation at the source (flow fixed as F)
flow, batch_ess_hist = train_reverse_KL_F(
    x_valid, source, target, flow,
    batch_size=2000, train_steps=200, lr=1e-3,
    mc_dt=1e-3, mc_steps=100,
)

# forward KL: each step manufactures its target batch by one-level AIS
# through the CURRENT flow (pushforward -> reweight -> resample -> Langevin;
# flow fixed as G)
flow, batch_ess_hist = train_forward_KL_G(
    x_valid, source, target, flow,
    batch_size=2000, train_steps=200, lr=1e-3,
    ladder=1, mc_dt=1e-3, mc_steps=100,
)
```

Both drivers are deterministic (per-step keys derive from a fixed internal
seed). Each fixed configuration compiles as one stage call and returns the
trained flow together with the per-step proposal-to-target batch-ESS history
(measured before AIS correction for the forward trainers). An optional
`Monitor` reports from inside the compiled scan via `jax.debug.callback`:

```python
flow, batch_ess_hist = train_reverse_KL_F(
    ..., monitor=Monitor(every=10, prefix="[reverse KL] ")
)
# [reverse KL] step    10   loss = +4.7476e+00   ESS = 0.3542
# [reverse KL] step    20   loss = +3.7126e+00   ESS = 0.4879
```

Across the public API, `dt` denotes an integration step size and `steps` a
count. Composite controls use `mc_dt` / `mc_steps`, `opt_alpha` /
`opt_steps`, and `train_steps`; cardinalities use `batch_size`, `pool_size`,
and `chunks`. The former `step` / `iters`, `mc_step` / `mc_iters`,
`opt_step` / `opt_iters`, `n_batch`, `n_pool`, and `chunk` keywords remain
accepted as compatibility aliases, and passing both spellings is an error.

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
    pool_size=24000, batch_size=2000, train_steps=500, lr=1e-4,
    ladder=1, mc_dt=1e-3, mc_steps=100,
    bg_param={"t_safe": 0.1, "shrink_factor": 0.7, "enlarge_factor": 1.5, "tau_ess": 0.6},
    flow_dir="run/flows",
)
# y_valid : advanced particle set (at target iff stages[-1]["t"] == 1)
# stages  : accepted-stage records. Full-validation scalars are
#           valid_selected_ess, valid_trained_ess, and valid_identity_ess.
#           Attempt-aligned diagnostics are t_hist, batch_ess_hist,
#           valid_trained_ess_hist, and valid_identity_ess_hist.
#           Every trained candidate, including rejected ones, is recoverable
#           through trained_flow_path_hist; selected_flow_path names the
#           committed trained-or-identity stage map.
```

The adaptive coefficient/retry loop stays in Python, while every accepted or
rejected training attempt invokes one compiled stage scan. Selection SMC,
stage advancement, and fixed-shape flow operations retain the committed
`filter_jit` treatment; retuned bridge coefficients are array leaves. The
full-set importance-weight helper evaluates `chunks` partitions sequentially to
bound peak memory.

## Compatibility with zflows

`jflows` is a mathematical JAX port, not a drop-in replacement for `zflows`.
The main migration points are:

- random operations take explicit key-first JAX PRNG arguments;
- Equinox flows are immutable, so identity initialization must be rebound as
  `flow = flow.zeros()`;
- public losses, importance weights, and AIS take a `Flow`, not `flow.t()`;
- losses return per-sample vectors and callers apply reductions explicitly;
- temperature is represented by scaling potentials rather than by a `beta`
  argument;
- potential names, F/G dispatch, and checkpoint formats differ;
- MCMC defaults to adjusted MALA, while zflows historically defaulted to ULA;
- NCSF is a genuine torus flow with periodic conditioning and a shared seam
  derivative, rather than the legacy raw-coordinate circular spline.

Forward-trainer `batch_ess_hist` means proposal-to-target importance ESS on the
source minibatch immediately before AIS correction. The temporary stage-record
alias `ess_history` contains only the accepted attempt's final row; new code
should use `batch_ess_hist`. Histories from jflows
before commit `f090ffa` used a post-AIS target-batch concentration statistic;
commit `f090ffa` (still package version 0.1.0) instead reconstructed the
pre-AIS proposal through a numerical inverse/forward round trip. Version 0.2.0
evaluates that proposal directly during the original push, avoiding inverse
integration error for CNF and OTFlow. These histories are not numerically
interchangeable. Experiment artifacts should record the package version and
commit and, for version 0.2.0 histories, the semantic tag
`proposal_pre_ais_v1`.

**Package layout.**

```
jflows
├── _artifacts.py
├── boltzmann.py
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
    ├── _compat.py
    ├── anneal.py
    ├── __init__.py
    ├── metrics.py
    ├── optimization.py
    ├── quench.py
    └── rejuvenation.py
```

## Installation

`jflows` is pure Python. A fresh pip-only virtual environment using the latest
compatible releases is the recommended setup. On Linux with an NVIDIA CUDA 13
driver:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda13]" equinox
```

Use `jax[cuda12]` for a CUDA 12 system, or plain `jax` for CPU-only work. Then
install the current checkout conventionally:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/xuda-ye-math/jflows.git "$HOME/src/jflows"
cd "$HOME/src/jflows"
pip install -e .
```

To run the plotting examples, install the optional plotting dependency with
`pip install -e ".[examples]"` instead.

Verify both accelerator selection and editable source provenance explicitly:

```bash
pip check
python
```

Then enter:

```python
>>> from pathlib import Path
>>> import jax
>>> import jflows
>>> print(jax.default_backend(), jax.devices())
>>> print(Path(jflows.__file__).resolve())
```

Repository examples and smoke modules use the same pattern:

```bash
python -m smoke.test_flow
```

**Checkpoint skeletons.** Equinox serializes array leaves into a caller-built
model skeleton. Reconstruct the same class and architecture before calling
`eqx.tree_deserialise_leaves`. NSF/NCSF orderings and masked matrices are array
leaves and are restored from the checkpoint. A `RealNVP` with `randmask=True`
must use the same constructor key because its coupling masks are static tuples.
CNF PRNG state is stored as ordinary uint32 key data and is supported by the
standard Equinox leaf serializer; reconstruct its skeleton with the same JAX
PRNG implementation (`threefry2x32`, `rbg`, etc.), although the seed itself may
differ.

Checkpoints written by jflows versions before box bounds were normalized to a
floating dtype need a one-time skeleton migration if their NSF/NCSF constructor
used integer lists. Build the same skeleton, temporarily replace its `a` and
`b` leaves with arrays of the legacy integer dtype via `eqx.tree_at`, deserialize,
then cast those two informational leaves back to the flow's `center.dtype`.
All transforms use the already-floating `center`/`halfwidth`; checkpoints made
with floating bounds, and all new checkpoints, need no migration.

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

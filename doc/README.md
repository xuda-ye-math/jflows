# jflows documentation

This directory is the standalone user documentation for the generic
`jflows` package. It expands the interface hierarchy introduced in the
[README Core features section](../README.md#core-features) and keeps the same order:

1. low-level building blocks;
2. medium-level single-stage trainers; and
3. high-level annealed Boltzmann generators.

The live source is authoritative. Public code should import from
`jflows.flow`, `jflows.potential`, `jflows.loss`, `jflows.utils`,
`jflows.train`, `jflows.artifacts`, or `jflows.boltzmann`. The retired
`jflows.training` namespace and the private `jflows.core` implementation are
not user interfaces.

## Documentation tree

```text
doc/
├── README.md                 # orientation, hierarchy, conventions
├── 01-low-level.md           # flows, potentials, losses, metrics, samplers
├── 02-medium-level.md        # direct train_* drivers and simple artifacts
├── 03-high-level.md          # Boltzmann generators and stage persistence
├── 04-smoke-tests.md         # executable contract checks by API area
└── 05-examples.md            # complete workflows and verified outputs
```

Read the numbered files in order. The first three are the API manual; the last
two connect each interface to executable repository evidence.

## Interface tree

```text
jflows
├── LOW LEVEL
│   ├── flow
│   │   ├── Flow
│   │   ├── NSF / NCSF
│   │   ├── CNF / OTFlow / RealNVP
│   │   └── public transform primitives
│   ├── potential
│   │   ├── Potential / potential_from
│   │   ├── Nlog_Uniform
│   │   ├── Nlog_Gaussian / Nlog_Gaussian_Mixture
│   │   └── potential algebra / linear_combination
│   ├── loss
│   │   ├── reverse_KL_F
│   │   ├── forward_KL_G
│   │   ├── forward_KLX_G
│   │   └── forward_X_G
│   └── utils
│       ├── weights / ESS / coverage / resampling
│       ├── Langevin / stochastic Heun / HMC
│       ├── SMC / flow-proposal AIS
│       ├── L-BFGS / AdamW
│       └── quench_and_temper
├── MEDIUM LEVEL
│   ├── train
│   │   ├── Monitor
│   │   ├── train_reverse_KL_F
│   │   ├── train_forward_KL_G
│   │   ├── train_forward_KLX_G
│   │   └── train_forward_KLXX_G
│   └── artifacts
│       ├── save_flow / load_flow
│       ├── save_samples / load_samples
│       └── save_history / load_history
└── HIGH LEVEL
    └── boltzmann
        ├── four adaptive boltzmann_* generators
        ├── four fixed-schedule boltzmann_*_fixed generators
        ├── accepted-stage records and identity fallback
        ├── write: create / stage / finish
        └── load: validate / load / fork / run / stage readers
```

The dependency direction is downward: a high-level generator uses a
medium-level trainer; a trainer uses low-level losses and sampling tools.
Direct use of a lower level remains supported when a custom algorithm needs
more control.

## Package tree

The corresponding public source layout is:

```text
jflows/
├── __init__.py
├── artifacts.py
├── boltzmann/
│   ├── __init__.py
│   ├── load.py
│   └── write.py
├── flow.py
├── loss.py
├── potential.py
├── train.py
├── utils/
│   ├── __init__.py
│   ├── anneal.py
│   ├── metrics.py
│   ├── optimization.py
│   ├── quench.py
│   └── rejuvenation.py
└── core/                     # private implementation, not a public import
```

## Core conventions

- A `Potential` is an energy `U`; its unnormalized density is proportional to
  `exp(-U(x))`.
- Batched energies map `[N, d]` to `[N]`. Flow maps preserve `[N, d]`.
- Random entry points take an explicit JAX PRNG key. Split or fold keys for
  logically independent operations.
- Equinox modules are immutable. Rebind `flow = flow.zeros()` and every
  returned trained flow.
- `F` means source to target. `G = F^{-1}` means target to source. For an
  F-native flow, generate with `flow(x)`; for a G-native flow, generate from
  source samples with `flow.inv(x)`.
- Primitive kernels use `dt` and `steps`; composite controls use `mc_dt`,
  `mc_steps`, `opt_dt`, `opt_steps`, `train_steps`, `batch_size`, and
  `pool_size`.
- `chunks` is the number of row partitions, not the number of rows in one
  partition. It is the only generic memory-partition keyword.
- MALA is the default Langevin mode. Positive taming is for ULA and therefore
  requires `adjust=False`.
- Finite-state safeguards are local and explicit: RQS inversion guards
  roundoff, MALA rejects nonfinite proposals, L-BFGS and AdamW reject
  nonfinite state transitions, resampling excludes nonfinite rows when a
  finite row exists, and a Boltzmann generator raises before yielding a
  nonfinite post-stage population.
- Prefer log weights and `compute_ESS_log` for numerically difficult targets.
  Normalized ESS lies in `[0, 1]`.

## Reading routes

- Building a custom sampler or loss: start with
  [Low-level interfaces](01-low-level.md).
- Training one direct source-to-target map: continue to
  [Medium-level interfaces](02-medium-level.md).
- Bridging a difficult target through accepted stages: continue to
  [High-level interfaces](03-high-level.md).
- Finding the executable contract for one function: use
  [Smoke tests](04-smoke-tests.md).
- Starting from a complete scientific script: use
  [Examples](05-examples.md).

## Runtime setup

The project targets accelerator-backed JAX. From a source checkout:

```bash
source ~/.envs/jflows/bin/activate
PYTHONPATH=/data/projects/jflows python your_script.py
```

For standalone programs that should avoid JAX's default full-device
preallocation, set the option before importing JAX:

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
```

Compilation is lazy. Warm up compiled calls before timing and synchronize with
`jax.block_until_ready(...)` at timing boundaries.

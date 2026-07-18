<p align="center"><img src="jflows.png" alt="jflows banner" width="800px"></p>
<p align="center"><sub><em>Banner designed by ChatGPT: the character Jax from "The Amazing Digital Circus", a nod to jflows being built on Google JAX.</em></sub></p>

# jflows

`jflows` is a JAX/Equinox package for unconditional normalizing flows,
energy-based sampling, and annealed Boltzmann generators. It provides a small
public API for building flows, defining unnormalized target energies, training
one transport stage, and connecting accepted stages into a complete source-to-
target sampler.

> **Status: experimental.** The package is developed and tested on Linux with
> NVIDIA CUDA JAX. JAX GPU is not supported on Windows, including WSL.
>
> This project was developed with Codex.

## Core features

- **First-class spline flows.** `NSF` models bounded Euclidean boxes and
  `NCSF` models periodic boxes or tori. `CNF`, `OTFlow`, and `RealNVP` provide
  complementary continuous and affine-coupling architectures.
- **One immutable flow interface.** Every flow provides `flow(x)`,
  `flow.call_and_ladj(x)`, `flow.inv(y)`, and `flow.inv_and_ladj(y)` and works
  as an Equinox pytree under JIT, vectorization, and differentiation.
- **Unified energy interface.** A `Potential` maps `[N,d]` samples to `[N]`
  energies for densities proportional to `exp(-U)`. Built-in uniform, Gaussian,
  and Gaussian-mixture potentials support explicit-key sampling, while
  `potential_from` wraps a plain batched energy function.
- **Potential algebra.** Addition, subtraction, scaling, and
  `linear_combination` construct bridge energies without introducing a second
  target abstraction.
- **Explicit randomness.** Every random constructor and sampler takes a JAX
  PRNG key. There is no package-global random state.
- **Sampling and diagnostics.** Importance weights, normalized ESS, coverage,
  resampling, Langevin/MALA, stochastic Heun, HMC, SMC, flow-proposal AIS,
  L-BFGS, AdamW, and quench-and-temper are available from `jflows.utils`.
- **Progressive training objectives.** Reverse KL, forward KL, KLX, and KLXX
  share one validation convention while adding progressively richer
  target-batch and density-ratio information.
- **Complete-stage persistence.** Boltzmann computation is separate from
  optional stage writers/loaders, so an interrupted stage can be recomputed
  from the last published population.

Flows are immutable. Identity initialization returns a new object:

```python
flow = flow.zeros()
```

`OTFlow` is the training exception: use `flow.near_identity()` so its
quadratic factor retains a nonzero gradient. Keep `OTFlow.zeros()` for an exact
identity map.

## Three-level interface

The public API is deliberately arranged from basic numerical components to a
complete annealed generator.

### LOW — building blocks

The low level contains flows, potentials, per-sample losses, metrics, MCMC,
SMC/AIS, optimization, and quench-and-temper:

```python
from jflows.flow import NSF, NCSF, CNF, OTFlow, RealNVP
from jflows.potential import Potential, Nlog_Gaussian, potential_from
from jflows.loss import reverse_KL_F, forward_KL_G, forward_KLX_G, forward_X_G
from jflows.utils import compute_ESS_log, importance_weights_log, langevin, smc
```

`F` denotes the source-to-target map and `G = F^{-1}` the target-to-source
map. An `_F` or `_G` suffix fixes the native training direction.

### MEDIUM — one-stage training

The medium level trains one map between a source and target potential:

```python
from jflows.train import (
    Monitor,
    train_reverse_KL_F,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
)
```

Every trainer returns a new flow and its optimizer-batch ESS history. Simple
flow, sample, and history serialization lives in `jflows.artifacts`.

### HIGH — annealed Boltzmann generation

The high level connects accepted bridge stages and compares every trained map
with an exact identity fallback using full-validation incremental ESS:

```python
from jflows.boltzmann import (
    boltzmann_reverse_KL_F,
    boltzmann_forward_KL_G,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
)
```

- reverse KL is the simplest source-sampled F baseline;
- forward KL manufactures target-side batches and trains G;
- KLX adds target-measure log-weight variation control; and
- KLXX adds a quench-and-temper/proposal mixture for wider mode and leakage
  regularization.

Adaptive and fixed-schedule variants use the same stage record and identity
selection rules. Complete-stage persistence is exposed separately through
`jflows.boltzmann.write` and `jflows.boltzmann.load`.

## Minimal setup

```bash
pip install "jax[cuda13]" equinox
pip install -e .
```

```python
import jax

from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.train import train_forward_KLX_G
from jflows.utils import compute_ESS_log, importance_weights_log

source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
target = potential_from(
    lambda x: 10.0 * (jax.numpy.linalg.norm(x, axis=-1) - 2.0) ** 2
)

flow = NSF(
    jax.random.key(0),
    [-4.0, -4.0],
    [4.0, 4.0],
    bins=16,
    transforms=4,
).zeros()
x_valid = source.samples(jax.random.key(1), 20000)

flow, batch_ess = train_forward_KLX_G(
    x_valid,
    source,
    target,
    flow,
    batch_size=500,
    train_steps=200,
    lr=1e-3,
    ladder=1,
    mc_dt=1e-3,
    mc_steps=50,
    coeff_lambda=1.0,
)

log_weights = importance_weights_log(
    x_valid, source, target, flow, type="G"
)
validation_ess = compute_ESS_log(log_weights)
y_valid = flow.inv(x_valid)
```

## Repository map

```text
jflows/
├── doc/
│   ├── README.md                 # detailed documentation map
│   ├── 01-low-level.md
│   ├── 02-medium-level.md
│   ├── 03-high-level.md
│   ├── 04-smoke-tests.md
│   └── 05-examples.md
├── jflows/
│   ├── flow.py                   # LOW
│   ├── potential.py              # LOW
│   ├── loss.py                   # LOW
│   ├── utils/                    # LOW
│   ├── train.py                  # MEDIUM
│   ├── artifacts.py              # MEDIUM
│   ├── boltzmann/                # HIGH
│   └── core/                     # private implementation
├── smoke/                        # narrow executable contracts
├── example/                      # complete workflows and results
└── README.md                     # this concise project introduction
```

## Read next

- **Detailed interface documentation:** [`doc/`](doc/README.md)
- **Executable API contracts:** [`smoke/`](smoke/)
- **Complete scripts, figures, and numerical results:**
  [`example/`](example/) and [`example/results.md`](example/results.md)

The detailed documentation gives exact signatures, direction conventions,
return values, memory controls, stage records, and persistence behavior. Smoke
tests establish narrow software contracts; examples demonstrate end-to-end
scientific workflows.

## Acknowledgements

`jflows` is strongly inspired by [zuko](https://github.com/probabilists/zuko).
The private flow, transform, and masked-MLP machinery in `jflows.core` is a
stripped-down port of zuko's clean and composable transform design.

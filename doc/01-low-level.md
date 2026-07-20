# Low-level interfaces

The low level contains the objects and numerical kernels from which custom
energy-based sampling pipelines are assembled. This page follows the README
Core features order: flows, explicit randomness, potentials, losses, then the
SMC toolkit.

## Public imports

```python
from jflows.flow import (
    Flow, NSF, NCSF, CNF, OTFlow, RealNVP,
    Transform, ComposedTransform,
    MonotonicRQSTransform, CircularRQSTransform,
)
from jflows.potential import (
    Potential, potential_from, linear_combination,
    Nlog_Uniform, Nlog_Gaussian, Nlog_Gaussian_Mixture,
)
from jflows.loss import (
    reverse_KL_F, forward_KL_G, forward_KLX_G, forward_X_G,
)
from jflows.utils import *
```

Use the explicit imports shown throughout this page in application code. The
wildcard above only summarizes the flat utility namespace.

## Flow interface

Every model subclasses the immutable Equinox `Flow` base. The common methods
are:

<div align="center">

<table>
<thead>
<tr><th>Call</th><th>Meaning</th><th>Return</th></tr>
</thead>
<tbody>
<tr><td><code>flow(x)</code></td><td>native forward map</td><td>transformed array <code>[N, d]</code></td></tr>
<tr><td><code>flow.call_and_ladj(x)</code></td><td>forward map and <code>log|det J|</code></td><td><code>(y, ladj)</code></td></tr>
<tr><td><code>flow.inv(y)</code></td><td>inverse map</td><td>pre-image array <code>[N, d]</code></td></tr>
<tr><td><code>flow.inv_and_ladj(y)</code></td><td>inverse map and inverse log-Jacobian</td><td><code>(x, ladj)</code></td></tr>
<tr><td><code>flow.t()</code></td><td>underlying composed transform</td><td><code>ComposedTransform</code></td></tr>
<tr><td><code>flow.zeros()</code></td><td>identity-parameterized copy</td><td>new flow</td></tr>
<tr><td><code>flow.with_trace_key(key)</code></td><td>replace a stochastic CNF probe</td><td>new flow or self</td></tr>
<tr><td><code>flow.needs_trace_key</code></td><td>whether the flow uses stochastic traces</td><td>Boolean</td></tr>
</tbody>
</table>

</div>

The methods accept both individual vectors where supported by the underlying
transform and batches in normal package workflows. Losses, weights, trainers,
and generators use batches.

### Direction convention

`F` and `G` name the transform represented by the trainable flow:

```text
source x  --F-->  target y
source x  <--G--  target y
              G = F^{-1}
```

- An `_F` objective applies the flow in its native direction during training.
  Generate target samples with `y = flow(x)`.
- A `_G` objective also applies the flow in its native direction during
  training, but that native direction is target to source. Generate target
  samples from source particles with `y = flow.inv(x)`.
- `importance_weights*` and `annealed_importance_sampling` take
  `type="F"` or `type="G"` because they support either convention.

### Flow constructors

```python
NSF(key, a, b, bins=8, slope=1e-3, transforms=4, randmask=True,
    hidden_features=(64, 64), activation=jax.nn.silu)

NCSF(key, a, b, bins=8, slope=1e-3, transforms=4, randmask=True,
     hidden_features=(64, 64), activation=jax.nn.silu)

CNF(key, dimension, frequency=3, nt=16, exact=True,
    hidden_features=(64, 64), activation=jax.nn.silu)

OTFlow(key, dimension, hidden=64, layer=3, rank=10, nt=8,
       time_bound=(0.0, 1.0))

RealNVP(key, dimension, transforms=4, randmask=True, mixing=None,
        hidden_features=(64, 64), activation=jax.nn.silu)
```

<div align="center">

<table>
<thead>
<tr><th>Model</th><th>Domain and main use</th><th>Important behavior</th></tr>
</thead>
<tbody>
<tr><td><code>NSF</code></td><td>bounded box <code>[a,b]^d</code>; first-class nonperiodic model</td><td>monotonic rational-quadratic spline; forward is parallel and inverse autoregressive</td></tr>
<tr><td><code>NCSF</code></td><td>periodic box/torus; first-class angular model</td><td>circular splines and periodic conditioner preserve seams</td></tr>
<tr><td><code>CNF</code></td><td><code>R^d</code>; flexible ODE flow</td><td>fixed-step RK4; <code>exact=False</code> uses a call-shared Hutchinson trace probe</td></tr>
<tr><td><code>OTFlow</code></td><td><code>R^d</code>; transport-inspired continuous flow</td><td>closed-form trace and fixed integration grid</td></tr>
<tr><td><code>RealNVP</code></td><td><code>R^d</code>; fast affine-coupling baseline</td><td>optional coordinate mixing; <code>mixing="lu"</code> needs float32 or float64</td></tr>
</tbody>
</table>

</div>

`NSF` and `NCSF` use `a` and `b` as coordinatewise bounds. `bins` controls
spline resolution, `transforms` controls stacked autoregressive layers, and
`hidden_features` gives the conditioner widths.

For a CNF, `nt` is the number of fixed RK4 steps. `exact=True` evaluates the
trace exactly; `exact=False` uses a stochastic estimator. Low-level loss and
weight functions accept `trace_key` so callers can refresh the probe.

### Initialization

```python
flow = NSF(key, a, b).zeros()
flow = NCSF(key, a, b).zeros()
flow = CNF(key, dimension).zeros()
flow = RealNVP(key, dimension).zeros()
```

All of these return a new exact-identity parameterization. `OTFlow` is the
exception for training:

```python
flow = OTFlow(key, dimension).near_identity()
```

Its quadratic factor is parameterized as `A.T @ A`; exact zero is an absorbing
zero-gradient point. `near_identity(quadratic_eps=1e-6)` preserves a usable
gradient. Keep `OTFlow.zeros()` only when an exact identity map is required,
such as a pure identity fallback.

### Public transform primitives

Advanced callers can compose stable transform primitives without importing
`jflows.core`:

```python
ComposedTransform(*transforms)
MonotonicRQSTransform(
    widths, heights, derivatives,
    bound=1.0, slope=1e-3, circular=False,
)
CircularRQSTransform(*phi, bound=math.pi, slope=1e-3)
```

`Transform` is the base protocol for custom transforms, not a directly usable
identity transform: a subclass must implement its map, inverse, and
log-Jacobian behavior. `ComposedTransform` and the two RQS constructors above are
concrete public constructors. Ordinary sampling and training should use a
`Flow` directly; transform primitives are intended for custom flow packages
and explicit composition.

The monotonic RQS inverse guards a valid-bin discriminant that can round
slightly below zero in finite precision, and clips the recovered normalized
coordinate to its selected bin. This protects ordinary float32 inversion from
cancellation-induced NaNs; it is not a repair for nonfinite inputs or invalid
spline parameters.

## Explicit PRNG keys

There is no package-global RNG. Constructors and random kernels take a JAX key
explicitly:

```python
key = jax.random.key(0)
key_flow, key_samples, key_train = jax.random.split(key, 3)

flow = NSF(key_flow, [-4.0, -4.0], [4.0, 4.0]).zeros()
x = source.samples(key_samples, 20000)
```

Do not reuse one key for logically independent draws. Inside repeatable loops,
derive operation keys with `jax.random.fold_in` or split a parent key.

## Potential interface

A `Potential` represents an energy `U(x)` for an unnormalized density
proportional to `exp(-U(x))`. A potential consumes a batch `[N,d]` and returns
one energy per row, `[N]`.

```python
class Ring(Potential):
    def __call__(self, x):
        radius = jnp.linalg.norm(x, axis=-1)
        return 10.0 * (radius - 2.0) ** 2

target = Ring()
energy = target(x)       # [N]
gradient = target.grad(x)  # [N, d]
```

`Potential.grad(x)` applies `jax.vmap` to the gradient of a one-row call,
`self(x_i[None])[0]`, and returns one gradient per row. Custom potentials
should therefore evaluate rows independently. `potential_from(fn)` is the
concise alternative:

```python
target = potential_from(
    lambda x: 10.0 * (jnp.linalg.norm(x, axis=-1) - 2.0) ** 2
)
```

### Built-in potentials

```python
Nlog_Uniform(a, b)
Nlog_Gaussian(mean, variance)
Nlog_Gaussian_Mixture(weights, mean, variance)
```

All built-ins implement `samples(key, N)`. Gaussian parameters are variances,
not standard deviations. Mixture `weights` are component weights; `mean` and
`variance` have one row per component.

`Nlog_Uniform(a,b)` samples uniformly from the box but intentionally evaluates
to zero everywhere, including outside the box. It is not a hard-support
barrier. Wrap periodic coordinates or provide an explicit confining energy
when support outside the box must be excluded. Gaussian-mixture weights may be
unnormalized; the constructor normalizes them internally.

### Potential algebra

Potential expressions remain `Potential` objects:

```python
bridge = (1.0 - t) * source + t * target
shifted = target - source
scaled = target / 2.0
combined = sum([source, target])

bridge = linear_combination(
    [target, source],
    [t, 1.0 - t],
)
```

Supported operations are scalar multiplication and division, addition,
subtraction, negation, and `sum(...)`. `linear_combination(potentials,
coeffs=None)` is the explicit annealing constructor. With `coeffs=None`, it
uses uniform coefficients `1 / len(potentials)`, so the result is an average,
not a sum. Nested linear combinations are flattened, and repeated references
to the same potential object are merged by object identity with their
coefficients added. The coefficient array is a pytree leaf, so changing bridge
coefficients can reuse one compiled structure.

## Per-sample losses

Low-level losses return a vector `[N]`. Call `.mean()` to obtain the scalar
objective used by a gradient update.

### `reverse_KL_F`

```python
reverse_KL_F(x, target, flow, trace_key=None)
```

For source samples `x`, it computes

```text
target(F(x)) - log|det J_F(x)|.
```

Its mean estimates reverse KL up to a constant. The map is evaluated as `F`;
no inverse appears in training.

### `forward_KL_G`

```python
forward_KL_G(y, source, flow, trace_key=None)
```

For target samples `y`, it computes

```text
source(G(y)) - log|det J_G(y)|.
```

Its mean estimates forward KL up to a constant. The map is evaluated as `G`.

### `forward_KLX_G`

```python
forward_KLX_G(
    y, source, target, flow, key,
    coeff_lambda=1.0, trace_key=None,
)
```

Define the log-density-ratio coordinate

```text
z(y) = source(G(y)) - target(y) - log|det J_G(y)|.
```

With a random batch permutation `perm`, the function returns

```text
z + coeff_lambda * abs(z - z[perm]).
```

The second term penalizes spread of the density ratio, which plain forward KL
does not directly control.

### `forward_X_G`

```python
forward_X_G(y, source, target, flow, key, trace_key=None)
```

This returns only `abs(z - z[perm])`. The sample population `y` can come from
the target or another weight measure. KLXX uses this idea with a mixture of a
quench-and-temper coverage measure and the detached flow proposal.

## Metrics and resampling

### Importance weights

```python
importance_weights_log(
    samples, source, target, flow, type,
    chunks=1, trace_key=None,
)
importance_weights(
    samples, source, target, flow, type,
    chunks=1, trace_key=None,
)
linear_weights_from_log(log_weights)
```

`samples` are always drawn from `source`. `type` specifies whether the trained
flow is F-native or G-native. Both interfaces return unnormalized weights.
The linear function uses a max-shifted exponential and is suitable for
self-normalized uses such as resampling. Keep log weights for difficult
targets.

```python
log_w = importance_weights_log(
    x, source, target, flow, type="G", chunks=8,
)
ess = compute_ESS_log(log_w)
y = flow.inv(x)
```

### ESS

```python
compute_ESS(weights)
compute_ESS_log(log_weights)
```

The package reports normalized effective sample size:

```text
ESS = (sum_i w_i)^2 / (N * sum_i w_i^2),
```

so the result is in `[0,1]`, not `[1,N]`. `compute_ESS_log` uses log-sum-exp
and is preferred when weights have a large dynamic range.

### Coverage

```python
coverage(y, x, k=5, chunks=1)
```

`x` is a wide-coverage reference population and `y` is the candidate
population. For every reference point, the function constructs its `k`-NN
ball within `x` and checks whether at least one candidate lies inside. The
returned fraction lies in `[0,1]`. Coverage complements ESS: it is a geometric
mode-reach diagnostic, while ESS measures proposal-to-target weight balance.

### Resampling

```python
resample(key, samples, weights, N=None)
```

This performs multinomial resampling with replacement using an inverse CDF.
`N` defaults to the input population size. Weights need not sum to one.
When at least one sample row is finite, nonfinite rows receive zero selection
mass. Positive infinities on eligible rows share the mass; invalid or
zero-total weight vectors fall back to uniform resampling over the eligible
finite rows. If every sample row is nonfinite, resampling cannot manufacture a
finite output.

## Rejuvenation kernels

### Langevin / MALA

```python
langevin_step(key, x, potential, dt=1e-3, adjust=True, taming=0)
langevin(
    key, samples, potential,
    dt=1e-3, steps=100, adjust=True, taming=0, chunks=1,
)
```

`langevin_step` returns `(samples, aux)`. With `adjust=True`, `aux` contains
per-chain acceptance and log-acceptance values. `langevin` returns the final
population after a compiled scan.

- `adjust=True`: Metropolis-adjusted Langevin (MALA), the default.
- `adjust=False`: unadjusted Langevin (ULA), with step-size bias.
- `taming>0`: tamed ULA drift for rapidly growing gradients; it is
  incompatible with `adjust=True`.

MALA maps a nonfinite Metropolis log-acceptance value to `-inf`, rejects that
proposal, and retains the corresponding input particle. This is a rejection
safeguard; it does not alter finite MALA transitions. ULA and stochastic Heun
remain unadjusted integrators and therefore do not have an accept/reject
fallback.

`rejuvenation` is a stable alias of `langevin`.

### Stochastic Heun

```python
stochastic_heun_step(key, x, potential, dt=1e-3)
stochastic_heun(key, samples, potential, dt=1e-3, steps=100, chunks=1)
```

The predictor-corrector reuses one Wiener increment and averages the drift at
the start and predictor. It is unadjusted and uses two gradient evaluations
per step.

### Hamiltonian Monte Carlo

```python
leapfrog(x, p, potential, dt, steps)
hmc_step(key, x, potential, dt=1e-2, leapfrog_steps=10)
hamiltonian_monte_carlo(
    key, samples, potential,
    dt=1e-2, leapfrog_steps=10, trajectories=10, chunks=1,
)
```

`hmc_step` returns `(samples, aux)` with acceptance diagnostics. The complete
routine repeats trajectories and returns the final samples. `hmc` is its
stable alias.

## Annealing routines

### Potential-space SMC

```python
sequential_monte_carlo(
    key, samples, source, target,
    ladder=1, mc_dt=1e-3, mc_steps=100,
    adjust=True, taming=0, chunks=1,
)
```

The result is `(samples, per_level_ess)`, with ESS shape `(ladder,)`. Each level
uses the bridge

```text
U_k = (1 - k/M) U_source + (k/M) U_target
```

and executes reweight, resample, then Langevin at that same intermediate
potential. `smc` is the stable alias.

### Flow-proposal AIS surrogate

```python
annealed_importance_sampling(
    key, samples, source, target, flow, type,
    ladder=1, mc_dt=1e-3, mc_steps=100,
    adjust=True, taming=0, chunks=1,
    trace_key=None, return_initial_log_weights=False,
)
```

This routine starts from source particles, pushes them through the trained
flow, applies fractional proposal-to-target weights, resamples, and
rejuvenates. Unlike classical SMC, rejuvenation targets the final target at
every level. It is therefore a deliberately biased surrogate rather than
exact AIS/SMC. It avoids gradients of a flow-proposal density or intermediate
geometric-path density, but target Langevin still evaluates `target.grad`.

By default it returns samples. With `return_initial_log_weights=True`, it
returns `(samples, initial_log_weights)`. Direct trainers use this initial
proposal diagnostic for batch ESS. `ais` is the stable alias.

## Batched optimization

### L-BFGS

```python
lbfgs_init(x, potential, memory=6)
lbfgs_step(state, potential, alpha=1.0, armijo=False)
lbfgs(
    samples, potential, alpha=1.0, steps=100,
    memory=6, armijo=False, chunks=1,
)
```

`LBFGS_State` stores one optimizer state per sample. The complete `lbfgs`
routine minimizes the potential independently for every row. `optimization`
is the stable alias. A candidate with nonfinite coordinates, cached energy, or
new gradient is rejected row by row, leaving that row at its preceding finite
state. In Armijo mode, such a rejection also reduces the carried trial scale
for the next iteration.

### AdamW

```python
adamw_init(x)
adamw_step(
    state, potential, lr=1e-2,
    beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
)
adamw(
    samples, potential, lr=1e-2, steps=100,
    beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.0, chunks=1,
)
```

`AdamW_State` is public for custom loops. This optimizer moves sample
positions down an energy, rather than optimizing flow parameters. An update is
committed atomically: if its gradient, moments, or resulting coordinates are
nonfinite, the complete state and bias-correction counter for that processed
chunk remain unchanged. A later finite step therefore resumes from the last
valid state.

## Quench and temper

```python
quench_and_temper(
    key, samples, target, melt,
    opt_dt=1.0, opt_steps=100,
    mc_dt=1e-3, mc_steps=100,
    mc_adjust=True, chunks=1,
)
```

The construction executes:

```text
input population
  -> Gaussian melt
  -> per-particle L-BFGS quench to target basins
  -> target Langevin temper
  -> wide-coverage population hat_mu
```

`chunks` is forwarded to both the quench and temper. `qt` is the stable alias.
KLXX uses this wide-coverage population for its additional X regularization.

## Chunking semantics

Every generic API uses only the spelling `chunks`. It means a count of row
partitions. Larger values create smaller partitions.

- Standalone `importance_weights_log` partitions rows in a plain Python loop;
  it does not add a per-chunk JIT wrapper. Boltzmann full-validation weighting
  uses eager sequential calls to its compiled chunk kernels.
- Rejuvenation and optimization calls partition rows but may sit inside an
  enclosing JIT. XLA can co-schedule buffers, so this is not a universal peak
  VRAM guarantee.
- Chunking does not change the intended statistical calculation. Random
  streams can differ because chunk keys are derived separately.

## Minimal custom pipeline

```python
import jax
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.loss import reverse_KL_F
from jflows.utils import (
    compute_ESS_log,
    importance_weights_log,
    langevin,
    linear_weights_from_log,
    resample,
)

source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
target = potential_from(
    lambda x: 10.0 * (jax.numpy.linalg.norm(x, axis=-1) - 2.0) ** 2
)

key_flow, key_samples, key_resample, key_mc = jax.random.split(
    jax.random.key(0), 4
)
flow = NSF(key_flow, [-4.0, -4.0], [4.0, 4.0]).zeros()
x = source.samples(key_samples, 20000)

per_sample_loss = reverse_KL_F(x[:500], target, flow)
log_w = importance_weights_log(x, source, target, flow, type="F", chunks=8)
ess = compute_ESS_log(log_w)
y = flow(x)
y = resample(key_resample, y, linear_weights_from_log(log_w))
y = langevin(key_mc, y, target, dt=1e-3, steps=50, chunks=8)
```

For direct optimization of a flow, use the medium-level trainers rather than
reimplementing their optimizer scans.

## Executable references

The closest low-level checks are:

- [flow and Jacobian tests](../smoke/test_flow.py)
- [periodic seam tests](../smoke/test_circular.py)
- [potential tests](../smoke/test_potential.py)
- [potential-algebra tests](../smoke/test_linear_combination.py)
- [loss formula tests](../smoke/test_loss.py)
- [metrics tests](../smoke/test_metrics.py)
- [rejuvenation tests](../smoke/test_rejuvenation.py)
- [annealing tests](../smoke/test_annealing.py)
- [optimization tests](../smoke/test_optimization.py)
- [flat utility API tests](../smoke/test_utils_api.py)

See [Smoke tests](04-smoke-tests.md) for the full routing table.

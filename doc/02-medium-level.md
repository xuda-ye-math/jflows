# Medium-level interfaces

The medium level trains one flow map over one source-to-target stage. It
composes low-level losses, target-batch construction, Adam updates, monitoring,
and optional clipping into one call. It does not choose an outer stage schedule
or persist a multi-stage run.

Public imports:

```python
from jflows.train import (
    Monitor,
    train_reverse_KL_F,
    train_forward_KL_G,
    train_forward_KLX_G,
    train_forward_KLXX_G,
    train_forward_KLL1_G,
    train_FAB_G,
    train_FABX_G,
)
from jflows.artifacts import (
    save_flow, load_flow,
    save_samples, load_samples,
    save_history, load_history,
)
```

## Shared contract

Every trainer takes:

- `x_valid`: the fixed source-side population used to draw fresh optimizer
  batches;
- `source` and `target`: `Potential` objects for the current stage;
- `flow`: the trainable map template;
- `batch_size`, `steps_total`, and `lr`: Adam training controls;
- an integer `seed`: deterministic internal key derivation; and
- optional `monitor`: a `Monitor` called from the compiled optimizer scan.

Every trainer returns:

```python
trained_flow, batch_ess_hist
```

`batch_ess_hist.shape == (steps_total,)`. It is an optimizer-batch diagnostic,
not a held-out acceptance metric. For forward objectives it is computed from
the full proposal-to-target log weights of the pushforward batch, before the
SMC resampling and target rejuvenation. Evaluate the returned flow on the complete validation
population when a final sampling metric is required.

The provided `flow` is preserved as the starting parameterization unless
`initialize_from_identity=True`. Because flows are immutable, always rebind
the return value.

## Execution model

The committed compilation boundaries are part of the interface behavior:

```text
reverse KL / forward KL / KLX
  -> one outer eqx.filter_jit call
  -> one lax.scan containing all Adam steps

KLXX
  -> eager chunked quench_and_temper pool construction
  -> one compiled lax.scan containing all Adam steps
```

Keeping KLXX quench and temper outside the enclosing optimizer JIT makes its
`chunks` partition active at the QT boundary. The scan still contains batch
selection, target-batch manufacture, loss/gradient evaluation, guarded Adam
updates, and monitor callbacks.

## Common controls

<div align="center">

<table>
<thead>
<tr><th>Control</th><th>Meaning</th></tr>
</thead>
<tbody>
<tr><td><code>batch_size</code></td><td>optimizer batch size sampled from <code>x_valid</code></td></tr>
<tr><td><code>steps_total</code></td><td>number of Adam updates</td></tr>
<tr><td><code>lr</code></td><td>Adam learning rate</td></tr>
<tr><td><code>ladder</code></td><td>SMC levels used to manufacture one forward-training batch</td></tr>
<tr><td><code>mc_dt</code></td><td>Langevin step size of every rejuvenation</td></tr>
<tr><td><code>mc_steps_1</code></td><td>MALA steps on the intermediate SMC levels only (at the target for the forward KL trainers, at the level's own distribution through the flow for FAB)</td></tr>
<tr><td><code>mc_steps_2</code></td><td>MALA steps of every other rejuvenation: the last SMC level at the target, the reverse KL source batch, the QT rows of the KLXX mixture batch, and the temper of the KLXX quench-and-temper pool (and its resampling when <code>coeff_qt &gt; 0</code>)</td></tr>
<tr><td><code>mc_adjust</code></td><td><code>True</code> for MALA, <code>False</code> for ULA</td></tr>
<tr><td><code>seed</code></td><td>deterministic trainer key namespace</td></tr>
<tr><td><code>checkpoint</code></td><td>rematerialize the loss calculation during reverse-mode differentiation</td></tr>
<tr><td><code>initialize_from_identity</code></td><td>replace the supplied flow with its trainable identity, or OTFlow near-identity, before training</td></tr>
<tr><td><code>t_start</code>, <code>t_end</code></td><td>labels for the stage interval, used by monitoring</td></tr>
<tr><td><code>u_clip</code></td><td>exclude optimizer-loss rows whose target energy exceeds the threshold</td></tr>
<tr><td><code>g_clip</code></td><td>global gradient-norm clipping threshold</td></tr>
<tr><td><code>chunks</code></td><td>row partitions for KLXX quench and temper</td></tr>
</tbody>
</table>

</div>

`u_clip` changes only the optimizer loss mask; it does not truncate the
validation population or redefine the target. Non-finite loss/gradient updates
are guarded so a rejected update leaves the previous parameters in place.

`checkpoint=True` can reduce activation memory by recomputing the loss during
the backward pass. It does not change the mathematical objective. Its runtime
and memory tradeoff depends on flow architecture and accelerator compiler.

## Monitor

```python
Monitor(every, prefix="", printer=print)
```

The monitor emits every `every`-th step and the final step; the first step
is not special:

```text
[reverse] [t: 0.000000 -> 1.000000] step    20   loss = ...   ESS = ...
```

It uses `jax.debug.callback` from inside the scan. A custom `printer` receives
one formatted string. Keep callback work small because it synchronizes host
reporting with device execution.

```python
monitor = Monitor(every=20, prefix="[forward KL] ")
```

## Reverse KL trainer

```python
train_reverse_KL_F(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    mc_dt, mc_steps_2,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

At every optimizer step:

1. draw `batch_size` rows without replacement from `x_valid`;
2. freshen the batch by `mc_steps_2` Langevin steps at `source`;
3. evaluate `reverse_KL_F(x, target, flow)`;
4. update the F-native flow with Adam; and
5. report the batch proposal ESS.

The flow is trained in the source-to-target direction. Generate with
`y = flow(x)`.

```python
flow, batch_ess = train_reverse_KL_F(
    x_valid, source, target, flow,
    batch_size=500,
    steps_total=200,
    lr=1e-3,
    mc_dt=1e-3,
    mc_steps_2=50,
    monitor=Monitor(20, "[reverse] "),
)
```

## Forward KL trainer

```python
train_forward_KL_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

At every step, the trainer draws a source batch and manufactures approximate
target samples with `sequential_monte_carlo` through the current G flow:
every level rejuvenates at the target, `mc_steps_1` MALA steps on the
intermediate levels and `mc_steps_2` on the last, so the intermediate
levels are a target surrogate and no level differentiates the flow. KLX,
KLL1, and KLXX use the same batch construction; only FAB uses the exact
levels of `sequential_monte_carlo_fab`.
It then minimizes `forward_KL_G` on that fresh population. The target batch is
not frozen across steps.

The trained flow is G-native. Generate from source particles with
`y = flow.inv(x)`.

```python
flow, batch_ess = train_forward_KL_G(
    x_valid, source, target, flow,
    batch_size=500,
    steps_total=200,
    lr=1e-3,
    ladder=1,
    mc_dt=1e-3,
    mc_steps_1=50,
    mc_steps_2=50,
    initialize_from_identity=True,
    u_clip=100.0,
    g_clip=100.0,
)
```

## KLX trainer

```python
train_forward_KLX_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

KLX uses the same fresh SMC target-batch path as forward KL, then adds the
pairwise X penalty on the log-density-ratio coordinate, the exact mean of
`|z_i - z_j|` over all pairs of the batch evaluated by one sort:

```text
mean(z) + coeff_lambda * mean_{i != j}(abs(z_i - z_j)).
```

`coeff_lambda=0` removes the X penalty. The output flow is G-native.

```python
flow, batch_ess = train_forward_KLX_G(
    x_valid, source, target, flow,
    batch_size=500,
    steps_total=200,
    lr=1e-3,
    ladder=1,
    mc_dt=1e-3,
    mc_steps_1=50,
    mc_steps_2=50,
    coeff_lambda=1.0,
)
```

## KLXX trainer

```python
train_forward_KLXX_G(
    x_valid, source, target, flow,
    pool_size, batch_size, steps_total,
    lr, ladder, melt, opt_dt, opt_steps,
    mc_dt, mc_steps_1, mc_steps_2,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    *,
    chunks=1,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

KLXX augments KLX with a second X functional evaluated on a coverage-oriented
mixture. Before the optimizer scan it constructs `hat_pi` with
`quench_and_temper`:

- `pool_size=0`: quench the complete `x_valid` population;
- `pool_size>0`: draw a separate resampled pool of that size from `x_valid`.

The temper of the pool uses `mc_steps_2`, and `coeff_qt > 0` resamples the
pool by the partial importance weights `exp(coeff_qt * (source - target))`
and rejuvenates it (`mc_steps_2` steps) before the melt. There is no second
chunk keyword: `chunks` is passed directly into QT.

During each training step:

1. manufacture an SMC target batch `y` through the current flow, whose
   pushforward proposal is kept as the detached `y_bar`;
2. draw `y_hat` from the fixed QT pool and rejuvenate it at the target for
   `mc_steps_2` steps;
3. mix `y_hat` and `y_bar` with probabilities `coeff_alpha` and
   `1 - coeff_alpha`; and
4. optimize the target KLX term plus `coeff_theta` times the mixture X term.

`coeff_lambda` weights the target-measure X term, `coeff_theta` weights the
mixture X term, and `coeff_alpha` is the QT proportion of the mixture. Both
X terms are the exact sorted pairwise means. The returned flow is G-native.

```python
flow, batch_ess = train_forward_KLXX_G(
    x_valid, source, target, flow,
    pool_size=10000,
    batch_size=500,
    steps_total=200,
    lr=1e-3,
    ladder=1,
    melt=2.0,
    opt_dt=0.5,
    opt_steps=100,
    mc_dt=1e-3,
    mc_steps_1=50,
    mc_steps_2=100,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    chunks=16,
)
```

## KLL1 trainer

```python
train_forward_KLL1_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

KLL1 has the signature and the fresh SMC target-batch path of KLX, and
replaces the pairwise X penalty by the centered L1 log-dispersion of LDR-L1:

```text
mean(z) + coeff_lambda * mean(abs(z - mean(z))).
```

The batch mean is the center and is differentiated through. `coeff_lambda=0`
removes the dispersion term. The output flow is G-native.

## FAB trainer

```python
train_FAB_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    *,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

FAB has the signature of forward KL. At every step it manufactures a batch
from `pi^2 / nu` with `sequential_monte_carlo_fab` through the current G
flow (`ladder` levels to the target, then `ladder` further levels on to
`pi^2 / nu`, `mc_steps_1` MALA steps at the level's own distribution on
the intermediate levels of each phase, `mc_steps_2` on the last) and
minimizes the mean log-ratio `z` over that batch. Its
parameter gradient
is that of `-mean(log nu(y))` over `pi^2 / nu` samples, the alpha = 2
divergence surrogate of FAB; there is no replay buffer. The reported batch
ESS is that of the phase-1 proposal, as for the direct trainers. The output
flow is G-native.

```python
flow, batch_ess = train_FAB_G(
    x_valid, source, target, flow,
    batch_size=500,
    steps_total=200,
    lr=1e-3,
    ladder=1,
    mc_dt=1e-3,
    mc_steps_1=50,
    mc_steps_2=50,
)
```

## FABX trainer

```python
train_FABX_G(
    x_valid, source, target, flow,
    pool_size, batch_size, steps_total,
    lr, ladder, melt, opt_dt, opt_steps,
    mc_dt, mc_steps_1, mc_steps_2,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    mc_adjust=True,
    monitor=None,
    seed=0,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    *,
    chunks=1,
    initialize_from_identity=False,
    t_start=0.0,
    t_end=1.0,
)
```

FABX is KLXX with the FAB loss in place of the KLX target term: the same
quench and temper pool, the same mixture of `y_hat` and the detached
phase-1 proposal `y_bar` with probabilities `coeff_alpha` and
`1 - coeff_alpha`, and the objective

```text
mean(z over the pi^2 / nu batch) + coeff_theta * mean_{i != j}(abs(z_i - z_j) over the mixture).
```

There is no target-measure X term and no `coeff_lambda`. The output flow is
G-native.

## Initialization and comparison fairness

`initialize_from_identity=True` starts each direct call from the flow's
trainable exact-identity parameterization, except that OTFlow uses its
trainable near-identity. It primarily standardizes initialization and can
improve stability for singular or difficult potentials. It does not guarantee
a higher final ESS.

When it is `False`, the trainer uses the supplied flow parameters exactly.
This supports deliberate warm starts. For `OTFlow`, identity initialization
uses `near_identity()` internally so the quadratic head retains a gradient.

## Held-out evaluation

Batch ESS is a training trace. Evaluate the final proposal separately:

```python
from jflows.utils import compute_ESS_log, importance_weights_log

log_w = importance_weights_log(
    x_valid,
    source,
    target,
    flow,
    type="G",
    chunks=8,
)
valid_ess = compute_ESS_log(log_w)
y_valid = flow.inv(x_valid)
```

Use `type="F"` and `flow(x_valid)` for an F-native reverse-KL flow. Pair ESS
with coverage, target observables, or known mode diagnostics when proposal
support matters.

## Medium-level artifacts

`jflows.artifacts` saves individual outputs without introducing a run
protocol.

```python
save_flow(path, flow)
load_flow(path, template)

save_samples(path, samples)
load_samples(path)

save_history(path, **history)
load_history(path)
```

### Flow serialization

Equinox serializes array leaves, not the Python architecture. Loading requires
a matching template:

```python
from jflows.artifacts import save_flow, load_flow

save_flow("flow.eqx", flow)

template = NSF(
    jax.random.key(0),
    [-4.0, -4.0],
    [4.0, 4.0],
    bins=16,
    transforms=6,
    hidden_features=(128, 128),
).zeros()
flow = load_flow("flow.eqx", template)
```

Reconstruct the same class and architecture. For `RealNVP(randmask=True)`, the
constructor key must reproduce its static coupling masks. For a CNF, use the
same PRNG implementation when rebuilding the skeleton.

### Arrays and histories

```python
save_samples("samples.npy", y_valid)
save_history(
    "history.npz",
    batch_ess=batch_ess,
    validation_ess=jax.numpy.asarray([valid_ess]),
)

y_valid = load_samples("samples.npy")
history = load_history("history.npz")
```

Samples are stored as NumPy `.npy`; named histories use `.npz`. These helpers
do not manage stages, manifests, retries, or continuation. Use the high-level
Boltzmann persistence modules for complete-stage storage.

## Choosing a trainer

<div align="center">

<table>
<thead>
<tr><th>Goal</th><th>Trainer</th></tr>
</thead>
<tbody>
<tr><td>simple mode-seeking baseline</td><td><code>train_reverse_KL_F</code></td></tr>
<tr><td>mass-covering target fit from manufactured target batches</td><td><code>train_forward_KL_G</code></td></tr>
<tr><td>stabilize the target density-ratio shape</td><td><code>train_forward_KLX_G</code></td></tr>
<tr><td>add explicit wide-coverage and leakage-sensitive X regularization</td><td><code>train_forward_KLXX_G</code></td></tr>
<tr><td>forward KL with the LDR-L1 log-dispersion penalty</td><td><code>train_forward_KLL1_G</code></td></tr>
<tr><td>alpha = 2 divergence surrogate on <code>pi^2 / nu</code> samples</td><td><code>train_FAB_G</code></td></tr>
<tr><td>FAB with the KLXX mixture variation</td><td><code>train_FABX_G</code></td></tr>
</tbody>
</table>

</div>

If one direct stage is too difficult, move to the high-level staged
Boltzmann generator rather than adding hidden outer loops around a trainer.

## Executable references

- [direct trainer and generator contracts](../smoke/test_train.py)

Complete direct-training workflows appear in
[2D_single.py](../example/2D_single.py),
[3D_periodic.py](../example/3D_periodic.py), and
[CNF_vs_OTFlow.py](../example/CNF_vs_OTFlow.py).

# High-level interfaces

The high level builds a staged Boltzmann generator from medium-level stage
trainers. It chooses or consumes stage points, trains an incremental map,
compares that map with an exact identity fallback on the complete validation
population, advances particles, and records every accepted stage.

The eight computation functions are exported directly by
`jflows.boltzmann`. Complete-stage persistence is separate in
`jflows.boltzmann.write` and `jflows.boltzmann.load`.

## Interpolation and stage model

The generator follows the linear potential interpolation

```text
U_t = (1 - t) U_source + t U_target,    0 <= t <= 1.
```

A stage from `t_start` to `t_end` trains one incremental flow between those
two stage distributions. It is not a new global source-to-final map. The
selected flows form an ordered deterministic proposal chain, but the returned
particle population also passes through weighting, resampling, and Langevin
after every selected map.

```text
pi_0
  --stage 1 flow / reweight / resample / Langevin--> particles at t1
  --stage 2 flow / reweight / resample / Langevin--> particles at t2
  ...
  --stage K flow / reweight / resample / Langevin--> particles at t=1
```

Consequently, `valid_selected_ess` is an incremental stage ESS. It does not
measure the global source-to-final chain. Compute fresh full-chain weights if
that global diagnostic is needed. Composing the selected flows gives a useful
deterministic proposal map, but it does not reproduce the stochastic
`y_valid` population by itself.

## Adaptive-staging generators

```python
from jflows.boltzmann import (
    boltzmann_identity,
    boltzmann_reverse_KL_F,
    boltzmann_forward_KL_G,
    boltzmann_forward_KLX_G,
    boltzmann_forward_KLXX_G,
    boltzmann_forward_KLL1_G,
    boltzmann_FAB_G,
    boltzmann_FABX_G,
)
```

`boltzmann_identity` is the flow-free reference controller. The other
functions share one adaptive-staging controller. They differ in the
medium-level trainer dispatched inside each stage: the direction in which the
flow is parameterized, how the optimizer obtains its training samples, and
which loss is differentiated. The stage-point proposal, the
trained-versus-identity comparison over the complete validation set, the
validation ESS gate `tau_valid`, and the accepted-particle advancement
(reweight, resample, `mc_steps_2` Langevin steps) are otherwise the same.

### Background: why four generators?

The package assumes that the source is easy to sample but the target is known
only through its energy. For a difficult target, learning one direct map can
fail because the source and target barely overlap. The Boltzmann generator
therefore replaces one hard fit with a sequence of easier stages.

The four public functions are four ways to train the map inside one such
stage. In simple terms:

- **Reverse KL** learns from the particles already available on the source
  side of the stage. It is the simplest and least expensive baseline.
- **Forward KL** first manufactures approximate target-side particles, then
  learns from them. It spends more computation to encourage mass coverage.
- **KLX** adds a penalty that asks importance weights to be more uniform over
  those target-side particles.
- **KLXX** also tests the density ratio on a wider mixture designed to expose
  missed modes and proposal leakage.

They are not four unrelated pipelines. They share the same stage selection,
validation, identity fallback, acceptance, and particle advancement. Only the
stage-training objective becomes progressively richer.

## Identity-only adaptive-staging generator

```python
boltzmann_identity(
    x_valid,
    source,
    target,
    mc_dt,
    mc_steps_2,
    *,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    seed=0,
)
```

This function runs the same adaptive stage policy without constructing or
training a flow. For each candidate `a -> b`, it:

1. evaluates exact identity log weights `U_a(x)-U_b(x)` on all `x_valid`;
2. requires their ESS to reach `tau_valid`, shrinking `b-a` on rejection;
3. resamples the complete population; and
4. applies `mc_steps_2` Langevin steps at `U_b`, using MALA when
   `mc_adjust=True`.

It returns `(samples, stages)`. Its signature deliberately omits `flow`,
`batch_size`, `steps_total`, `lr`, optimizer controls, and initialization
controls. `chunks` is used for the complete-set identity weights and the
rejuvenation. This makes `boltzmann_identity` the direct adaptive-staging
reweight/resample/MCMC baseline rather than a flow generator configured with
zero training steps.

Identity records contain `t`, `t_start`, `valid_selected_ess`,
`valid_identity_ess`, `valid_sample_count`, `selected="identity"`, `t_hist`,
`valid_identity_ess_hist`, `attempt_status_hist`, and
`elapsed_seconds`. They contain no trained ESS, batch history, flow, or
continuation flow.

### Design philosophy

The interface follows five principles:

1. **Start with the simplest objective.** Reverse KL is available without
   hidden target-sample machinery. Forward KL, KLX, and KLXX add work only when
   their additional information is wanted.
2. **Train in the flow's native direction.** Reverse KL trains `F` directly;
   the three forward objectives train `G` directly. The optimizer never needs
   to differentiate through an autoregressive inverse.
3. **Separate training from judgment.** Different losses propose different
   maps, but the same ESS over the complete validation set compares every proposal with the
   same exact identity fallback.
4. **Make regularization progressive and explicit.** `coeff_lambda` adds the
   target X term; KLXX then exposes `pool_size`, QT controls, and mixture
   coefficients rather than hiding those choices.
5. **Keep the controller comparable.** With matched stage policy, flow,
   validation population, and training budget, differences mainly reflect the
   objective and its data-construction cost rather than a different acceptance
   rule.

#### Short selection rule

```text
Need the simplest baseline?                    reverse KL
Need a forward mass-covering objective?        forward KL
Need more uniform target-side log weights?     KLX
Need explicit wide-mode/leakage regularizing?  KLXX
```

The last three descriptions are motivations, not guarantees. Final quality
must still be established with held-out ESS and geometric or problem-specific
coverage diagnostics.

### The four objectives at a glance

For one proposed stage, write

```text
U_a = U_t_start,    U_b = U_t_end,
pi_a proportional to exp(-U_a),
pi_b proportional to exp(-U_b).
```

The current validation population approximates `pi_a`. The stage must learn a
map between `pi_a` and `pi_b`.

<div align="center">

<table>
<thead>
<tr><th>Generator</th><th>Native flow</th><th>Optimizer population</th><th>Differentiated objective</th><th>Distinct controls</th></tr>
</thead>
<tbody>
<tr><td><code>boltzmann_reverse_KL_F</code></td><td><code>F: pi_a -&gt; pi_b</code></td><td>current <code>pi_a</code> particles, source-freshened by Langevin</td><td>reverse KL</td><td>no objective regularizer, no <code>ladder</code>, and no <code>mc_steps_1</code></td></tr>
<tr><td><code>boltzmann_forward_KL_G</code></td><td><code>G: pi_b -&gt; pi_a</code></td><td>approximate <code>pi_b</code> batches manufactured by flow-proposal SMC</td><td>forward KL</td><td><code>ladder</code>, <code>u_clip</code>, <code>g_clip</code></td></tr>
<tr><td><code>boltzmann_forward_KLX_G</code></td><td><code>G: pi_b -&gt; pi_a</code></td><td>the same SMC-manufactured target batches</td><td>forward KL plus target-measure X variation</td><td><code>coeff_lambda</code> plus the forward KL controls</td></tr>
<tr><td><code>boltzmann_forward_KLXX_G</code></td><td><code>G: pi_b -&gt; pi_a</code></td><td>SMC target batches plus a QT/proposal mixture</td><td>KLX plus a second mixture-measure X variation</td><td><code>pool_size</code>, <code>melt</code>, <code>opt_dt</code>, <code>opt_steps</code>, <code>coeff_theta</code>, <code>coeff_alpha</code>, <code>coeff_qt</code>, <code>chunks</code></td></tr>
<tr><td><code>boltzmann_forward_KLL1_G</code></td><td><code>G: pi_b -&gt; pi_a</code></td><td>the same SMC-manufactured target batches</td><td>forward KL plus the LDR-L1 log-dispersion</td><td><code>coeff_lambda</code> plus the forward KL controls</td></tr>
<tr><td><code>boltzmann_FAB_G</code></td><td><code>G: pi_b -&gt; pi_a</code></td><td><code>pi_b^2 / nu</code> batches manufactured by the two-phase SMC</td><td>FAB mean log-ratio (alpha = 2 surrogate)</td><td>the forward KL controls</td></tr>
<tr><td><code>boltzmann_FABX_G</code></td><td><code>G: pi_b -&gt; pi_a</code></td><td><code>pi_b^2 / nu</code> batches plus a QT/proposal mixture</td><td>FAB plus the mixture-measure X variation</td><td>the KLXX controls without <code>coeff_lambda</code></td></tr>
</tbody>
</table>

</div>

The progression is therefore:

```text
reverse KL
  = source-sampled F training

forward KL
  = SMC-manufactured target-sampled G training

KLX
  = forward KL + density-ratio variation on the target measure

KLXX
  = KLX + density-ratio variation on a QT/proposal mixture
```

The three forward objectives are conceptually nested: forward KL → KLX →
KLXX. Reverse KL is the separate source-sampled F baseline. The implementations
also use different trainer key namespaces and data-generation paths, so
setting a regularization coefficient to zero does not imply bitwise equality
with another generator.

### Shared controller, objective-specific trainer

Every adaptive-staging generator performs this outer sequence:

```text
propose t_end
  -> optionally test endpoint with potential-space SMC
  -> construct U_a and U_b
  -> run the selected objective's direct trainer
  -> evaluate trained stage flow on every current validation particle
  -> evaluate exact identity on the same particles
  -> select the higher-ESS map
  -> accept or shrink/retry using selected ESS
  -> push, reweight, resample, and rejuvenate at U_b
```

The objective changes the trained candidate, but not the scientific quantity
used to choose between that candidate and identity. All four generators use
complete-validation incremental importance ESS for that decision. Their
optimizer-batch ESS traces remain monitors only.

The validation direction matches the trained representation:

- reverse KL evaluates importance weights with `type="F"` and advances with
  `flow(samples)`;
- forward KL, KLX, and KLXX evaluate with `type="G"` and advance with
  `flow.inv(samples)`.

The identity comparison uses no flow inverse and is the same incremental
importance-reweighting identity baseline for all four objectives. Resampling
and rejuvenation occur only after the trained-or-identity map is selected.

### Reverse KL

```python
boltzmann_reverse_KL_F(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    mc_dt, mc_steps_2,
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    seed=0,
)
```

For current-stage particles `x ~ pi_a`, the reverse trainer minimizes

```text
L_reverse(F) = mean[U_b(F(x)) - log|det J_F(x)|].
```

Each optimizer step samples rows from the current validation population and
freshens them with `mc_steps_2` Langevin steps targeting `U_a`. It never needs manufactured
target samples and differentiates the flow only in its native forward
direction. The selected stage flow maps `pi_a` toward `pi_b`, so stage
advancement applies `flow(x)`.

Reverse KL is the least elaborate of the four stage objectives and is a useful
mode-seeking baseline. That qualitative tendency is not an acceptance rule:
the outer controller still measures ESS over the complete validation set and can select
identity or reject the proposed endpoint.

### Forward KL

```python
boltzmann_forward_KL_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    seed=0,
)
```

The forward trainer represents the inverse map `G: pi_b -> pi_a`. It cannot
draw exact `pi_b` batches directly, so every Adam step:

1. samples source-side particles from the current validation population;
2. applies flow-proposal SMC through the current G flow;
3. reweights, resamples, and rejuvenates (`mc_steps_1` MALA steps at the
   level's own distribution of the geometric path on the intermediate
   levels, `mc_steps_2` steps at `pi_b` on the last) to manufacture an
   approximate `pi_b` batch `y`; and
4. minimizes

```text
L_forward(G) = mean[U_a(G(y)) - log|det J_G(y)|].
```

The omitted `-U_b(y)` term is constant with respect to the flow parameters for
the fixed manufactured batch. `ladder` controls the number of SMC levels used
inside every optimizer step. The SMC routine rejuvenates every level with
MALA at the level's own distribution of the geometric path between the
flow proposal and `pi_b`, so every level is exact; on the intermediate
levels the Langevin drift is differentiated through the flow, and the last
level runs under `U_b` alone.

This objective is normally chosen when mass coverage is more important than a
pure reverse-KL baseline. It costs more per optimizer step because target-batch
manufacture includes flow evaluation, weighting, resampling, and Langevin.
`u_clip` can screen high-energy manufactured samples from the optimizer loss,
and `g_clip` bounds the global gradient norm. They do not change the
selection rule over the complete validation set, but they can change the trained candidate and
therefore its validation ESS and the selected outcome.

### KLX

```python
boltzmann_forward_KLX_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    seed=0,
)
```

KLX uses the same SMC-manufactured `pi_b` batches and the same G direction as
forward KL. It defines the stage log-density-ratio coordinate

```text
z(y) = U_a(G(y)) - U_b(y) - log|det J_G(y)|
```

and minimizes

```text
L_KLX(G)
  = mean[z(y)]
  + coeff_lambda * mean_{i != j}[|z(y_i) - z(y_j)|],
```

where the second term is the exact mean over all pairs of the current batch,
evaluated by one sort (`pairwise_variation`). The first term has the forward
KL gradient. The X term penalizes variation of the log density ratio across
target-measure samples: a proposal with a more nearly constant ratio has more
uniform importance weights.

`coeff_lambda` controls only this target-measure X term. KLX adds little array
storage compared with forward KL, but its per-step loss includes one sort of
the batch log-ratios. It is intended to improve weight uniformity; it does
not mathematically guarantee that every finite run will have higher
validation ESS than forward KL.

### KLXX

```python
boltzmann_forward_KLXX_G(
    x_valid, source, target, flow,
    pool_size, batch_size, steps_total,
    lr, ladder, melt, opt_dt, opt_steps,
    mc_dt, mc_steps_1, mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    seed=0,
)
```

KLXX retains the KLX target-batch objective and adds a second X functional on
a broader, deliberately constructed measure. At the beginning of every stage
attempt, before the compiled Adam scan, it builds a fixed quench and temper
pool `hat_pi` for `U_b`:

```text
current stage population
  -> (coeff_qt > 0) resample by exp(coeff_qt * (U_a - U_b)),
     then mc_steps_2 Langevin steps at (1 - coeff_qt) U_a + coeff_qt U_b
  -> Gaussian melt
  -> L-BFGS quench into target basins
  -> Langevin temper at U_b (mc_steps_2 steps)
  -> hat_pi pool
```

At each optimizer step it then forms two populations:

- `y_hat`: samples from the QT pool, freshly Langevin-rejuvenated at `U_b`
  for `mc_steps_2` steps;
- `y_bar`: the current detached G proposal, the pushforward of the
  current-stage source batch that the SMC started from.

It mixes them with probabilities `coeff_alpha` and `1 - coeff_alpha`:

```text
omega = coeff_alpha * hat_pi + (1 - coeff_alpha) * bar_nu.
```

The complete loss is

```text
L_KLXX(G)
  = mean[z(y)]
  + coeff_lambda * mean_{i != j}[|z(y_i) - z(y_j)|]
  + coeff_theta * mean_{i != j}[|z(y_omega,i) - z(y_omega,j)|],
```

with both pair means evaluated exactly by sorting.

The QT component emphasizes mode discovery through basin coverage. The
detached proposal component exposes regions already produced by the flow and
penalizes leakage through the same density-ratio variation. Sample locations
are not differentiated through; the current candidate flow is differentiated
when evaluating `z` on those locations.

KLXX is the most computationally and memory intensive option:

- QT runs once for every stage attempt, including stage retries;
- `pool_size=0` applies QT to the complete current validation population;
- `pool_size>0` draws a separate pool with replacement from that population;
- `melt`, `opt_dt`, and `opt_steps` control the melt/quench construction,
  and `coeff_qt` the partial importance resampling that precedes it;
- `mc_dt` with `mc_steps_1` controls the intermediate SMC levels of the
  forward-batch manufacture, and `mc_dt` with `mc_steps_2` its last level,
  the `y_hat` freshening, the QT pool, and the stage advance; and
- `chunks` partitions QT and the outer operations over the complete validation set.

Use KLXX when explicit wide-coverage and proposal-leakage diagnostics justify
that added cost. Its richer objective still passes through the same
trained-versus-identity validation and stage ESS gate as the other
generators.

The KLXX pool semantics are identical to the medium-level trainer:
`pool_size=0` quenches the complete current validation population; a positive
value draws a separate pool. `chunks` reaches the QT call and the full-set
stage operations.

### KLL1

```python
boltzmann_forward_KLL1_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_lambda=1.0,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    seed=0,
)
```

KLL1 has the KLX signature and the same SMC-manufactured `pi_b` batches, and
minimizes forward KL plus `coeff_lambda` times the centered L1
log-dispersion of LDR-L1, `mean[|z(y) - mean z|]`, with the batch mean
differentiated through. It exists for comparison with LDR-L1 under the same
staging controller.

### FAB

```python
boltzmann_FAB_G(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2,
    *,
    initialize_from_identity=True,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    seed=0,
)
```

FAB has the forward KL signature. Inside each stage attempt the trainer
manufactures batches from `pi_b^2 / nu` with `sequential_monte_carlo_fab`
(`ladder` levels to `pi_b`, then `ladder` further levels on to
`pi_b^2 / nu`, `mc_steps_1` MALA steps at the level's own distribution on
the intermediate levels of each phase, `mc_steps_2` on the last) and
minimizes the mean log-ratio over them, the alpha = 2
divergence
surrogate of FAB without a replay buffer. Stage selection, the validation
ESS gate, and the stage advance are those of the other generators.

### FABX

```python
boltzmann_FABX_G(
    x_valid, source, target, flow,
    pool_size, batch_size, steps_total,
    lr, ladder, melt, opt_dt, opt_steps,
    mc_dt, mc_steps_1, mc_steps_2,
    *,
    initialize_from_identity=True,
    coeff_theta=1.0,
    coeff_alpha=0.5,
    coeff_qt=0.0,
    mc_adjust=True,
    monitor=None,
    bg_param=None,
    chunks=1,
    checkpoint=False,
    u_clip=inf,
    g_clip=inf,
    seed=0,
)
```

FABX is KLXX with the FAB loss in place of the KLX target term: the same
quench and temper pool and the same `y_hat` / `y_bar` mixture, and the
objective is the FAB mean log-ratio plus `coeff_theta` times the mixture
X variation. There is no target-measure X term and no `coeff_lambda`.

All generators return:

```python
y_valid, stages
```

`y_valid` is the particle set after the last accepted stage. A run has reached
the target only when:

```python
complete = bool(stages and stages[-1]["t"] == 1.0)
```

If the adaptive-staging controller exhausts its stage or retry budget, it
returns the accepted prefix rather than claiming completion.

### Choosing among the four

<div align="center">

<table>
<thead>
<tr><th>Primary need</th><th>Starting choice</th><th>Reason</th></tr>
</thead>
<tbody>
<tr><td>simple, inexpensive stage baseline</td><td><code>boltzmann_reverse_KL_F</code></td><td>source-sampled native-F loss; no SMC target-batch construction inside Adam</td></tr>
<tr><td>forward mass-covering objective</td><td><code>boltzmann_forward_KL_G</code></td><td>manufactures target batches and trains the native G direction</td></tr>
<tr><td>explicit target-measure weight-uniformity penalty</td><td><code>boltzmann_forward_KLX_G</code></td><td>adds variation control for the stage log-density ratio</td></tr>
<tr><td>mode discovery plus proposal-leakage regularization</td><td><code>boltzmann_forward_KLXX_G</code></td><td>adds QT coverage and detached-proposal mixture samples</td></tr>
<tr><td>LDR-L1 comparison</td><td><code>boltzmann_forward_KLL1_G</code></td><td>the KLX signature with the centered L1 log-dispersion in place of the X variation</td></tr>
<tr><td>FAB comparison</td><td><code>boltzmann_FAB_G</code></td><td>the forward KL signature with <code>pi_b^2 / nu</code> batches from the two-phase SMC</td></tr>
<tr><td>FAB with mixture coverage</td><td><code>boltzmann_FABX_G</code></td><td>the KLXX signature without <code>coeff_lambda</code></td></tr>
</tbody>
</table>

</div>

This is a workflow guide, not a universal performance ordering. Target
geometry, flow capacity, batch size, stage spacing, MCMC quality, and optimizer
budget can change which objective yields the best held-out ESS. Compare the
four under matched stage policy and architecture, and interpret ESS together
with coverage or target-specific observables.

## Adaptive stage policy

`bg_param` overrides any subset of the default policy:

```python
{
    "t_safe": 0.2,
    "shrink_factor": 0.7,
    "enlarge_factor": 1.5,
    "tau_valid": 0.6,
    "t_tol": 0.01,
    "max_stages": 30,
    "max_retry": 6,
}
```

<div align="center">

<table>
<thead>
<tr><th>Key</th><th>Role</th></tr>
</thead>
<tbody>
<tr><td><code>t_safe</code></td><td>first proposed endpoint</td></tr>
<tr><td><code>shrink_factor</code></td><td>multiply a rejected interval length before retrying</td></tr>
<tr><td><code>enlarge_factor</code></td><td>grow the next interval from the last accepted interval</td></tr>
<tr><td><code>tau_valid</code></td><td>minimum selected trained-or-identity validation ESS</td></tr>
<tr><td><code>t_tol</code></td><td>snap a proposed endpoint near one to exactly one</td></tr>
<tr><td><code>max_stages</code></td><td>maximum accepted-stage index attempted</td></tr>
<tr><td><code>max_retry</code></td><td>maximum training attempts for one stage</td></tr>
</tbody>
</table>

</div>

An unknown key in `bg_param` raises `KeyError`. The trained controller uses
one gate: after training, complete-validation ESS compares the trained flow
and exact identity. Their better ESS must clear `tau_valid`; otherwise the
interval shrinks and training retries with a fresh operation key.

Batch ESS emitted during optimization is never an acceptance gate.
For `boltzmann_identity`, the gate is simply the complete-validation
identity ESS; no training or trained-versus-identity comparison occurs.

## Identity fallback and initialization

At every trained attempt, the controller computes log weights over the complete validation set
for:

- the trained stage flow; and
- an exact identity map, equivalent to pure importance reweighting for that stage transition.

The higher-ESS map advances the particles. Therefore a stage does not select a
trained map that is worse than the identity fallback on the validation metric.

`initialize_from_identity=True` controls the optimizer start, not the
independent identity comparison. It starts each attempt from a trainable exact
identity for ordinary flows and from `OTFlow.near_identity()` for OTFlow, whose
exact zero quadratic factor has no training gradient. With `False`, the next
stage warm-starts from the preceding continuation flow. Retries reuse the
immutable stage-entry template. The independent validation fallback remains
the exact `.zeros()` identity for every flow.

If the identity wins, `flow` in the record is the exact identity map while
`continuation_flow` is the trainable exact-identity or OTFlow near-identity
initializer used as the next warm-start. If the trained map wins, both refer
to the trained candidate.

## Accepted-stage advancement

After an attempt is accepted, the controller:

1. applies the selected map in its F or G generation direction;
2. converts its selected log weights to stable linear weights;
3. resamples the complete particle population; and
4. applies Langevin at the accepted stage target `U_t_end`.

`mc_adjust=True` gives MALA advancement; `False` gives ULA. `chunks` partitions
weighting over the complete validation set, flow application, and rejuvenation. The generator
derives deterministic operation keys from `seed`, stage, attempt, and operation
namespace. After advancement is synchronized, every coordinate in the new
population must be finite. Otherwise the generator raises `FloatingPointError`
before yielding the stage record, so persistence cannot label or store that
invalid population as a complete stage.

## Fixed-schedule generators

Fixed generators consume explicit stage points and never shrink or retry.
They still compare the trained map with identity and advance with the better
map.

```python
from jflows.boltzmann import (
    boltzmann_reverse_KL_F_fixed,
    boltzmann_forward_KL_G_fixed,
    boltzmann_forward_KLX_G_fixed,
    boltzmann_forward_KLXX_G_fixed,
)
```

```python
boltzmann_reverse_KL_F_fixed(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    mc_dt, mc_steps_2, t_list,
    *, initialize_from_identity=True,
    mc_adjust=True, monitor=None, chunks=1,
    checkpoint=False, seed=0,
)

boltzmann_forward_KL_G_fixed(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2, t_list,
    *, initialize_from_identity=True,
    mc_adjust=True, monitor=None, chunks=1,
    checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)

boltzmann_forward_KLX_G_fixed(
    x_valid, source, target, flow,
    batch_size, steps_total, lr,
    ladder, mc_dt, mc_steps_1, mc_steps_2, t_list,
    *, initialize_from_identity=True,
    coeff_lambda=1.0,
    mc_adjust=True, monitor=None, chunks=1,
    checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)

boltzmann_forward_KLXX_G_fixed(
    x_valid, source, target, flow,
    pool_size, batch_size, steps_total,
    lr, ladder, melt, opt_dt, opt_steps,
    mc_dt, mc_steps_1, mc_steps_2, t_list,
    *, initialize_from_identity=True,
    coeff_lambda=1.0, coeff_theta=1.0, coeff_alpha=0.5, coeff_qt=0.0,
    mc_adjust=True, monitor=None, chunks=1,
    checkpoint=False, u_clip=inf, g_clip=inf, seed=0,
)
```

Supply an increasing `t_list` in `(0,1]`. The controller always selects the
first remaining endpoint greater than the current `t`. A schedule ending below
one deliberately produces an incomplete accepted prefix.

## Stage records

Each element of `stages` has the following canonical fields:

```python
{
    "t": float,
    "t_start": float,
    "valid_selected_ess": float,
    "valid_trained_ess": float,
    "valid_identity_ess": float,
    "valid_sample_count": int,
    "selected": "trained" | "identity",
    "flow": Flow,
    "continuation_flow": Flow,
    "t_hist": Array,
    "batch_ess_hist": Array,
    "valid_trained_ess_hist": Array,
    "valid_identity_ess_hist": Array,
    "attempt_status_hist": tuple[str, ...],
    "elapsed_seconds": float,
    "selected_flow_path": str | None,
    "continuation_flow_path": str | None,
    "validation_samples_path": str | None,
}
```

### Scalar fields

- `t_start`, `t`: accepted stage interval.
- `valid_trained_ess`, `valid_identity_ess`: final attempt's two complete-set
  ESS values.
- `valid_selected_ess`: the maximum of those two values.
- `valid_sample_count`: number of rows used for validation and advancement.
- `selected`: which map advanced the stage.
- `elapsed_seconds`: wall time for the accepted stage including its attempts.

### Attempt-aligned histories

- `t_hist`: attempted endpoints.
- `batch_ess_hist`: shape `(attempts, steps_total)`.
- `valid_trained_ess_hist`, `valid_identity_ess_hist`: one value per attempt.
- `attempt_status_hist`: `"accepted"` or `"rejected"` per training attempt.

The three path fields are `None` for pure in-memory computation. The storage
layer fills them with paths relative to the run root.

## Minimal adaptive-staging workflow

```python
import jax
from jflows.boltzmann import boltzmann_forward_KLX_G
from jflows.flow import NSF
from jflows.potential import Nlog_Gaussian, potential_from
from jflows.train import Monitor

source = Nlog_Gaussian([0.0, 0.0], [1.0, 1.0])
target = potential_from(
    lambda x: 10.0 * (jax.numpy.linalg.norm(x, axis=-1) - 2.0) ** 2
)
x_valid = source.samples(jax.random.key(1), 40000)
flow = NSF(
    jax.random.key(0), [-4.0, -4.0], [4.0, 4.0],
    bins=16, transforms=6, hidden_features=(128, 128),
).zeros()

y_valid, stages = boltzmann_forward_KLX_G(
    x_valid, source, target, flow,
    batch_size=1000,
    steps_total=300,
    lr=1e-3,
    ladder=1,
    mc_dt=1e-3,
    mc_steps_1=50,
    mc_steps_2=100,
    coeff_lambda=1.0,
    bg_param={
        "t_safe": 0.2,
        "shrink_factor": 0.7,
        "enlarge_factor": 1.5,
        "tau_valid": 0.6,
    },
    chunks=8,
    monitor=Monitor(50, "[KLX] "),
)

if not stages or stages[-1]["t"] != 1.0:
    raise RuntimeError("Boltzmann stage schedule did not reach the target")
```

## Complete-stage persistence

The nine generator functions are computation-only. They do not accept
`run_dir`, `problem_id`, or `resume`. Persistence is an explicit high-level
layer for advanced workflows.

Use a new empty run directory. `create` does not enforce emptiness: on a reused
directory it can overwrite the initial artifacts and manifest while leaving
an old `stages/` directory present. The `load.run` iterator contract expects
each callback yield to represent one complete accepted stage, but `write.stage`
does not independently validate acceptance; it persists any structurally
suitable record supplied by the caller. A stage becomes resumable only after
it is appended to the manifest, so an unlisted interrupted stage is ignored
and recomputed from the last manifest-listed stage.

### Writer API

```python
from jflows.boltzmann.write import create, stage, finish

run = create(run_dir, problem_id, config, samples, flow)
saved_record = stage(run_dir, run, record, samples)
finish(run_dir, run, status)
```

- `create` writes `initial_samples.npy` and `run.json`; it also writes
  `initial_flow.eqx` when `flow` is not `None`.
- `stage` atomically writes validation samples, histories, and stage metadata,
  plus selected and continuation flows for trained records, then appends the
  stage to the manifest.
- `finish` changes the manifest status, normally to `"complete"` or
  `"exhausted"`.

### Loader API

```python
from jflows.boltzmann.load import (
    manifest,
    validate,
    load,
    fork,
    load_stage_flow,
    load_validation_samples,
    load_training_history,
    run,
)
```

```python
manifest(run_dir) -> dict
validate(run_dir) -> dict
load(run_dir, template=None)
fork(run_dir, destination, problem_id=None) -> dict
load_stage_flow(run_dir, stage, role, template)
load_validation_samples(run_dir, stage=None, *, mmap_mode=None)
load_training_history(run_dir, stage) -> dict
run(run_dir, problem_id, config, samples, flow, iterate, *, resume=False)
```

- `manifest` reads `run.json` without loading arrays.
- `validate` verifies that every file referenced by the complete-stage
  manifest exists.
- `load` returns `(samples, continuation_flow, stages)` at the last complete
  stage. A trained run requires a matching flow template; an identity run is
  loaded without a template and returns `continuation_flow=None`.
- `fork` copies a validated run to a new destination and resets its status to
  running. The destination must not already exist.
- `load_stage_flow` uses one-based stage indices and `role="selected"` or
  `role="continuation"`.
- `load_validation_samples(..., stage=None)` reads the initial population; a
  one-based stage reads that post-stage population. `mmap_mode` is passed to
  NumPy.
- `load_training_history` restores numerical attempt histories plus status and
  selection records.

Stable compatibility names are `inspect_run = manifest`,
`validate_run = validate`, and `fork_run = fork`.

### Persistence controller

`load.run` is the package-level controller for custom stage iterators:

```python
samples, records = run(
    run_dir,
    problem_id,
    config,
    initial_samples,
    initial_flow,
    iterate,
    resume=False,
)
```

The callback contract is:

```python
iterate(samples, flow, accepted_t, start_stage)
    -> iterator yielding (samples, record, continuation_flow)
```

With `resume=True`, `run` loads the last complete population, continuation
flow, records, accepted stage points, and next one-based stage index before
calling the iterator. It writes every newly yielded complete stage. This is an
advanced persistence hook; ordinary callers should use one of the nine
generators directly.

## Stored run tree

```text
run-directory/
├── run.json
├── initial_flow.eqx
├── initial_samples.npy
└── stages/
    ├── stage_000001/
    │   ├── stage.json
    │   ├── selected.eqx
    │   ├── continuation.eqx
    │   ├── samples.npy
    │   └── history.npz
    └── stage_000002/
        └── ...
```

An identity run omits `initial_flow.eqx`, `selected.eqx`, and
`continuation.eqx`; its run tree contains only the manifest, initial samples,
and per-stage metadata, samples, and histories.

The manifest stores relative paths. Flow files contain Equinox array leaves,
so loading still requires the original architecture template.

## Failure and completion rules

- An empty `stages` list means no stage was accepted.
- A nonempty list ending below `t=1` is an incomplete prefix.
- Stage rejection is expected behavior when overlap is insufficient; it is
  recorded in the accepted stage's attempt histories if a later retry succeeds.
- Fixed schedules do not reject on ESS, but an exhausted schedule can still
  stop below the target.
- Identity selection is a valid accepted result, not a training crash.
- Batch ESS, training loss, and the selected validation ESS answer different
  questions; do not substitute one for another.

## Executable references

- [trainers, adaptive and fixed schedules, policy keys](../smoke/test_train.py)
- [complete-stage persistence](../smoke/test_boltzmann_artifacts.py)
- [identity persistence](../smoke/test_boltzmann_identity_artifacts.py)
- [4D adaptive-staging Boltzmann example](../example/4D_boltzmann.py)
- [4D verified figure and results](../example/results.md#4d_boltzmann--adaptive-staging-bg)

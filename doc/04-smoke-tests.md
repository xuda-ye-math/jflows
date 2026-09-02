# Smoke-test reference

The [smoke directory](../smoke/) is the executable contract for the public
API. Smoke modules use deliberately small numerical problems: they establish
imports, shapes, algebra, compilation, and behavioral invariants, but their
ESS values are not scientific benchmark results.

Run a module from the repository root:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m smoke.test_loss
```

The project targets accelerator-backed JAX. Do not force CPU merely to avoid
compilation. First calls include JIT compilation.

## Test map

<div align="center">

<table>
<thead>
<tr><th>Module</th><th>Primary interface coverage</th><th>Main contracts</th></tr>
</thead>
<tbody>
<tr><td><a href="../smoke/test_potential.py"><code>test_potential</code></a></td><td>built-in and custom potentials</td><td>energies, gradients, sampling shapes, Gaussian variance convention, JIT behavior</td></tr>
<tr><td><a href="../smoke/test_linear_combination.py"><code>test_linear_combination</code></a></td><td>potential vector-space algebra</td><td>coefficient semantics, flattening, gradients, interpolation evaluation, visualization</td></tr>
<tr><td><a href="../smoke/test_loss.py"><code>test_loss</code></a></td><td>four low-level losses and <code>pairwise_variation</code></td><td>per-sample shape, F/G formulas, exact sorted X term against the brute-force pair mean, stochastic trace-key route</td></tr>
<tr><td><a href="../smoke/test_metrics.py"><code>test_metrics</code></a></td><td>weights, ESS, coverage, resampling</td><td>analytic ESS cases, log/linear agreement, F/G weight agreement, numerical degeneracies, k-NN coverage, resampling frequencies</td></tr>
<tr><td><a href="../smoke/test_rejuvenation.py"><code>test_rejuvenation</code></a></td><td>Langevin, stochastic Heun, HMC</td><td>step/full-kernel consistency, MALA acceptance, tamed ULA, nonfinite MALA/HMC rejection, aliases</td></tr>
<tr><td><a href="../smoke/test_optimization.py"><code>test_optimization</code></a></td><td>L-BFGS and AdamW sample optimization</td><td>state kernels, complete loops, Armijo behavior, chunked optimization, nonfinite candidate rejection and recovery</td></tr>
<tr><td><a href="../smoke/test_utils_api.py"><code>test_utils_api</code></a></td><td>flat <code>jflows.utils</code> namespace</td><td>canonical signatures, semantic aliases, flow-proposal SMC return contract, <code>coeff_qt</code> quench and temper, representative execution</td></tr>
<tr><td><a href="../smoke/test_train.py"><code>test_train</code></a></td><td>direct trainers, <code>Monitor</code>, adaptive and fixed generators</td><td>two-step compiled scans of all four trainers, ESS histories, <code>mc_steps_1</code>/<code>mc_steps_2</code>, <code>coeff_qt</code>, a complete KLXX schedule, a fixed schedule, rejection of a retired policy key</td></tr>
<tr><td><a href="../smoke/test_boltzmann_identity_artifacts.py"><code>test_boltzmann_identity_artifacts</code></a></td><td>flow-free complete-stage persistence</td><td>identity save/load/resume with no flow artifacts</td></tr>
<tr><td><a href="../smoke/test_boltzmann_artifacts.py"><code>test_boltzmann_artifacts</code></a></td><td>complete-stage persistence</td><td>create/write/load, interruption boundary, continuation flow, and stage readers</td></tr>
</tbody>
</table>

</div>

## Routing by interface level

Every module is small: each runs in seconds on an accelerator, most of it
compilation.

### Low level

```text
potential               -> test_potential, test_linear_combination
loss                    -> test_loss
weight/ESS/coverage     -> test_metrics
MCMC                    -> test_rejuvenation
SMC / quench and temper -> test_utils_api
sample optimization     -> test_optimization
flat utility surface    -> test_utils_api
```

### Medium and high level

```text
direct training         -> test_train
stage controllers       -> test_train
stage persistence       -> test_boltzmann_artifacts
identity persistence    -> test_boltzmann_identity_artifacts
```

## What to verify in output

Different modules print different diagnostics, but the recurring
postconditions are:

- arrays have the documented shapes and finite values;
- forward/inverse reconstruction error is bounded for the tested flow;
- forward and inverse log-Jacobians have opposite signs;
- normalized ESS lies in `[0,1]`;
- trainer history length equals `steps_total`;
- adaptive-staging records align histories by attempt;
- a claimed complete Boltzmann run ends at `t=1.0`;
- stored run manifests reference every required complete-stage file; and
- aliases behave identically to their canonical long names.

## Isolated verification

Examples and tests can produce logs, figures, bytecode, and compiled caches.
For verification that must leave the public checkout unchanged, copy the
package and selected module to a temporary repository root:

```bash
repo_root=$(pwd)
temp_root=$(mktemp -d)
mkdir -p "$temp_root/jflows"
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  "$repo_root/jflows" \
  "$repo_root/smoke" \
  "$repo_root/pyproject.toml" \
  "$temp_root/jflows/"

cd "$temp_root/jflows"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$temp_root/jflows" \
python -m smoke.test_metrics
```

Use a task-specific temporary directory and remove it only after confirming it
contains no project-owned data. Do not treat existing repository log files as
the current result unless the matching run was directly verified.

## Scope of smoke evidence

A single smoke module supports claims only about the functions and
configurations it executes. It does not establish large-scale memory use,
scientific convergence, benchmark quality, or behavior of every flow class.
Use the [examples](05-examples.md) for complete workflows and dedicated
benchmark scripts for scientific claims.

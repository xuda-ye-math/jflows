# Smoke-test reference

The [smoke directory](../smoke/) is the executable contract for the public
API. Smoke modules use deliberately small numerical problems: they establish
imports, shapes, algebra, compilation, and behavioral invariants, but their
ESS values are not scientific benchmark results.

Run a module from the repository root:

```bash
source ~/.envs/jflows/bin/activate
PYTHONPATH=/data/projects/jflows \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m smoke.test_flow
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
<tr><td><a href="../smoke/test_flow.py"><code>test_flow</code></a></td><td><code>NSF</code>, <code>NCSF</code>, <code>CNF</code>, <code>OTFlow</code>, <code>RealNVP</code></td><td>constructors, identity initialization, forward/inverse reconstruction, opposite log-Jacobian signs, JIT and gradients</td></tr>
<tr><td><a href="../smoke/test_circular.py"><code>test_circular</code></a></td><td><code>NCSF</code> and circular RQS</td><td>periodic seams, wrapped representatives, Jacobian continuity, circular invertibility</td></tr>
<tr><td><a href="../smoke/test_potential.py"><code>test_potential</code></a></td><td>built-in and custom potentials</td><td>energies, gradients, sampling shapes, Gaussian variance convention, JIT behavior</td></tr>
<tr><td><a href="../smoke/test_linear_combination.py"><code>test_linear_combination</code></a></td><td>potential vector-space algebra</td><td>coefficient semantics, flattening, gradients, bridge evaluation, visualization</td></tr>
<tr><td><a href="../smoke/test_loss.py"><code>test_loss</code></a></td><td>four low-level losses</td><td>per-sample shape, F/G formulas, permutation-based X term, stochastic trace-key route</td></tr>
<tr><td><a href="../smoke/test_loss_training.py"><code>test_loss_training</code></a></td><td>loss differentiability across flows</td><td>finite gradients, trainability, architecture comparison on one target</td></tr>
<tr><td><a href="../smoke/test_metrics.py"><code>test_metrics</code></a></td><td>weights, ESS, coverage, resampling</td><td>analytic ESS cases, log/linear agreement, F/G weight agreement, numerical degeneracies, k-NN coverage, resampling frequencies</td></tr>
<tr><td><a href="../smoke/test_rejuvenation.py"><code>test_rejuvenation</code></a></td><td>Langevin, stochastic Heun, HMC</td><td>step/full-kernel consistency, MALA acceptance, tamed ULA, nonfinite MALA/HMC rejection, aliases</td></tr>
<tr><td><a href="../smoke/test_annealing.py"><code>test_annealing</code></a></td><td>SMC and flow-proposal AIS</td><td>bridge transport, level ESS, F/G directions, initial weight return, CNF/OTFlow numerical paths</td></tr>
<tr><td><a href="../smoke/test_optimization.py"><code>test_optimization</code></a></td><td>L-BFGS and AdamW sample optimization</td><td>state kernels, complete loops, Armijo behavior, chunked optimization, nonfinite candidate rejection and recovery</td></tr>
<tr><td><a href="../smoke/test_utils_api.py"><code>test_utils_api</code></a></td><td>flat <code>jflows.utils</code> namespace</td><td>canonical signatures, semantic aliases, representative execution</td></tr>
<tr><td><a href="../smoke/test_train.py"><code>test_train</code></a></td><td>direct stage trainers and <code>Monitor</code></td><td>compiled scans, ESS histories, deterministic seeds, F/G sampling directions</td></tr>
<tr><td><a href="../smoke/test_clip.py"><code>test_clip</code></a></td><td><code>u_clip</code> and <code>g_clip</code></td><td>loss screening, stable global gradient clipping, trainer/generator propagation</td></tr>
<tr><td><a href="../smoke/test_checkpoint.py"><code>test_checkpoint</code></a></td><td>flow serialization and stochastic CNF state</td><td>Equinox leaf serialization, template reconstruction, approximate-CNF trace keys, ordinary packed training</td></tr>
<tr><td><a href="../smoke/test_edge_cases.py"><code>test_edge_cases</code></a></td><td>cross-module numerical contracts</td><td>RQS inverse roundoff, degenerate weights, nonfinite-row resampling, extreme values, small configurations, identity starts</td></tr>
<tr><td><a href="../smoke/test_boltzmann.py"><code>test_boltzmann</code></a></td><td>adaptive-staging and fixed-schedule Boltzmann generators</td><td>stage records, ESS selection, identity fallback, fixed advancement, completion</td></tr>
<tr><td><a href="../smoke/test_boltzmann_identity.py"><code>test_boltzmann_identity</code></a></td><td>identity-only adaptive-staging Boltzmann generator</td><td>flow-free signature, ESS shrinking, deterministic advancement, and absence of trainer calls</td></tr>
<tr><td><a href="../smoke/test_boltzmann_identity_artifacts.py"><code>test_boltzmann_identity_artifacts</code></a></td><td>flow-free complete-stage persistence</td><td>identity save/load/resume with no flow artifacts</td></tr>
<tr><td><a href="../smoke/test_boltzmann_artifacts.py"><code>test_boltzmann_artifacts</code></a></td><td>complete-stage persistence</td><td>create/write/load, interruption boundary, continuation flow, and stage readers</td></tr>
<tr><td><a href="../smoke/test_boltzmann_chunks.py"><code>test_boltzmann_chunks</code></a></td><td>KLXX memory-control and advancement path</td><td>one <code>chunks</code> spelling, forwarding into quench-and-temper, and rejection of a nonfinite post-stage population</td></tr>
<tr><td><a href="../smoke/test_chunk.py"><code>test_chunk</code></a></td><td>chunked full-set weights</td><td>chunk-count equivalence and eager device-memory partition behavior</td></tr>
<tr><td><a href="../smoke/test_public_api.py"><code>test_public_api</code></a></td><td>package namespace</td><td>client-free backend report, <code>jflows.train</code>/<code>jflows.boltzmann</code> split, retired <code>jflows.training</code>, trainer signatures, KLXX pool semantics</td></tr>
</tbody>
</table>

</div>

## Routing by interface level

### Low level

Use these first when changing foundational numerical behavior:

```text
flow model              -> test_flow, test_circular
potential               -> test_potential, test_linear_combination
loss                    -> test_loss, test_loss_training
weight/ESS/coverage     -> test_metrics
MCMC                    -> test_rejuvenation
SMC/AIS                 -> test_annealing
sample optimization     -> test_optimization
flat utility surface    -> test_utils_api
```

### Medium level

```text
direct training         -> test_train
screening/clipping      -> test_clip
serialization/CNF keys  -> test_checkpoint
KLXX QT chunks          -> test_boltzmann_chunks
numerical boundaries    -> test_edge_cases
```

### High level

```text
stage controller        -> test_boltzmann
identity controller     -> test_boltzmann_identity
stage persistence       -> test_boltzmann_artifacts
identity persistence    -> test_boltzmann_identity_artifacts
full-set chunking       -> test_chunk, test_boltzmann_chunks
public module split     -> test_public_api
```

## What to verify in output

Different modules print different diagnostics, but the recurring
postconditions are:

- arrays have the documented shapes and finite values;
- forward/inverse reconstruction error is bounded for the tested flow;
- forward and inverse log-Jacobians have opposite signs;
- normalized ESS lies in `[0,1]`;
- trainer history length equals `train_steps`;
- adaptive-staging records align histories by attempt;
- a claimed complete Boltzmann run ends at `t=1.0`;
- stored run manifests reference every required complete-stage file; and
- aliases behave identically to their canonical long names.

## Isolated verification

Examples and tests can produce logs, figures, bytecode, and compiled caches.
For verification that must leave the public checkout unchanged, copy the
package and selected module to a temporary repository root:

```bash
source ~/.envs/jflows/bin/activate
temp_root=$(mktemp -d /tmp/jflows-smoke.XXXXXX)
mkdir -p "$temp_root/jflows"
rsync -a \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  /data/projects/jflows/jflows \
  /data/projects/jflows/smoke \
  /data/projects/jflows/pyproject.toml \
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

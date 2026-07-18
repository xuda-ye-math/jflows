# Example workflows

The [example directory](../example/) contains complete programs that connect
the low-, medium-, and high-level interfaces. The examples are the best place
to copy end-to-end structure; the smoke tests are the better source for narrow
contracts and edge cases.

Run examples as modules from the repository root:

```bash
source ~/.envs/jflows/bin/activate
PYTHONPATH=/data/projects/jflows \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m example.2D_single
```

The scripts write outputs next to themselves. Use a temporary copied checkout
when repository cleanliness matters.

## Example map

<div align="center">

<table>
<thead>
<tr><th>Example</th><th>Interface levels</th><th>Main subjects</th><th>Outputs</th></tr>
</thead>
<tbody>
<tr><td><a href="../example/2D_single.py"><code>2D_single.py</code></a></td><td>low + medium</td><td>NSF, Gaussian mixture, reverse KL vs forward KL, ESS</td><td><code>2D_single.png</code>, log</td></tr>
<tr><td><a href="../example/3D_periodic.py"><code>3D_periodic.py</code></a></td><td>low + medium</td><td>NCSF on a torus, log weights, resampling, target MALA</td><td><code>3D_periodic.png</code>, log</td></tr>
<tr><td><a href="../example/4D_boltzmann.py"><code>4D_boltzmann.py</code></a></td><td>low + medium + high</td><td>adaptive reverse/forward Boltzmann ladders, stage ESS, identity comparison</td><td><code>4D_boltzmann.png</code>, log</td></tr>
<tr><td><a href="../example/CNF_vs_OTFlow.py"><code>CNF_vs_OTFlow.py</code></a></td><td>low + medium</td><td>continuous flows, checkpointing, held-out ESS, dimension scaling</td><td><code>CNF_vs_OTFlow.csv</code>, log</td></tr>
<tr><td><a href="../example/flow_scaling_law.py"><code>flow_scaling_law.py</code></a></td><td>low</td><td>NSF forward/inverse latency, warmup, synchronization, architecture scaling</td><td><code>flow_scaling_law.csv</code>, log</td></tr>
</tbody>
</table>

</div>

The current detailed numerical record is
[example/results.md](../example/results.md). CSV files are machine-readable;
the Markdown record explains the setup and scientific interpretation.

## 2D single-stage training

[Source](../example/2D_single.py) ·
[Figure](../example/2D_single.png) ·
[Results](../example/results.md#2d_single--reverse-kl-vs-forward-kl)

This example builds a bounded NSF and a three-component Gaussian-mixture
target, then trains two independent copies:

```text
Gaussian source
├── train_reverse_KL_F -> F-native proposal -> flow(x)
└── train_forward_KL_G -> G-native proposal -> flow.inv(x)
```

It demonstrates:

- explicit constructor and sample keys;
- identity initialization;
- `Monitor` output during a compiled trainer scan;
- the difference between optimizer-batch ESS and final full-set ESS;
- the F/G generation-direction rule; and
- visual comparison of target and generated samples.

The verified 2026-07-18 run reported final ESS `0.9425` for reverse KL and
`0.9469` for forward KL. The figure shows that all three target modes are
represented.

## 3D periodic training

[Source](../example/3D_periodic.py) ·
[Figure](../example/3D_periodic.png) ·
[Results](../example/results.md#3d_periodic--the-ncsf-on-the-torus)

This example uses `NCSF` for three periodic coordinates. It demonstrates:

- defining a periodic target energy through `Potential`;
- circular spline geometry and wrapped coordinate representatives;
- reverse and forward direct training;
- log-space importance weights and `compute_ESS_log`;
- weighted resampling; and
- MALA freshening at the final target.

The verified run reported final ESS `0.8980` for reverse KL and `0.9046` for
forward KL. The plot uses periodic projections rather than treating the domain
as ordinary unconstrained Euclidean space.

## 4D adaptive Boltzmann generator

[Source](../example/4D_boltzmann.py) ·
[Figure](../example/4D_boltzmann.png) ·
[Results](../example/results.md#4d_boltzmann--annealed-bg-with-the-adaptive-ladder)

The two-charge problem is the canonical high-level example. It demonstrates:

```text
source validation population
  -> adaptive endpoint proposal
  -> one direct stage trainer
  -> trained vs identity full-validation ESS
  -> accept/retry
  -> reweight/resample/MALA advancement
  -> next bridge stage
```

It runs adaptive reverse KL and forward KL generators separately and records
their accepted bridge endpoints and incremental ESS values.

The verified reverse run accepted:

```text
t   = [0.098, 0.245, 0.4655, 0.7963, 1.0]
ESS = [0.780, 0.931, 0.947, 0.971, 0.991]
```

The initial candidates `0.2` and `0.14` were rejected attempts before the
accepted first endpoint `0.098`; rejected candidates do not appear in the
accepted ladder.

The verified forward run accepted:

```text
t   = [0.2, 0.5, 0.95, 1.0]
ESS = [0.763, 0.902, 0.962, 0.997]
```

These ESS values are stage-local. The example's geometric plots are necessary
to interpret whether the transported particles represent the intended
two-charge structure.

## CNF versus OTFlow

[Source](../example/CNF_vs_OTFlow.py) ·
[CSV](../example/CNF_vs_OTFlow.csv) ·
[Results](../example/results.md#cnf_vs_otflow--continuous-flows-across-dimension)

This sweep compares two continuous-flow families over dimensions
`4, 8, 16, 32, 64, 128`. It demonstrates:

- `CNF` and `OTFlow` construction;
- the `OTFlow.near_identity()` training requirement;
- optional checkpointing for continuous-flow losses;
- JIT compilation and synchronized timing;
- full-validation log weights and ESS; and
- CSV output after the complete sweep.

In the verified run, ESS at `d=128` was `0.4291` for CNF and `0.5927` for
OTFlow. See the CSV for every dimension and training time. Timings include the
script's stated boundaries and should only be compared on the same runtime and
hardware.

## NSF forward/inverse scaling

[Source](../example/flow_scaling_law.py) ·
[CSV](../example/flow_scaling_law.csv) ·
[Results](../example/results.md#flow_scaling_law--forward-vs-inverse-map-latency)

This example isolates map latency rather than training quality. It sweeps
dimensions and conditioner widths, warms up each compiled call, synchronizes,
and times forward and inverse maps separately.

NSF is MAF-style: its forward map is parallel while inversion is
autoregressive. For width `64x64`, the verified run measured:

```text
d=4:   forward 0.179 ms, inverse 0.436 ms
d=128: forward 0.407 ms, inverse 41.539 ms
```

This asymmetry informs the choice of F-native versus G-native training and
sampling direction. The CSV contains all `3 x 6` architecture/dimension cells.

## Selecting an example

<div align="center">

<table>
<thead>
<tr><th>If you need...</th><th>Start from...</th></tr>
</thead>
<tbody>
<tr><td>a minimal bounded flow training script</td><td><code>2D_single.py</code></td></tr>
<tr><td>periodic coordinates and post-training MALA</td><td><code>3D_periodic.py</code></td></tr>
<tr><td>adaptive accepted-stage training</td><td><code>4D_boltzmann.py</code></td></tr>
<tr><td>continuous-flow checkpointing and held-out ESS</td><td><code>CNF_vs_OTFlow.py</code></td></tr>
<tr><td>correct accelerator latency measurement</td><td><code>flow_scaling_law.py</code></td></tr>
</tbody>
</table>

</div>

## Example evidence and reproducibility

- Example constants are explicit experimental choices, not universal tuning
  defaults.
- First calls include compilation; do not compare an un-warmed call with a
  warmed call.
- Accelerator kernels and JAX versions can cause small numerical or timing
  differences even when the accepted algorithmic path is unchanged.
- A PNG is evidence for geometric support only after direct visual inspection.
- CSV and Markdown numbers should be updated from the same completed run.
- A completed example does not prove every smoke contract; use
  [04-smoke-tests.md](04-smoke-tests.md) for narrow API evidence.

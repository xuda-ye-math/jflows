# jflows — reconstruction plan

Port `zflows` (`/mnt/projects/zflows`, v0.6.2, PyTorch) to a JAX package `jflows`
in `/mnt/projects/jflows`, mirroring the zflows layout (two deliberate layout
changes, §1: `utils.py` expands into a `utils/` package; `tests/` splits into
`smoke/` + `examples/`).

- **Backend**: `jax` + `equinox` (user-confirmed). `eqx.Module` is the closest JAX
  analogue of `torch.nn.Module`, so the zflows class structure ports nearly 1:1.
- **Environment**: `~/.envs/jax` — Python 3.14.6, jax 0.10.2 (CUDA 13, 1 GPU),
  equinox 0.13.8, numpy 2.5.0, scipy 1.18.0, matplotlib 3.11.0.
- **Reference**: zflows source + `TREE.md` are the authoritative spec. The port keeps
  every public name, argument name, default value, and docstring convention identical
  unless §3 lists a divergence (JAX-forced or requested).
- Folder creation is confined to `/mnt/projects/jflows` (this build is the explicit
  purpose of the project folder).

---

## 1. Target layout (mirror of zflows)

```
jflows
├── pyproject.toml           # name jflows; deps: jax, equinox, numpy
├── README.md                # rewritten from zflows README for the JAX API
├── TREE.md                  # source-tree doc, same format as zflows/TREE.md
├── LICENSE                  # MIT, copied
├── jflows
│   ├── __init__.py          # re-exports flow, loss, potential, utils
│   ├── flow.py              # Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
│   ├── potential.py         # Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
│   │                        #   Nlog_Gaussian_Mixture, linear_combination (potential algebra)
│   ├── loss.py              # reverse_KL, forward_KL (type='F'/'G'), OT_loss — per-sample returns, shape (N,)
│   ├── train.py             # train_reverse_KL — packed single-stage training (no zflows counterpart)
│   ├── utils
│   │   ├── __init__.py      # re-exports the flat jflows.utils namespace
│   │   ├── metrics.py       # importance_weights (+log, type='F'/'G'), compute_ESS (+log),
│   │   │                    #   coverage (k-NN, Naeem et al. 2020), resample  [CESS dropped]
│   │   ├── optimization.py  # lbfgs (+ LBFGS_State / lbfgs_init / lbfgs_step);
│   │   │                    #   adamw (+ AdamW_State / adamw_init / adamw_step; no zflows counterpart)
│   │   ├── rejuvenation.py  # langevin, stochastic_heun, hamiltonian_monte_carlo
│   │   │                    #   (+ *_step kernels, leapfrog; aliases rejuvenation, hmc)
│   │   └── annealing.py     # sequential_monte_carlo, annealed_importance_sampling (type='F'/'G')
│   │                        #   (key-first, taming pass-through; AIS rejuvenates at the target)
│   └── core
│       ├── __init__.py
│       ├── transforms.py    # Transform hierarchy (ComposedTransform, RQS, coupling, …)
│       ├── flows.py         # lazy conditioner modules → concrete Transforms
│       ├── nn.py            # Linear / MLP / MaskedLinear / MaskedMLP
│       ├── numerics.py      # bisection, gauss_legendre, rk4_fixed, broadcast, unpack
│       └── otflow.py        # ResNN, OTPhi (trHess), OTFlowTransform
├── parity_check             # cross-framework zflows fixtures + checks (§5)
│   ├── parity_dump_*.py     # torch side — writes the parity_*.npz fixtures
│   └── parity_check_*.py    # jax side — transplants weights and asserts
├── smoke              # standalone jflows-only tests (test_*, user-run)
│   └── test_*.py            # flows, potentials, circular conditions, losses, …
└── example                  # runnable examples — X-regularization conventions
    └── 2D_single.py         #   (the newer reference; zflows' tests/ are legacy):
                             #   parameters block, fresh source batch per step
                             #   (no epochs), final ESS on a fresh N_VALID batch
```

Not mirrored: `.archive/`, `.venv/`, `zflows.egg-info/`, `Dockerfile`,
`conda.recipe/`, `git_init.sh`, `logo.png` (add later only on request);
`compare_compiled_{loss,inverse}.py` (torch.compile-specific tests).

---

## 2. Torch → JAX/equinox translation table

| zflows (torch)                            | jflows (jax + equinox)                                    |
|-------------------------------------------|-----------------------------------------------------------|
| `nn.Module`                                | `eqx.Module` (frozen PyTree dataclass)                     |
| `nn.Linear`, `nn.Sequential` MLP           | `eqx.nn.Linear` / custom `MLP(eqx.Module)` in `core/nn.py` |
| `MaskedLinear` (autoregressive masks)      | equinox module with mask as a stored array; mask applied in `__call__` |
| `torch.Tensor` ops                         | `jnp` ops (API nearly identical; `searchsorted`, `cumsum`, `logsumexp` all exist) |
| `autograd.Function` (Bisection)            | `jax.lax.custom_root` (implicit-diff root finder) or `jax.custom_vjp` |
| `autograd.Function` (GaussLegendre)        | `jax.custom_vjp` around fixed-node quadrature              |
| `rk4_fixed` Python loop                    | `jax.lax.scan` fixed-step RK4                              |
| FFJORD trace estimator (`autograd.grad`)   | `jax.jvp` / `jax.vjp` inside the ODE drift                 |
| OT-Flow `trHess`                           | direct port — closed-form trace, plain `jnp` math          |
| `torch.compile` machinery: `loss_compile{,_beta}`, `check_compile_available`, `set_cache_size_limit` | **dropped** — obsolete under JAX; `jax.jit` is applied directly where needed |
| `suppress_warnings`                        | **dropped** — the torch.compile noise it muted doesn't exist under JAX |
| `Potential.grad` via autograd              | `jax.grad` of the summed potential (or `vmap(grad)`)       |
| `enable_grad` / `enable_eval` / `enable_for_ladj` / `enable_inv_ladj` (torch.compile fast-path setup) | **dropped** — `grad`/`eval`/`call_and_ladj` are jit-compiled by default |
| in-place MCMC loops (langevin, HMC, SMC, AIS) | `jax.lax.scan` / `fori_loop` over functional state, jitted end-to-end |
| `chunk` argument (memory-bounded batching) | `jax.lax.map(..., batch_size=...)` or explicit chunked loop — same argument kept |
| `.state_dict()` / `.pth` checkpoints       | `eqx.tree_serialise_leaves` / `.eqx` files                 |
| `.to(device)`                              | not needed (JAX places arrays on GPU by default); omitted  |
| batched L-BFGS (`utils.lbfgs`)             | direct port of the two-loop recursion with `lax.scan` (keep zflows' own algorithm; do not substitute a library optimizer) |

---

## 3. Deliberate API divergences (JAX-forced or requested; documented in README + TREE)

1. **Explicit PRNG keys.** No global seed in JAX. Every random entry point gains a
   `key: jax.Array` argument: flow constructors (weight init, `randmask` permutations),
   `Potential.samples`, `resample`, `langevin`, `hamiltonian_monte_carlo`,
   `stochastic_heun`, `sequential_monte_carlo`, `annealed_importance_sampling`.
   Convention: `key` is the first positional argument (equinox style).
2. **Per-sample losses.** Every function in `loss.py` returns the per-sample loss
   vector, shape `(N,)` aligned with the batch — no internal reduction; callers
   take `.mean()` themselves.
3. **No compile machinery.** `loss_compile{,_beta}`, `check_compile_available`,
   `set_cache_size_limit`, `suppress_warnings`, and the `enable_grad`/
   `enable_eval`/`enable_for_ladj`/`enable_inv_ladj` fast-path setters do not
   exist in jflows — `jax.jit` covers compilation, and the warning noise the
   torch stack produced has no JAX counterpart.
4. **`utils` is a package.** `jflows.utils` splits into `metrics.py` /
   `optimization.py` / `rejuvenation.py` / `annealing.py`;
   `utils/__init__.py` re-exports the flat namespace so
   `jflows.utils.<fn>` call sites read the same as in zflows.
   `compute_CESS` / `compute_CESS_log` are dropped (user-requested);
   `coverage` (k-NN mode-collapse diagnostic, Naeem et al. 2020, ported from
   the X-regularization 2D benchmarks) is added.
5. **Immutability.** `eqx.Module` is frozen: `zeros()` returns a new instance
   instead of mutating in place. Retuning a linear combination's coefficients
   is a rebuild (or an `eqx.tree_at` replacement of the `coeffs` leaf) — the
   pytree structure is unchanged, so jitted consumers do not recompile.
6. **No `beta` temperature arguments** (user-requested). Every potential `U` is
   always the energy of `exp(-U)`: `beta` / `beta_source` / `beta_target`
   disappear from `loss.py`, the importance weights, the samplers
   (`langevin`, `hamiltonian_monte_carlo`, `stochastic_heun`, SMC, AIS), and
   `Gaussian.samples`. Temperature scaling, where needed, is expressed through
   the potential itself (e.g. `Linear_Combination` coefficients).
7. **Conditioner plumbing.** JAX pytrees cannot hold parent back-references, so
   `MaskedAutoregressiveTransform` / `GeneralCouplingTransform` take
   `univariate` as a plain callable (e.g. `MonotonicRQSTransform`) plus explicit
   `bound` / `slope` fields, instead of zflows' bound-method closure
   (`self._univariate`). Same math, same parameters.
8. **Activation arguments** become callables (`jax.nn.silu`) instead of
   `nn.Module` classes.
9. **dtype**: float32 default, matching torch. No `jax_enable_x64` flag inside the
   package; parity checks in `smoke/` may enable it locally.
10. Checkpoint format is `.eqx` (tree-serialised leaves), not `.pth`.
11. **Hutchinson probe (CNF `exact=False`).** The trace-estimator noise is drawn
    from the flow's stored PRNG key — fixed per instance, not resampled per
    call; rebuild with a fresh key for a new probe. The default `exact=True`
    path is unaffected.
12. **`Nlog_` potential naming** (user-requested). The concrete potentials are
    `Nlog_Uniform` / `Nlog_Gaussian` / `Nlog_Gaussian_Mixture` — the `Nlog_`
    prefix (negative log) marks the translation from distribution language to
    potential language: each is the energy `U = -log p` of its density, always
    paired with `exp(-U)`. The `device` constructor argument is dropped (JAX
    places arrays), and `Potential.grad` needs no enabling step.
13. **Combined F/G interface** (user-requested). `reverse_KL` / `forward_KL` /
    `importance_weights{,_log}` / `annealed_importance_sampling` take the flow
    as `transform` plus `type: str` ('F' = forward map source -> target,
    'G' = inverse map target -> source) instead of paired `_F`/`_G` functions;
    aliases `smc` and `ais`. The high-level functions take the `Flow` itself —
    the `ComposedTransform` layer stays internal/advanced (`t()`), and `Flow`
    exposes `__call__` / `call_and_ladj` / `inv` / `inv_and_ladj` as thin
    delegations to it.
14. **Potential algebra** (user-requested redesign). Potentials form a vector
    space over instances: `linear_combination(potentials, coeffs)` and the
    arithmetic operators on `Potential` (`c*U`, `U+V`, `U-V`, `-U`, `U/c`,
    `sum([...])`) build FLAT combinations — nested combinations are absorbed,
    and repeated instances (by object identity) merge with summed
    coefficients, so `(0.5*U1 + 0.5*U2) + (0.5*U2 + 0.5*U3)` has exactly the
    terms `(U1, U2, U3)` with coefficients `(0.5, 1.0, 0.5)`. There is no
    public `Linear_Combination` class and no `set_coeffs`; coefficients are a
    `(n,)` array leaf and may be traced (bridges built inside jit are fine).
14. **Two-level sampler interface** (user-requested). `optimization.py` and
    `rejuvenation.py` expose both the high-level loops (zflows names and
    returns: `lbfgs`/`optimization`, `langevin`/`rejuvenation`,
    `stochastic_heun`, `hamiltonian_monte_carlo`/`hmc`) and the low-level
    kernels they scan over (`LBFGS_State`/`lbfgs_init`/`lbfgs_step`;
    `langevin_step`, `stochastic_heun_step`, `hmc_step`, `leapfrog`). Steps
    return `(x, aux)` with MH diagnostics; every loop is exactly a lax.scan
    over its public step (loop ≡ manual composition, tested), with keys
    derived as fold_in(key, chunk_index) then split(chunk_key, iters).
    Naming convention (outside `core/`, which keeps zuko's names): classes
    are Alice_Bob style, instances/functions alice_bob.
15. **NCSF circular spline** (user-approved fix; diverges from zflows/zuko).
    `MonotonicRQSTransform(..., circular=True)` wrap-shares the first
    unconstrained derivative onto the last knot — `d_0 = d_K`, one learnable
    seam slope — and NCSF emits `shapes=[(bins,), (bins,), (bins,)]`. Each
    coordinate map is a C¹ circle diffeomorphism with a trainable seam
    density (Rezende et al., 2020; the convention normflows implements).
    zflows/zuko pin both seam slopes to 1, which freezes the model density at
    ±π to the base value. The Phase-3 zflows cross-check therefore builds the
    legacy composition explicitly for the NCSF machinery comparison;
    `smoke/test_circular.py` tests the circular conditions standalone.

Everything else — names, call signatures, defaults, `F`/`G` conventions,
the `t() -> ComposedTransform` access path — stays identical.

---

## 4. Build order (each phase gated by a parity check before the next)

Dependency order is bottom-up, same as the zflows internal dependency graph:

- **Phase 0 — scaffolding.** `pyproject.toml` (name `jflows`, deps `jax`, `equinox`,
  `numpy`), `LICENSE`, package skeleton. jflows is NOT pip-installed: everything
  runs from the repo root against the local source tree.
  Gate: `~/.envs/jax/bin/python -c "import jflows"` from the repo root.
- **Phase 1 — `core/numerics.py` + `core/nn.py`.** bisection (custom_root),
  gauss_legendre (custom_vjp), rk4_fixed (scan), broadcast, unpack, Partial;
  Linear/MLP/MaskedLinear/MaskedMLP. Gate: fixture parity (§5) on quadrature,
  root-finding (values + gradients), masked-MLP autoregressive property.
- **Phase 2 — `core/transforms.py`.** Transform base + ComposedTransform,
  Identity, Additive, MonotonicAffine, MonotonicRQS (per-coord `bound`),
  CircularShift, Autoregressive, Coupling, FreeFormJacobian, Rotation, LULinear,
  Dependent. Gate: for each transform, round-trip `inv(f(x)) ≈ x` and
  `ladj` vs autodiff jacobian; RQS/CircularShift parity against torch fixtures.
- **Phase 3 — `core/flows.py` + `core/otflow.py` + `flow.py`.** Lazy conditioner
  modules, OT-Flow machinery, public NSF/NCSF/CNF/OTFlow/RealNVP.
  Gate: weight-transplant test (§5) — identical parameters ⇒ identical
  forward/inverse/ladj outputs vs zflows, per flow class.
- **Phase 4 — `potential.py`.** Step 4a: Potential base (`__call__`/`grad`),
  potential_from, Nlog_Uniform, Nlog_Gaussian, Nlog_Gaussian_Mixture (§3.12).
  Step 4b: linear_combination + operator algebra on Potential (§3.13).
  Gate: energy/grad parity fixtures; sampler moments (mean/cov) statistical match.
- **Phase 5 — `loss.py`.** reverse/forward_KL_{F,G}, OT_loss — each returns the
  per-sample loss vector, shape (N,). Gate: batch-mean of the jflows vector equals
  the zflows scalar loss on transplanted weights + fixed input batches.
- **Phase 6 — `utils/` package.** Step 6a: `metrics.py` (compute_ESS (+log),
  importance_weights (+log, type='F'/'G'), coverage, resample; CESS dropped).
  Step 6b/6c: `optimization.py` + `rejuvenation.py` with the two-level
  interface (§3.14). Step 6d: `annealing.py` (sequential_monte_carlo +
  annealed_importance_sampling (type='F'/'G'); key-first, taming pass-through, AIS
  rejuvenation at the target). suppress_warnings dropped (§3.3).
  Gate: deterministic pieces (weights, ESS, lbfgs on a quadratic) exact-parity;
  stochastic pieces validated statistically (§5).
- **Phase 7 — `example/` + docs.** Runnable examples following the
  X-regularization conventions (the newer reference — zflows' tests/ are
  legacy): parameters block, no epoch / training-set notion (fresh source
  batch per step), AIS-supplied forward KL data, final ESS on a fresh
  N_VALID batch. Built incrementally under user direction (first:
  2D_single.ipynb — reverse KL vs forward KL, single stage). Write
  README.md + TREE.md at the end. (The `smoke/` `test_*` suite is built
  incrementally with each phase, not here.) Gate: every script runs
  END-to-END on the jax env GPU with a live status log and reported ESS.

---

## 5. Verification protocol (cross-framework parity)

Two frameworks, two envs — parity runs through saved `.npz` fixtures:

1. **Fixture dump (torch side).** A script run with the zflows interpreter
   (per `/torch` skill) samples random inputs + parameters, evaluates the zflows
   module, saves `{inputs, params, outputs}` to `parity_check/parity_<module>.npz`.
   Read-only w.r.t. the zflows repo — nothing is written into `/mnt/projects/zflows`.
2. **Parity check (jax side).** `~/.envs/jax/bin/python` loads the fixture, builds
   the jflows twin, transplants the parameters (numpy is the common currency),
   asserts `allclose` — rtol/atol 1e-5 (float32) on values, 1e-4 on gradients;
   float64 spot-checks where a discrepancy needs adjudication.
3. **Weight transplant for whole flows.** zflows `state_dict → npz → eqx.tree_at`
   mapping per flow class; then forward/inverse/ladj agreement on shared inputs.
4. **Stochastic algorithms** (samplers, SMC, AIS): distributional checks — moments,
   ESS, log-Z estimates vs analytic values (Gaussian/GMM targets) within Monte-Carlo
   error bars, plus torch-vs-jax consistency on the same target.
5. Every runnable verification script follows the observability rules: flushed
   timestamped log to file + stdout, START/progress/DONE lines, no tqdm, launched
   via the tracked background mechanism.
6. Official invocation: jflows-side tests run from the repo root as
   `~/.envs/jax/bin/python -m smoke.<test_name>` /
   `-m parity_check.<check_name>` / `-m example.<name>`,
   mirroring zflows' `python -m tests.<name>` convention — they always
   exercise the local source tree, never an installed copy. They use the
   default JAX backend (GPU when available; `JAX_PLATFORMS=cpu` forces CPU)
   with GPU memory preallocation disabled.

---

## 6. Open points (defaults chosen, flag if you want otherwise)

- **Optimizer for the training loops in `examples/`**: zflows tests use `torch.optim.Adam`;
  the jax env has no optax. Default: a ~30-line Adam inside the test helper
  (keeps jflows deps at jax+equinox+numpy). Alternative: `pip install optax`
  into `~/.envs/jax` — say the word and the examples use optax instead.
- **`key` placement**: first positional argument (equinox convention). Alternative:
  trailing keyword-only `key=` to keep positional signatures byte-identical to zflows.
- **4D_Boltzmann_generator.pth**: its jflows counterpart will be regenerated as
  `.eqx` by training the ported script, not converted from the torch checkpoint.

# Project status

Last updated: 2026-07-12T22:38:24-04:00

## Current state

- Branch: `main`, tracking `origin/main`.
- Pre-checkpoint HEAD: `0302829fe440b6241172b652ff914db1ebecc273`
  (`Repair forward AIS and quench correctness`).
- The reviewed refactor and this handoff are currently uncommitted: 16 tracked
  files are modified and five source/smoke/status files are untracked. The main paths are
  `jflows/train.py`, `jflows/boltzmann.py`, `jflows/utils/`, `_artifacts.py`,
  the public examples/README, and package smoke tests.
- Public controls now use `dt`/`steps` for primitive integration,
  `mc_dt`/`mc_steps`, `opt_alpha`/`opt_steps`, `train_steps`, `batch_size`,
  `pool_size`, and `chunks`. Retired keyword spellings remain strict
  compatibility aliases and duplicate old/new spellings are rejected.
- Boltzmann stage records distinguish optimizer `batch_ess_hist` from
  full-validation `valid_selected_ess`, `valid_trained_ess`, and
  `valid_identity_ess`. Only the full-validation trained-versus-identity ESS
  controls post-training acceptance.
- Optional `flow_dir` persistence atomically saves every trained attempt,
  including rejected and identity-losing candidates, plus the selected map,
  per-attempt monitor data, relative paths, and a terminal run manifest.
- Verification passed in the pip-only `~/.envs/jflows` environment with the
  live source tree: `smoke.test_utils_names`, `smoke.test_train`,
  `smoke.test_annealing`, `smoke.test_boltzmann`,
  `smoke.test_boltzmann_artifacts`, `smoke.test_chunk`, `compileall`, and
  `git diff --check`.
- Two independent final reviews and a cross-package review found no remaining
  major or medium scientific/API issue.
- A verified pre-refactor standalone snapshot is available at
  `/mnt/games/jflows_071226`.

## Pending

- Pending outside this repository: rerun the historical
  `/mnt/projects/X-regularization/Codes` suite only when the user authorizes
  that scientific rerun. This interface refactor does not itself invalidate
  the old results.
- Pending outside this repository: resume `Molecular_BG` experiments only
  under separate instructions.
- Low-priority observability polish: explicitly label `max_stages` as the
  terminal reason in the generic run manifest.
- Remove temporary stage/keyword compatibility aliases only in a future major
  API version.

## Timeline

### 2026-07-12T22:38:24-04:00 — Naming, ESS history, and flow-artifact refactor verified

- Standardized utility, trainer, and Boltzmann-generator controls while
  preserving positional behavior and old keyword compatibility.
- Added attempt-aligned `t_hist`, `batch_ess_hist`, validation-ESS histories,
  statuses, and saved-flow paths to all adaptive and fixed BG families.
- Added atomic, non-overwriting attempt persistence and exact flow reload
  regression tests; saving enabled/disabled produces identical numerical
  results.
- Restored full public `help()` documentation through compatibility wrappers.
- Independent package and cross-package audits found no major or medium issue.

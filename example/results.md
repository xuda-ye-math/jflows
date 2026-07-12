# Example results

Numerical results of the runnable examples in this folder.

## 2D_single — reverse KL vs forward KL

Single-stage flow training on a 2D three-mode target, comparing the reverse KL and forward KL objectives under the 100% energy-driven workflow of [`2D_single.py`](2D_single.py).

### Setup

- **Source** $\mu_0 \propto e^{-U_0}$: isotropic Gaussian, $U_0(x) = \lVert x\rVert^2 / (2\sigma^2)$, $\sigma = 2$.
- **Target** $\mu_1 \propto e^{-U_1}$: three-mode Gaussian mixture placed like the Julia three-dot sign — equal weights, means $(0, 2.4)$, $(-2.2, -1.4)$, $(2.2, -1.4)$, per-coordinate variance $0.3$. The modes are separated by $\sim 8$ standard deviations, so a mode-seeking objective can lose mass while a mass-covering one should not.
- **Flow**: NSF on $[-5, 5]^2$, 16 bins, 4 autoregressive transforms, $(64, 64)$ conditioners, identity-initialised (`zeros()`).
- **Training**: one packed call per method — `N_VALID = 40000` fixed source set, `N_BATCH = 2000` per Adam step, `STEPS = 200`, `LR = 1e-3`, Langevin rejuvenation `MC_STEP = 1e-3`, `MC_ITERS = 100`, single-level AIS (`LADDER = 1`).

Both trainers regenerate their batch inside every Adam step. The reverse KL flow acts as $F$ (source → target, `train_reverse_KL_F`): each step draws an `N_BATCH` subset of the fixed set and freshens it with Langevin steps at the source. The forward KL flow acts as $G$ (target → source, `train_forward_KL_G`): each step manufactures its target batch by single-level AIS through the *current* flow — pushforward, self-normalized reweighting, multinomial resampling, Langevin rejuvenation at the target. Neither objective touches the inverse map during training, and the final ESS is computed on the full fixed set through the flow importance weights.

### Results

<div align="center">

| objective  | final ESS ($N = 40000$) | batch ESS along training |
| :--------: | :---------------------: | :----------------------: |
| reverse KL |         0.9242          |   0.35 -> 0.88 -> 0.94   |
| forward KL |         0.9455          |   0.84 -> 0.96 -> 0.97   |

</div>

<p align="center"><img src="2D_single.png" alt="2D single-stage training" width="1000px"></p>

The left panel shows the per-step batch-ESS histories returned by the trainers (x-axis exactly $[0, 200]$); the middle and right panels show the `N_VALID` pushforward samples of each method (blue: $y = F(x)$; red: $y = G^{-1}(x)$) over the target energy background, with 5000 source samples in gray for context.

### Reading the result

Both flows populate all three modes, and the AIS-fed forward KL ends with the higher ESS: its data pipeline supplies (approximately) target-distributed samples covering every mode from the first step — its batch ESS starts already at $\sim 0.84$ — while the reverse KL starts near $0.35$ and pays its mode-seeking tax as the barriers grow. On a harder target — farther modes, or modes the initial proposal never reaches — the reverse objective can collapse entirely, and the forward objective inherits whatever the AIS chain covers; that regime (and the regularization that addresses it) is the subject of the X-regularization line of work these conventions come from.

Two practical notes. First, wall clock: each packed 200-step training compiles in a few seconds and executes in 1.5–2.5 s on an RTX 5070 Ti; the whole script (both trainings, evaluation, figure) runs in about 12 s. Second, reproducibility: the trainers are deterministic within a process (fixed internal seed — two identical calls agree bit for bit), while across separate launches XLA kernel autotuning can introduce last-ulp differences that 200 training steps amplify, so the final ESS varies by a few times $10^{-2}$ between runs (reverse KL $\approx 0.92$–$0.93$, forward KL $\approx 0.95$).

## 3D_periodic — the NCSF on the torus

Run by [`3D_periodic.py`](3D_periodic.py): its purpose is to show that the **NCSF actually works** — the circular-spline flow trains, inverts, and reweights correctly on a genuinely periodic domain.

### Setup

- **Source** $\mu_0$: uniform on the periodic box $[-\pi, \pi]^3$.
- **Target** $\mu_1 \propto e^{-U_1}$: the von Mises ridge mixture on the 3-torus,
  $U_1(x) = -\log[\, e^{\kappa\cos(x_1 - x_2)} + e^{\kappa\cos(x_2 - x_3)} + e^{\kappa\cos(x_3 - x_1)} \,]$, $\kappa = 4$ — three pairwise ridges that wrap around the torus.
- **Flow**: NCSF on $[-\pi, \pi]^3$, 8 bins, 4 autoregressive transforms, $(128, 128)$ conditioners, identity-initialised.
- **Training**: one packed call per objective — `N_VALID = 40000` fixed source set, `N_BATCH = 2000` per Adam step, `STEPS = 200`, `LR = 1e-3` — followed by the reweighting pipeline: importance weights → ESS → multinomial resampling → MALA rejuvenation at the target (`1e-3 × 100`).

### Results

<div align="center">

| objective  | final ESS ($N = 40000$) |
| :--------: | :---------------------: |
| reverse KL |         0.8978          |
| forward KL |         0.9051          |

</div>

<p align="center"><img src="3D_periodic.png" alt="3D periodic NCSF" width="1000px"></p>

Both panels show the resampled and rejuvenated particle sets (left: reverse KL; right: forward KL) concentrating on the wrap-around ridge tubes of the target — the structure a non-periodic flow cannot represent without seam artifacts. The healthy ESS of both objectives on this domain is the point: the NCSF's circular splines carry the periodic geometry end to end.

## 4D_boltzmann — annealed BG with the adaptive ladder

The 4D two-charge target, sampled by [`4D_boltzmann.py`](4D_boltzmann.py) with BOTH annealed generators — `boltzmann_reverse_KL_F` and `boltzmann_forward_KL_G` — along the bridge ladder $U_t = (1-t)\,U_0 + t\,U_1$, with the coefficient $t$ selected adaptively instead of a fixed schedule $c_k = k/12$.

### Setup

- **Source** $\mu_0$: standard Gaussian on $\mathbb R^4$.
- **Target**: two particles $x = (x_1, x_2)$, $x_i \in \mathbb R^2$, on a soft annulus with regularized 3D Coulomb repulsion,
  $U_1(x) = a\,[(\lVert x_1\rVert^2 - r_0^2)^2 + (\lVert x_2\rVert^2 - r_0^2)^2] + q^2 / \sqrt{\lVert x_1 - x_2\rVert^2 + \varepsilon^2}$
  with $r_0 = 2$, $a = 1$, $q^2 = 4$, $\varepsilon = 10^{-3}$ (identical to the original).
- **Flow**: NSF on $[-3, 3]^4$, 8 bins, 6 autoregressive transforms, $(64, 64)$ conditioners, identity-initialised.
- **Boltzmann generators**: the stage flows are connected step by step — stage $k$ selects $t_k$ through the SMC gate (`tau_smc`, `LADDER = 1` level, on an `N_POOL`-sized selection pool drawn from the particle set), trains the warm-started flow as the incremental map $\mu_{t_{k-1}} \to \mu_{t_k}$ on the advancing particle set, accepts on the incremental importance-sampling ESS (`tau_ess`), and advances the set by reweight → resample → MALA at $U_{t_k}$ (`mc_adjust = True`; the Metropolis gate keeps the near-singular Coulomb tail out of the particle set). The reverse KL stages train on Langevin-freshened batches of the set; the forward KL stages train on target batches manufactured per Adam step by AIS through the current flow (SMC gate and AIS share `LADDER`). Parameters: `N_VALID = 120000`, `N_POOL = 24000`, `N_BATCH = 2000`, `STEPS = 500`, `LR = 1e-4`, MALA `1e-3 × 100`; ladder `t_safe = 0.2`, `shrink_factor = 0.7`, `enlarge_factor = 1.5`, `tau_smc = 0.2`, `tau_ess = 0.6`.

### Results

Both ladders reach $t = 1$ in four stages (~8-10 s each on the full 120000-particle set; the stage trainer, weight evaluation, and advance each compile once and are reused across all stages). The rejection machinery earns its keep in the reverse run: its safe start $t = 0.2$ trains but misses the acceptance bar and shrinks to $0.14$, while the AIS-fed forward run accepts the full safe start and climbs faster — every other stage passes on its first attempt:

<div align="center">

| reverse KL, stage $k$ |   1   |   2   |   3   |   4   |
| :-------------------: | :---: | :---: | :---: | :---: |
|         $t_k$         | 0.14  | 0.35  | 0.665 |  1.0  |
|          ESS          | 0.634 | 0.904 | 0.957 | 0.985 |

</div>

<div align="center">

| forward KL, stage $k$ |   1   |   2   |   3   |   4   |
| :-------------------: | :---: | :---: | :---: | :---: |
|         $t_k$         | 0.20  | 0.50  | 0.95  |  1.0  |
|          ESS          | 0.763 | 0.897 | 0.962 | 0.996 |

</div>

The accepted ESS is per stage the better of the trained flow and the identity map (pure SMC reweighting): after training, each stage keeps whichever has the higher incremental ESS, so a stage is never worse than SMC. Here the trained flow wins every level except the forward run's last, where the identity fallback edges it ($0.996$ against the trained $0.986$).

<p align="center"><img src="4D_boltzmann.png" alt="4D Boltzmann generator" width="1000px"></p>

Each row (top: reverse KL; bottom: forward KL) shows the adaptive ladder ($t_k$ and the per-stage incremental ESS), the particle-1 marginal at $t = 1$ concentrated on the annulus $\lVert x_1 \rVert = r_0$ (dashed circle), and the relative angle $\Delta\theta$ between the two particles, peaked at $\pm\pi$ with vanishing density at $0$ — the antipodal Coulomb minimum.

### Reading the result

The ESS trace follows the reference behaviour of the original fixed-ladder run — a lower leading level, then a high plateau — while the ESS-gated selection compresses the schedule: the leading increment is small (`t_safe`), the accepted step then grows by the enlarge factor, and the final extrapolation snaps to $t = 1$, so four or five stages cover what the fixed schedule spent twelve levels on. The generator's sample output is the advanced particle set; the per-stage incremental flows and their acceptance ESS are returned in the stage records.

## CNF_vs_OTFlow — continuous flows across dimension

CNF versus OTFlow on a fixed multi-modal target as the dimension grows, run by [`CNF_vs_OTFlow.py`](CNF_vs_OTFlow.py). Both continuous flows are trained by the **same** objective, plain reverse KL, so the comparison isolates the one variable that differs: the velocity-field architecture.

### Setup

- **Source** $\mu_0$: standard Gaussian on $\mathbb R^d$.
- **Target** $\mu_1 \propto e^{-U_1}$: a factorized multi-well potential whose mode count stays fixed while $d$ is swept,
  $U_1(x) = \sum_{i < 3} \beta_{\mathrm w}\,((x_i / s)^2 - 1)^2 + \sum_{i \ge 3} \tfrac12 x_i^2$, with $s = 1.5$, $\beta_{\mathrm w} = 1.5$. The first three coordinates are symmetric double wells (minima at $\pm s$), giving $2^3 = 8$ modes; the remaining $d - 3$ coordinates are standard Gaussian and only raise the dimension. The barrier is deliberately shallow so mode-seeking reverse KL must cover all eight modes rather than collapse onto a subset.
- **Flows**: `CNF` (FFJORD, free-form MLP velocity, exact $O(d)$ augmented-Jacobian trace) with `frequency = 4`; `OTFlow` (velocity $= -\nabla\Phi$ with a closed-form Hessian trace) with `hidden = 64`, `layer = 3`, `rank = min(10, d+1)`. Both integrate with the same fixed-step RK4 (`nt = 12`) and $(64, 64)$-width ODE nets, identity-initialised.
- **Training**: one packed `train_reverse_KL_F` call per cell — plain reverse KL with no rejuvenation (`MC_ITERS = 0`), `N_VALID = 40000` fixed source pool, `N_BATCH = 512` per Adam step, `STEPS = 1000`, `LR = 2e-3`, `checkpoint = True` (the CNF exact-trace path). The final ESS is the proposal importance-sampling ESS on a held-out $N = 20000$ source set.

### Results

<div align="center">

| flow   | $d=4$  | $d=8$  | $d=16$ | $d=32$ | $d=64$ | $d=128$ |
| :----: | :----: | :----: | :----: | :----: | :----: | :-----: |
| CNF    | 0.9705 | 0.9579 | 0.9311 | 0.8824 | 0.7412 | 0.4293  |
| OTFlow | 0.9713 | 0.9595 | 0.9456 | 0.9145 | 0.8588 | 0.6166  |

</div>

The tidy numbers are also written to [`CNF_vs_OTFlow.csv`](CNF_vs_OTFlow.csv) (`flow, dimension, ess, train_seconds`) for downstream plotting.

### Reading the result

Pushed out to $d = 128$, the two flows start nearly tied at low dimension ($\approx 0.97$ at $d = 4$) and then separate: OTFlow stays consistently above the CNF, and while both lose ESS as the dimension climbs, the CNF falls faster — from $0.97$ down to $0.43$ at $d = 128$, versus OTFlow's $0.97 \to 0.62$. The gap widens monotonically with $d$ (already $0.74$ vs $0.86$ at $d = 64$), so OTFlow's closed-form-trace / potential-gradient parameterisation is markedly more dimension-robust than the CNF's free-form velocity — the mode-seeking CNF's importance weights spike sooner as the ambient dimension grows. The high-dimensional cells converge slowly and use the full $1000$-step budget; per-cell wall time is tens of seconds and grows with $d$, the CNF's $O(d)$ exact trace closing most of the cost gap to OTFlow by $d = 128$. As with the other examples these are single-seed numbers that shift by a few times $10^{-2}$ between launches, so the reproducible finding is the widening OTFlow-over-CNF ordering, not the third decimal.

## flow_scaling_law — forward vs inverse map latency

Forward versus inverse map latency of an NSF as the dimension grows, run by [`flow_scaling_law.py`](flow_scaling_law.py). jflows has no `torch.compile`, so this measures the pure jitted cost of the two fused maps, each returned with its $\log|\det J|$.

### Setup

- **Flow**: NSF on $[-3, 3]^d$, 12 bins, 4 autoregressive transforms, random-initialised (a nontrivial bijection, so the inverse does real work), with the conditioner width swept over $(64, 64)$, $(128, 128)$, $(256, 256)$.
- **Maps**: `forward_map` $=$ `flow.call_and_ladj(x)` and `inverse_map` $=$ `flow.inv_and_ladj(y)`, each `eqx.filter_jit`-compiled once per (width, $d$) cell.
- **Timing**: fixed `BATCH = 2000`, `WARMUP = 20` untimed calls to absorb the XLA compile, `TIMED = 50` timed calls, a single `block_until_ready` after the batch; latency is the mean wall time per call. Dimension swept over $d \in \{4, 8, 16, 32, 64, 128\}$.

### Results

Forward map, mean ms per call:

<div align="center">

| width   |  $d=4$ |  $d=8$ | $d=16$ | $d=32$ | $d=64$ | $d=128$ |
| :-----: | :----: | :----: | :----: | :----: | :----: | :-----: |
| 64x64   | 0.153  | 0.157  | 0.179  | 0.221  | 0.272  |  0.420  |
| 128x128 | 0.166  | 0.171  | 0.199  | 0.231  | 0.317  |  0.451  |
| 256x256 | 0.176  | 0.192  | 0.207  | 0.260  | 0.365  |  0.552  |

</div>

Inverse map, mean ms per call:

<div align="center">

| width   | $d=4$ | $d=8$ | $d=16$ | $d=32$ | $d=64$ | $d=128$ |
| :-----: | :---: | :---: | :----: | :----: | :----: | :-----: |
| 64x64   | 0.432 | 0.720 | 1.585  | 4.016  | 11.815 | 40.124  |
| 128x128 | 0.471 | 0.846 | 1.865  | 4.585  | 13.338 | 46.438  |
| 256x256 | 0.554 | 1.071 | 2.205  | 5.907  | 17.529 | 59.332  |

</div>

The full grid, with the inv/fwd ratio, is written to [`flow_scaling_law.csv`](flow_scaling_law.csv) (`hidden_features, dimension, forward_ms, inverse_ms, inv_over_fwd`).

### Reading the result

The forward map is a single parallel pass and stays essentially flat in dimension — at $(64, 64)$ it moves only $0.153 \to 0.420$ ms from $d = 4$ to $d = 128$ ($\sim 3\times$ over a $32\times$ dimension increase). The inverse is autoregressive: a MAF-style flow inverts one coordinate at a time, so it runs $d$ sequential conditioner passes and climbs steeply — $0.432 \to 40.1$ ms at $(64, 64)$, roughly $90\times$, and steepening as $d$ rises. The resulting inv/fwd penalty opens from $\sim 3\times$ at $d = 4$ to $\sim 100\times$ at $d = 128$ across all three widths. Widening the conditioner ($64 \to 256$) raises both maps but far less than dimension does — at $d = 128$ the inverse grows only $40 \to 59$ ms and the forward $0.42 \to 0.55$ ms — so dimension, through the sequential autoregressive inversion, is the dominant cost, not MLP width. This is the intrinsic forward/inverse asymmetry of autoregressive spline flows, and it is exactly why the losses in the other examples are differentiated in each flow's native forward direction and never invert during training. The absolute milliseconds are GPU-specific; the scaling — a near-flat forward and a steeply growing inverse, dominated by dimension — is the reproducible finding.

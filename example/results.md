# Example results

Numerical results of the runnable examples in this folder.

## 2D_single — reverse KL vs forward KL

Single-stage flow training on a 2D three-mode target, comparing the reverse KL and forward KL objectives under the 100% energy-driven workflow of [`2D_single.py`](2D_single.py).

### Setup

- **Source** $\mu_0 \propto e^{-U_0}$: isotropic Gaussian, $U_0(x) = \lVert x\rVert^2 / (2\sigma^2)$, $\sigma = 2$.
- **Target** $\mu_1 \propto e^{-U_1}$: three-mode Gaussian mixture placed like the Julia three-dot sign — equal weights, means $(0, 2.4)$, $(-2.2, -1.4)$, $(2.2, -1.4)$, per-coordinate variance $0.3$. The modes are separated by $\sim 8$ standard deviations, so a mode-seeking objective can lose mass while a mass-covering one should not.
- **Flow**: NSF on $[-5, 5]^2$, 16 bins, 4 autoregressive transforms, $(64, 64)$ conditioners, identity-initialised (`zeros()`).
- **Training**: one packed call per method — `N_VALID = 40000` fixed source set, `N_BATCH = 2000` per Adam step, `STEPS = 200`, `LR = 1e-3`, Langevin rejuvenation `MC_STEP = 1e-3`, `MC_ITERS = 100`, single-rung AIS (`LADDER = 1`).

Both trainers regenerate their batch inside every Adam step. The reverse KL flow acts as $F$ (source → target, `type='F'`): each step draws an `N_BATCH` subset of the fixed set and freshens it with Langevin steps at the source. The forward KL flow acts as $G$ (target → source, `type='G'`): each step manufactures its target batch by single-rung AIS through the *current* flow — pushforward, self-normalized reweighting, multinomial resampling, Langevin rejuvenation at the target. Neither objective touches the inverse map during training, and the final ESS is computed on the full fixed set through the flow importance weights.

### Results

| objective | final ESS ($N = 40000$) | batch ESS along training |
| --------- | :---: | :---: |
| reverse KL | 0.9324 | 0.19 → 0.83 → 0.93 |
| forward KL | 0.9500 | 0.74 → 0.97 → 0.96 |

<p align="center"><img src="2D_single.png" alt="2D single-stage training" width="1000px"></p>

The left panel shows the per-step batch-ESS histories returned by the trainers (x-axis exactly $[0, 200]$); the middle and right panels show the `N_VALID` pushforward samples of each method (blue: $y = F(x)$; red: $y = G^{-1}(x)$) over the target energy background, with 5000 source samples in gray for context.

### Reading the result

Both flows populate all three modes, and the AIS-fed forward KL ends with the higher ESS: its data pipeline supplies (approximately) target-distributed samples covering every mode from the first step — its batch ESS starts already at $\sim 0.75$ — while the reverse KL starts near $0.19$ and pays its mode-seeking tax as the barriers grow. On a harder target — farther modes, or modes the initial proposal never reaches — the reverse objective can collapse entirely, and the forward objective inherits whatever the AIS chain covers; that regime (and the regularization that addresses it) is the subject of the X-regularization line of work these conventions come from.

Two practical notes. First, wall clock: each packed 200-step training compiles in a few seconds and executes in 1.5–2.5 s on an RTX 5070 Ti; the whole script (both trainings, evaluation, figure) runs in about 12 s. Second, reproducibility: the trainers are deterministic within a process (fixed internal seed — two identical calls agree bit for bit), while across separate launches XLA kernel autotuning can introduce last-ulp differences that 200 training steps amplify, so the final ESS varies by a few times $10^{-2}$ between runs (reverse KL $\approx 0.92$–$0.93$, forward KL $\approx 0.95$).

## 4D_boltzmann — annealed BG with the adaptive ladder

The 4D two-charge target of the zflows reference test, sampled by [`4D_boltzmann.py`](4D_boltzmann.py) with BOTH annealed generators — `boltzmann_reverse_KL` and `boltzmann_forward_KL` — along the bridge ladder $U_t = (1-t)\,U_0 + t\,U_1$, with the coefficient $t$ selected adaptively instead of the original fixed schedule $c_k = k/12$.

### Setup

- **Source** $\mu_0$: standard Gaussian on $\mathbb R^4$.
- **Target**: two particles $x = (x_1, x_2)$, $x_i \in \mathbb R^2$, on a soft annulus with regularized 3D Coulomb repulsion,
  $U_1(x) = a\,[(\lVert x_1\rVert^2 - r_0^2)^2 + (\lVert x_2\rVert^2 - r_0^2)^2] + q^2 / \sqrt{\lVert x_1 - x_2\rVert^2 + \varepsilon^2}$
  with $r_0 = 2$, $a = 1$, $q^2 = 4$, $\varepsilon = 10^{-3}$ (identical to the original).
- **Flow**: NSF on $[-3, 3]^4$, 8 bins, 6 autoregressive transforms, $(64, 64)$ conditioners, identity-initialised.
- **Boltzmann generators**: the stage flows are connected step by step — stage $k$ selects $t_k$ through the SMC gate (`tau_smc`, `LADDER = 1` rung), trains the warm-started flow as the incremental map $\mu_{t_{k-1}} \to \mu_{t_k}$ on the advancing particle set, accepts on the incremental importance-sampling ESS (`tau_ess`), and advances the set by reweight → resample → MALA at $U_{t_k}$ (`mc_adjust = True`; the Metropolis gate keeps the near-singular Coulomb tail out of the particle set, as in the zflows reference). The reverse KL stages train on Langevin-freshened batches of the set; the forward KL stages train on target batches manufactured per Adam step by AIS through the current flow (SMC gate and AIS share `LADDER`). Parameters: `N_VALID = 120000`, `N_BATCH = 2000`, `STEPS = 500`, `LR = 1e-4`, MALA `1e-3 × 100`; ladder `t_safe = 0.2`, `shrink_factor = 0.7`, `enlarge_factor = 1.5`, `tau_smc = 0.2`, `tau_ess = 0.6`.

### Results

Both ladders reach $t = 1$ in four stages (~8-10 s each on the full 120000-particle set; the stage trainer, weight evaluation, and advance each compile once and are reused across all stages). The rejection machinery earns its keep in the reverse run: its safe start $t = 0.2$ trains but misses the acceptance bar and shrinks to $0.14$, while the AIS-fed forward run accepts the full safe start and climbs faster — every other stage passes on its first attempt:

| reverse KL, stage $k$ | 1 | 2 | 3 | 4 |
| --------- | :---: | :---: | :---: | :---: |
| $t_k$     | 0.14 | 0.35 | 0.665 | 1.0 |
| ESS       | 0.657 | 0.904 | 0.957 | 0.984 |

| forward KL, stage $k$ | 1 | 2 | 3 | 4 |
| --------- | :---: | :---: | :---: | :---: |
| $t_k$     | 0.20 | 0.50 | 0.95 | 1.0 |
| ESS       | 0.764 | 0.898 | 0.963 | 0.987 |

<p align="center"><img src="4D_boltzmann.png" alt="4D Boltzmann generator" width="1000px"></p>

Each row (top: reverse KL; bottom: forward KL) shows the adaptive ladder ($t_k$ and the per-stage incremental ESS), the particle-1 marginal at $t = 1$ concentrated on the annulus $\lVert x_1 \rVert = r_0$ (dashed circle), and the relative angle $\Delta\theta$ between the two particles, peaked at $\pm\pi$ with vanishing density at $0$ — the antipodal Coulomb minimum.

### Reading the result

The ESS trace follows the reference behaviour of the original fixed-ladder run — a lower leading rung, then a high plateau — while the ESS-gated selection compresses the schedule: the leading increment is small (`t_safe`), the accepted step then grows by the enlarge factor, and the final extrapolation snaps to $t = 1$, so four stages cover what the fixed schedule spent twelve rungs on. The generator's sample output is the advanced particle set; the per-stage incremental flows and their acceptance ESS are returned in the stage records.

## 3D_periodic — the NCSF on the torus

The jflows rewrite of the zflows periodic reference test, run by [`3D_periodic.py`](3D_periodic.py): its purpose is to show that the **NCSF actually works** — the circular-spline flow trains, inverts, and reweights correctly on a genuinely periodic domain.

### Setup

- **Source** $\mu_0$: uniform on the periodic box $[-\pi, \pi]^3$.
- **Target** $\mu_1 \propto e^{-U_1}$: the von Mises ridge mixture on the 3-torus,
  $U_1(x) = -\log[\, e^{\kappa\cos(x_1 - x_2)} + e^{\kappa\cos(x_2 - x_3)} + e^{\kappa\cos(x_3 - x_1)} \,]$, $\kappa = 4$ — three pairwise ridges that wrap around the torus.
- **Flow**: NCSF on $[-\pi, \pi]^3$, 8 bins, 4 autoregressive transforms, $(128, 128)$ conditioners, identity-initialised.
- **Training**: one packed call per objective — `N_VALID = 40000` fixed source set, `N_BATCH = 2000` per Adam step, `STEPS = 200`, `LR = 1e-3` — followed by the reweighting pipeline: importance weights → ESS → multinomial resampling → MALA rejuvenation at the target (`1e-3 × 100`).

### Results

| objective | final ESS ($N = 40000$) |
| --------- | :---: |
| reverse KL | 0.8335 |
| forward KL | 0.9074 |

<p align="center"><img src="3D_periodic.png" alt="3D periodic NCSF" width="1000px"></p>

Both panels show the resampled and rejuvenated particle sets (left: reverse KL; right: forward KL) concentrating on the wrap-around ridge tubes of the target — the structure a non-periodic flow cannot represent without seam artifacts. The healthy ESS of both objectives on this domain is the point: the NCSF's circular splines carry the periodic geometry end to end, matching the behaviour of the zflows original at the same $\kappa$, architecture, and training budget.

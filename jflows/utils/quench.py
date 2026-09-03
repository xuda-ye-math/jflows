"""Quench and temper for jflows — the wide-coverage mode-discovery measure.

Builds hat_pi, the wide-coverage measure of the X-regularized training
loss (a mixture weight of `forward_X_G`), in three moves on a batch:
melt scatters the particles across the landscape with Gaussian noise,
quench drives every particle to a mode center of the target by batched
L-BFGS (global optimization by multi-start), and temper spreads the
particles around each center with a short Langevin run (MALA by
default). Mode discovery comes from the optimizer, so no chain has to
cross an energy barrier.

An optional partial importance resampling precedes the melt: with
`coeff_qt > 0` the particles are resampled by the tempered weights
exp(coeff_qt * (source - target)) and rejuvenated under the tempered
potential (1 - coeff_qt) * source + coeff_qt * target, so the cloud that
is melted already leans towards the target.

Built on the other utils modules: `lbfgs` (optimization) for the quench,
`langevin` (rejuvenation) for the temper and the rejuvenation after the
resampling, and `resample` (metrics).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..potential import Potential, linear_combination
from .metrics import linear_weights_from_log, resample
from .optimization import lbfgs
from .rejuvenation import langevin


__all__ = [
    "qt",
    "quench_and_temper",
]


def quench_and_temper(
    key: Array,
    samples: Array,
    target: Potential,
    melt: float,
    opt_dt: float = 1.0,
    opt_steps: int = 100,
    mc_dt: float = 1e-3,
    mc_steps: int = 100,
    mc_adjust: bool = True,
    chunks: int = 1,
    *,
    source: Potential | None = None,
    coeff_qt: float = 0.0,
) -> Array:
    """
    Quench and temper targeting exp(-U(x)): melt -> quench -> temper, with
    an optional partial importance resampling first.

        if coeff_qt > 0:                    # partial importance resampling
            w  <- exp(coeff_qt * (source(x) - target(x)))
            x  <- resample(x, w)
            x  <- langevin(x, (1 - coeff_qt) * source + coeff_qt * target)
        x <- samples + melt * noise         # melt:   scatter across R^d
        x <- lbfgs(x, target, armijo=True)  # quench: drive each particle to
                                            #         a mode center of target
        x <- langevin(x, target)            # temper: spread particles
                                            #         around each mode

    The output approximates the wide-coverage measure hat_pi: every mode
    whose basin the melted cloud touches receives particles, weighted by
    basin volume rather than by mode energy. Pass the result (or a
    resampled subset) as the hat_pi half of the mixture batch of
    `forward_X_G`.

    Input:
        key:       PRNG key (split into one melt key and one temper key)
        samples:   Array [N, d]   initial particles (e.g. source samples)
        target:    Potential      target potential U
        melt:      float          melt scale — std of the Gaussian scatter;
                                  large enough that the cloud reaches every
                                  basin of interest
        opt_dt:    float          L-BFGS initial trial step size (armijo
                                  backtracking line search)
        opt_steps: int            L-BFGS iterations of the quench
        mc_dt:     float          Langevin step size of the temper
        mc_steps:  int            Langevin steps of the temper
        mc_adjust: bool           if True, temper with MALA (unbiased);
                                  if False, ULA
        chunks:    int            split the batch into this many chunks in
                                  the quench, the temper, and the weights
                                  (bounds peak memory; statistically
                                  equivalent)
        source:    Potential      the potential the input particles follow;
                                  required when coeff_qt > 0
        coeff_qt:  float in [0, 1] exponent of the partial importance
                                  weights exp(coeff_qt * (source - target))
                                  applied before the melt; 0 (default) skips
                                  the resampling, 1 is the full importance
                                  resampling from the source to the target.
                                  The resampled particles are rejuvenated
                                  for `mc_steps` Langevin steps under the
                                  tempered potential
                                  (1 - coeff_qt) * source + coeff_qt * target
    Output:
        samples: Array [N, d]   the tempered particles ~ hat_pi
    """
    key_melt, key_mc = jax.random.split(key)
    x = samples
    if coeff_qt > 0:
        if source is None:
            raise ValueError("coeff_qt > 0 requires the source potential")
        key_resample, key_rejuvenate = jax.random.split(jax.random.fold_in(key, 1))
        log_w = jnp.concatenate([
            coeff_qt * (source(part) - target(part))
            for part in jnp.array_split(x, chunks, axis=0)
        ])
        x = resample(key_resample, x, linear_weights_from_log(log_w))
        x = langevin(
            key_rejuvenate, x,
            linear_combination([target, source], [coeff_qt, 1.0 - coeff_qt]),
            dt=mc_dt, steps=mc_steps, adjust=mc_adjust, chunks=chunks,
        )
    x = x + melt * jax.random.normal(key_melt, x.shape, dtype=x.dtype)
    x = lbfgs(
        x, target, alpha=opt_dt, steps=opt_steps, armijo=True, chunks=chunks
    )
    return langevin(
        key_mc, x, target, dt=mc_dt, steps=mc_steps,
        adjust=mc_adjust, chunks=chunks,
    )


# alias: the short name of the quench and temper construction
qt = quench_and_temper

"""Quench and temper for jflows — the wide-coverage mode-discovery measure.

Builds hat_mu, the wide-coverage measure of the X-regularized training
loss (a mixture weight of `forward_X_G`), in three moves on a batch:
melt scatters the particles across the landscape with Gaussian noise,
quench drives every particle to a mode center of the target by batched
L-BFGS (global optimization by multi-start), and temper spreads the
particles around each center with a short Langevin run (MALA by
default). Mode discovery comes from the optimizer, so no chain has to
cross an energy barrier.

Built on the other utils modules: `lbfgs` (optimization) for the quench
and `langevin` (rejuvenation) for the temper.
"""

from __future__ import annotations

import jax
from jax import Array

from ..potential import Potential
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
    opt_step: float = 1.0,
    opt_iters: int = 100,
    mc_step: float = 1e-3,
    mc_iters: int = 100,
    mc_adjust: bool = True,
    chunk: int = 1,
) -> Array:
    """
    Quench and temper targeting exp(-U(x)): melt -> quench -> temper.

        x <- samples + melt * noise         # melt:   scatter across R^d
        x <- lbfgs(x, target, armijo=True)  # quench: drive each particle to
                                            #         a mode center of target
        x <- langevin(x, target)            # temper: spread particles
                                            #         around each mode

    The output approximates the wide-coverage measure hat_mu: every mode
    whose basin the melted cloud touches receives particles, weighted by
    basin volume rather than by mode energy. Pass the result (or a
    resampled subset) as the hat_mu half of the mixture batch of
    `forward_X_G`.

    Input:
        key:       PRNG key (split into one melt key and one temper key)
        samples:   Array [N, d]   initial particles (e.g. source samples)
        target:    Potential      target potential U
        melt:      float          melt scale — std of the Gaussian scatter;
                                  large enough that the cloud reaches every
                                  basin of interest
        opt_step:  float          L-BFGS initial trial alpha (armijo
                                  backtracking line search)
        opt_iters: int            L-BFGS iterations of the quench
        mc_step:   float          Langevin step size of the temper
        mc_iters:  int            Langevin steps of the temper
        mc_adjust: bool           if True, temper with MALA (unbiased);
                                  if False, ULA
        chunk:     int            split the batch into this many chunks in
                                  the quench and the temper (bounds peak
                                  memory; statistically equivalent)
    Output:
        samples: Array [N, d]   the tempered particles ~ hat_mu
    """
    key_melt, key_mc = jax.random.split(key)
    x = samples + melt * jax.random.normal(key_melt, samples.shape, dtype=samples.dtype)
    x = lbfgs(x, target, step=opt_step, iters=opt_iters, armijo=True, chunk=chunk)
    return langevin(key_mc, x, target, step=mc_step, iters=mc_iters,
                    adjust=mc_adjust, chunk=chunk)


# alias: the short name of the quench-and-temper construction
qt = quench_and_temper

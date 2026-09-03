"""Smoke test of `boltzmann_FAB_G` and `boltzmann_FABX_G` — run from the repo
root as `python -m smoke.test_fab_boltzmann`.

The 4D two-charge target and the adaptive-staging parameters of
`example/4D_boltzmann.py` are imported (500 Adam steps per stage attempt,
batch 2000, a population of 120,000, `tau_valid` 0.6). Both generators must
complete their stage schedule at t = 1, the flow of the last stage must push
the source population to the target with an independently recomputed ESS at
or above `tau_valid`, the trained map must be selected on the last stage,
the records must carry the stage flows, and the FABX run must use its
quench-and-temper pool (pool size 0, melt 1.0, 50 quench steps, `coeff_qt`
0.5).
"""

import importlib.util
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from jflows.boltzmann import boltzmann_FAB_G, boltzmann_FABX_G
from jflows.flow import NSF
from jflows.train import Monitor
from jflows.utils import compute_ESS, importance_weights

spec = importlib.util.spec_from_file_location(
    "example_4d", "/data/projects/jflows/example/4D_boltzmann.py"
)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    print(f"{name}: OK")


def main():
    x_valid = E.u0.samples(jax.random.key(2), E.VALID_SIZE)

    def new_flow(key):
        return NSF(key, a=[-E.NSF_LIM] * 4, b=[E.NSF_LIM] * 4, bins=E.BINS,
                   transforms=E.TRANSFORMS, hidden_features=E.HIDDEN_FEATURES).zeros()

    common = dict(
        batch_size=E.BATCH_SIZE, steps_total=E.STEPS_TOTAL, lr=E.LR, ladder=E.LADDER,
        mc_dt=E.MC_DT, mc_steps_1=E.MC_STEPS_1, mc_steps_2=E.MC_STEPS_2,
        bg_param=E.BG_PARAM,
    )
    started = time.time()
    y_fab, fab = boltzmann_FAB_G(
        x_valid, E.u0, E.u1, new_flow(jax.random.key(1)),
        monitor=Monitor(100, "[FAB] "), **common,
    )
    print(f"FAB: {time.time() - started:.1f}s, t = {[round(s['t'], 4) for s in fab]}, "
          f"ESS = {[round(s['valid_selected_ess'], 3) for s in fab]}, "
          f"selected = {[s['selected'] for s in fab]}")
    check("FAB schedule reaches t = 1", fab and fab[-1]["t"] == 1.0)
    ess_fab = compute_ESS(importance_weights(x_valid, E.u0, E.u1, fab[-1]["flow"], "G"))
    check(f"FAB final flow: recomputed target ESS {float(ess_fab):.3f} >= tau_valid",
          ess_fab >= E.BG_PARAM["tau_valid"])
    check("FAB trains the last stage", fab[-1]["selected"] == "trained")
    check("FAB records carry the selected flow", all("flow" in s and "continuation_flow" in s for s in fab))
    check("FAB population finite", bool(jax.numpy.all(jax.numpy.isfinite(y_fab))))

    started = time.time()
    y_fabx, fabx = boltzmann_FABX_G(
        x_valid, E.u0, E.u1, new_flow(jax.random.key(1)), 0,
        melt=1.0, opt_dt=0.5, opt_steps=50, coeff_theta=1.0, coeff_alpha=0.5,
        coeff_qt=0.5, chunks=4, monitor=Monitor(100, "[FABX] "), **common,
    )
    print(f"FABX: {time.time() - started:.1f}s, t = {[round(s['t'], 4) for s in fabx]}, "
          f"ESS = {[round(s['valid_selected_ess'], 3) for s in fabx]}, "
          f"selected = {[s['selected'] for s in fabx]}")
    check("FABX schedule reaches t = 1", fabx and fabx[-1]["t"] == 1.0)
    ess_fabx = compute_ESS(importance_weights(x_valid, E.u0, E.u1, fabx[-1]["flow"], "G"))
    check(f"FABX final flow: recomputed target ESS {float(ess_fabx):.3f} >= tau_valid",
          ess_fabx >= E.BG_PARAM["tau_valid"])
    check("FABX trains the last stage", fabx[-1]["selected"] == "trained")
    check("FABX records carry the selected flow", all("flow" in s and "continuation_flow" in s for s in fabx))
    check("FABX population finite", bool(jax.numpy.all(jax.numpy.isfinite(y_fabx))))


if __name__ == "__main__":
    main()

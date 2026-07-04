"""Phase 2 fixture dump (torch side) — run with ~/.envs/torch/bin/python.

Evaluates zflows core.transforms (MonotonicRQS with scalar AND per-coord
bound, CircularShift, MonotonicAffine) on fixed inputs and saves
{params, inputs, outputs, grads} to smoke_tests/parity_phase2.npz.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/projects/zflows")
from zflows.core.transforms import (  # noqa: E402
    CircularShiftTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase2_dump.log")
OUT = os.path.join(HERE, "parity_phase2.npz")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    log(f"START parity_dump_phase2 | torch {torch.__version__} | out={OUT}")
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    data: dict[str, np.ndarray] = {}

    N, d, K = 5, 3, 8
    widths = torch.randn(N, d, K)
    heights = torch.randn(N, d, K)
    derivs = torch.randn(N, d, K - 1)
    bound_vec = torch.tensor([1.5, 3.0, 0.8])
    x = (torch.rand(N, d) * 2 - 1) * bound_vec * 0.95
    data["rqs_widths"], data["rqs_heights"], data["rqs_derivs"] = (
        widths.numpy(), heights.numpy(), derivs.numpy(),
    )
    data["rqs_bound_vec"] = bound_vec.numpy()
    data["rqs_x"] = x.detach().numpy()

    # ── RQS with per-coord bound: y, ladj, inverse, grad of ladj w.r.t x ──
    log("MonotonicRQS (per-coord bound) fixture")
    xg = x.clone().requires_grad_()
    t = MonotonicRQSTransform(widths, heights, derivs, bound=bound_vec, slope=1e-3)
    y, ladj = t.call_and_ladj(xg)
    ladj.sum().backward()
    data["rqs_vec_y"] = y.detach().numpy()
    data["rqs_vec_ladj"] = ladj.detach().numpy()
    data["rqs_vec_gradx"] = xg.grad.numpy()
    data["rqs_vec_xback"] = t.inv(y.detach()).numpy()

    # ── RQS with scalar bound ──
    log("MonotonicRQS (scalar bound) fixture")
    xs = (torch.rand(N, d) * 2 - 1) * 2.0 * 0.95
    t2 = MonotonicRQSTransform(widths, heights, derivs, bound=2.0, slope=1e-3)
    y2, ladj2 = t2.call_and_ladj(xs)
    data["rqs_sc_x"] = xs.numpy()
    data["rqs_sc_y"] = y2.detach().numpy()
    data["rqs_sc_ladj"] = ladj2.detach().numpy()

    # ── CircularShift (per-coord bound) ──
    log("CircularShift fixture")
    c = CircularShiftTransform(bound=bound_vec)
    xc = torch.randn(N, d) * 3
    data["cs_x"] = xc.numpy()
    data["cs_y"] = c(xc).numpy()

    # ── MonotonicAffine ──
    log("MonotonicAffine fixture")
    shift, scale = torch.randn(N, d), torch.randn(N, d) * 3
    m = MonotonicAffineTransform(shift, scale, slope=1e-3)
    xm = torch.randn(N, d)
    ym, ladjm = m.call_and_ladj(xm)
    data["aff_shift"], data["aff_scale"] = shift.numpy(), scale.numpy()
    data["aff_x"] = xm.numpy()
    data["aff_y"] = ym.detach().numpy()
    data["aff_ladj"] = ladjm.detach().numpy()

    np.savez(OUT, **data)
    log(f"DONE — wrote {OUT} ({len(data)} arrays)")


if __name__ == "__main__":
    main()

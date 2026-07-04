"""Phase 3 fixture dump (torch side) — run with ~/.envs/torch/bin/python.

Builds every zflows public flow class (randmask=False for structural
determinism), records its parameters and its forward / ladj / inverse
outputs on fixed inputs, and saves everything to
smoke_tests/parity_phase3.npz for the jax-side weight-transplant check.
Float64, CPU only.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/projects/zflows")
from zflows.core.nn import Linear as ZLinear  # noqa: E402
from zflows.flow import CNF, NCSF, NSF, OTFlow, RealNVP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase3_dump.log")
OUT = os.path.join(HERE, "parity_phase3.npz")

data: dict[str, np.ndarray] = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def dump_masked_mlp(prefix: str, mmlp) -> None:
    linears = [m for m in mmlp if hasattr(m, "mask")]
    data[f"{prefix}_n"] = np.array(len(linears))
    for j, m in enumerate(linears):
        data[f"{prefix}_w{j}"] = m.weight.detach().numpy()
        data[f"{prefix}_b{j}"] = m.bias.detach().numpy()
        data[f"{prefix}_mask{j}"] = m.mask.numpy()


def dump_mlp(prefix: str, mlp) -> None:
    linears = [m for m in mlp if isinstance(m, ZLinear)]
    data[f"{prefix}_n"] = np.array(len(linears))
    for j, m in enumerate(linears):
        data[f"{prefix}_w{j}"] = m.weight.detach().numpy()
        data[f"{prefix}_b{j}"] = m.bias.detach().numpy()


def dump_io(prefix: str, flow, x: torch.Tensor) -> None:
    F = flow.t()
    y, ladj = F.call_and_ladj(x)
    xb = F.inv(y)
    data[f"{prefix}_x"] = x.numpy()
    data[f"{prefix}_y"] = y.detach().numpy()
    data[f"{prefix}_ladj"] = ladj.detach().numpy()
    data[f"{prefix}_xb"] = xb.detach().numpy()


def main() -> None:
    log(f"START parity_dump_phase3 | torch {torch.__version__} | out={OUT}")
    torch.manual_seed(2)
    torch.set_default_dtype(torch.float64)

    # ── NSF ──
    log("NSF fixture")
    a = torch.tensor([-2.0, -1.5, -3.0])
    b = torch.tensor([2.0, 1.5, 3.0])
    nsf = NSF(a, b, bins=8, slope=1e-3, transforms=3, randmask=False,
              hidden_features=(32, 16))
    for i, maf in enumerate(nsf._maf):
        dump_masked_mlp(f"nsf_maf{i}", maf.hyper)
    x = (torch.rand(16, 3) * 2 - 1) * nsf.halfwidth * 0.95 + nsf.center
    dump_io("nsf", nsf, x)

    # ── NCSF ──
    log("NCSF fixture")
    pi3 = torch.full((3,), math.pi)
    ncsf = NCSF(-pi3, pi3, bins=8, slope=1e-3, transforms=3, randmask=False,
                hidden_features=(32, 16))
    for i, maf in enumerate(ncsf._maf):
        dump_masked_mlp(f"ncsf_maf{i}", maf.hyper)
    xc = (torch.rand(16, 3) * 2 - 1) * math.pi * 0.95
    dump_io("ncsf", ncsf, xc)

    # ── RealNVP (lu and rotation mixing) ──
    for kind in ("lu", "rotation"):
        log(f"RealNVP ({kind}) fixture")
        nvp = RealNVP(dimension=4, transforms=3, randmask=False, mixing=kind,
                      hidden_features=(32, 16))
        n_gct = 0
        n_mix = 0
        for layer in nvp._layers:
            if hasattr(layer, "hyper"):
                dump_mlp(f"nvp_{kind}_gct{n_gct}", layer.hyper)
                data[f"nvp_{kind}_gctmask{n_gct}"] = layer.mask.numpy()
                n_gct += 1
            else:
                with torch.no_grad():
                    layer.weight.add_(0.1 * torch.randn_like(layer.weight))
                data[f"nvp_{kind}_mix{n_mix}"] = layer.weight.detach().numpy()
                n_mix += 1
        data[f"nvp_{kind}_ngct"] = np.array(n_gct)
        data[f"nvp_{kind}_nmix"] = np.array(n_mix)
        dump_io(f"nvp_{kind}", nvp, torch.randn(16, 4))

    # ── CNF ──
    log("CNF fixture")
    cnf = CNF(dimension=3, frequency=3, nt=8, exact=True, hidden_features=(16, 16))
    dump_mlp("cnf_ode", cnf._ffj.ode)
    dump_io("cnf", cnf, torch.randn(8, 3))

    # ── OTFlow ──
    log("OTFlow fixture")
    otf = OTFlow(dimension=3, hidden=16, layer=3, rank=4, nt=8)
    phi = otf._ot.phi
    data["otf_A"] = phi.A.detach().numpy()
    data["otf_cw"] = phi.c.weight.detach().numpy()
    data["otf_ww"] = phi.w.weight.detach().numpy()
    data["otf_nres"] = np.array(len(phi.N.layers))
    for j, m in enumerate(phi.N.layers):
        data[f"otf_res_w{j}"] = m.weight.detach().numpy()
        data[f"otf_res_b{j}"] = m.bias.detach().numpy()
    xo = torch.randn(8, 3)
    dump_io("otf", otf, xo)
    F = otf.t().transforms[0]
    y, ladj, cost, hjb = F.call_full(xo)
    data["otf_cost"] = cost.detach().numpy()
    data["otf_hjb"] = hjb.detach().numpy()

    np.savez(OUT, **data)
    log(f"DONE — wrote {OUT} ({len(data)} arrays)")


if __name__ == "__main__":
    main()

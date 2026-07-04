"""Phase 5 fixture dump (torch side) — run with ~/.envs/torch/bin/python.

Builds a zflows NSF + OTFlow (float64, randmask=False), a Gaussian target,
and records the scalar KL / OT losses (beta = 1) together with all
parameters and inputs, for the jax-side batch-mean parity check.
Note: zflows potentials cast their buffers to float32 — the buffers
actually used are dumped, so both sides compute with identical values.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/projects/zflows")
from zflows.flow import NSF, OTFlow  # noqa: E402
from zflows.loss import OT_loss, forward_KL_F, forward_KL_G, reverse_KL_F, reverse_KL_G  # noqa: E402
from zflows.potential import Gaussian  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase5_dump.log")
OUT = os.path.join(HERE, "parity_phase5.npz")

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


def main() -> None:
    log(f"START parity_dump_phase5 | torch {torch.__version__} | out={OUT}")
    torch.manual_seed(5)
    torch.set_default_dtype(torch.float64)

    a = torch.tensor([-2.0, -1.5, -3.0])
    b = torch.tensor([2.0, 1.5, 3.0])
    nsf = NSF(a, b, bins=8, slope=1e-3, transforms=3, randmask=False,
              hidden_features=(32, 16))
    for i, maf in enumerate(nsf._maf):
        dump_masked_mlp(f"nsf_maf{i}", maf.hyper)

    target = Gaussian(torch.zeros(3), torch.ones(3) * 0.8)
    data["tgt_mean"] = target.mean.numpy()   # float32 buffers as actually used
    data["tgt_var"] = target.variance.numpy()

    x = (torch.rand(16, 3) * 2 - 1) * nsf.halfwidth * 0.95 + nsf.center
    y = (torch.rand(16, 3) * 2 - 1) * nsf.halfwidth * 0.95 + nsf.center
    data["x"], data["y"] = x.numpy(), y.numpy()

    log("KL losses on NSF")
    F = nsf.t()
    data["rkl_F"] = reverse_KL_F(x, target, F).detach().numpy()
    data["rkl_G"] = reverse_KL_G(x, target, F).detach().numpy()
    data["fkl_F"] = forward_KL_F(y, target, F).detach().numpy()
    data["fkl_G"] = forward_KL_G(y, target, F).detach().numpy()

    log("OT_loss on OTFlow")
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
    data["xo"] = xo.numpy()
    data["ot_loss"] = OT_loss(xo, target, otf, alpha_C=0.7, alpha_R=0.4).detach().numpy()

    np.savez(OUT, **data)
    log(f"DONE — wrote {OUT} ({len(data)} arrays)")


if __name__ == "__main__":
    main()

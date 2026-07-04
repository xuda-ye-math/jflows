"""Phase 6a fixture dump (torch side) — run with ~/.envs/torch/bin/python.

Evaluates zflows compute_ESS / compute_ESS_log on fixed weights, and the
importance_weights_{F,G} (+log) family on a fixed NSF (beta = 1), saving
everything to parity_check/parity_phase6.npz for the jax-side check.
(resample is stochastic — validated statistically in smoke/test_metrics.py.)
Extended as the remaining Phase 6 modules land. Float64, CPU only.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/projects/zflows")
from zflows.flow import NSF  # noqa: E402
from zflows.potential import Gaussian  # noqa: E402
from zflows.utils import (  # noqa: E402
    compute_ESS,
    compute_ESS_log,
    importance_weights_F,
    importance_weights_log_F,
    importance_weights_log_G,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase6_dump.log")
OUT = os.path.join(HERE, "parity_phase6.npz")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    log(f"START parity_dump_phase6 | torch {torch.__version__} | out={OUT}")
    torch.manual_seed(6)
    torch.set_default_dtype(torch.float64)
    data: dict[str, np.ndarray] = {}

    log("ESS fixtures")
    w = torch.rand(512) + 1e-3
    logw = torch.randn(512) * 3.0
    data["w"] = w.numpy()
    data["logw"] = logw.numpy()
    data["ess"] = compute_ESS(w).numpy()
    data["ess_log"] = compute_ESS_log(logw).numpy()
    data["ess_from_logw"] = compute_ESS(logw.exp()).numpy()

    log("importance-weight fixtures (NSF, beta = 1)")
    a = torch.tensor([-2.0, -1.5, -3.0])
    b = torch.tensor([2.0, 1.5, 3.0])
    nsf = NSF(a, b, bins=8, slope=1e-3, transforms=3, randmask=False,
              hidden_features=(32, 16))
    for i, maf in enumerate(nsf._maf):
        linears = [m for m in maf.hyper if hasattr(m, "mask")]
        data[f"nsf_maf{i}_n"] = np.array(len(linears))
        for j, m in enumerate(linears):
            data[f"nsf_maf{i}_w{j}"] = m.weight.detach().numpy()
            data[f"nsf_maf{i}_b{j}"] = m.bias.detach().numpy()

    source = Gaussian(torch.zeros(3), torch.ones(3))
    target = Gaussian(torch.tensor([0.5, -0.5, 0.0]), torch.ones(3) * 0.7)
    data["src_mean"], data["src_var"] = source.mean.numpy(), source.variance.numpy()
    data["tgt_mean"], data["tgt_var"] = target.mean.numpy(), target.variance.numpy()

    x = (torch.rand(64, 3) * 2 - 1) * nsf.halfwidth * 0.95 + nsf.center
    data["x"] = x.numpy()
    F = nsf.t()
    data["iw_log_F"] = importance_weights_log_F(x, source, target, F).detach().numpy()
    data["iw_log_G"] = importance_weights_log_G(x, source, target, F).detach().numpy()
    data["iw_F"] = importance_weights_F(x, source, target, F).detach().numpy()

    np.savez(OUT, **data)
    log(f"DONE — wrote {OUT} ({len(data)} arrays)")


if __name__ == "__main__":
    main()

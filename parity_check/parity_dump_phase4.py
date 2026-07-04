"""Phase 4a fixture dump (torch side) — run with ~/.envs/torch/bin/python.

Evaluates zflows potentials (Uniform, Gaussian, Gaussian_Mixture) on fixed
inputs and saves {params, inputs, energies, grads} to
smoke_tests/parity_phase4.npz for the jax-side check. Float32 end-to-end
(the zflows potential constructors cast their buffers to float32), CPU
only. (The jflows twins carry the Nlog_ prefix; the math is identical.)
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/projects/zflows")
from zflows.potential import (  # noqa: E402
    Gaussian,
    Gaussian_Mixture,
    Linear_Combination,
    Uniform,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase4_dump.log")
OUT = os.path.join(HERE, "parity_phase4.npz")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    log(f"START parity_dump_phase4 | torch {torch.__version__} | out={OUT}")
    torch.manual_seed(4)
    data: dict[str, np.ndarray] = {}

    N, d, K = 32, 3, 4
    x = torch.randn(N, d)
    data["x"] = x.numpy()

    # ── Uniform: constant energy (grad trivially zero, checked jax-side) ──
    log("Uniform fixture")
    a = torch.tensor([-2.0, -1.0, -3.0])
    b = torch.tensor([2.0, 1.5, 3.0])
    uni = Uniform(a, b)
    data["uni_a"], data["uni_b"] = a.numpy(), b.numpy()
    data["uni_U"] = uni(x).numpy()

    # ── Gaussian: energy + grad ──
    log("Gaussian fixture")
    mean = torch.randn(d)
    variance = torch.rand(d) + 0.5
    gau = Gaussian(mean, variance)
    xg = x.clone().requires_grad_()
    U = gau(xg)
    U.sum().backward()
    data["gau_mean"], data["gau_var"] = mean.numpy(), variance.numpy()
    data["gau_U"] = U.detach().numpy()
    data["gau_grad"] = xg.grad.numpy()

    # ── Gaussian_Mixture: energy + grad (unnormalized weights on purpose) ──
    log("Gaussian_Mixture fixture")
    weights = torch.rand(K) + 0.2
    means = torch.randn(K, d) * 2
    variances = torch.rand(K, d) + 0.3
    gmm = Gaussian_Mixture(weights, means, variances)
    xm = x.clone().requires_grad_()
    Um = gmm(xm)
    Um.sum().backward()
    data["gmm_w"], data["gmm_mean"], data["gmm_var"] = (
        weights.numpy(), means.numpy(), variances.numpy(),
    )
    data["gmm_U"] = Um.detach().numpy()
    data["gmm_grad"] = xm.grad.numpy()

    # ── Linear_Combination: energy + grad, plus a nested combination ──
    log("Linear_Combination fixture")
    lc = Linear_Combination([gau, gmm], [0.7, -0.3])
    xl = x.clone().requires_grad_()
    Ul = lc(xl)
    Ul.sum().backward()
    data["lc_U"] = Ul.detach().numpy()
    data["lc_grad"] = xl.grad.numpy()
    nested = Linear_Combination([Linear_Combination([gau], [2.0]), gmm], [0.5, 1.0])
    data["lcn_U"] = nested(x).detach().numpy()

    np.savez(OUT, **data)
    log(f"DONE — wrote {OUT} ({len(data)} arrays)")


if __name__ == "__main__":
    main()

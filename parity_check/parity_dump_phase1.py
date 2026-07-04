"""Phase 1 fixture dump (torch side) — run with ~/.envs/torch/bin/python.

Evaluates zflows core.numerics + core.nn on fixed inputs and saves
{inputs, params, outputs, grads} to smoke_tests/parity_phase1.npz for the
jax-side check (parity_check_phase1.py). CPU only.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("/mnt/projects/zflows"))
from zflows.core.nn import MLP, Linear, MaskedMLP  # noqa: E402
from zflows.core.numerics import bisection, gauss_legendre, rk4_fixed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "parity_phase1_dump.log")
OUT = os.path.join(HERE, "parity_phase1.npz")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main() -> None:
    log(f"START parity_dump_phase1 | torch {torch.__version__} | out={OUT}")
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    data: dict[str, np.ndarray] = {}

    # ── gauss_legendre: value + grads w.r.t. a, b (f cubic → rule exact) ──
    log("gauss_legendre fixture")
    a = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
    b = torch.tensor([2.0, 3.0, 0.5], requires_grad=True)
    f = lambda t: t**3 - 2.0 * t + 0.5
    area = gauss_legendre(f, a, b, n=3)
    area.sum().backward()
    data["gl_a"], data["gl_b"] = a.detach().numpy(), b.detach().numpy()
    data["gl_area"] = area.detach().numpy()
    data["gl_grad_a"], data["gl_grad_b"] = a.grad.numpy(), b.grad.numpy()

    # ── bisection: root of x^3 + alpha x = y, grads w.r.t. y and alpha ──
    log("bisection fixture")
    alpha = torch.tensor(0.7, requires_grad=True)
    y = torch.tensor([0.5, 2.0, 9.0], requires_grad=True)
    froot = lambda x: x**3 + alpha * x
    x_root = bisection(froot, y, 0.0, 3.0, n=60, phi=(alpha,))
    x_root.sum().backward()
    data["bi_y"] = y.detach().numpy()
    data["bi_alpha"] = alpha.detach().numpy()
    data["bi_root"] = x_root.detach().numpy()
    data["bi_grad_y"] = y.grad.numpy()
    data["bi_grad_alpha"] = alpha.grad.numpy()

    # ── rk4_fixed: dx/dt = tanh(x) cos(t), value + grad w.r.t. x0 ──
    log("rk4_fixed fixture")
    x0 = torch.randn(4, 3, requires_grad=True)
    fode = lambda t, x: torch.tanh(x) * torch.cos(t)
    xT = rk4_fixed(fode, x0, 0.0, 1.0, nt=16)
    xT.sum().backward()
    data["rk4_x0"] = x0.detach().numpy()
    data["rk4_xT"] = xT.detach().numpy()
    data["rk4_grad_x0"] = x0.grad.numpy()

    # ── Linear (stacked) ──
    log("Linear (stacked) fixture")
    lin = Linear(5, 4, bias=True, stack=3)
    xs = torch.randn(7, 3, 5)
    data["lin_w"] = lin.weight.detach().numpy()
    data["lin_b"] = lin.bias.detach().numpy()
    data["lin_x"] = xs.numpy()
    data["lin_y"] = lin(xs).detach().numpy()

    # ── MLP: value + grad w.r.t. input ──
    log("MLP fixture")
    mlp = MLP(3, 2, hidden_features=(16, 8), activation=torch.nn.SiLU)
    linears = [m for m in mlp if isinstance(m, Linear)]
    for i, m in enumerate(linears):
        data[f"mlp_w{i}"] = m.weight.detach().numpy()
        data[f"mlp_b{i}"] = m.bias.detach().numpy()
    data["mlp_n"] = np.array(len(linears))
    xm = torch.randn(9, 3, requires_grad=True)
    ym = mlp(xm)
    ym.sum().backward()
    data["mlp_x"] = xm.detach().numpy()
    data["mlp_y"] = ym.detach().numpy()
    data["mlp_grad_x"] = xm.grad.numpy()

    # ── MaskedMLP: masks (unique-row ordering parity) + value ──
    log("MaskedMLP fixture")
    order = torch.tensor([2, 0, 3, 1])
    adjacency = order[:, None] > order
    adjacency = torch.repeat_interleave(adjacency, repeats=3, dim=0)  # total=3
    mmlp = MaskedMLP(adjacency, hidden_features=(32, 16), activation=torch.nn.SiLU)
    mlinears = [m for m in mmlp if hasattr(m, "mask")]
    for i, m in enumerate(mlinears):
        data[f"mm_w{i}"] = m.weight.detach().numpy()
        data[f"mm_b{i}"] = m.bias.detach().numpy()
        data[f"mm_mask{i}"] = m.mask.numpy()
    data["mm_n"] = np.array(len(mlinears))
    data["mm_adjacency"] = adjacency.numpy()
    xa = torch.randn(6, 4)
    data["mm_x"] = xa.numpy()
    data["mm_y"] = mmlp(xa).detach().numpy()

    np.savez(OUT, **data)
    log(f"DONE — wrote {OUT} ({len(data)} arrays)")


if __name__ == "__main__":
    main()

"""jflows — JAX normalizing flows for unconditional energy-based sampling.

Built on JAX and equinox.

Public surface:

    jflows.flow      : Flow, NSF, NCSF, CNF, OTFlow, RealNVP, ComposedTransform
    jflows.potential : Potential, potential_from, Nlog_Uniform, Nlog_Gaussian,
                       Nlog_Gaussian_Mixture, linear_combination (+ the operator
                       algebra c*U, U+V, U-V, -U, U/c, sum([...]))
    jflows.loss      : reverse_KL_F, forward_KL_G, forward_KLX_G,
                       forward_X_G, pairwise_variation
    jflows.train     : Monitor and all train_* stage drivers
    jflows.boltzmann : adaptive-staging and fixed-schedule boltzmann_* generators
    jflows.artifacts : save and load medium-level training outputs
    jflows.backend   : selected/available JAX backends and accelerator model
    jflows.utils     : metrics / optimization / rejuvenation / anneal / quench

Internals (`jflows.core.*`) adapt zuko's clean flow and transform design.
The public computation API uses explicit PRNG keys and takes `Flow` objects
directly for losses, importance weights, SMC, and training.
"""

from importlib.metadata import entry_points
from importlib.util import find_spec
import os
from shutil import which
import subprocess

import equinox
import jax

from . import artifacts, boltzmann, flow, loss, potential, train, utils
from .version import __version__


def backend():
    """Print the available JAX backends and selected accelerator."""
    plugins = tuple(item.name.lower() for item in entry_points(group="jax_plugins"))
    available = ["CPU"]
    details = ["- CPU — cpu"]

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
    if cuda_visible and any("cuda" in name for name in plugins) and which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        models = tuple(
            dict.fromkeys(line for line in result.stdout.splitlines() if line)
        )
        if result.returncode == 0 and models:
            available.append("CUDA")
            details.append(f"- CUDA — {', '.join(models)}")

    rocm_visible = os.environ.get("ROCR_VISIBLE_DEVICES") not in ("", "-1")
    if rocm_visible and any("rocm" in name for name in plugins) and which("rocm-smi"):
        available.append("ROCm")
        details.append("- ROCm")

    if find_spec("libtpu"):
        available.append("TPU")
        details.append("- TPU")

    configured = os.environ.get("JAX_PLATFORMS", "").split(",")[0].lower()
    selected = {"cpu": "CPU", "cuda": "CUDA", "rocm": "ROCm", "tpu": "TPU"}.get(
        configured
    )
    if selected not in available:
        selected = next(
            name for name in ("CUDA", "ROCm", "TPU", "CPU") if name in available
        )

    print(f"JAX {jax.__version__}")
    print(f"Equinox {equinox.__version__}")
    print(f"Selected backend: {selected}")
    print(f"Available backends: {', '.join(available)}")
    for detail in details:
        print(detail)


__all__ = [
    "__version__",
    "artifacts",
    "backend",
    "boltzmann",
    "flow",
    "loss",
    "potential",
    "train",
    "utils",
]

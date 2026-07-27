# jflows on CUDA 13, complete install (latest packages, default Python)
#
# Everything both packages declare: the JAX CUDA 13 GPU backend, OpenMM with
# its CUDA 13 platform, jflows[examples] and jflows-md[bundles] -- the last of
# which adds ParmEd and AmberTools for bundle construction.
#
# build the container
#   docker build -t xudayemath/jflows:full docker/
# push to docker hub
#   docker push xudayemath/jflows:full
# interactive shell
#   docker run --rm -it --gpus all xudayemath/jflows:full
# mount current dir and work in it
#   docker run --rm -it --gpus all -v "$PWD:/workspace" xudayemath/jflows:full
# python REPL
#   docker run --rm -it --gpus all xudayemath/jflows:full python
#
# "base", not "devel": every CUDA component this stack compiles with arrives as
# a pip wheel.  jax[cuda13] brings ptxas and nvlink in nvidia-cuda-nvcc and
# libdevice.10.bc in nvidia-nvvm, and OpenMM compiles its kernels through the
# NVRTC it bundles, so no system nvcc is ever consulted.  The image therefore
# needs only the driver, which the NVIDIA Container Toolkit injects at
# "--gpus all"; CUDA 13 requires host driver >= 580.
FROM nvidia/cuda:13.3.0-base-ubuntu26.04

# Non-interactive apt, unbuffered Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Python + toolchain + connection/auth tools (Ubuntu 26.04: python3.14) ---
# gcc and g++ are required: ParmEd publishes no wheel for this platform, so pip
# builds it from source and compiles src/_rdparm.cpp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev gcc g++ \
        ca-certificates curl wget git openssh-client gnupg micro && \
    rm -rf /var/lib/apt/lists/*

# --- Virtual env named "jflows", matching ~/.envs/jflows ---
ENV VIRTUAL_ENV=/opt/jflows
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m pip install --upgrade pip setuptools wheel

# --- JAX built against CUDA 13 (pulls the bundled nvidia-*-cu13 wheels) ---
RUN pip install "jax[cuda13]"

# --- Core scientific stack ---
RUN pip install numpy scipy scikit-learn pandas matplotlib h5py tqdm

# --- OpenMM with the CUDA 13 platform ---
# Plain "openmm" ships the Reference and CPU platforms only.  The cuda13 extra
# adds the OpenMM-CUDA-13 plugin, which is what the jflows_md OpenMM backend
# needs; jflows-md[openmm] on its own asks for bare "openmm" and would leave
# the container without a CUDA platform.
RUN pip install "openmm[cuda13]"

# --- jflows and jflows_md, every extra (from PyPI) ---
# examples adds matplotlib; bundles is a superset of the openmm extra and adds
# ParmEd and AmberTools for bundle_build.  jax, numpy, matplotlib and openmm
# are already satisfied above, so this adds equinox, ParmEd and AmberTools.
RUN pip install "jflows[examples]" "jflows-md[bundles]"

WORKDIR /workspace

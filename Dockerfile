# jflows on CUDA 13 (latest packages, default Python)
#
# build the container
#   docker build -t xudayemath/jflows:cu13 .
# push to docker hub
#   docker push xudayemath/jflows:cu13
# interactive shell
#   docker run --rm -it --gpus all xudayemath/jflows:cu13
# mount current dir and work in it
#   docker run --rm -it --gpus all -v "$PWD:/workspace" xudayemath/jflows:cu13
# python REPL
#   docker run --rm -it --gpus all xudayemath/jflows:cu13 python
#
# The CUDA runtime image is enough: the jax[cuda13] wheels bundle cuDNN,
# cuBLAS, and the other NVIDIA libraries as pip packages, so no nvcc or
# CUDA headers (the devel image) are needed. The host supplies the driver.
FROM nvidia/cuda:13.3.0-runtime-ubuntu26.04

# Non-interactive apt, unbuffered Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Python + connection/auth tools (Ubuntu 26.04 default: python3.14) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        ca-certificates curl wget git openssh-client gnupg micro && \
    rm -rf /var/lib/apt/lists/*

# --- Virtual env named "jflows", matching ~/.envs/jflows ---
ENV VIRTUAL_ENV=/opt/jflows
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m pip install --upgrade pip setuptools wheel

# --- JAX with the bundled CUDA 13 libraries (jaxlib + jax-cuda13-plugin/pjrt) ---
RUN pip install "jax[cuda13]"

# --- Core scientific stack ---
RUN pip install equinox numpy scipy matplotlib tqdm

# --- jflows (from PyPI) ---
# jax, equinox, and numpy are already satisfied above, so this only adds jflows itself.
RUN pip install jflows

# Do not let JAX grab the whole GPU on import; the drivers set this too.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace

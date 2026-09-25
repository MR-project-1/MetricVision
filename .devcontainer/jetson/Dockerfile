# ---------------------------------------------------------------------------
# Development/deployment image for the Jetson AGX Orin (JetPack 6, aarch64).
# The base tag MUST match the L4T release on the device: check with
#   cat /etc/nv_tegra_release      (e.g. "R36 (release), REVISION: 4.0" -> r36.4.0)
# JetPack 6 is Ubuntu 22.04 + Python 3.10, identical to the laptop image.
# ---------------------------------------------------------------------------

# L4T release, overridable at build time: --build-arg L4T_TAG=r36.3.0
ARG L4T_TAG=r36.4.0

# NVIDIA's JetPack image: CUDA, cuDNN and TensorRT for the Jetson GPU.
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

# Never block the build on interactive apt prompts.
ENV DEBIAN_FRONTEND=noninteractive

# Same system packages as the laptop image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential git curl ca-certificates \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv's official image is multi-arch, so this pulls the aarch64 binaries on the Jetson.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# Same uv settings as the laptop (see laptop/Dockerfile for the reasoning).
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=3.10 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Put the project environment first on PATH.
ENV PATH="/opt/venv/bin:${PATH}"

# Default working directory; the devcontainer mounts the repository here.
WORKDIR /workspaces/vobench

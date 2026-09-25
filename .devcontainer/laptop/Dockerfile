# ---------------------------------------------------------------------------
# Development image for an x86_64 laptop with an NVIDIA GPU.
# Ubuntu 22.04 + Python 3.10 matches ROS 2 Humble and the Jetson image, so code
# and lockfile behave the same on both machines.
# ---------------------------------------------------------------------------

# CUDA 12.4 runtime + cuDNN (needed later by the learned extractors); requires host driver >= 550.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Never block the build on interactive apt prompts.
ENV DEBIAN_FRONTEND=noninteractive

# System packages: Python 3.10 (Ubuntu 22.04 default), build tools for any source wheels,
# git for version control, and glib, which OpenCV links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        build-essential git curl ca-certificates \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv by copying its static binaries from the official image (version pinned to match uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# uv settings:
#  - use the system Python 3.10 (not a downloaded one) so ROS 2 rclpy can be added later,
#  - keep the environment outside the bind-mounted workspace so host and container never clash,
#  - copy instead of hard-linking because the cache lives on a separate volume,
#  - precompile bytecode for faster startup.
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON=3.10 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Put the project environment first on PATH so `python` and `vobench` work without `uv run`.
ENV PATH="/opt/venv/bin:${PATH}"

# Default working directory; the devcontainer mounts the repository here.
WORKDIR /workspaces/metricvision

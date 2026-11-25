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
        python3 python3-dev python3-venv python3-tk \
        build-essential git curl ca-certificates \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble (Ubuntu 22.04's distribution) for the live wheelchair node: ros-base (rclpy, launch, rosbag2,
# tf2, message_filters), rviz2 + rqt_image_view for the debug view, and colcon to build ros2_ws/.
# It uses the same system Python 3.10 as the uv environment, so the VO (in /opt/venv) can import rclpy.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common gnupg locales \
    && locale-gen en_US.UTF-8 \
    && add-apt-repository -y universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base ros-humble-rviz2 ros-humble-rqt-image-view ros-humble-sensor-msgs-py \
        ros-humble-rosbag2-storage-default-plugins python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Every interactive shell gets ROS 2 and the metricvision_ros package (built by postCreateCommand into
# /opt/ros2_ws, outside the bind-mounted checkout so it never clashes with a host build in ros2_ws/).
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "[ -f /opt/ros2_ws/install/setup.bash ] && source /opt/ros2_ws/install/setup.bash" >> /root/.bashrc \
    && echo 'echo "metricvision: ros2 launch metricvision_ros vo.launch.py [bag:=\$PWD/ros_bags/<run>]"' >> /root/.bashrc

# Install uv by copying its static binaries from the official image (version pinned to match uv.lock).
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# uv settings:
#  - use the system Python 3.10 (not a downloaded one) so it can import ROS 2 Humble's rclpy,
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

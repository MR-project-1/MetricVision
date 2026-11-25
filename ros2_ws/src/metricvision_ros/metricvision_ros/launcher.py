"""``ros2 run metricvision_ros vo_node``: start the VO node inside the project's uv environment.

colcon installs this entry point for the system Python, which has rclpy but not the VO's dependencies
(OpenCV 5, SciPy, torch, ...). Those live in the repository's uv environment, which is created from the
same Python 3.10 and so can import rclpy from the sourced ROS 2 install. This launcher therefore replaces
itself with ``<uv env>/bin/python -m metricvision.ros.vo_node <ROS arguments>``, run from the repository
root (so the relative paths in configs/ resolve).

Overrides: ``METRICVISION_ROOT`` (repository), ``METRICVISION_PYTHON`` (interpreter).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_repo(path: Path) -> bool:
    """The MetricVision repository root has pyproject.toml and src/metricvision."""
    return (path / "pyproject.toml").is_file() and (path / "src" / "metricvision").is_dir()


def repo_root() -> Path | None:
    """Locate the repository.

    Order: ``METRICVISION_ROOT``; upwards from this file (``--symlink-install`` resolves into the source tree,
    a plain install lives in ros2_ws/install inside the repository); upwards from the working directory;
    the dev container mount. Nothing is recorded at build time, because the host and the dev container
    see the same checkout under different paths.
    """
    if os.environ.get("METRICVISION_ROOT"):
        return Path(os.environ["METRICVISION_ROOT"])
    for start in (Path(__file__).resolve(), Path(__file__).absolute(), Path.cwd()):
        for path in (start, *start.parents):
            if _is_repo(path):
                return path
    fallback = Path("/workspaces/metricvision")
    return fallback if _is_repo(fallback) else None


def venv_python(root: Path) -> Path | None:
    """Find the uv environment's interpreter (host: <repo>/.venv, dev container: /opt/venv)."""
    candidates = [os.environ.get("METRICVISION_PYTHON", "")]
    if os.environ.get("UV_PROJECT_ENVIRONMENT"):
        candidates.append(os.path.join(os.environ["UV_PROJECT_ENVIRONMENT"], "bin", "python"))
    candidates += [str(root / ".venv" / "bin" / "python"), "/opt/venv/bin/python"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def main() -> None:
    """Exec the VO node."""
    _exec("metricvision.ros.vo_node")


def plot_main() -> None:
    """Exec the live comparison graph window."""
    _exec("metricvision.ros.vo_plot")


def _exec(module: str) -> None:
    """Replace this process with ``<uv env python> -m <module> <ROS arguments>`` in the repository root."""
    root = repo_root()
    python = venv_python(root) if root is not None else None
    if root is None or python is None:
        sys.exit(
            f"metricvision_ros: repository or uv environment not found (repo={root}, python={python}).\n"
            f"Run `uv sync` in the repository, or set METRICVISION_ROOT / METRICVISION_PYTHON."
        )
    os.chdir(root)
    # The wheelchair stack (lidar, DLIO, nav2, cameras) shares the CPU: BLAS/OpenMP default to one thread per core,
    # which only adds contention. Two threads each is enough for the VO's small matrices.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    os.execv(str(python), [str(python), "-m", module, *sys.argv[1:]])


if __name__ == "__main__":
    main()

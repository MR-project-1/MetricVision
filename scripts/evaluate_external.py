"""Evaluate a trajectory produced by an external system (libviso2, ORB-SLAM3, Basalt, ...) with our metrics.

The result is written like one of our runs (results/<dataset tag>/<name>/ with trajectory.tum/.kitti, frame_stats.csv
and summary.json), so make_report_assets.py and compare_runs.py treat it exactly like our own methods.

Usage:
    uv run python scripts/evaluate_external.py --config configs/kitti.yaml:$KITTI_ROOT \
        --trajectory orbslam3_kitti00.txt --format kitti --name ext_orbslam3
    uv run python scripts/evaluate_external.py --config configs/euroc.yaml:$EUROC_ROOT \
        --trajectory CameraTrajectory.txt --format tum --frame camera --name ext_orbslam3

Formats: ``kitti`` (12 values per line, one line per frame, timestamps taken from the dataset) or ``tum``
(t x y z qx qy qz qw). ``--frame camera`` (default): poses of the left camera (converted to the ground-truth body
frame like our own estimates); ``--frame body``: already in the ground-truth frame. Run the external system with loop
closing disabled so that it is compared as odometry.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # CLI.
from pathlib import Path  # Paths.

import numpy as np  # Arrays.
from scipy.spatial.transform import Rotation  # Quaternions.

from metricvision.backend.stats import FrameStats  # Per-frame record.
from metricvision.config import ExperimentConfig  # Dataset construction.
from metricvision.core.geometry import make_pose  # Poses.
from metricvision.core.types import Trajectory  # Trajectory type.
from metricvision.datasets import DATASETS  # Registry.
from metricvision.evaluation.trajectory_metrics import evaluate_trajectory  # Metrics.
from metricvision.evaluation.writers import write_json, write_kitti, write_rows_csv, write_tum  # Writers.
from metricvision.pipeline import RunSummary  # Summary format.


def _timestamps(dataset) -> np.ndarray:  # noqa: ANN001
    """Frame timestamps of a dataset (cheap attribute when available, else by loading frames)."""
    stamps = getattr(dataset, "_timestamps", None)
    if stamps is not None:
        return np.asarray(stamps, dtype=np.float64)
    return np.array([dataset.load_frame(i).timestamp for i in range(len(dataset))])


def read_trajectory(path: Path, fmt: str, timestamps: np.ndarray) -> Trajectory:
    """Read a KITTI- or TUM-format trajectory.

    Args:
        path: File.
        fmt: ``kitti`` or ``tum``.
        timestamps: Dataset timestamps (KITTI format).

    Returns:
        Trajectory of T_world_frame.

    Raises:
        ValueError: If a KITTI file has more poses than the dataset has frames.
    """
    rows = np.loadtxt(path, comments="#", ndmin=2)
    trajectory = Trajectory()
    if fmt == "kitti":
        if len(rows) > len(timestamps):
            raise ValueError(f"{path} has {len(rows)} poses but the dataset has {len(timestamps)} frames")
        for t, row in zip(timestamps, rows, strict=False):
            trajectory.append(float(t), np.vstack([row.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]]))
    else:
        for row in rows:
            trajectory.append(float(row[0]), make_pose(Rotation.from_quat(row[4:8]).as_matrix(), row[1:4]))
    return trajectory


def main() -> None:
    """Evaluate and write the run folder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="<config.yaml>:<dataset root>")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--format", choices=["kitti", "tum"], required=True)
    parser.add_argument("--frame", choices=["camera", "body"], default="camera")
    parser.add_argument("--name", required=True, help="run folder name, e.g. ext_orbslam3")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ms-per-frame", type=float, default=float("nan"), help="runtime reported by the external system, if known")
    args = parser.parse_args()
    config_path, _, root = args.config.partition(":")
    config = ExperimentConfig.from_yaml(config_path).with_overrides(dataset_root=root or None)
    dataset = DATASETS.create(config.dataset.name, **config.dataset.params)
    raw = read_trajectory(args.trajectory, args.format, _timestamps(dataset))
    estimate = raw.conjugated(dataset.body_from_camera) if args.frame == "camera" else raw
    reference = dataset.ground_truth()
    errors = evaluate_trajectory(estimate, reference) if reference is not None else None
    stats = [
        FrameStats(i, t, 0, 0, 0, float("nan"), True, False, float("nan"), float("nan"), float("nan"), args.ms_per_frame)
        for i, t in enumerate(estimate.timestamps)
    ]
    summary = RunSummary.from_stats(dataset.tag, args.name, args.name, "external", "external", stats, 0, errors)
    out = args.results / dataset.tag / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_tum(estimate, out / "trajectory.tum")
    write_kitti(estimate, out / "trajectory.kitti")
    write_rows_csv([s.to_dict() for s in stats], out / "frame_stats.csv")
    write_json(summary.to_dict(), out / "summary.json")
    ate = errors.ate_rmse_m if errors is not None else float("nan")
    print(f"{args.name} on {dataset.tag}: {len(estimate)} poses, ATE {ate:.3f} m -> {out}")


if __name__ == "__main__":
    main()

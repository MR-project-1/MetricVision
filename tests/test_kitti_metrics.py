"""Tests for the KITTI devkit segment metric."""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Pose construction.

from metricvision.core.geometry import make_pose  # Pose helper.
from metricvision.evaluation.kitti_metrics import kitti_segment_errors  # Unit under test.


def _straight_line(num_frames: int, step: float) -> list[np.ndarray]:
    """Poses moving forward along z by ``step`` metres per frame.

    Args:
        num_frames: Number of poses.
        step: Forward motion per frame.

    Returns:
        4x4 poses.
    """
    return [make_pose(np.eye(3), np.array([0.0, 0.0, i * step])) for i in range(num_frames)]  # Straight drive.


def test_perfect_estimate_has_zero_error() -> None:
    """An exact estimate must score zero on both devkit errors."""
    gt = _straight_line(1000, 1.0)  # 999 m drive.
    errors = kitti_segment_errors(gt, gt)  # Compare with itself.
    assert errors.num_segments > 0 and errors.translation_percent == 0.0 and errors.rotation_deg_per_100m == 0.0  # No error.


def test_scale_error_gives_matching_percentage() -> None:
    """Driving 2% too far must give a ~2% translation error."""
    gt = _straight_line(1000, 1.0)  # Ground truth.
    estimate = _straight_line(1000, 1.02)  # 2% scale drift.
    errors = kitti_segment_errors(estimate, gt)  # Evaluate.
    assert abs(errors.translation_percent - 2.0) < 0.05  # About 2%.

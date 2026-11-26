"""Stereo self-calibration recovers a moved camera from matches alone."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from metricvision.calibration.stereo_selfcal import estimate_extrinsic
from metricvision.core.geometry import make_pose
from metricvision.datasets.rosbag_stereo import RigCalibration


def _rig(right_from_left: np.ndarray) -> RigCalibration:
    k = np.array([[386.0, 0.0, 322.0], [0.0, 385.0, 243.0], [0.0, 0.0, 1.0]])
    return RigCalibration(k, np.zeros(5), k.copy(), np.zeros(5), 640, 480, right_from_left, np.eye(4))


def test_selfcal_recovers_a_tilted_camera() -> None:
    """Truth: the right camera pitched 2.5 deg and yawed 1 deg away from the stored calibration."""
    rng = np.random.default_rng(0)
    stored = make_pose(Rotation.from_euler("xyz", [1.5, 8.0, -6.7], degrees=True).as_matrix(), np.array([-0.415, 0.047, 0.029]))
    truth = make_pose(Rotation.from_euler("xyz", [2.5, 1.0, 0.0], degrees=True).as_matrix(), np.zeros(3)) @ stored
    points = np.column_stack([rng.uniform(-3, 3, 600), rng.uniform(-1.5, 1.5, 600), rng.uniform(1.5, 12, 600)])  # Left camera.
    right = points @ truth[:3, :3].T + truth[:3, 3]
    xl = points[:, :2] / points[:, 2:]
    xr = right[:, :2] / right[:, 2:]
    xr += rng.normal(0, 0.3 / 386.0, xr.shape)  # 0.3 px matching noise.
    xr[:40] += rng.uniform(-0.05, 0.05, (40, 2))  # Some outlier matches.

    result = estimate_extrinsic(_rig(stored), xl, xr)

    assert result is not None and result.reliable
    assert result.dy_before_px > 10  # The stored calibration misaligns the rows ...
    assert result.dy_after_px < 0.5  # ... the re-estimate aligns them.
    error = Rotation.from_matrix(result.calibration.right_from_left[:3, :3] @ truth[:3, :3].T).magnitude()
    assert np.degrees(error) < 0.05
    np.testing.assert_allclose(np.linalg.norm(result.calibration.right_from_left[:3, 3]), np.linalg.norm(stored[:3, 3]))  # Length kept.


def test_selfcal_needs_matches() -> None:
    assert estimate_extrinsic(_rig(np.eye(4)), np.zeros((5, 2)), np.zeros((5, 2))) is None

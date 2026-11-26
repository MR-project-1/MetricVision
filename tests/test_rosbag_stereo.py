"""Tests for the ROS 2 bag stereo loader helpers (no bag files needed)."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from metricvision.core.geometry import make_pose, orthonormalize_pose
from metricvision.datasets.rosbag_stereo import (
    RigCalibration,
    StereoRectifier,
    image_message_to_gray,
    interpolate_poses,
    pair_by_timestamp,
)


def test_pairing_is_mutual_and_respects_tolerance() -> None:
    left = np.arange(10) / 15.0  # 15 Hz.
    right = left + 0.023  # Constant 23 ms offset, like the wheelchair cameras.
    right = np.delete(right, 4)  # A dropped right frame.
    pairs = pair_by_timestamp(left, right, 0.035)
    assert len(pairs) == 9  # Left frame 4 has no partner within 35 ms.
    assert all(abs(left[i] - right[j]) <= 0.035 for i, j in pairs)
    assert len({j for _, j in pairs}) == len(pairs)  # Each right frame used once.
    assert pair_by_timestamp(left, right, 0.01) == []  # Offset larger than the tolerance.


def test_interpolation_matches_constant_twist() -> None:
    times = np.array([0.0, 1.0, 2.0])
    poses = [make_pose(Rotation.from_euler("z", 30 * k, degrees=True).as_matrix(), [k, 2 * k, 0]) for k in range(3)]
    out, inside = interpolate_poses(times, poses, np.array([-0.5, 0.5, 1.25, 3.0]))
    assert inside.tolist() == [False, True, True, False]  # No extrapolation.
    np.testing.assert_allclose(out[0][:3, 3], [0.5, 1.0, 0.0])
    yaw = Rotation.from_matrix(out[1][:3, :3]).as_euler("xyz", degrees=True)[2]
    assert abs(yaw - 37.5) < 1e-9


def test_image_conversion_handles_encodings_and_padding() -> None:
    rgb = np.random.default_rng(0).integers(0, 255, (4, 5, 3), dtype=np.uint8)
    padded = np.zeros((4, 5 * 3 + 1), dtype=np.uint8)  # One byte of row padding.
    padded[:, :15] = rgb.reshape(4, 15)
    msg = SimpleNamespace(height=4, width=5, step=16, encoding="rgb8", data=padded.tobytes())
    np.testing.assert_array_equal(image_message_to_gray(msg), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    bgr = SimpleNamespace(height=4, width=5, step=16, encoding="bgr8", data=padded.tobytes())
    np.testing.assert_array_equal(image_message_to_gray(bgr), cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))


def test_orthonormalize_pose_repairs_drift() -> None:
    step = make_pose(Rotation.from_euler("xyz", [0.01, 0.02, 0.03]).as_matrix(), [0.1, 0, 0])
    step[:3, :3] *= 1 + 1e-6  # Slightly non-orthonormal, as after round-off.
    pose = np.eye(4)
    for _ in range(2000):
        pose = orthonormalize_pose(pose @ step)
    rot = pose[:3, :3]
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(rot) - 1) < 1e-12


def test_rectifier_removes_vertical_disparity_of_a_rotated_rig() -> None:
    k = np.array([[385.0, 0, 320], [0, 385.0, 240], [0, 0, 1]])
    d = np.zeros(5)
    rot = Rotation.from_rotvec(np.radians([5.4, 8.6, -7.5])).as_matrix()  # Like the wheelchair rig.
    right_from_left = make_pose(rot, rot @ np.array([-0.42, 0.0, 0.0]))  # Right camera 0.42 m to the right.
    calib = RigCalibration(k, d, k, d, 640, 480, right_from_left, np.eye(4))
    rect = StereoRectifier(calib)
    assert abs(rect.camera.baseline - 0.42) < 1e-6
    # Project random points in front of both cameras, map them through the rectification.
    pts = np.random.default_rng(1).uniform([-1, -1, 3], [1, 1, 8], (50, 3))
    xl, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), k, d)
    xr, _ = cv2.projectPoints(pts, cv2.Rodrigues(rot)[0], right_from_left[:3, 3].reshape(3, 1), k, d)
    r1, r2, p1, p2, *_ = cv2.stereoRectify(k, d, k, d, (640, 480), rot, right_from_left[:3, 3].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    ul = cv2.undistortPoints(xl, k, d, R=r1, P=p1).reshape(-1, 2)
    ur = cv2.undistortPoints(xr, k, d, R=r2, P=p2).reshape(-1, 2)
    np.testing.assert_allclose(ul[:, 1], ur[:, 1], atol=1e-6)  # Same row.
    depth = rect.camera.fx * rect.camera.baseline / (ul[:, 0] - ur[:, 0])
    z_rect = (r1 @ pts.T).T[:, 2]
    np.testing.assert_allclose(depth, z_rect, rtol=1e-6)  # Metric depth from disparity.


def test_resolution_adapter_maps_realsense_1280x720_onto_640x480() -> None:
    """cam2's factory intrinsics: the 1280x720 mode is 0.6x scale + shift of the 640x480 one, and pixels land right."""
    import cv2

    from metricvision.datasets.rosbag_stereo import apply_resolution_adapter, resolution_adapter

    k_hd = np.array([[644.98926, 0.0, 641.94568], [0.0, 644.26904, 370.45084], [0.0, 0.0, 1.0]])
    k_vga = np.array([[387.20432, 0.0, 321.16736], [0.0, 386.77194, 246.27051], [0.0, 0.0, 1.0]])
    d = np.array([-0.05566, 0.06760, -0.00054, 0.00004, -0.02226])
    affine = resolution_adapter(k_hd, d, k_vga, d)
    assert affine is not None
    assert affine[0, 0] == pytest.approx(0.6, abs=1e-3) and affine[1, 2] == pytest.approx(24.0, abs=0.5)

    # The same 3D points seen in both modes: the map moves 1280x720 pixels onto the 640x480 pixels.
    points = np.array([[0.3, -0.2, 2.0], [-0.5, 0.1, 3.0], [0.0, 0.0, 1.0]])
    hd = (k_hd @ points.T).T
    vga = (k_vga @ points.T).T
    hd, vga = hd[:, :2] / hd[:, 2:], vga[:, :2] / vga[:, 2:]
    mapped = hd @ affine[:, :2].T + affine[:, 2]
    np.testing.assert_allclose(mapped, vga, atol=0.05)

    # And the image resampling agrees with the map (blob centroid).
    image = np.zeros((720, 1280), np.uint8)
    cv2.circle(image, (800, 300), 12, 255, -1)
    out = apply_resolution_adapter(image, affine, (640, 480))
    assert out.shape == (480, 640)
    weight = out.astype(np.float64)
    ys, xs = np.mgrid[: out.shape[0], : out.shape[1]]
    centroid = np.array([(weight * xs).sum(), (weight * ys).sum()]) / weight.sum()  # Intensity-weighted: unbiased.
    expected = affine[:, :2] @ np.array([800.0, 300.0]) + affine[:, 2]
    np.testing.assert_allclose(centroid, expected, atol=0.05)


def test_resolution_adapter_rejects_a_different_lens() -> None:
    """Different distortion (another camera / calibration) must not be silently remapped."""
    from metricvision.datasets.rosbag_stereo import resolution_adapter

    k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    assert resolution_adapter(k, np.array([0.1, 0.0, 0.0, 0.0, 0.0]), k, np.zeros(5)) is None
    squashed = np.array([[600.0, 0.0, 320.0], [0.0, 450.0, 240.0], [0.0, 0.0, 1.0]])
    assert resolution_adapter(squashed, np.zeros(5), k, np.zeros(5)) is None

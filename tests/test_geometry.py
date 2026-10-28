"""Unit tests for the SE(3) helpers."""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Numerical checks.
from scipy.spatial.transform import Rotation  # Random rotations.

from metricvision.core.geometry import (  # Functions under test.
    invert_pose,
    make_pose,
    pose_from_rvec_tvec,
    quaternion_xyzw_from_rotation,
    rotation_angle_deg,
    rotation_from_quaternion_wxyz,
    rvec_tvec_from_pose,
)


def _random_pose(seed: int) -> np.ndarray:
    """Create a reproducible random rigid transform.

    Args:
        seed: Random seed.

    Returns:
        4x4 rigid transform.
    """
    rng = np.random.default_rng(seed)  # Seeded generator.
    rotation = Rotation.random(random_state=seed).as_matrix()  # Uniform random rotation.
    return make_pose(rotation, rng.normal(size=3))  # Random translation.


def test_invert_pose_gives_identity() -> None:
    """T @ T^-1 must be the identity."""
    pose = _random_pose(0)  # Any rigid transform.
    np.testing.assert_allclose(pose @ invert_pose(pose), np.eye(4), atol=1e-12)  # Round trip.


def test_rvec_tvec_round_trip() -> None:
    """Pose -> (rvec, tvec) -> pose must be lossless."""
    pose = _random_pose(1)  # Any rigid transform.
    rvec, tvec = rvec_tvec_from_pose(pose)  # Convert to OpenCV form.
    np.testing.assert_allclose(pose_from_rvec_tvec(rvec, tvec), pose, atol=1e-10)  # Convert back.


def test_rotation_angle() -> None:
    """A 30 degree rotation about z must measure 30 degrees."""
    rotation = Rotation.from_euler("z", 30, degrees=True).as_matrix()  # Known angle.
    assert abs(rotation_angle_deg(rotation) - 30.0) < 1e-9  # Measured angle matches.


def test_quaternion_orders_agree() -> None:
    """xyzw output and wxyz input conventions must be consistent."""
    rotation = _random_pose(2)[:3, :3]  # Any rotation.
    qx, qy, qz, qw = quaternion_xyzw_from_rotation(rotation)  # Scalar-last.
    np.testing.assert_allclose(rotation_from_quaternion_wxyz(np.array([qw, qx, qy, qz])), rotation, atol=1e-12)  # Scalar-first back.

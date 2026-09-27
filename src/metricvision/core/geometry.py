"""Small SE(3) helpers built on NumPy, OpenCV and SciPy."""

from __future__ import annotations

import cv2  
import numpy as np
from scipy.spatial.transform import Rotation  # Robust quaternion conversions.

from metricvision.core.types import FloatArray  # Shared array alias.

def make_pose(rotation: FloatArray, translation: FloatArray) -> FloatArray:
    """Assemble a 4x4 homogeneous transform from a rotation and a translation.

    Args:
        rotation: 3x3 rotation matrix.
        translation: Translation vector with 3 elements.

    Returns:
        The 4x4 float64 transform [[R, t], [0, 1]].
    """
    pose = np.eye(4, dtype=np.float64)  # Start from identity so the bottom row is [0, 0, 0, 1].
    pose[:3, :3] = rotation  # Top-left 3x3 block is the rotation.
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)  # Last column is the translation.
    return pose  # Finished transform.

def invert_pose(pose: FloatArray) -> FloatArray:
    """Invert a rigid transform using its structure instead of a general matrix inverse.

    Args:
        pose: 4x4 rigid transform T_a_b.

    Returns:
        The inverse transform T_b_a.
    """
    rotation_t = pose[:3, :3].T  # Inverse of a rotation is its transpose.
    return make_pose(rotation_t, -rotation_t @ pose[:3, 3])  # New translation is -R^T t.

def pose_from_rvec_tvec(rvec: FloatArray, tvec: FloatArray) -> FloatArray:
    """Convert OpenCV's (rotation vector, translation vector) pair to a 4x4 transform.

    Args:
        rvec: Axis-angle rotation vector with 3 elements.
        tvec: Translation vector with 3 elements.

    Returns:
        The equivalent 4x4 transform.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))  # Axis-angle to matrix.
    return make_pose(np.asarray(rotation, dtype=np.float64), tvec)  # Combine with the translation.

def rvec_tvec_from_pose(pose: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert a 4x4 transform to OpenCV's (rotation vector, translation vector) pair.

    Args:
        pose: 4x4 rigid transform.

    Returns:
        A tuple (rvec, tvec), each a float64 array of shape (3, 1).
    """
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(pose[:3, :3]))  # Matrix to axis-angle.
    tvec = pose[:3, 3].reshape(3, 1).astype(np.float64)  # Translation as a column vector.
    return rvec.astype(np.float64), tvec  # OpenCV expects float64 column vectors.

def rotation_angle_deg(rotation: FloatArray) -> float:
    """Return the magnitude of a rotation in degrees.

    Args:
        rotation: 3x3 rotation matrix.

    Returns:
        Rotation angle in degrees, in [0, 180].
    """
    cos_angle = (np.trace(rotation) - 1.0) / 2.0  # From trace(R) = 1 + 2 cos(theta).
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))  # Clip guards against rounding outside [-1, 1].

def quaternion_xyzw_from_rotation(rotation: FloatArray) -> FloatArray:
    """Convert a rotation matrix to a unit quaternion in (x, y, z, w) order (TUM format).

    Args:
        rotation: 3x3 rotation matrix.

    Returns:
        Quaternion as a float64 array [qx, qy, qz, qw].
    """
    return Rotation.from_matrix(rotation).as_quat()  # SciPy's native order is scalar-last (x, y, z, w).

def rotation_from_quaternion_wxyz(quaternion_wxyz: FloatArray) -> FloatArray:
    """Convert a scalar-first quaternion (w, x, y, z), as used by EuRoC, to a rotation matrix.

    Args:
        quaternion_wxyz: Quaternion [qw, qx, qy, qz].

    Returns:
        3x3 rotation matrix.
    """
    w, x, y, z = (float(v) for v in quaternion_wxyz)  # Unpack scalar-first order.
    return Rotation.from_quat([x, y, z, w]).as_matrix()  # Reorder to SciPy's scalar-last convention.
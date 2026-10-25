"""Rectified-stereo triangulation with first-order (Jacobian) covariance propagation."""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Vectorised maths over all points.

from metricvision.core.types import BoolArray, FloatArray, RealArray, StereoCamera  # Shared types.


def triangulate_rectified(
    points_left: RealArray,
    disparity: RealArray,
    camera: StereoCamera,
    min_disparity: float = 0.5,
) -> tuple[FloatArray, BoolArray]:
    """Back-project left-image pixels with known disparity into the left camera frame.

    For a rectified pair: Z = fx * B / d, X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy.

    Args:
        points_left: (N, 2) pixel coordinates (u, v) in the left image.
        disparity: (N,) disparities d = u_left - u_right in pixels.
        camera: Rectified stereo camera.
        min_disparity: Smaller disparities are rejected (too far / unreliable).

    Returns:
        A tuple (points, valid): (N, 3) points (NaN where invalid) and the (N,) validity mask.
    """
    u = np.asarray(points_left[:, 0], dtype=np.float64)  # Columns.
    v = np.asarray(points_left[:, 1], dtype=np.float64)  # Rows.
    d = np.asarray(disparity, dtype=np.float64)  # Disparities.
    valid = np.isfinite(d) & (d > min_disparity)  # Positive, non-tiny disparity only.
    points = np.full((u.size, 3), np.nan, dtype=np.float64)  # Default: invalid.
    z = camera.fx * camera.baseline / d[valid]  # Depth from disparity.
    points[valid, 0] = (u[valid] - camera.cx) * z / camera.fx  # X.
    points[valid, 1] = (v[valid] - camera.cy) * z / camera.fy  # Y.
    points[valid, 2] = z  # Z.
    return points, valid  # Points plus mask.


def triangulation_covariance(
    points_left: RealArray,
    disparity: RealArray,
    camera: StereoCamera,
    pixel_sigma: float = 0.5,
) -> FloatArray:
    """Propagate pixel noise to a 3x3 covariance per triangulated point.

    The measurement is m = (u_l, v, u_r) with independent noise sigma on each
    coordinate, so Sigma_X = sigma^2 * J J^T with J = d(X, Y, Z) / d(u_l, v, u_r).
    Depth uncertainty grows quadratically with depth: sigma_Z ~ Z^2 / (fx B) * sqrt(2) sigma.

    Args:
        points_left: (N, 2) left pixel coordinates.
        disparity: (N,) disparities in pixels (must be positive where used).
        camera: Rectified stereo camera.
        pixel_sigma: Standard deviation of each pixel coordinate, in pixels.

    Returns:
        (N, 3, 3) covariance matrices in the left camera frame (metres^2).
    """
    u = np.asarray(points_left[:, 0], dtype=np.float64) - camera.cx  # Centred column.
    v = np.asarray(points_left[:, 1], dtype=np.float64) - camera.cy  # Centred row.
    d = np.asarray(disparity, dtype=np.float64)  # Disparity.
    b, fx, fy = camera.baseline, camera.fx, camera.fy  # Shorthands.
    jac = np.zeros((d.size, 3, 3), dtype=np.float64)  # Rows: X, Y, Z. Columns: u_l, v, u_r.
    jac[:, 0, 0] = b / d - u * b / d**2  # dX/du_l (u_l appears in both numerator and disparity).
    jac[:, 0, 2] = u * b / d**2  # dX/du_r.
    jac[:, 1, 0] = -v * fx * b / (fy * d**2)  # dY/du_l.
    jac[:, 1, 1] = fx * b / (fy * d)  # dY/dv.
    jac[:, 1, 2] = v * fx * b / (fy * d**2)  # dY/du_r.
    jac[:, 2, 0] = -fx * b / d**2  # dZ/du_l.
    jac[:, 2, 2] = fx * b / d**2  # dZ/du_r.
    return pixel_sigma**2 * jac @ jac.transpose(0, 2, 1)  # sigma^2 J J^T.

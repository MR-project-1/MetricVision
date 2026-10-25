"""Per-frame front-end quality metrics that do not depend on the back end."""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # Essential-matrix RANSAC.
import numpy as np  # Mask arithmetic.

from metricvision.core.types import Correspondences, FloatArray  # Shared types.


def essential_inlier_ratio(
    correspondences: Correspondences,
    camera_matrix: FloatArray,
    threshold_px: float = 1.0,
    confidence: float = 0.999,
) -> float:
    """Fraction of matches consistent with a single essential matrix (2D-2D RANSAC).

    This measures match quality from image geometry alone, independent of
    stereo depth and PnP, so it isolates the front end. Caveat: when the camera
    barely moves, the essential matrix is poorly constrained and the ratio is
    optimistic, so interpret it together with the trajectory metrics.

    Args:
        correspondences: Matches between two images taken by the same camera.
        camera_matrix: 3x3 intrinsic matrix.
        threshold_px: RANSAC inlier threshold (distance to the epipolar line, in pixels).
        confidence: RANSAC confidence level.

    Returns:
        Inlier ratio in [0, 1], or NaN if there are fewer than 5 matches (the minimal set).
    """
    if len(correspondences) < 5:  # The five-point algorithm needs at least five matches.
        return float("nan")  # Undefined rather than zero.
    _, mask = cv2.findEssentialMat(  # Robustly fit an essential matrix.
        correspondences.points_a.astype(np.float64),  # Points in image A.
        correspondences.points_b.astype(np.float64),  # Points in image B.
        camera_matrix,  # Shared intrinsics.
        method=cv2.RANSAC,  # Robust estimation.
        prob=confidence,  # Desired confidence.
        threshold=threshold_px,  # Inlier distance in pixels.
    )
    if mask is None:  # RANSAC failed to produce any model.
        return 0.0  # No match is geometrically consistent.
    return float(np.count_nonzero(mask)) / float(len(correspondences))  # Inliers divided by all matches.

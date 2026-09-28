from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import dataclass  # Config and result containers.

import cv2  # solvePnPRansac and Levenberg-Marquardt refinement.
import numpy as np  # Array handling.

from metricvision.core.geometry import pose_from_rvec_tvec, rvec_tvec_from_pose  # OpenCV pose conversions.
from metricvision.core.types import BoolArray, FloatArray  # Shared types.

@dataclass(frozen=True)
class PnPConfig:
    """Settings for robust PnP. Keep identical across extractors so only the front end varies.

    Attributes:
        reprojection_error_px: RANSAC inlier threshold on reprojection error, in pixels.
        iterations: Maximum RANSAC iterations.
        confidence: RANSAC confidence level.
        min_inliers: Fewer inliers than this counts as a tracking failure.
        refine: Whether to polish the pose on inliers with Levenberg-Marquardt.
    """

    reprojection_error_px: float = 2.0  # Inlier threshold.
    iterations: int = 200  # RANSAC budget.
    confidence: float = 0.999  # RANSAC confidence.
    min_inliers: int = 15  # Failure threshold.
    refine: bool = True  # Non-linear refinement on/off.
    
@dataclass(frozen=True)
class PoseEstimate:
    """Result of one PnP solve.

    Attributes:
        success: Whether a pose with enough inliers was found.
        camera_from_reference: T_cam_ref, mapping reference-frame points into the current camera.
        inlier_mask: Boolean mask over the input correspondences.
        num_inliers: Number of RANSAC inliers.
    """

    success: bool  # Outcome flag.
    camera_from_reference: FloatArray  # Estimated relative pose (identity on failure).
    inlier_mask: BoolArray  # Per-correspondence inlier flags.
    num_inliers: int  # Inlier count.

    @classmethod
    def failure(cls, num_points: int) -> PoseEstimate:
        """Create a failed estimate.

        Args:
            num_points: Number of input correspondences, to size the mask.

        Returns:
            An estimate with success=False and no inliers.
        """
        return cls(False, np.eye(4, dtype=np.float64), np.zeros(num_points, dtype=bool), 0)  # Neutral values.
    
class PnPPoseEstimator:
    """Estimates camera pose from 3D points (reference frame) and their 2D projections (current frame)."""
    
    def __init__(self, config: PnPConfig | None = None) -> None:
        """Store the configuration.

        Args:
            config: PnP settings; defaults are used when omitted.
        """
        self._config = config or PnPConfig()  # Fall back to defaults.
        
    def estimate(
        self,
        points_3d: FloatArray,
        points_2d: FloatArray,
        camera_matrix: FloatArray,
        initial_guess: FloatArray | None = None,
    ) -> PoseEstimate:
        pass
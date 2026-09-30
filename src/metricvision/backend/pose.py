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
        """Solve PnP robustly, then optionally refine on inliers.

        Args:
            points_3d: (N, 3) points in the reference camera frame.
            points_2d: (N, 2) observations of those points in the current image.
            camera_matrix: 3x3 intrinsic matrix (images are rectified, so no distortion).
            initial_guess: Optional T_cam_ref prior (e.g. from a constant-velocity model).

        Returns:
            The pose estimate with its inlier mask.
        """
        num_points = int(points_3d.shape[0])
        if num_points < max(self._config.min_inliers, 6):
            return PoseEstimate.failure(num_points)
        object_points = np.ascontiguousarray(points_3d, dtype=np.float64)  # OpenCV wants contiguous float64.
        image_points = np.ascontiguousarray(points_2d, dtype=np.float64)  # Same for 2D points.
        use_guess = initial_guess is not None
        rvec0, tvec0 = rvec_tvec_from_pose(initial_guess) if initial_guess is not None else (np.zeros((3, 1)), np.zeros((3, 1)))  # Prior or zeros.
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(  # Robust PnP.
            object_points,  # 3D points in the reference frame.
            image_points,  # 2D points in the current image.
            camera_matrix,  # Intrinsics.
            None,  # No distortion: images are rectified.
            rvec=rvec0.copy(),  # Initial rotation (used only if use_guess).
            tvec=tvec0.copy(),  # Initial translation (used only if use_guess).
            useExtrinsicGuess=use_guess,  # Seed the iterative solver with the prior.
            iterationsCount=self._config.iterations,  # RANSAC budget.
            reprojectionError=self._config.reprojection_error_px,  # Inlier threshold.
            confidence=self._config.confidence,  # RANSAC confidence.
            flags=cv2.SOLVEPNP_ITERATIVE,  # Levenberg-Marquardt based solver.
        )
        if not ok or inliers is None or len(inliers) < self._config.min_inliers: # Solver failed or support too weak.
            return PoseEstimate.failure(num_points)  # Report failure.
        inlier_index = inliers.reshape(-1)  # OpenCV returns an (K, 1) index array.
        if self._config.refine:  # Polish the RANSAC pose using all inliers.
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_index], image_points[inlier_index], camera_matrix, None, rvec, tvec
            )  # Minimise reprojection error.
        mask = np.z0eros(num_points, dtype=bool)  # Per-correspondence inlier flags.
        mask[inlier_index] = True  # Mark inliers.
        pose = pose_from_rvec_tvec(np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64))  # 4x4 T_cam_ref.
        return PoseEstimate(True, pose, mask, int(inlier_index.size))  # Successful estimate.
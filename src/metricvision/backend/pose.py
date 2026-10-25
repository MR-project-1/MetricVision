from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import dataclass  # Config and result containers.

import cv2  # solvePnPRansac and Levenberg-Marquardt refinement.
import numpy as np  # Array handling.

from metricvision.backend.lm import LMConfig, refine_pose_lm  # Rigid-rig Levenberg-Marquardt with analytic Jacobians.
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
        refine: Inlier refinement: ``"none"``, ``"opencv"`` (cv2.solvePnPRefineLM, left image only) or
            ``"lm"`` (our covariance-weighted rigid-rig LM, using the right image too when available).
            Booleans are accepted for backward compatibility (True = "opencv").
        lm_iterations: Maximum LM iterations for ``refine="lm"``.
        lm_pixel_sigma: Observation noise (pixels) assumed by the LM weighting.
        lm_huber_delta: Huber threshold of the LM, in standard deviations.
    """

    reprojection_error_px: float = 2.0  # Inlier threshold.
    iterations: int = 200  # RANSAC budget.
    confidence: float = 0.999  # RANSAC confidence.
    min_inliers: int = 15  # Failure threshold.
    refine: bool | str = "opencv"  # Refinement method.
    lm_iterations: int = 20  # LM budget.
    lm_pixel_sigma: float = 1.0  # LM pixel noise.
    lm_huber_delta: float = 2.0  # LM robust threshold.

    @property
    def refine_method(self) -> str:
        """Return the refinement method as a string.

        Returns:
            One of ``"none"``, ``"opencv"`` or ``"lm"``.

        Raises:
            ValueError: If the configured value is unknown.
        """
        if isinstance(self.refine, bool):  # Legacy on/off flag.
            return "opencv" if self.refine else "none"  # True meant OpenCV's LM.
        if self.refine not in ("none", "opencv", "lm"):  # Typo guard.
            raise ValueError(f"Unknown pnp.refine '{self.refine}'; use none, opencv or lm")  # Clear error.
        return self.refine  # Already a valid name.


@dataclass(frozen=True)
class PoseEstimate:
    """Result of one PnP solve.

    Attributes:
        success: Whether a pose with enough inliers was found.
        camera_from_reference: T_cam_ref, mapping reference-frame points into the current camera.
        inlier_mask: Boolean mask over the input correspondences.
        num_inliers: Number of RANSAC inliers.
        refined_inlier_mask: Inliers of the LM chi^2 test at the refined pose (only with ``refine="lm"``).
    """

    success: bool  # Outcome flag.
    camera_from_reference: FloatArray  # Estimated relative pose (identity on failure).
    inlier_mask: BoolArray  # Per-correspondence inlier flags.
    num_inliers: int  # Inlier count.
    refined_inlier_mask: BoolArray | None = None  # Post-LM inliers, if LM ran.

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
        points_2d_right: FloatArray | None = None,
        baseline: float = 0.0,
        covariances: FloatArray | None = None,
    ) -> PoseEstimate:
        """Solve PnP robustly, then optionally refine on inliers.

        Args:
            points_3d: (N, 3) points in the reference camera frame.
            points_2d: (N, 2) observations of those points in the current image.
            camera_matrix: 3x3 intrinsic matrix (images are rectified, so no distortion).
            initial_guess: Optional T_cam_ref prior (e.g. from a constant-velocity model).
            points_2d_right: Optional (N, 2) observations in the current right image (rigid-rig LM only).
            baseline: Stereo baseline in metres (needed with ``points_2d_right``).
            covariances: Optional (N, 3, 3) covariances of ``points_3d`` used to weight the LM.

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
        if not ok or inliers is None or len(inliers) < self._config.min_inliers:  # Solver failed or support too weak.
            return PoseEstimate.failure(num_points)  # Report failure.
        inlier_index = inliers.reshape(-1)  # OpenCV returns an (K, 1) index array.
        method = self._config.refine_method  # Which refinement to run.
        if method == "opencv":  # Polish the RANSAC pose using all inliers.
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_index], image_points[inlier_index], camera_matrix, None, rvec, tvec
            )  # Minimise reprojection error.
        mask = np.zeros(num_points, dtype=bool)  # Per-correspondence inlier flags.
        mask[inlier_index] = True  # Mark inliers.
        pose = pose_from_rvec_tvec(np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64))  # 4x4 T_cam_ref.
        refined_mask = None  # Only set by the LM path.
        if method == "lm":  # Covariance-weighted rigid-rig LM on the RANSAC inliers.
            lm_config = LMConfig(  # Settings from the PnP config.
                max_iterations=self._config.lm_iterations,
                pixel_sigma=self._config.lm_pixel_sigma,
                huber_delta=self._config.lm_huber_delta,
            )
            result = refine_pose_lm(  # Analytic-Jacobian LM.
                pose,
                object_points[inlier_index],
                image_points[inlier_index],
                camera_matrix,
                observations_right=None if points_2d_right is None else np.asarray(points_2d_right, dtype=np.float64)[inlier_index],
                baseline=baseline,
                covariances=None if covariances is None else covariances[inlier_index],
                config=lm_config,
            )
            pose = result.camera_from_reference  # Refined T_cam_ref.
            refined_mask = np.zeros(num_points, dtype=bool)  # Map LM inliers back to all correspondences.
            refined_mask[inlier_index[result.inlier_mask]] = True  # RANSAC inlier and chi^2 inlier.
        return PoseEstimate(True, pose, mask, int(inlier_index.size), refined_mask)  # Successful estimate.

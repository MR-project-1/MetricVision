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
<<<<<<< HEAD
        lm_pixel_sigma: Observation noise (pixels) assumed by the LM weighting. ``None`` (default) uses the
            odometry's stereo noise, so triangulation covariance and LM weighting share one noise model.
        lm_huber_delta: Huber threshold of the LM, in standard deviations (``inf`` disables the robust kernel).
        ransac: ``"opencv"`` (cv2.solvePnPRansac with the iterative solver, inliers judged on the left image) or
            ``"stereo"`` (minimal AP3P hypotheses scored on the rigid-rig stereo residual (u_l, v_l, u_r), so
            the rig model is enforced inside RANSAC; the motion prior is tested as the first hypothesis).
        ransac_seed: Seed of the stereo RANSAC sampler (runs are deterministic).
=======
        lm_pixel_sigma: Observation noise (pixels) assumed by the LM weighting.
        lm_huber_delta: Huber threshold of the LM, in standard deviations.
>>>>>>> origin/main
    """

    reprojection_error_px: float = 2.0  # Inlier threshold.
    iterations: int = 200  # RANSAC budget.
    confidence: float = 0.999  # RANSAC confidence.
    min_inliers: int = 15  # Failure threshold.
    refine: bool | str = "opencv"  # Refinement method.
    lm_iterations: int = 20  # LM budget.
<<<<<<< HEAD
    lm_pixel_sigma: float | None = None  # LM pixel noise (None = shared stereo noise model).
    lm_huber_delta: float = 2.0  # LM robust threshold.
    ransac: str = "opencv"  # opencv | stereo.
    ransac_seed: int = 0  # Deterministic sampling.

    def __post_init__(self) -> None:
        """Validate the RANSAC variant.

        Raises:
            ValueError: If ``ransac`` is unknown.
        """
        if self.ransac not in ("opencv", "stereo"):  # Typo guard.
            raise ValueError(f"Unknown pnp.ransac '{self.ransac}'; use opencv or stereo")  # Clear error.
=======
    lm_pixel_sigma: float = 1.0  # LM pixel noise.
    lm_huber_delta: float = 2.0  # LM robust threshold.
>>>>>>> origin/main

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
<<<<<<< HEAD
        covariance: Pose covariance from the LM information matrix (only with ``refine="lm"``).
        whitened_sq_norms: Squared whitened LM residual per correspondence (NaN where not refined).
        residual_dof: Dimension of each LM residual (3 for stereo, 2 for mono).
=======
>>>>>>> origin/main
    """

    success: bool  # Outcome flag.
    camera_from_reference: FloatArray  # Estimated relative pose (identity on failure).
    inlier_mask: BoolArray  # Per-correspondence inlier flags.
    num_inliers: int  # Inlier count.
    refined_inlier_mask: BoolArray | None = None  # Post-LM inliers, if LM ran.
<<<<<<< HEAD
    covariance: FloatArray | None = None  # (6, 6) covariance of the left-perturbation delta of T_cam_ref (LM only).
    whitened_sq_norms: FloatArray | None = None  # (N,) squared whitened residuals at the LM solution (NaN = not used).
    residual_dof: int = 3  # Residual dimension of whitened_sq_norms.
=======
>>>>>>> origin/main

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
<<<<<<< HEAD
        self._rng = np.random.default_rng(self._config.ransac_seed)  # Deterministic stereo-RANSAC sampler.
=======
>>>>>>> origin/main

    def estimate(
        self,
        points_3d: FloatArray,
        points_2d: FloatArray,
        camera_matrix: FloatArray,
        initial_guess: FloatArray | None = None,
        points_2d_right: FloatArray | None = None,
        baseline: float = 0.0,
        covariances: FloatArray | None = None,
<<<<<<< HEAD
        pixel_sigma: float | None = None,
=======
>>>>>>> origin/main
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
<<<<<<< HEAD
            pixel_sigma: Observation noise for the LM when ``config.lm_pixel_sigma`` is None (shared noise model).
=======
>>>>>>> origin/main

        Returns:
            The pose estimate with its inlier mask.
        """
        num_points = int(points_3d.shape[0])
        if num_points < max(self._config.min_inliers, 6):
            return PoseEstimate.failure(num_points)
        object_points = np.ascontiguousarray(points_3d, dtype=np.float64)  # OpenCV wants contiguous float64.
        image_points = np.ascontiguousarray(points_2d, dtype=np.float64)  # Same for 2D points.
<<<<<<< HEAD
        if self._config.ransac == "stereo":  # Rigid-rig minimal RANSAC.
            found = stereo_ransac(
                object_points,
                image_points,
                None if points_2d_right is None else np.asarray(points_2d_right, dtype=np.float64),
                camera_matrix,
                baseline,
                initial_guess,
                self._config,
                self._rng,
            )
            if found is None:  # Not enough support.
                return PoseEstimate.failure(num_points)  # Report failure.
            ransac_pose, inliers = found  # Best hypothesis and its inliers (indices).
            rvec, tvec = rvec_tvec_from_pose(ransac_pose)  # OpenCV parameterisation for the refinement paths.
        else:  # OpenCV RANSAC with the iterative solver (left image only).
            use_guess = initial_guess is not None
            rvec0, tvec0 = (
                rvec_tvec_from_pose(initial_guess) if initial_guess is not None else (np.zeros((3, 1)), np.zeros((3, 1)))
            )  # Prior or zeros.
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
=======
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
>>>>>>> origin/main
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
<<<<<<< HEAD
            sigma = self._config.lm_pixel_sigma if self._config.lm_pixel_sigma is not None else (pixel_sigma or 1.0)  # Shared noise model.
            lm_config = LMConfig(  # Settings from the PnP config.
                max_iterations=self._config.lm_iterations,
                pixel_sigma=sigma,
=======
            lm_config = LMConfig(  # Settings from the PnP config.
                max_iterations=self._config.lm_iterations,
                pixel_sigma=self._config.lm_pixel_sigma,
>>>>>>> origin/main
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
<<<<<<< HEAD
            sq_norms = np.full(num_points, np.nan)  # Whitened residuals mapped back to all correspondences.
            if result.whitened_sq_norms is not None:  # Available from the LM.
                sq_norms[inlier_index] = result.whitened_sq_norms  # RANSAC inliers only.
            return PoseEstimate(
                True, pose, mask, int(inlier_index.size), refined_mask, result.covariance, sq_norms, result.residual_dof
            )  # Successful estimate with uncertainty.
        return PoseEstimate(True, pose, mask, int(inlier_index.size), refined_mask)  # Successful estimate.


def _stereo_errors(
    camera_from_reference: FloatArray,
    points_3d: FloatArray,
    points_left: FloatArray,
    points_right: FloatArray | None,
    camera_matrix: FloatArray,
    baseline: float,
) -> tuple[FloatArray, BoolArray]:
    """Per-point worst-coordinate reprojection error of a pose hypothesis under the rigid-rig model.

    The residual is (u_l, v_l, u_r) for stereo observations (the right image adds its column only: in a rectified
    rig v_r = v_l), or (u_l, v_l) without a right observation. The infinity norm makes the test equivalent to
    "every coordinate within the threshold", which keeps the threshold in plain pixels.

    Args:
        camera_from_reference: Hypothesis T_cur_ref.
        points_3d: (N, 3) reference-frame points.
        points_left: (N, 2) current left observations.
        points_right: (N, 2) current right observations or None.
        camera_matrix: 3x3 intrinsics of the rectified pair.
        baseline: Stereo baseline in metres.

    Returns:
        (N,) errors in pixels (inf behind the camera) and the (N,) in-front mask.
    """
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]  # Intrinsics.
    p = points_3d @ camera_from_reference[:3, :3].T + camera_from_reference[:3, 3]  # Points in the current left camera.
    front = p[:, 2] > 1e-6  # Cheirality.
    z = np.where(front, p[:, 2], 1.0)  # Safe depth.
    u = fx * p[:, 0] / z + cx  # Predicted left column.
    err = np.maximum(np.abs(u - points_left[:, 0]), np.abs(fy * p[:, 1] / z + cy - points_left[:, 1]))  # Left error.
    if points_right is not None:  # Rigid rig: the right column follows from the same pose and the baseline.
        err = np.maximum(err, np.abs(u - fx * baseline / z - points_right[:, 0]))  # u_r = u_l - fx B / z.
    return np.where(front, err, np.inf), front  # Points behind the camera are never inliers.


def stereo_ransac(
    points_3d: FloatArray,
    points_left: FloatArray,
    points_right: FloatArray | None,
    camera_matrix: FloatArray,
    baseline: float,
    prior: FloatArray | None,
    config: PnPConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, np.ndarray] | None:
    """RANSAC with minimal AP3P hypotheses and inlier scoring on the rigid-rig stereo residual.

    Compared with ``cv2.solvePnPRansac`` + the iterative solver, this (i) uses a minimal 3-point solver, so each
    hypothesis is cheap and outlier-free samples are found quickly, (ii) judges inliers with both images through
    the fixed baseline, so a match that is wrong only in the right image is rejected before refinement, and
    (iii) tests the constant-velocity prior first, which usually lets adaptive termination stop after a few
    iterations. The returned pose is the best hypothesis; refinement is left to the caller.

    Args:
        points_3d: (N, 3) reference-frame points.
        points_left: (N, 2) current left observations.
        points_right: (N, 2) current right observations, or None (left-only scoring).
        camera_matrix: 3x3 intrinsics.
        baseline: Stereo baseline in metres.
        prior: Optional T_cur_ref prior tested as the first hypothesis.
        config: PnP settings (threshold, iterations, confidence, min_inliers).
        rng: Random generator (seeded by the estimator for reproducibility).

    Returns:
        (T_cur_ref, inlier indices) of the best hypothesis, or None without enough support.
    """
    n = points_3d.shape[0]  # Correspondence count.
    threshold = config.reprojection_error_px  # Pixel threshold.
    best_pose: FloatArray | None = None  # Best hypothesis so far.
    best_inliers = np.zeros(0, dtype=np.int64)  # Its inliers.
    best_score = np.inf  # Truncated-quadratic (MSAC) score; lower is better.

    def consider(pose: FloatArray) -> None:
        """Score a hypothesis with MSAC and keep it if it is the best so far."""
        nonlocal best_pose, best_inliers, best_score
        err, _ = _stereo_errors(pose, points_3d, points_left, points_right, camera_matrix, baseline)  # Errors.
        score = float(np.sum(np.minimum(err, threshold) ** 2))  # MSAC: inliers by error, outliers at the cap.
        if score < best_score:  # Better hypothesis.
            best_pose, best_score = pose, score  # Keep it.
            best_inliers = np.flatnonzero(err < threshold)  # And its support.

    if prior is not None:  # Constant-velocity prediction is often already close.
        consider(np.asarray(prior, dtype=np.float64))  # Free hypothesis.
    log_fail = np.log(1.0 - config.confidence)  # For adaptive termination.
    max_iterations = config.iterations  # Upper bound.
    iteration = 0  # Counter.
    while iteration < max_iterations:  # Main loop.
        iteration += 1  # Count.
        sample = rng.choice(n, 3, replace=False)  # Minimal sample.
        try:  # AP3P can fail on degenerate (collinear) samples.
            count, rvecs, tvecs = cv2.solveP3P(
                points_3d[sample], points_left[sample], camera_matrix, None, flags=cv2.SOLVEPNP_AP3P
            )  # Up to four solutions.
        except cv2.error:  # Degenerate sample.
            continue  # Next sample.
        for rvec, tvec in zip(rvecs[:count], tvecs[:count], strict=False):  # Every solution.
            consider(pose_from_rvec_tvec(np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)))  # Score it.
        ratio = best_inliers.size / n  # Current inlier ratio.
        if ratio > 0:  # Adaptive number of iterations for the requested confidence.
            denom = np.log(max(1e-12, 1.0 - ratio**3))  # log(1 - w^3).
            max_iterations = min(config.iterations, int(np.ceil(log_fail / denom)) if denom < 0 else 1)  # Shrinks as w grows.
    if best_pose is None or best_inliers.size < config.min_inliers:  # Not enough support.
        return None  # Failure.
    return best_pose, best_inliers  # Best hypothesis.
=======
        return PoseEstimate(True, pose, mask, int(inlier_index.size), refined_mask)  # Successful estimate.
>>>>>>> origin/main

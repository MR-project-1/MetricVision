"""Sparse stereo visual odometry (FAST stereo pairs, KLT circular tracking, robust PnP + rigid-rig LM).

Per frame:
1. Each track carries a landmark anchor (``backend.anchors``): re-triangulated in the previous frame
   (``previous``), kept from the track's birth frame (``birth``), or fused over all re-observations
   (``fused``), as a Gaussian world point (``xyz``) or an inverse depth on the birth ray (``inverse_depth``).
2. The front end tracks the stereo pairs into the current frame with the circular L/R check, seeded by
   projecting the anchors with a constant-velocity prediction.
3. The pose comes from RANSAC (OpenCV, or minimal AP3P scored on the stereo residual) and the covariance-weighted
   rigid-rig LM. ``rig_mode`` switches to independent per-camera estimation or SE(3) averaging for the
   ablation of the rigid-rig constraint; the implied extrinsic deviation is recorded.
4. Optional: long tracks are re-aligned against their birth template (removes accumulated KLT drift); anchors
   are fused with pose-uncertainty inflation; the pixel noise is re-estimated from whitened LM residuals.
5. Only inliers survive as tracks; new pairs are detected when tracks run low.
On failure the pose coasts on the motion model, all tracks are re-detected and the failure is counted.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import time  # Runtime measurements.

import numpy as np  # Array maths.

from metricvision.backend.anchors import (  # Landmark models.
    AnchorState,
    fuse_inverse_depth,
    fuse_xyz,
    inverse_depth_points,
    median_chi2,
    pose_uncertainty_covariance,
)
from metricvision.backend.motion_alternatives import (  # RQ2 baseline estimators on the same inliers.
    mahalanobis_3d,
    refine_translation,
    rotation_from_essential,
    umeyama,
)
from metricvision.backend.odometry import OdometryConfig, TrackingResult  # Shared settings and result type.
from metricvision.backend.pose import PnPPoseEstimator, PoseEstimate, _stereo_errors  # PnP + RANSAC (+ LM).
from metricvision.backend.stats import FrameStats  # Per-frame measurements.
from metricvision.backend.window_ba import SlidingWindowBA, WindowBAConfig  # Keyframe bundle adjustment.
from metricvision.core.geometry import invert_pose, make_pose, rotation_angle_deg  # Pose helpers.
from metricvision.core.stereo import triangulate_rectified, triangulation_covariance  # Sparse triangulation.
from metricvision.core.types import (  # Shared types.
    Correspondences,
    FloatArray,
    ImageArray,
    StereoCamera,
    StereoFrame,
    Trajectory,
)
from metricvision.evaluation.frontend_metrics import essential_inlier_ratio  # Front-end-only metric.
from metricvision.frontend.alignment import align_warped, predict_affine, refine_right  # Track re-alignment.
from metricvision.frontend.stereo_klt import StereoFrontend, StereoTracks  # Front end.


def _elapsed_ms(start: float) -> float:
    """Milliseconds since a ``time.perf_counter()`` reading.

    Args:
        start: Earlier perf_counter value.

    Returns:
        Elapsed milliseconds.
    """
    return (time.perf_counter() - start) * 1000.0  # Seconds to milliseconds.


def _chordal_mean(rotations: list[FloatArray]) -> FloatArray:
    """Chordal L2 mean of rotations (projection of their sum onto SO(3)).

    Args:
        rotations: 3x3 rotation matrices.

    Returns:
        Mean rotation.
    """
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))  # Polar decomposition of the sum.
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt  # Closest proper rotation.


class SparseStereoOdometry:
    """Stereo VO driven by a sparse ``StereoFrontend``; same API as ``StereoVisualOdometry``."""

    def __init__(self, camera: StereoCamera, frontend: StereoFrontend, config: OdometryConfig | None = None) -> None:
        """Wire the pipeline together.

        Args:
            camera: Rectified stereo camera.
            frontend: Sparse stereo front end.
            config: Back-end settings; defaults when omitted.

        Raises:
            ValueError: If the anchor mode is unknown.
        """
        self._camera = camera  # Camera model.
        self._camera_matrix = camera.intrinsic_matrix  # Cached K.
        self._frontend = frontend  # Front end.
        self._config = config or OdometryConfig()  # Settings.
        self._pose_estimator = PnPPoseEstimator(self._config.pnp)  # PnP solver.
        self._right_from_left = make_pose(np.eye(3), np.array([-camera.baseline, 0.0, 0.0]))  # Calibrated T_RL.
        ba_config = WindowBAConfig(  # Keyframe bundle adjustment (window 0 = off).
            window=self._config.ba_window,
            keyframe_every=self._config.ba_keyframe_every,
            min_translation_m=self._config.ba_min_translation_m,
            iterations=self._config.ba_iterations,
            huber_delta=self._config.pnp.lm_huber_delta if np.isfinite(self._config.pnp.lm_huber_delta) else 1e9,
        )
        self._ba = SlidingWindowBA(ba_config, self._camera_matrix, camera.baseline)
        if self._config.sparse_point_anchor not in ("previous", "birth", "fused"):  # Typo guard.
            raise ValueError(f"sparse_point_anchor must be 'previous', 'birth' or 'fused', got '{self._config.sparse_point_anchor}'")
        self.reset()  # Initialise state.

    # ------------------------------------------------------------------ public API
    @property
    def trajectory(self) -> Trajectory:
        """Return the estimated left-camera trajectory.

        Returns:
            T_world_cam per processed frame.
        """
        return self._trajectory  # Accumulated poses.

    @property
    def num_failures(self) -> int:
        """Return the number of tracking failures.

        Returns:
            Failure count.
        """
        return self._num_failures  # Counter.

    @property
    def num_keyframes(self) -> int:
        """Return the number of replenishment frames (frames where new features were detected).

        Returns:
            Replenishment count.
        """
        return self._num_keyframes  # Counter.

    @property
    def noise_sigma_px(self) -> float:
        """Return the pixel noise currently assumed.

        Returns:
            Standard deviation in pixels.
        """
        return self._sigma  # Fixed or adaptive.

    @property
    def pose_covariances(self) -> list[FloatArray | None]:
        """Return the per-frame LM pose covariance (left perturbation of T_cam_world), None where unavailable.

        Returns:
            One entry per processed frame.
        """
        return self._pose_covs  # For consistency (NEES) evaluation.

    def reset(self) -> None:
        """Forget all state before a new sequence."""
        self._frontend.reset()  # Front-end state.
        self._ba.reset()  # Keyframe window.
        self._next_track_id = 0  # Persistent track identifiers.
        self._pose_covs: list[FloatArray | None] = []  # Per-frame pose covariance.
        self._previous: tuple[ImageArray, ImageArray] | None = None  # Previous (left, right) images.
        self._tracks = StereoTracks.empty()  # Pairs in the previous frame.
        self._state = AnchorState.empty()  # Per-track anchors (rows match self._tracks).
        self._poses: dict[int, FloatArray] = {}  # Birth-frame poses T_world_cam.
        self._images: dict[int, ImageArray] = {}  # Birth-frame left images (re-alignment only).
        self._last_pose = np.eye(4, dtype=np.float64)  # T_world_prev.
        self._last_pose_cov: FloatArray | None = None  # Covariance of the previous pose (pose inflation).
        self._velocity = np.eye(4, dtype=np.float64)  # T_prev_cur of the last step.
        self._sigma = float(self._config.stereo_sigma_px)  # Pixel noise model.
        self._trajectory = Trajectory()  # Output.
        self._num_failures = 0  # Counters.
        self._num_keyframes = 0

    def process(self, frame: StereoFrame) -> TrackingResult:
        """Process the next frame.

        Args:
            frame: Rectified stereo frame.

        Returns:
            Estimated pose and measurements.
        """
        if self._previous is None:  # First frame: detect, define the world origin.
            start = time.perf_counter()  # Time the step.
            tracks, _, depth_ms = self._replenish(frame, StereoTracks.empty())  # Initial stereo pairs.
            self._previous = (frame.left, frame.right)  # Remember images.
            pose = np.eye(4, dtype=np.float64)  # World = first camera.
            self._tracks, self._state = tracks, self._new_anchors(tracks, frame, pose, None)  # Anchor them at the origin.
            self._trajectory.append(frame.timestamp, pose)  # Record.
            self._pose_covs.append(None)  # The origin is fixed.
            stats = FrameStats(frame.index, frame.timestamp, 0, 0, 0, float("nan"), True, True, 0.0, 0.0, depth_ms, _elapsed_ms(start))
            return TrackingResult(pose, stats)  # Report.
        return self._track(frame)  # Normal tracking.

    # ------------------------------------------------------------------ anchors
    def _triangulate(self, tracks: StereoTracks) -> tuple[FloatArray, FloatArray]:
        """Triangulate stereo pairs in their camera with the current noise model.

        Args:
            tracks: Stereo pairs.

        Returns:
            (N, 3) camera-frame points (NaN where invalid) and (N, 3, 3) covariances (NaN where invalid).
        """
        points, valid = triangulate_rectified(tracks.left, tracks.disparity, self._camera)  # Camera-frame points.
        covariances = np.full((len(tracks), 3, 3), np.nan, dtype=np.float64)  # Default: invalid.
        covariances[valid] = triangulation_covariance(tracks.left[valid], tracks.disparity[valid], self._camera, self._sigma)
        return points, covariances  # Camera frame.

    def _observe_world(
        self, tracks: StereoTracks, world_from_camera: FloatArray, pose_cov: FloatArray | None
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Triangulate pairs and express them in the world, with optional pose-uncertainty inflation.

        Args:
            tracks: Stereo pairs observed by the camera at ``world_from_camera``.
            world_from_camera: T_world_cam.
            pose_cov: Covariance of that camera's pose (left perturbation of T_cam_world), or None.

        Returns:
            World points, world covariances (including inflation), camera points, camera covariances.
        """
        points_c, cov_c = self._triangulate(tracks)  # Camera frame.
        rotation, translation = world_from_camera[:3, :3], world_from_camera[:3, 3]  # Pose parts.
        points_w = points_c @ rotation.T + translation  # World (NaN propagates).
        cov_w = rotation @ cov_c @ rotation.T  # Rotated covariance.
        if self._config.anchor_pose_inflation and pose_cov is not None:  # Pose uncertainty into the point.
            ok = np.isfinite(points_c).all(axis=1)  # Valid rows only.
            cov_w[ok] += pose_uncertainty_covariance(points_c[ok], world_from_camera, pose_cov)
        return points_w, cov_w, points_c, cov_c  # Both frames.

    def _new_anchors(self, tracks: StereoTracks, frame: StereoFrame, world_from_camera: FloatArray, pose_cov: FloatArray | None) -> AnchorState:
        """Create anchors for tracks born in ``frame``.

        Args:
            tracks: Newly detected pairs.
            frame: Birth frame.
            world_from_camera: Its pose.
            pose_cov: Its pose covariance (or None).

        Returns:
            Anchor state rows for the new tracks.
        """
        n = len(tracks)  # New track count.
        points_w, cov_w, points_c, cov_c = self._observe_world(tracks, world_from_camera, pose_cov)  # Birth triangulation.
        depth = points_c[:, 2]  # Birth depth (NaN where invalid).
        inverse_depth = 1.0 / depth  # rho.
        inverse_depth_var = cov_c[:, 2, 2] / depth**4  # var(1/Z) = var(Z) / Z^4.
        birth_pose_cov = np.zeros((n, 3, 3))  # Extra covariance from the birth pose.
        if self._config.anchor_pose_inflation and pose_cov is not None:  # Same inflation for the inverse-depth model.
            ok = np.isfinite(points_c).all(axis=1)
            birth_pose_cov[ok] = pose_uncertainty_covariance(points_c[ok], world_from_camera, pose_cov)
        if n:  # Remember what re-alignment and the inverse-depth model need.
            self._poses[frame.index] = world_from_camera.copy()  # Birth pose.
            if self._config.realign_every > 0:  # Birth template image.
                self._images[frame.index] = frame.left
        return AnchorState(
            points=points_w,
            covariances=cov_w,
            birth_frame=np.full(n, frame.index, dtype=np.int64),
            birth_left=tracks.left.astype(np.float64),
            inverse_depth=inverse_depth,
            inverse_depth_var=inverse_depth_var,
            birth_pose_cov=birth_pose_cov,
            age=np.zeros(n, dtype=np.int64),
            track_id=self._new_ids(n),
        )

    def _new_ids(self, n: int) -> np.ndarray:
        """Allocate persistent track identifiers.

        Args:
            n: How many.

        Returns:
            (n,) identifiers.
        """
        ids = np.arange(self._next_track_id, self._next_track_id + n, dtype=np.int64)
        self._next_track_id += n
        return ids

    def _anchor_points(self) -> tuple[FloatArray, FloatArray]:
        """World points and covariances the current tracks are matched against.

        Returns:
            (N, 3) points (NaN = unusable) and (N, 3, 3) covariances.
        """
        if self._config.sparse_point_anchor == "previous":  # Pure frame-to-frame: re-triangulate now.
            points, covariances, _, _ = self._observe_world(self._tracks, self._last_pose, self._last_pose_cov)
            return points, covariances
        if self._config.anchor_param == "inverse_depth":  # Points on the birth rays.
            return inverse_depth_points(self._state, self._poses, self._camera, self._sigma)
        return self._state.points, self._state.covariances  # Gaussian world anchors.

    def _forget_unused(self) -> None:
        """Drop birth poses and images no track refers to any more."""
        alive = set(np.unique(self._state.birth_frame).tolist())  # Frames still referenced.
        for store in (self._poses, self._images):  # Both caches.
            for key in [k for k in store if k not in alive]:
                del store[key]

    # ------------------------------------------------------------------ pose estimation
    def _estimate(
        self, points: FloatArray, covariances: FloatArray, observed: StereoTracks, prior: FloatArray, previous_left: FloatArray | None = None
    ) -> tuple[PoseEstimate, FloatArray, np.ndarray, tuple[float, float]]:
        """Estimate T_cur_world with the configured rig model.

        Args:
            points: (M, 3) world points.
            covariances: (M, 3, 3) their covariances.
            observed: Current observations of those points.
            prior: Predicted T_cur_world.
            previous_left: (M, 2) the same tracks in the previous left image (``pose_method: soft``).

        Returns:
            The (left) estimate, the pose T_cur_world to use, the inlier mask, and the implied extrinsic deviation
            (mm, deg) for the independent/averaged modes (NaN for rigid).
        """
        cov = covariances if self._config.lm_covariance_weighting else None  # Whitening on/off.
        left = observed.left.astype(np.float64)  # Left observations.
        right = observed.right.astype(np.float64)  # Right observations.
        mode = self._config.rig_mode  # Rig model.
        if mode == "rigid":  # One pose, both images.
            est = self._pose_estimator.estimate(
                points,
                left,
                self._camera_matrix,
                prior,
                points_2d_right=right if self._config.lm_right_residuals else None,
                baseline=self._camera.baseline,
                covariances=cov,
                pixel_sigma=self._sigma,
            )
            inliers = est.refined_inlier_mask if est.refined_inlier_mask is not None else est.inlier_mask
            if est.success and self._config.pose_method != "pnp":  # Alternative estimator on the RANSAC inliers.
                pose, inliers = self._alternative(est, points, covariances, observed, previous_left)
                est = PoseEstimate(True, pose, est.inlier_mask, est.num_inliers)  # No LM covariance for these estimators.
                return est, pose, inliers, (float("nan"), float("nan"))
            return est, est.camera_from_reference, inliers, (float("nan"), float("nan"))
        # Independent per-camera estimation (12 DoF): each camera sees the same landmarks through its own pose.
        est_l = self._pose_estimator.estimate(points, left, self._camera_matrix, prior, covariances=cov, pixel_sigma=self._sigma)
        prior_r = self._right_from_left @ prior  # Predicted T_right_world.
        est_r = self._pose_estimator.estimate(points, right, self._camera_matrix, prior_r, covariances=cov, pixel_sigma=self._sigma)
        if not (est_l.success and est_r.success):  # Both must succeed.
            failed = est_l if not est_l.success else est_r
            return failed, prior, np.zeros(len(points), dtype=bool), (float("nan"), float("nan"))
        mask_l = est_l.refined_inlier_mask if est_l.refined_inlier_mask is not None else est_l.inlier_mask
        mask_r = est_r.refined_inlier_mask if est_r.refined_inlier_mask is not None else est_r.inlier_mask
        implied = est_r.camera_from_reference @ invert_pose(est_l.camera_from_reference)  # Implied T_RL.
        deviation = invert_pose(self._right_from_left) @ implied  # Away from calibration.
        dev = (1000.0 * float(np.linalg.norm(deviation[:3, 3])), rotation_angle_deg(deviation[:3, :3]))  # mm, deg.
        pose = est_l.camera_from_reference  # Reported: the left camera.
        if mode == "averaged":  # Loose coupling: average the left estimate and the right-implied left pose.
            from_right = invert_pose(self._right_from_left) @ est_r.camera_from_reference  # T_left_world via the right camera.
            rotation = _chordal_mean([pose[:3, :3], from_right[:3, :3]])
            pose = make_pose(rotation, 0.5 * (pose[:3, 3] + from_right[:3, 3]))
        return est_l, pose, mask_l & mask_r, dev

    def _alternative(
        self, est: PoseEstimate, points: FloatArray, covariances: FloatArray, observed: StereoTracks, previous_left: FloatArray | None
    ) -> tuple[FloatArray, np.ndarray]:
        """Re-estimate the pose from the RANSAC inliers with a baseline estimator (``pose_method``).

        Outlier rejection is held fixed (same RANSAC inliers), so only the estimator differs from the rigid-rig LM.

        Args:
            est: PnP/RANSAC estimate (its inlier mask and pose are used as the inlier set and the initial guess).
            points: (M, 3) world landmarks.
            covariances: (M, 3, 3) their covariances.
            observed: Current observations.
            previous_left: (M, 2) previous left pixels of the same tracks.

        Returns:
            T_cam_world and the final inlier mask (stereo error below the RANSAC threshold).
        """
        mask = est.inlier_mask
        left, right = observed.left.astype(np.float64), observed.right.astype(np.float64)
        initial = est.camera_from_reference
        method = self._config.pose_method
        if method in ("umeyama3d", "mahalanobis3d"):  # 3D-3D: landmarks vs current triangulations.
            cam_points, cam_cov = self._triangulate(observed)
            ok = mask & np.isfinite(cam_points).all(axis=1)
            pose = umeyama(points[ok], cam_points[ok]) if ok.sum() >= 3 else initial
            if method == "mahalanobis3d" and ok.sum() >= 3:
                pose = mahalanobis_3d(points[ok], covariances[ok], cam_points[ok], cam_cov[ok], pose, huber_delta=self._config.pnp.lm_huber_delta)
        else:  # SOFT: rotation from the essential matrix, translation from stereo with the rotation fixed.
            pose = initial.copy()
            if previous_left is not None:
                rotation_cur_prev = rotation_from_essential(previous_left[mask], left[mask], self._camera_matrix)
                if rotation_cur_prev is not None:
                    pose[:3, :3] = rotation_cur_prev @ invert_pose(self._last_pose)[:3, :3]  # R_cur_world = R_cur_prev R_prev_world.
            cov = covariances[mask] if self._config.lm_covariance_weighting else None
            pose = refine_translation(
                pose,
                points[mask],
                left[mask],
                right[mask],
                self._camera_matrix,
                self._camera.baseline,
                cov,
                self._sigma,
                huber_delta=self._config.pnp.lm_huber_delta,
            )
        errors, _ = _stereo_errors(pose, points, left, right, self._camera_matrix, self._camera.baseline)
        return pose, mask & (errors < self._config.pnp.reprojection_error_px)

    def _bundle_adjust(
        self, frame: StereoFrame, survivors: StereoTracks, state: AnchorState, pose: FloatArray
    ) -> tuple[FloatArray, AnchorState, np.ndarray, int]:
        """Insert a keyframe (if the policy says so) and optimise the window.

        Args:
            frame: Current frame.
            survivors: Current inlier observations.
            state: Their anchors.
            pose: Current T_world_cam.

        Returns:
            Updated T_world_cam, updated anchors, keep mask (chi^2 outliers of the newest keyframe dropped), landmarks.
        """
        keep = np.ones(len(survivors), dtype=bool)
        camera_from_world = invert_pose(pose)
        if self._config.ba_window <= 0 or not self._ba.wants_keyframe(camera_from_world):
            return pose, state, keep, 0
        obs = np.column_stack([survivors.left, survivors.right[:, 0]]).astype(np.float64)  # (u_L, v_L, u_R).
        init_points, _, _, _ = self._observe_world(survivors, pose, None)  # Initial landmark positions.
        self._ba.add_keyframe(frame.index, camera_from_world, state.track_id, obs, init_points)
        result = self._ba.optimise(self._sigma)
        if result is None:
            return pose, state, keep, 0
        keyframes = self._ba.keyframes
        pose = invert_pose(keyframes[-1].camera_from_world)  # Refined current pose.
        for kf in keyframes:  # Birth poses of keyframes that are birth frames follow the refinement.
            if kf.frame_index in self._poses:
                self._poses[kf.frame_index] = invert_pose(kf.camera_from_world)
        refined = np.array([int(t) in self._ba.landmarks for t in state.track_id], dtype=bool)
        if refined.any() and self._config.sparse_point_anchor != "previous":  # Anchors take the BA landmark positions.
            points = state.points.copy()
            points[refined] = np.stack([self._ba.landmarks[int(t)] for t in state.track_id[refined]])
            rho = state.inverse_depth.copy()
            for i in np.flatnonzero(refined):  # Inverse depth on the (possibly refined) birth ray.
                birth = self._poses.get(int(state.birth_frame[i]))
                if birth is not None:
                    z = ((points[i] - birth[:3, 3]) @ birth[:3, :3])[2]
                    if z > 1e-3:
                        rho[i] = 1.0 / z
            state = state.replace(points=points, inverse_depth=rho)
        keep = ~np.isin(state.track_id, result.outlier_ids)
        return pose, state, keep, result.num_landmarks

    def _update_noise(self, est: PoseEstimate, inliers: np.ndarray) -> None:
        """Re-estimate the pixel noise from whitened LM residuals (``noise_model: adaptive``).

        If the noise model is right, the squared whitened residual of a d-dimensional inlier follows chi^2(d). The
        whole residual covariance scales with sigma^2 (pixel term and triangulation term alike), so the robust
        ratio median(e^2) / median(chi^2_d) estimates (sigma_true / sigma)^2.

        Args:
            est: Pose estimate with whitened residuals.
            inliers: Inlier mask over its correspondences.
        """
        if self._config.noise_model != "adaptive" or est.whitened_sq_norms is None:  # Fixed noise or no LM.
            return
        values = est.whitened_sq_norms[inliers]  # Inlier residuals.
        values = values[np.isfinite(values)]
        if values.size < 20:  # Too few for a stable estimate.
            return
        scale = float(np.sqrt(np.median(values) / median_chi2(est.residual_dof)))  # sigma_true / sigma.
        target = self._sigma * scale  # Implied noise.
        rate = self._config.noise_adapt_rate  # Smoothing.
        low, high = self._config.noise_bounds_px  # Clamp.
        self._sigma = float(np.clip((1.0 - rate) * self._sigma + rate * target, low, high))

    # ------------------------------------------------------------------ re-alignment
    def _realign(
        self, frame: StereoFrame, tracks: StereoTracks, state: AnchorState, camera_from_world: FloatArray
    ) -> tuple[StereoTracks, np.ndarray, int]:
        """Re-align long tracks against their birth templates and re-match their right observations.

        Args:
            frame: Current frame.
            tracks: Surviving pairs in the current frame.
            state: Their anchors.
            camera_from_world: Current T_cam_world.

        Returns:
            Corrected tracks, a keep mask (failed re-alignments are dropped) and the number re-aligned.
        """
        keep = np.ones(len(tracks), dtype=bool)  # Default: keep everything.
        left, right = tracks.left.astype(np.float64).copy(), tracks.right.astype(np.float64).copy()  # Working copies.
        realigned = 0  # Counter.
        candidates = (state.age >= self._config.realign_min_age) & np.isfinite(state.inverse_depth) & (state.inverse_depth > 0)
        for birth in np.unique(state.birth_frame[candidates]):  # One template image per birth frame.
            if int(birth) not in self._images:  # Image not cached (e.g. re-alignment enabled mid-run).
                continue
            rows = np.flatnonzero(candidates & (state.birth_frame == birth))
            world_from_birth = self._poses[int(birth)]  # T_world_birth.
            if self._config.sparse_point_anchor != "previous" and self._config.anchor_param == "xyz":  # Use the fused anchor depth.
                local = (state.points[rows] - world_from_birth[:3, 3]) @ world_from_birth[:3, :3]  # Anchor in the birth camera.
                depth = local[:, 2]
            else:  # Inverse depth on the birth ray.
                depth = 1.0 / state.inverse_depth[rows]
            usable = np.isfinite(depth) & (depth > 0.05)
            rows, depth = rows[usable], depth[usable]
            if rows.size == 0:
                continue
            affine, _ = predict_affine(self._camera_matrix, state.birth_left[rows], depth, camera_from_world @ world_from_birth)
            aligned, ok = align_warped(
                self._images[int(birth)], frame.left, state.birth_left[rows], affine, left[rows], max_shift_px=self._config.realign_max_shift_px
            )
            new_right, ok_r = refine_right(frame.left, frame.right, aligned, right[rows] + (aligned - left[rows]))  # Follow on the right.
            good = ok & ok_r & (np.abs(new_right[:, 1] - aligned[:, 1]) <= 1.5) & (aligned[:, 0] - new_right[:, 0] > 0.5)  # Epipolar.
            left[rows[good]], right[rows[good]] = aligned[good], new_right[good]  # Corrected observations.
            keep[rows[~good]] = False  # Drifted or occluded tracks end here.
            realigned += int(good.sum())
        return StereoTracks(left.astype(np.float32), right.astype(np.float32)), keep, realigned

    # ------------------------------------------------------------------ main loop
    def _replenish(self, frame: StereoFrame, tracks: StereoTracks) -> tuple[StereoTracks, bool, float]:
        """Detect new stereo pairs when tracks run low.

        Args:
            frame: Current frame.
            tracks: Surviving pairs in the current frame.

        Returns:
            Updated tracks, whether detection ran, and its time in milliseconds.
        """
        if len(tracks) >= self._frontend.min_tracks:  # Enough tracks.
            return tracks, False, 0.0
        start = time.perf_counter()  # Time stereo detection + matching.
        new = self._frontend.detect(frame.left, frame.right, tracks)  # New pairs away from existing ones.
        self._num_keyframes += 1  # Count replenishment frames.
        return tracks.concatenate(new), True, _elapsed_ms(start)

    def _track(self, frame: StereoFrame) -> TrackingResult:
        """Track the previous frame's pairs into ``frame`` and estimate its pose.

        Args:
            frame: Current frame.

        Returns:
            Pose and measurements.
        """
        assert self._previous is not None  # Guaranteed by process().
        start = time.perf_counter()  # Whole-step timer.
        tracks = self._tracks  # Previous-frame pairs.
        points_world, cov_world = self._anchor_points()  # Landmarks to match against.
        has_depth = np.isfinite(points_world).all(axis=1) & np.isfinite(cov_world).all(axis=(1, 2))  # Usable anchors.
        predicted_pose = self._last_pose @ self._velocity  # Constant-velocity prediction of T_world_cur.
        prior = invert_pose(predicted_pose)  # Predicted T_cur_world.
        predicted_left, predicted_ok = self._project(prior, points_world)  # Where tracks should appear.
        predicted_left = np.where(predicted_ok[:, None], predicted_left, tracks.left).astype(np.float32)

        match_start = time.perf_counter()  # Front-end timer.
        current, keep = self._frontend.track(self._previous, (frame.left, frame.right), tracks, predicted_left)  # Circular KLT.
        self._frontend.synchronize()
        match_ms = _elapsed_ms(match_start)

        pose_start = time.perf_counter()  # Pose timer.
        kept_index = np.flatnonzero(keep)  # Rows of the previous tracks that survived tracking.
        usable = has_depth[keep]  # Survivors with a valid anchor.
        if self._config.max_depth_baselines > 0:  # Classic depth gate (ablation of covariance weighting).
            depth = (points_world[keep] @ prior[:3, :3].T + prior[:3, 3])[:, 2]  # Predicted depth in the current camera.
            usable &= np.nan_to_num(depth, nan=np.inf) < self._config.max_depth_baselines * self._camera.baseline
        used_index = kept_index[usable]  # Rows used for pose estimation.
        observed = current.subset(usable)  # Their current observations.
        estimate, camera_from_world, inliers, extrinsic_dev = self._estimate(
            points_world[used_index], cov_world[used_index], observed, prior, tracks.left[used_index].astype(np.float64)
        )
        pose_ms = _elapsed_ms(pose_start)

        realigned, realign_dropped, fused, ba_landmarks, ba_ms = 0, 0, 0, 0, 0.0  # Counters.
        if estimate.success:  # Pose found.
            pose = invert_pose(camera_from_world)  # T_world_cur.
            survivors = observed.subset(inliers)  # Outliers (and moving objects) stop being tracked.
            state = self._state.subset(used_index[inliers])  # Their anchors travel with them.
            state = state.replace(age=state.age + 1)  # One frame older.
            pose_cov = estimate.covariance  # Uncertainty of this pose.
            if self._config.realign_every > 0 and frame.index % self._config.realign_every == 0:  # Drift correction.
                survivors, ok, realigned = self._realign(frame, survivors, state, camera_from_world)
                realign_dropped = int((~ok).sum())
                survivors, state = survivors.subset(ok), state.subset(ok)
            if self._config.sparse_point_anchor == "fused" and len(survivors):  # Fuse this frame's triangulation.
                obs_points, obs_cov, _, _ = self._observe_world(survivors, pose, pose_cov)
                if self._config.anchor_param == "inverse_depth":
                    state, fused = fuse_inverse_depth(state, self._poses, obs_points, obs_cov)
                else:
                    new_points, new_cov = fuse_xyz(state.points, state.covariances, obs_points, obs_cov)
                    fused = int(np.isfinite(obs_points).all(axis=1).sum())
                    state = state.replace(points=new_points, covariances=new_cov)
            self._update_noise(estimate, inliers)  # Adaptive noise model.
            ba_start = time.perf_counter()
            pose, state, ok, ba_landmarks = self._bundle_adjust(frame, survivors, state, pose)  # Keyframe window.
            survivors, state = survivors.subset(ok), state.subset(ok)
            ba_ms = _elapsed_ms(ba_start)
        else:  # Tracking failure.
            pose = predicted_pose  # Coast.
            pose_cov = None
            self._num_failures += 1
            self._ba.reset()  # The window cannot bridge a failure.
            survivors, state = StereoTracks.empty(), AnchorState.empty()  # Re-detect everything.
        self._velocity = invert_pose(self._last_pose) @ pose  # Update the motion model.
        self._last_pose, self._last_pose_cov = pose, pose_cov  # Remember.
        new_tracks, replenished, depth_ms = self._replenish(frame, survivors)  # Top up features.
        if not estimate.success and not replenished:  # Always re-detect after a failure.
            new_tracks = self._frontend.detect(frame.left, frame.right, StereoTracks.empty())
        born = new_tracks.subset(np.arange(len(new_tracks)) >= len(survivors))  # Only the new pairs.
        self._tracks = new_tracks  # Survivors first, then new pairs.
        self._state = state.concatenate(self._new_anchors(born, frame, pose, pose_cov))  # Anchors in the same order.
        self._forget_unused()  # Bounded memory.
        self._previous = (frame.left, frame.right)  # Advance.
        total_ms = _elapsed_ms(start)  # Stop before measurement-only work.

        ratio = (  # Front-end quality on the temporal (prev L -> cur L) tracks, excluded from runtime.
            essential_inlier_ratio(
                Correspondences(tracks.left[keep], current.left, np.ones(len(current), dtype=np.float32)),
                self._camera_matrix,
                self._config.essential_threshold_px,
            )
            if self._config.compute_essential_inlier_ratio
            else float("nan")
        )
        vertical = (current.left[:, 1] - current.right[:, 1]).astype(np.float64)  # Rig-consistency signal.
        v_median = float(np.median(vertical)) if vertical.size else float("nan")
        v_mad = float(1.4826 * np.median(np.abs(vertical - v_median))) if vertical.size else float("nan")
        sigma_t = sigma_r = float("nan")  # Pose uncertainty.
        if pose_cov is not None and np.isfinite(pose_cov).all():
            sigma_t = 1000.0 * float(np.sqrt(max(np.trace(pose_cov[:3, :3]), 0.0)))
            sigma_r = float(np.degrees(np.sqrt(max(np.trace(pose_cov[3:, 3:]), 0.0))))
        self._trajectory.append(frame.timestamp, pose)  # Record.
        self._pose_covs.append(pose_cov)  # For NEES.
        stats = FrameStats(  # Measurements.
            frame_index=frame.index,
            timestamp=frame.timestamp,
            num_matches=int(keep.sum()),  # Tracks surviving the circular check.
            num_matches_with_depth=int(usable.sum()),  # With a usable anchor.
            num_pnp_inliers=estimate.num_inliers,  # RANSAC inliers.
            essential_inlier_ratio=ratio,
            tracking_ok=estimate.success,
            is_keyframe=replenished,  # Replenishment frame.
            match_ms=match_ms,
            pose_ms=pose_ms,
            depth_ms=depth_ms,  # Stereo detection + matching time.
            total_ms=total_ms,
            vertical_disparity_median_px=v_median,
            vertical_disparity_mad_px=v_mad,
            pose_sigma_trans_mm=sigma_t,
            pose_sigma_rot_deg=sigma_r,
            noise_sigma_px=self._sigma,
            extrinsic_dev_trans_mm=extrinsic_dev[0],
            extrinsic_dev_rot_deg=extrinsic_dev[1],
            num_realigned=realigned,
            num_realign_dropped=realign_dropped,
            num_fused=fused,
            ba_ms=ba_ms,
            num_ba_landmarks=ba_landmarks,
        )
        return TrackingResult(pose, stats)

    def _project(self, camera_from_reference: FloatArray, points: FloatArray) -> tuple[FloatArray, np.ndarray]:
        """Project reference-frame points into the current left image.

        Args:
            camera_from_reference: T_cur_ref.
            points: (N, 3) points (NaN rows allowed).

        Returns:
            (N, 2) pixels and an (N,) mask of finite points in front of the camera.
        """
        p = points @ camera_from_reference[:3, :3].T + camera_from_reference[:3, 3]  # Transform.
        ok = np.isfinite(p).all(axis=1) & (p[:, 2] > 0.1)  # Usable projections.
        z = np.where(ok, p[:, 2], 1.0)  # Safe depth.
        pixels = np.column_stack([self._camera.fx * p[:, 0] / z + self._camera.cx, self._camera.fy * p[:, 1] / z + self._camera.cy])
        return pixels.astype(np.float32), ok

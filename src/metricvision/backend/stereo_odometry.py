"""Frame-to-frame sparse stereo visual odometry (FAST + ORB stereo, KLT circular tracking, PnP + rig LM).

Per frame:
1. Stereo pairs of the previous frame are triangulated (with first-order covariance).
2. The front end tracks them into the current frame with the circular L/R check,
   seeded by projecting the 3D points with a constant-velocity prediction.
3. PnP + RANSAC on (3D, current left 2D) gives the initial pose; the
   covariance-weighted rigid-rig LM then refines it with both current images.
   The 3D points are either re-triangulated in the previous frame (``previous``
   anchor: pure frame-to-frame) or kept from each track's birth frame in world
   coordinates (``birth`` anchor), which stops small per-frame biases from chaining.
4. Only LM inliers survive as tracks; new pairs are detected when tracks run low.
5. Optionally (``window_size >= 2``), every ``window_keyframe_interval`` frames the frame
   becomes a keyframe of a sliding-window rigid-rig bundle adjustment over keyframe poses and
   landmarks, with optional gated monocular depth priors (``depth_prior``, RoMeO-style). The
   refined keyframe pose is output and the refined landmarks replace the tracking anchors.
On failure the pose coasts on the motion model, all tracks are re-detected and
the failure is counted, exactly as in the dense keyframe pipeline.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import time  # Runtime measurements.

import numpy as np  # Array maths.
import numpy.typing as npt  # Id arrays.

from metricvision.backend.keyframe_window import DepthPriorConfig, KeyframeWindow, WindowStats  # Windowed BA.
from metricvision.backend.odometry import OdometryConfig, TrackingResult  # Shared settings and result type.
from metricvision.backend.pose import PnPPoseEstimator  # PnP + RANSAC (+ LM).
from metricvision.backend.stats import FrameStats  # Per-frame measurements.
from metricvision.backend.window_ba import WindowBAConfig  # BA settings.
from metricvision.core.geometry import invert_pose, orthonormalize_pose  # Rigid inverse, SO(3) projection.
from metricvision.core.stereo import triangulate_rectified, triangulation_covariance  # Sparse triangulation.
from metricvision.core.types import (  # Shared types.
    BoolArray,
    Correspondences,
    Float32Array,
    FloatArray,
    FrameDebug,
    ImageArray,
    StereoCamera,
    StereoFrame,
    Trajectory,
)
from metricvision.depth.base import DepthEstimator  # Monocular depth priors.
from metricvision.dynamic import DynamicMasker, pixels_in_mask  # Movable-object masking.
from metricvision.evaluation.frontend_metrics import essential_inlier_ratio  # Front-end-only metric.
from metricvision.frontend.stereo_klt import StereoFrontend, StereoTracks  # Front end.


def _elapsed_ms(start: float) -> float:
    """Milliseconds since a ``time.perf_counter()`` reading.

    Args:
        start: Earlier perf_counter value.

    Returns:
        Elapsed milliseconds.
    """
    return (time.perf_counter() - start) * 1000.0  # Seconds to milliseconds.


class SparseStereoOdometry:
    """Frame-to-frame stereo VO driven by a sparse ``StereoFrontend``; same API as ``StereoVisualOdometry``."""

    def __init__(
        self,
        camera: StereoCamera,
        frontend: StereoFrontend,
        config: OdometryConfig | None = None,
        depth_estimator: DepthEstimator | None = None,
        masker: DynamicMasker | None = None,
    ) -> None:
        """Wire the pipeline together.

        Args:
            camera: Rectified stereo camera.
            frontend: Sparse stereo front end.
            config: Back-end settings (PnP/LM thresholds, stereo noise, windowed BA); defaults when omitted.
            depth_estimator: Monocular depth network for BA priors (used only with ``depth_prior``).
            masker: Segmentation of movable objects (used only with ``dynamic_mask``).

        Raises:
            ValueError: If the anchoring mode is unknown or depth priors are requested without an estimator.
        """
        self._camera = camera  # Camera model.
        self._camera_matrix = camera.intrinsic_matrix  # Cached K.
        self._frontend = frontend  # Front end.
        self._config = config or OdometryConfig()  # Settings.
        self._pose_estimator = PnPPoseEstimator(self._config.pnp)  # PnP solver.
        if self._config.sparse_point_anchor not in ("previous", "birth", "fused"):  # Typo guard.
            raise ValueError(f"sparse_point_anchor must be 'previous', 'birth' or 'fused', got '{self._config.sparse_point_anchor}'")  # Clear error.
        if self._config.depth_prior and depth_estimator is None:  # Priors need a network.
            raise ValueError("depth_prior requires a depth estimator (e.g. --depth depth_anything_v2)")  # Clear error.
        self._depth_estimator = depth_estimator if self._config.depth_prior else None  # Only used for priors.
        self._masker = masker if self._config.dynamic_mask else None  # Movable-object masking.
        self.reset()  # Initialise state.

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

    def reset(self) -> None:
        """Forget all state before a new sequence."""
        self._frontend.reset()  # Front-end state.
        self._previous: tuple[ImageArray, ImageArray] | None = None  # Previous (left, right) images.
        self._tracks = StereoTracks.empty()  # Pairs in the previous frame.
        self._anchors: FloatArray = np.zeros((0, 3), dtype=np.float64)  # World point per track (birth anchoring; NaN = no depth).
        self._anchor_covariances: FloatArray = np.zeros((0, 3, 3), dtype=np.float64)  # Their world-frame covariances.
        self._last_pose = np.eye(4, dtype=np.float64)  # T_world_prev.
        self._velocity = np.eye(4, dtype=np.float64)  # T_prev_cur of the last step.
        self._trajectory = Trajectory()  # Output.
        self._num_failures = 0  # Counters.
        self._num_keyframes = 0  # Counters.
        self._ids: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)  # Persistent id per track (parallel to the tracks).
        self._next_id = 0  # Next unused id.
        self._frames_since_window_keyframe = 0  # Window keyframe spacing.
        self._window = self._new_window()  # Windowed BA state (None when disabled).
        self._dynamic_mask: BoolArray | None = None  # Movable-object mask of the current frame.

    def _new_window(self) -> KeyframeWindow | None:
        """Create an empty keyframe window, or None when windowed BA is disabled.

        Returns:
            The window.
        """
        if self._config.window_size < 2:  # Disabled.
            return None
        prior = DepthPriorConfig(  # Gating settings from the odometry config.
            rel_sigma=self._config.depth_prior_rel_sigma,
            gate_sigmas=self._config.depth_prior_gate_sigmas,
            scale_align=self._config.depth_prior_scale_align,
        )
        ba = WindowBAConfig(max_iterations=self._config.window_iterations, pixel_sigma=self._config.pnp.lm_pixel_sigma)  # BA settings.
        return KeyframeWindow(self._config.window_size, ba, prior)  # Empty window.

    def _window_keyframe(self, frame: StereoFrame, pose: FloatArray) -> tuple[FloatArray | None, WindowStats, float, float]:
        """Insert the current frame into the window and run windowed BA.

        Args:
            frame: Current frame (its tracks are already in ``self._tracks``).
            pose: Current T_world,cam.

        Returns:
            Refined T_world,cam (None if BA did not run), statistics, network time (ms) and BA time (ms).
        """
        assert self._window is not None  # Caller checks.
        depth_ms = 0.0  # Network time.
        depth_map = None  # No priors by default.
        if self._depth_estimator is not None:  # Monocular metric depth for this keyframe.
            start = time.perf_counter()  # Time the network.
            depth_map = self._depth_estimator.compute(frame.left, frame.right, self._camera)  # (H, W) metres.
            depth_ms = _elapsed_ms(start)  # Network time.
        start = time.perf_counter()  # Time BA (including bookkeeping).
        pixels = np.column_stack([self._tracks.left, self._tracks.right[:, 0]]).astype(np.float64)  # (u_l, v_l, u_r).
        stats = self._window.add_keyframe(frame.index, pose, self._ids, pixels, self._anchors, self._anchor_covariances, depth_map)  # Insert.
        refined, stats = self._window.optimise(self._camera_matrix, self._camera.baseline, stats)  # BA.
        if refined is not None:  # Refined landmarks become the tracking anchors.
            stored = [self._window.point(int(i)) for i in self._ids]  # Window estimates (None if not in the window).
            updated = np.array([p is not None for p in stored], dtype=bool)  # Which tracks have them.
            if updated.any():  # Copy them over.
                self._anchors = self._anchors.copy()  # Do not alias.
                self._anchors[updated] = np.array([p for p in stored if p is not None])  # Refined points.
        return refined, stats, depth_ms, _elapsed_ms(start)  # Report.

    def _replenish(self, frame: StereoFrame, tracks: StereoTracks) -> tuple[StereoTracks, bool, float]:
        """Detect new stereo pairs when tracks run low.

        Args:
            frame: Current frame.
            tracks: Surviving pairs in the current frame.

        Returns:
            Updated tracks, whether detection ran, and its time in milliseconds.
        """
        if len(tracks) >= self._frontend.min_tracks:  # Enough tracks.
            return tracks, False, 0.0  # Nothing to do.
        start = time.perf_counter()  # Time stereo detection + matching.
        new = self._frontend.detect(frame.left, frame.right, tracks, self._dynamic_mask)  # New pairs away from existing ones and moving objects.
        self._num_keyframes += 1  # Count replenishment frames.
        return tracks.concatenate(new), True, _elapsed_ms(start)  # Combined tracks.

    def _to_world(self, tracks: StereoTracks, world_from_camera: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Triangulate stereo pairs and express points and covariances in the world frame.

        Args:
            tracks: Stereo pairs observed by the camera at ``world_from_camera``.
            world_from_camera: T_world_cam of that frame.

        Returns:
            (N, 3) world points (NaN rows where depth is invalid) and (N, 3, 3) covariances.
        """
        points, valid = triangulate_rectified(tracks.left, tracks.disparity, self._camera)  # Camera-frame points.
        covariances = np.full((len(tracks), 3, 3), np.nan, dtype=np.float64)  # Default: invalid.
        covariances[valid] = triangulation_covariance(
            tracks.left[valid], tracks.disparity[valid], self._camera, self._config.stereo_sigma_px
        )  # Camera frame.
        rotation, translation = world_from_camera[:3, :3], world_from_camera[:3, 3]  # Pose parts.
        return points @ rotation.T + translation, rotation @ covariances @ rotation.T  # World frame (NaN propagates).

    def _set_tracks(
        self,
        tracks: StereoTracks,
        world_from_camera: FloatArray,
        anchors: tuple[FloatArray, FloatArray] | None,
        ids: npt.NDArray[np.int64] | None = None,
    ) -> None:
        """Store tracks for the next frame, anchoring new ones at the current pose.

        Args:
            tracks: All pairs in the current frame (survivors first, then new detections).
            world_from_camera: Current T_world_cam.
            anchors: Anchors of the leading survivor rows, or None if there are none.
            ids: Ids of the leading survivor rows (new rows get fresh ids).
        """
        kept = 0 if anchors is None else len(anchors[0])  # Survivors with existing anchors.
        fresh = np.arange(self._next_id, self._next_id + len(tracks) - kept, dtype=np.int64)  # Ids for new tracks.
        self._next_id += fresh.size  # Reserve them.
        self._ids = fresh if ids is None or kept == 0 else np.concatenate([ids[:kept], fresh])  # Parallel to the tracks.
        new_points, new_cov = self._to_world(tracks.subset(np.arange(len(tracks)) >= kept), world_from_camera)  # Anchor new tracks here.
        self._tracks = tracks  # Pairs.
        self._anchors = new_points if anchors is None else np.vstack([anchors[0], new_points])  # Points.
        self._anchor_covariances = new_cov if anchors is None else np.concatenate([anchors[1], new_cov])  # Covariances.

    def _fuse(
        self, anchors: tuple[FloatArray, FloatArray], tracks: StereoTracks, world_from_camera: FloatArray
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Fuse a new stereo triangulation of each track into its anchor (information-form update).

        X = (S1^-1 + S2^-1)^-1 (S1^-1 X1 + S2^-1 X2): far, uncertain anchors sharpen as tracks
        are re-observed, while precise ones barely move.

        Args:
            anchors: Current anchor points and covariances (world frame), one per track.
            tracks: The same tracks observed in the current frame.
            world_from_camera: Current T_world_cam.

        Returns:
            Fused anchor points and covariances, and an (N,) mask of tracks whose new observation is consistent
            with their landmark (squared Mahalanobis distance below ``dynamic_gate_chi2``).
        """
        points, covariances = anchors  # Prior anchors.
        new_points, new_cov = self._to_world(tracks, world_from_camera)  # Fresh observation.
        ok = np.isfinite(new_points).all(axis=1) & np.isfinite(points).all(axis=1)  # Both valid.
        consistent = np.ones(len(points), dtype=bool)  # Untestable tracks count as consistent.
        if not ok.any():  # Nothing to fuse.
            return points, covariances, consistent  # Unchanged.
        diff = new_points[ok] - points[ok]  # Landmark vs. new triangulation.
        d2 = np.einsum("ni,nij,nj->n", diff, np.linalg.inv(covariances[ok] + new_cov[ok]), diff)  # Squared Mahalanobis distance.
        consistent[ok] = d2 < self._config.dynamic_gate_chi2  # Static, unoccluded points agree.
        info_old, info_new = np.linalg.inv(covariances[ok]), np.linalg.inv(new_cov[ok])  # Information matrices.
        fused_cov = np.linalg.inv(info_old + info_new)  # Combined covariance.
        fused = np.einsum(
            "nij,nj->ni", fused_cov, np.einsum("nij,nj->ni", info_old, points[ok]) + np.einsum("nij,nj->ni", info_new, new_points[ok])
        )  # Weighted mean.
        points, covariances = points.copy(), covariances.copy()  # Do not modify the caller's arrays.
        points[ok], covariances[ok] = fused, fused_cov  # Update.
        return points, covariances, consistent  # Fused anchors and the consistency test.

    def _project(self, camera_from_reference: FloatArray, points: FloatArray) -> tuple[Float32Array, np.ndarray]:
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
        pixels = np.column_stack([self._camera.fx * p[:, 0] / z + self._camera.cx, self._camera.fy * p[:, 1] / z + self._camera.cy])  # Pinhole.
        return pixels.astype(np.float32), ok  # Pixels and mask.

    def process(self, frame: StereoFrame) -> TrackingResult:
        """Process the next frame.

        Args:
            frame: Rectified stereo frame.

        Returns:
            Estimated pose and measurements.
        """
        if self._previous is None:  # First frame: detect, define the world origin.
            start = time.perf_counter()  # Time the step.
            if self._masker is not None:  # Movable objects in the first frame.
                self._dynamic_mask = self._masker.mask(frame.left)
            tracks, _, depth_ms = self._replenish(frame, StereoTracks.empty())  # Initial stereo pairs.
            self._previous = (frame.left, frame.right)  # Remember images.
            pose = np.eye(4, dtype=np.float64)  # World = first camera.
            self._set_tracks(tracks, pose, None)  # Anchor them at the origin.
            if self._window is not None:  # First window keyframe (gauge).
                _, _, prior_ms, _ = self._window_keyframe(frame, pose)  # Nothing to optimise yet.
                depth_ms += prior_ms  # Network time.
            self._trajectory.append(frame.timestamp, pose)  # Record.
            stats = FrameStats(frame.index, frame.timestamp, 0, 0, 0, float("nan"), True, True, 0.0, 0.0, depth_ms, _elapsed_ms(start))  # Init stats.
            return TrackingResult(pose, stats)  # Report.
        return self._track(frame)  # Normal tracking.

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
        track_ids = self._ids  # Their ids.
        if self._config.sparse_point_anchor in ("birth", "fused"):  # Points anchored where each track was born (and fused since).
            points_world, cov_world = self._anchors, self._anchor_covariances  # Stored anchors.
        else:  # Pure frame-to-frame: re-triangulate in the previous frame.
            points_world, cov_world = self._to_world(tracks, self._last_pose)  # Fresh points.
        has_depth: BoolArray = np.asarray(np.isfinite(points_world).all(axis=1), dtype=bool)  # Tracks with a valid 3D point.
        predicted_pose = self._last_pose @ self._velocity  # Constant-velocity prediction of T_world_cur.
        prior = invert_pose(predicted_pose)  # Predicted T_cur_world (the PnP unknown).
        predicted_left, predicted_ok = self._project(prior, points_world)  # Where tracks should appear.
        predicted_left = np.where(predicted_ok[:, None], predicted_left, tracks.left).astype(np.float32)  # Fall back to zero motion.

        match_start = time.perf_counter()  # Front-end timer.
        current, keep = self._frontend.track(self._previous, (frame.left, frame.right), tracks, predicted_left)  # Circular KLT.
        self._frontend.synchronize()  # Honest timing for GPU front ends.
        dynamic_rejected = 0  # Tracks dropped on movable objects or by the dynamic/occlusion gate.
        if self._masker is not None:  # Stop tracking points on people and vehicles.
            self._dynamic_mask = self._masker.mask(frame.left)  # Current frame.
            on_dynamic = pixels_in_mask(self._dynamic_mask, current.left)  # Tracks now on a movable object.
            keep = keep.copy()  # Do not modify the front end's array.
            keep[np.flatnonzero(keep)[on_dynamic]] = False  # Drop them...
            current = current.subset(~on_dynamic)  # ...from both views of the track list.
            dynamic_rejected += int(on_dynamic.sum())  # Count them.
        match_ms = _elapsed_ms(match_start)  # Front-end time.

        pose_start = time.perf_counter()  # Pose timer.
        usable = has_depth[keep]  # Survivors with a valid previous triangulation.
        points_3d = points_world[keep][usable]  # 3D points (world frame).
        covariances = cov_world[keep][usable]  # Their uncertainty.
        observed = current.subset(usable)  # Current left/right observations.
        estimate = self._pose_estimator.estimate(  # PnP + RANSAC, then rig LM.
            points_3d,
            observed.left.astype(np.float64),
            self._camera_matrix,
            prior,
            points_2d_right=observed.right.astype(np.float64) if self._config.lm_right_residuals else None,  # Rig or mono LM.
            baseline=self._camera.baseline,
            covariances=covariances if self._config.lm_covariance_weighting else None,  # Whitening on/off.
        )
        pose_ms = _elapsed_ms(pose_start)  # Pose time.

        if estimate.success:  # Pose found.
            pose = invert_pose(estimate.camera_from_reference)  # T_world_cur.
            inliers = estimate.refined_inlier_mask if estimate.refined_inlier_mask is not None else estimate.inlier_mask  # Best available inlier set.
            survivors = observed.subset(inliers)  # Outliers (and moving objects) stop being tracked.
            survivor_anchors: tuple[FloatArray, FloatArray] = (points_3d[inliers], covariances[inliers])  # Their anchors travel with them.
            survivor_ids = track_ids[keep][usable][inliers]  # And their ids.
            if self._config.sparse_point_anchor == "fused":  # Fuse this frame's stereo triangulation into each anchor.
                fused_points, fused_cov, consistent = self._fuse(survivor_anchors, survivors, pose)  # Inverse-covariance update.
                survivor_anchors = (fused_points, fused_cov)  # Fused anchors.
                if self._config.dynamic_gate:  # Moving points and occlusion jumps fail the consistency test.
                    survivors = survivors.subset(consistent)  # Stop tracking them.
                    survivor_anchors = (fused_points[consistent], fused_cov[consistent])  # Keep anchors aligned.
                    survivor_ids = survivor_ids[consistent]  # And ids.
                    dynamic_rejected += int((~consistent).sum())  # Count them.
        else:  # Tracking failure.
            pose = orthonormalize_pose(predicted_pose)  # Coast. Re-orthonormalised so long coasts stay on SO(3).
            self._num_failures += 1  # Count it.
            survivors = StereoTracks.empty()  # Re-detect everything.
            survivor_anchors = (np.zeros((0, 3)), np.zeros((0, 3, 3)))  # No anchors survive.
            survivor_ids = np.zeros(0, dtype=np.int64)  # No ids survive.
            self._window = self._new_window()  # Start a new BA segment after a failure.
        previous_pose = self._last_pose  # T_world_prev, for the motion model.
        new_tracks, replenished, depth_ms = self._replenish(frame, survivors)  # Top up features.
        if not estimate.success and not replenished:  # Always re-detect after a failure.
            new_tracks = self._frontend.detect(frame.left, frame.right, StereoTracks.empty(), self._dynamic_mask)  # Fresh start.
        self._set_tracks(new_tracks, pose, survivor_anchors, survivor_ids)  # Keep survivors' anchors and ids, anchor new tracks here.
        window_stats, ba_ms = WindowStats(), 0.0  # No BA by default.
        self._frames_since_window_keyframe += 1  # Keyframe spacing.
        if self._window is not None and (
            self._frames_since_window_keyframe >= self._config.window_keyframe_interval or self._window.num_keyframes == 0
        ):
            self._frames_since_window_keyframe = 0  # Reset spacing.
            refined, window_stats, prior_ms, ba_ms = self._window_keyframe(frame, pose)  # Insert + BA.
            depth_ms += prior_ms  # Network time counts as depth time.
            if refined is not None:  # Output the BA pose for this keyframe.
                pose = refined  # Refined T_world,cam.
        self._velocity = invert_pose(previous_pose) @ pose  # Update the motion model.
        self._last_pose = pose  # Remember.
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
        self._trajectory.append(frame.timestamp, pose)  # Record.
        stats = FrameStats(  # Measurements.
            frame_index=frame.index,
            timestamp=frame.timestamp,
            num_matches=int(keep.sum()),  # Tracks surviving the circular check.
            num_matches_with_depth=int(usable.sum()),  # With valid previous depth.
            num_pnp_inliers=estimate.num_inliers,  # RANSAC inliers.
            essential_inlier_ratio=ratio,
            tracking_ok=estimate.success,
            is_keyframe=replenished,  # Replenishment frame.
            match_ms=match_ms,
            pose_ms=pose_ms,
            depth_ms=depth_ms,  # Stereo detection + matching (+ depth network) time.
            total_ms=total_ms,
            ba_ms=ba_ms,
            ba_landmarks=window_stats.landmarks,
            prior_factors=window_stats.prior_factors,
            prior_scale=window_stats.prior_scale,
            dynamic_rejected=dynamic_rejected,
        )
        inliers_all = np.zeros(len(current), dtype=bool)  # Over the circular-check survivors.
        if estimate.success:  # Only a found pose has inliers.
            inliers_all[np.flatnonzero(usable)[inliers]] = True  # Map back.
        camera_from_world = invert_pose(pose)  # T_cur_world.
        added = new_tracks.subset(np.arange(len(new_tracks)) >= len(survivors))  # This frame's detections.
        debug = FrameDebug(
            frame.index - 1,  # Previous frame.
            tracks.left[keep],
            current.left,
            current.right,
            points_world[keep] @ camera_from_world[:3, :3].T + camera_from_world[:3, 3],  # Anchors seen from here (NaN kept).
            inliers_all,
            added.left,
            added.right,
        )
        return TrackingResult(pose, stats, debug)  # Report.

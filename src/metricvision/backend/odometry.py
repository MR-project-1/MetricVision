from __future__ import annotations  # Postponed evaluation of type hints.

import time  # High-resolution timers for runtime measurements.
from collections.abc import Mapping  # Input type for from_dict.
from dataclasses import dataclass, field  # Config and result containers.
from typing import Any  # Loosely typed config dictionaries.

import numpy as np  # Array maths.

from metricvision.backend.keyframe import KeyframeConfig, KeyframePolicy  # Keyframe selection.
from metricvision.backend.pose import PnPConfig, PnPPoseEstimator  # Robust pose estimation.
from metricvision.backend.stats import FrameStats  # Per-frame measurements.
from metricvision.core.geometry import invert_pose  # Rigid inverse.
from metricvision.core.stereo import triangulation_covariance  # Depth uncertainty for the LM weighting.
from metricvision.core.types import (  # Shared types.
    BoolArray,
    Float32Array,
    FloatArray,
    ImageView,
    StereoCamera,
    StereoFrame,
    Trajectory,
)
from metricvision.depth.base import DepthEstimator  # Fixed depth source.
from metricvision.evaluation.frontend_metrics import essential_inlier_ratio  # Front-end-only metric.
from metricvision.frontend.base import FeatureMatcher  # The component under test.


@dataclass(frozen=True)
class OdometryConfig:
    """Back-end settings. Keep identical across extractors so only the front end varies.

    Attributes:
        pnp: Robust PnP settings.
        keyframe: Keyframe selection thresholds.
        essential_threshold_px: Inlier threshold for the essential-matrix quality metric.
        compute_essential_inlier_ratio: Whether to compute that metric (adds time, excluded from runtime).
        stereo_sigma_px: Pixel noise assumed for stereo measurements when propagating 3D point covariance.
        lm_right_residuals: Sparse pipeline only: include current right-image residuals in the LM (rigid rig) or not (mono).
        lm_covariance_weighting: Whiten LM residuals with the propagated point covariance (else pixel noise only).
        sparse_point_anchor: Sparse pipeline only: ``previous`` re-triangulates every track in the previous frame
            (pure frame-to-frame), ``birth`` keeps each track's 3D point (in world coordinates) from the frame
            where it was first triangulated, so long tracks constrain many poses without chaining their errors;
            ``fused`` additionally fuses each new stereo observation into the anchor by inverse covariance.
        anchor_param: Sparse pipeline: landmark model of birth/fused anchors, ``xyz`` (Gaussian world point) or
            ``inverse_depth`` (point on the birth ray, inverse depth fused in 1-D; faithful for far points).
        anchor_pose_inflation: Add the triangulating camera's pose uncertainty (LM covariance) to every new
            point observation, so repeated observations do not make anchors over-confident.
        noise_model: ``fixed`` uses ``stereo_sigma_px`` throughout; ``adaptive`` re-estimates the pixel noise from
            the whitened LM residuals of the inliers (robust median), so triangulation covariance, LM weighting
            and chi^2 gates stay calibrated on every dataset.
        noise_adapt_rate: Exponential smoothing factor of the adaptive noise estimate.
        noise_bounds_px: Clamp of the adaptive noise estimate.
        max_depth_baselines: Depth gate for pose estimation: points farther than this many baselines are not used
            (0 disables; the classic heuristic that covariance weighting should beat).
        realign_every: Re-align long tracks against their birth template every N frames (0 disables).
        realign_min_age: Only tracks at least this many frames old are re-aligned.
        realign_max_shift_px: Larger corrections are treated as failures and the track is dropped.
        rig_mode: ``rigid`` (one pose, both images through the fixed baseline), ``independent`` (left and right
            camera poses estimated separately; the left one is reported and the implied extrinsic deviation
            is recorded) or ``averaged`` (the two independent estimates averaged on SE(3)).
        pose_method: Sparse pipeline, rigid mode: estimator applied to the RANSAC inliers. ``pnp`` (the configured
            PnP refinement, e.g. the rigid-rig LM), ``umeyama3d`` (unweighted 3D-3D alignment), ``mahalanobis3d``
            (covariance-weighted 3D-3D) or ``soft`` (rotation from the essential matrix, translation from stereo).
        ba_window: Sparse pipeline: keyframes in the sliding-window bundle adjustment (0 disables it; 5-7 typical).
        ba_keyframe_every: New keyframe every N frames...
        ba_min_translation_m: ...or after this much camera motion.
        ba_iterations: Levenberg-Marquardt iterations per window optimisation.
    """

    pnp: PnPConfig = field(default_factory=PnPConfig)  # PnP settings.
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)  # Keyframe thresholds.
    essential_threshold_px: float = 1.0  # Metric threshold in pixels.
    compute_essential_inlier_ratio: bool = True  # Metric on/off.
    stereo_sigma_px: float = 0.5  # Disparity/pixel noise for covariance propagation.
    lm_right_residuals: bool = True  # Rigid-rig (stereo) residuals on/off.
    lm_covariance_weighting: bool = True  # Covariance whitening on/off.
    sparse_point_anchor: str = "previous"  # previous | birth | fused.
    anchor_param: str = "xyz"  # xyz | inverse_depth.
    anchor_pose_inflation: bool = False  # Pose uncertainty into point covariance.
    noise_model: str = "fixed"  # fixed | adaptive.
    noise_adapt_rate: float = 0.05  # EMA factor.
    noise_bounds_px: tuple[float, float] = (0.1, 3.0)  # Clamp.
    max_depth_baselines: float = 0.0  # Depth gate (0 = off).
    realign_every: int = 0  # Track re-alignment period (0 = off).
    realign_min_age: int = 3  # Minimum track age for re-alignment.
    realign_max_shift_px: float = 2.0  # Re-alignment acceptance.
    rig_mode: str = "rigid"  # rigid | independent | averaged.
    pose_method: str = "pnp"  # pnp | umeyama3d | mahalanobis3d | soft.
    ba_window: int = 0  # Keyframe bundle-adjustment window (0 = off).
    ba_keyframe_every: int = 3  # Keyframe period (frames).
    ba_min_translation_m: float = 1.0  # ...or keyframe after this much motion.
    ba_iterations: int = 6  # LM iterations per window optimisation.

    def __post_init__(self) -> None:
        """Validate option names.

        Raises:
            ValueError: If an option has an unknown value.
        """
        choices = {
            "anchor_param": ("xyz", "inverse_depth"),
            "noise_model": ("fixed", "adaptive"),
            "rig_mode": ("rigid", "independent", "averaged"),
            "pose_method": ("pnp", "umeyama3d", "mahalanobis3d", "soft"),
        }  # Allowed values.
        for name, allowed in choices.items():  # Check each option.
            if getattr(self, name) not in allowed:
                raise ValueError(f"odometry.{name} must be one of {allowed}, got '{getattr(self, name)}'")
        object.__setattr__(self, "noise_bounds_px", tuple(float(v) for v in self.noise_bounds_px))  # YAML lists -> tuple.

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> OdometryConfig:
        """Build a config from a nested dictionary (e.g. parsed YAML).

        Args:
            data: Mapping with optional ``pnp`` and ``keyframe`` sub-mappings and scalar options.

        Returns:
            The parsed configuration.
        """
        values = dict(data or {})  # Copy so we can pop keys safely.
        pnp = PnPConfig(**(values.pop("pnp", None) or {}))  # Nested PnP settings.
        keyframe = KeyframeConfig(**(values.pop("keyframe", None) or {}))  # Nested keyframe settings.
        return cls(pnp=pnp, keyframe=keyframe, **values)  # Remaining keys are scalar options.


@dataclass(frozen=True)
class Keyframe:
    """A reference frame that subsequent frames are tracked against.

    Attributes:
        frame_index: Index of the frame that became the keyframe.
        view: Left image with identifier (so the front end can reuse cached features).
        depth: Stereo depth map of the left image, metres, NaN where invalid.
        world_from_camera: Pose T_world_kf.
    """

    frame_index: int  # Source frame.
    view: ImageView  # Left image.
    depth: Float32Array  # Depth map.
    world_from_camera: FloatArray  # Pose.


@dataclass(frozen=True)
class TrackingResult:
    """Output of processing one frame.

    Attributes:
        world_from_camera: Estimated pose T_world_cam of the left camera.
        stats: Measurements recorded for this frame.
    """

    world_from_camera: FloatArray  # Pose estimate.
    stats: FrameStats  # Measurements.


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading.

    Args:
        start: Earlier perf_counter value.

    Returns:
        Elapsed time in milliseconds.
    """
    return (time.perf_counter() - start) * 1000.0  # Seconds to milliseconds.


class StereoVisualOdometry:
    """Keyframe-based stereo VO where only the front end changes between experiments."""

    def __init__(
        self,
        camera: StereoCamera,
        matcher: FeatureMatcher,
        depth_estimator: DepthEstimator,
        config: OdometryConfig | None = None,
    ) -> None:
        """Wire the pipeline together.

        Args:
            camera: Rectified stereo camera model.
            matcher: Front end under test.
            depth_estimator: Fixed stereo depth source.
            config: Back-end settings; defaults are used when omitted.
        """
        self._camera = camera  # Camera model.
        self._camera_matrix = camera.intrinsic_matrix  # Cached K.
        self._matcher = matcher  # Front end.
        self._depth_estimator = depth_estimator  # Depth source.
        self._config = config or OdometryConfig()  # Settings.
        self._pose_estimator = PnPPoseEstimator(self._config.pnp)  # PnP solver.
        self._keyframe_policy = KeyframePolicy(self._config.keyframe)  # Keyframe policy.
        self._keyframe: Keyframe | None = None  # No keyframe until the first frame arrives.
        self._last_pose = np.eye(4, dtype=np.float64)  # Pose of the previous frame, T_world_prev.
        self._velocity = np.eye(4, dtype=np.float64)  # Last inter-frame motion, T_prev_cur.
        self._trajectory = Trajectory()  # Estimated left-camera poses.
        self._num_failures = 0  # Tracking-failure counter.
        self._num_keyframes = 0  # Keyframe counter.

    @property
    def trajectory(self) -> Trajectory:
        """Return the estimated trajectory of the left camera (world = first camera).

        Returns:
            Trajectory of T_world_cam poses, one per processed frame.
        """
        return self._trajectory  # Accumulated poses.

    @property
    def num_failures(self) -> int:
        """Return the number of tracking failures so far.

        Returns:
            Failure count.
        """
        return self._num_failures  # Counter value.

    @property
    def num_keyframes(self) -> int:
        """Return the number of keyframes created so far.

        Returns:
            Keyframe count.
        """
        return self._num_keyframes  # Counter value.

    def reset(self) -> None:
        """Forget all state so a new sequence can be processed."""
        self._matcher.reset()  # Drop cached features.
        self._keyframe = None  # Force re-initialisation.
        self._last_pose = np.eye(4, dtype=np.float64)  # Restart at the origin.
        self._velocity = np.eye(4, dtype=np.float64)  # Zero motion prior.
        self._trajectory = Trajectory()  # Empty trajectory.
        self._num_failures = 0  # Reset counters.
        self._num_keyframes = 0  # Reset counters.

    def process(self, frame: StereoFrame) -> TrackingResult:
        """Process the next frame of the sequence.

        Args:
            frame: Rectified stereo frame.

        Returns:
            The estimated pose and this frame's measurements.
        """
        if self._keyframe is None:  # First frame of the sequence.
            return self._initialize(frame)  # Becomes the first keyframe at the origin.
        return self._track(frame)  # Normal tracking.

    def _create_keyframe(self, frame: StereoFrame, world_from_camera: FloatArray) -> float:
        """Make ``frame`` the new keyframe, computing its stereo depth.

        Args:
            frame: Frame to promote.
            world_from_camera: Its pose T_world_cam.

        Returns:
            Time spent computing depth, in milliseconds.
        """
        start = time.perf_counter()  # Time the depth computation.
        depth = self._depth_estimator.compute(frame.left, frame.right, self._camera)  # Dense depth for the left image.
        depth_ms = _elapsed_ms(start)  # Stop the timer.
        self._keyframe = Keyframe(frame.index, ImageView(frame.index, frame.left), depth, world_from_camera.copy())  # Store the keyframe.
        self._num_keyframes += 1  # Count it.
        return depth_ms  # Report the cost.

    def _initialize(self, frame: StereoFrame) -> TrackingResult:
        """Handle the first frame: it defines the world origin and becomes the first keyframe.

        Args:
            frame: First frame of the sequence.

        Returns:
            Identity pose and initial measurements.
        """
        start = time.perf_counter()  # Time the whole step.
        pose = np.eye(4, dtype=np.float64)  # World frame = first camera frame.
        depth_ms = self._create_keyframe(frame, pose)  # First keyframe.
        self._last_pose = pose  # Seed the motion model.
        self._trajectory.append(frame.timestamp, pose)  # Record the pose.
        stats = FrameStats(  # Nothing is matched on the first frame.
            frame_index=frame.index,
            timestamp=frame.timestamp,  # Identification.
            num_matches=0,
            num_matches_with_depth=0,
            num_pnp_inliers=0,  # No matching yet.
            essential_inlier_ratio=float("nan"),  # Undefined without a pair.
            tracking_ok=True,
            is_keyframe=True,  # Initialisation always succeeds.
            match_ms=0.0,
            pose_ms=0.0,
            depth_ms=depth_ms,
            total_ms=_elapsed_ms(start),  # Timings.
        )
        return TrackingResult(pose, stats)  # Report.

    def _lift_to_3d(self, pixels: Float32Array, depth: Float32Array) -> tuple[FloatArray, BoolArray]:
        """Back-project keyframe pixels to 3D points using the depth map.

        Depth is sampled with nearest-neighbour lookup; bilinear interpolation
        would blend foreground and background depths at object edges.

        Args:
            pixels: (N, 2) pixel coordinates in the keyframe.
            depth: Keyframe depth map in metres (NaN = invalid).

        Returns:
            A tuple (points, valid): (M, 3) points for the M valid pixels, and an (N,) mask of which pixels were valid.
        """
        columns = np.rint(pixels[:, 0]).astype(np.int64)  # Nearest pixel column.
        rows = np.rint(pixels[:, 1]).astype(np.int64)  # Nearest pixel row.
        inside = (columns >= 0) & (columns < depth.shape[1]) & (rows >= 0) & (rows < depth.shape[0])  # Within the image.
        z = np.full(pixels.shape[0], np.nan, dtype=np.float64)  # Default: invalid.
        z[inside] = depth[rows[inside], columns[inside]]  # Sample depth where possible.
        valid = np.isfinite(z)  # Keep only pixels with a valid depth.
        x = (pixels[valid, 0] - self._camera.cx) * z[valid] / self._camera.fx  # Pinhole back-projection, x.
        y = (pixels[valid, 1] - self._camera.cy) * z[valid] / self._camera.fy  # Pinhole back-projection, y.
        return np.column_stack([x, y, z[valid]]), valid  # (M, 3) points plus mask.

    def _track(self, frame: StereoFrame) -> TrackingResult:
        """Estimate the pose of a frame relative to the current keyframe.

        Args:
            frame: Frame to track.

        Returns:
            Estimated pose and measurements.
        """
        assert self._keyframe is not None  # Guaranteed by process().
        keyframe = self._keyframe  # Local alias.
        start = time.perf_counter()  # Time the whole step.

        match_start = time.perf_counter()  # Time the front end.
        matches = self._matcher.match(keyframe.view, ImageView(frame.index, frame.left))  # Keyframe -> current correspondences.
        self._matcher.synchronize()  # Wait for GPU work so the timing is honest.
        match_ms = _elapsed_ms(match_start)  # Front-end time.

        pose_start = time.perf_counter()  # Time lifting + PnP.
        points_3d, valid = self._lift_to_3d(matches.points_a, keyframe.depth)  # 3D points in the keyframe frame.
        predicted_pose = self._last_pose @ self._velocity  # Constant-velocity prediction of T_world_cur.
        prior = invert_pose(predicted_pose) @ keyframe.world_from_camera  # Same prediction as T_cur_kf, the PnP unknown.
        covariances = None  # Only the LM refinement uses point covariances.
        if (
            self._config.pnp.refine_method == "lm" and self._config.lm_covariance_weighting and len(points_3d) > 0
        ):  # Recover disparity from SGBM depth and propagate pixel noise.
            disparity = self._camera.fx * self._camera.baseline / points_3d[:, 2]  # d = fx B / Z.
            covariances = triangulation_covariance(matches.points_a[valid], disparity, self._camera, self._config.stereo_sigma_px)  # (M, 3, 3).
        estimate = self._pose_estimator.estimate(  # Robust PnP (+ optional covariance-weighted LM).
            points_3d,
            matches.points_b[valid].astype(np.float64),
            self._camera_matrix,
            prior,
            covariances=covariances,
            pixel_sigma=self._config.stereo_sigma_px,  # One noise model for triangulation covariance and LM weighting.
        )
        pose_ms = _elapsed_ms(pose_start)  # Pose time.

        if estimate.success:  # PnP found a well-supported pose.
            pose = keyframe.world_from_camera @ invert_pose(estimate.camera_from_reference)  # T_world_cur = T_world_kf @ T_kf_cur.
        else:  # Tracking failure.
            pose = predicted_pose  # Coast on the motion model.
            self._num_failures += 1  # Count the failure.
        self._velocity = invert_pose(self._last_pose) @ pose  # Update the motion model (unchanged on failure).
        self._last_pose = pose  # Remember for the next prediction.

        frames_since = frame.index - keyframe.frame_index  # Keyframe age.
        insert = (not estimate.success) or self._keyframe_policy.should_insert(  # Re-anchor on failure or when the policy says so.
            invert_pose(keyframe.world_from_camera) @ pose,
            estimate.num_inliers,
            frames_since,  # T_kf_cur, support, age.
        )
        depth_ms = self._create_keyframe(frame, pose) if insert else 0.0  # Depth is only needed for keyframes.
        total_ms = _elapsed_ms(start)  # Stop before measurement-only work below.

        ratio = (  # Front-end quality metric, excluded from runtime.
            essential_inlier_ratio(matches, self._camera_matrix, self._config.essential_threshold_px)  # 2D-2D RANSAC.
            if self._config.compute_essential_inlier_ratio  # Only if enabled.
            else float("nan")  # Otherwise undefined.
        )

        self._trajectory.append(frame.timestamp, pose)  # Record the pose.
        stats = FrameStats(  # Collect measurements.
            frame_index=frame.index,
            timestamp=frame.timestamp,  # Identification.
            num_matches=len(matches),
            num_matches_with_depth=int(valid.sum()),  # Match counts.
            num_pnp_inliers=estimate.num_inliers,
            essential_inlier_ratio=ratio,  # Quality.
            tracking_ok=estimate.success,
            is_keyframe=insert,  # Flags.
            match_ms=match_ms,
            pose_ms=pose_ms,
            depth_ms=depth_ms,
            total_ms=total_ms,  # Timings.
        )
        return TrackingResult(pose, stats)  # Report.

"""Per-frame measurements recorded by the odometry pipeline."""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import asdict, dataclass  # Record container and dict conversion.


@dataclass(frozen=True)
class FrameStats:
    """Everything measured while processing one frame.

    Attributes:
        frame_index: Frame position in the sequence.
        timestamp: Frame time in seconds.
        num_matches: Correspondences returned by the front end (keyframe -> current).
        num_matches_with_depth: Matches whose keyframe pixel has valid stereo depth.
        num_pnp_inliers: Matches consistent with the PnP pose.
        essential_inlier_ratio: 2D-2D RANSAC inlier ratio (front-end-only quality); NaN if undefined.
        tracking_ok: Whether PnP succeeded; False counts as a tracking failure.
        is_keyframe: Whether this frame became the new keyframe.
        match_ms: Front-end time (extraction + matching).
        pose_ms: 3D lifting + PnP time.
        depth_ms: Stereo depth time (non-zero only on keyframes).
        total_ms: Whole pipeline time for the frame, excluding measurement-only metrics.
        ba_ms: Windowed bundle adjustment time (sparse pipeline keyframes only).
        ba_landmarks: Landmarks optimised by the windowed BA at this frame (0 if it did not run).
        prior_factors: Monocular depth-prior factors in that BA.
        prior_scale: Median stereo / network depth ratio at this keyframe (NaN if not computed).
    """

    frame_index: int  # Position in sequence.
    timestamp: float  # Seconds.
    num_matches: int  # Raw match count.
    num_matches_with_depth: int  # Matches usable by PnP.
    num_pnp_inliers: int  # Geometrically verified matches.
    essential_inlier_ratio: float  # Front-end quality in [0, 1].
    tracking_ok: bool  # Success flag.
    is_keyframe: bool  # Keyframe flag.
    match_ms: float  # Front-end runtime.
    pose_ms: float  # Pose-estimation runtime.
    depth_ms: float  # Depth runtime.
    total_ms: float  # End-to-end runtime.
    ba_ms: float = 0.0  # BA runtime.
    ba_landmarks: int = 0  # BA size.
    prior_factors: int = 0  # Depth priors used.
    prior_scale: float = float("nan")  # Network scale check.

    def to_dict(self) -> dict[str, float | int | bool]:
        """Convert to a plain dictionary for CSV output.

        Returns:
            Field name -> value mapping.
        """
        return asdict(self)  # Dataclass fields as a dict.

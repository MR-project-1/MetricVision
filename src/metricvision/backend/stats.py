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
        vertical_disparity_median_px: Median of v_l - v_r over the tracked stereo pairs (sparse pipeline). In a
            perfectly rectified, rigid rig it is zero; a drift is a direct symptom of a changed extrinsic.
        vertical_disparity_mad_px: Robust spread (1.4826 MAD) of v_l - v_r.
        pose_sigma_trans_mm: 1-sigma translation uncertainty of the frame's pose (LM information, trace-based).
        pose_sigma_rot_deg: 1-sigma rotation uncertainty of the frame's pose.
        noise_sigma_px: Pixel noise used for this frame (changes with ``noise_model: adaptive``).
        extrinsic_dev_trans_mm: Independent/averaged rig modes: translation of the implied T_RL away from calibration.
        extrinsic_dev_rot_deg: Same for rotation.
        num_realigned: Tracks re-aligned against their birth template this frame.
        num_realign_dropped: Tracks dropped because re-alignment failed (occlusion or drift beyond tolerance).
        num_fused: Anchor observations fused this frame.
        ba_ms: Keyframe bundle-adjustment time (0 when no keyframe was inserted).
        num_ba_landmarks: Landmarks in the bundle-adjustment problem (0 when it did not run).
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
    vertical_disparity_median_px: float = float("nan")  # Rig-consistency signal (RQ3).
    vertical_disparity_mad_px: float = float("nan")  # Its robust spread.
    pose_sigma_trans_mm: float = float("nan")  # Pose uncertainty (translation).
    pose_sigma_rot_deg: float = float("nan")  # Pose uncertainty (rotation).
    noise_sigma_px: float = float("nan")  # Noise model in use.
    extrinsic_dev_trans_mm: float = float("nan")  # Implied extrinsic deviation (independent / averaged modes).
    extrinsic_dev_rot_deg: float = float("nan")  # Same, rotation.
    num_realigned: int = 0  # Re-aligned tracks.
    num_realign_dropped: int = 0  # Tracks dropped by re-alignment.
    num_fused: int = 0  # Fused anchor observations.
    ba_ms: float = 0.0  # Bundle-adjustment time.
    num_ba_landmarks: int = 0  # Bundle-adjustment problem size.

    def to_dict(self) -> dict[str, float | int | bool]:
        """Convert to a plain dictionary for CSV output.

        Returns:
            Field name -> value mapping.
        """
        return asdict(self)  # Dataclass fields as a dict.

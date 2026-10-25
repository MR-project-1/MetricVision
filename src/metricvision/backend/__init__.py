"""The fixed back end: stereo depth + PnP tracking against keyframes."""

from metricvision.backend.keyframe import KeyframeConfig, KeyframePolicy  # Keyframe selection.
from metricvision.backend.odometry import OdometryConfig, StereoVisualOdometry, TrackingResult  # Main pipeline.
from metricvision.backend.pose import PnPConfig, PnPPoseEstimator, PoseEstimate  # Pose estimation.
from metricvision.backend.stats import FrameStats  # Per-frame measurements.

__all__ = [  # Public API of the back end.
    "FrameStats",
    "KeyframeConfig",
    "KeyframePolicy",
    "OdometryConfig",
    "PnPConfig",
    "PnPPoseEstimator",
    "PoseEstimate",
    "StereoVisualOdometry",
    "TrackingResult",
]

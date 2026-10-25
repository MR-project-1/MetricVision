from __future__ import annotations

from dataclasses import dataclass  # Config container.

import numpy as np

from metricvision.core.geometry import rotation_angle_deg  # Rotation magnitude.
from metricvision.core.types import FloatArray  # Shared types.


@dataclass(frozen=True)
class KeyframeConfig:
    """Thresholds that trigger a new keyframe. Whichever fires first wins.

    Attributes:
        min_inliers: Insert when PnP inliers drop below this (tracking is getting weak).
        max_translation_m: Insert after moving this far from the keyframe.
        max_rotation_deg: Insert after rotating this much from the keyframe.
        max_frames: Insert after this many frames regardless.
    """

    min_inliers: int = 100  # Weak-tracking trigger.
    max_translation_m: float = 1.0  # Distance trigger.
    max_rotation_deg: float = 15.0  # Rotation trigger.
    max_frames: int = 20  # Age trigger.


class KeyframePolicy:
    """Decides when the current frame should replace the keyframe."""

    def __init__(self, config: KeyframeConfig | None = None) -> None:
        """Store the thresholds.

        Args:
            config: Keyframe thresholds; defaults are used when omitted.
        """
        self._config = config or KeyframeConfig()  # Fall back to defaults.

    def should_insert(self, keyframe_from_current: FloatArray, num_inliers: int, frames_since_keyframe: int) -> bool:
        """Return True if a new keyframe should be created at the current frame.

        Args:
            keyframe_from_current: Pose of the current camera relative to the keyframe, T_kf_cur.
            num_inliers: PnP inliers for the current frame.
            frames_since_keyframe: Frames elapsed since the keyframe.

        Returns:
            Whether to insert a keyframe.
        """
        config = self._config  # Local alias for readability.
        if num_inliers < config.min_inliers:  # Tracking support is thinning out.
            return True  # Re-anchor before tracking is lost.
        if float(np.linalg.norm(keyframe_from_current[:3, 3])) > config.max_translation_m:  # Moved far away.
            return True  # Keyframe overlap is shrinking.
        if rotation_angle_deg(keyframe_from_current[:3, :3]) > config.max_rotation_deg:  # Turned a lot.
            return True  # Keyframe overlap is shrinking.
        return frames_since_keyframe >= config.max_frames  # Age limit as a final fallback.

"""ORB front end: FAST corners with rotated BRIEF binary descriptors."""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # ORB implementation.

from metricvision.frontend.base import MATCHERS, OpenCVFeatureMatcher  # Registry and shared matching logic.


@MATCHERS.register("orb")
class OrbMatcher(OpenCVFeatureMatcher):
    """ORB features matched by Hamming distance."""

    name = "orb"  # Used in output folder names.

    def __init__(
        self,
        max_keypoints: int = 2000,
        scale_factor: float = 1.2,
        num_levels: int = 8,
        fast_threshold: int = 20,
        ratio: float = 0.8,
        mutual_check: bool = True,
    ) -> None:
        """Configure ORB.

        Args:
            max_keypoints: Keypoint budget per image (keep equal across extractors for fairness).
            scale_factor: Scale ratio between pyramid levels.
            num_levels: Number of pyramid levels.
            fast_threshold: FAST corner detection threshold.
            ratio: Lowe ratio-test threshold.
            mutual_check: Whether to require mutual nearest neighbours.
        """
        detector = cv2.ORB.create(  # Build the OpenCV ORB detector/descriptor.
            nfeatures=max_keypoints,  # Keypoint budget.
            scaleFactor=scale_factor,  # Pyramid scale step.
            nlevels=num_levels,  # Pyramid depth.
            fastThreshold=fast_threshold,  # Corner sensitivity.
        )
        super().__init__(detector, cv2.NORM_HAMMING, ratio=ratio, mutual_check=mutual_check)  # Binary descriptors use Hamming distance.

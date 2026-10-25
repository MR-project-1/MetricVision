"""SIFT front end: difference-of-Gaussians keypoints with 128-D gradient histograms."""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # SIFT implementation (in the main OpenCV package since 4.4).

from metricvision.frontend.base import MATCHERS, OpenCVFeatureMatcher  # Registry and shared matching logic.


@MATCHERS.register("sift")
class SiftMatcher(OpenCVFeatureMatcher):
    """SIFT features matched by Euclidean (L2) distance."""

    name = "sift"  # Used in output folder names.

    def __init__(
        self,
        max_keypoints: int = 2000,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        ratio: float = 0.8,
        mutual_check: bool = True,
    ) -> None:
        """Configure SIFT.

        Args:
            max_keypoints: Keypoint budget per image (keep equal across extractors for fairness).
            contrast_threshold: Rejects low-contrast keypoints; lower detects more in dark images.
            edge_threshold: Rejects edge-like keypoints.
            ratio: Lowe ratio-test threshold.
            mutual_check: Whether to require mutual nearest neighbours.
        """
        detector = cv2.SIFT.create(  # Build the OpenCV SIFT detector/descriptor.
            nfeatures=max_keypoints,  # Keypoint budget.
            contrastThreshold=contrast_threshold,  # Contrast filter.
            edgeThreshold=edge_threshold,  # Edge filter.
        )
        super().__init__(detector, cv2.NORM_L2, ratio=ratio, mutual_check=mutual_check)  # Float descriptors use L2 distance.
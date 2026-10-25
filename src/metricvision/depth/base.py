"""Abstract interface for stereo depth estimation.

Depth comes from a fixed stereo algorithm, never from the front end being
tested, so the extractor only influences temporal matching.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from abc import ABC, abstractmethod  # Abstract base class machinery.

from metricvision.core.registry import Registry  # Name -> class lookup for configs.
from metricvision.core.types import Float32Array, ImageArray, StereoCamera  # Shared types.


class DepthEstimator(ABC):
    """Computes a dense depth map for the left image of a rectified stereo pair."""

    @abstractmethod
    def compute(self, left: ImageArray, right: ImageArray, camera: StereoCamera) -> Float32Array:
        """Estimate per-pixel depth.

        Args:
            left: Rectified left grayscale image.
            right: Rectified right grayscale image.
            camera: Rectified stereo camera model.

        Returns:
            Float32 depth map of shape (H, W) in metres; NaN marks invalid pixels.
        """


DEPTH_ESTIMATORS: Registry[DepthEstimator] = Registry("depth estimator")  # Global registry.
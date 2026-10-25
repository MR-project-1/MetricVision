"""Semi-global block matching (SGBM) depth, OpenCV's standard stereo matcher."""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # StereoSGBM implementation.
import numpy as np  # Disparity to depth conversion.

from metricvision.core.types import Float32Array, ImageArray, StereoCamera  # Shared types.
from metricvision.depth.base import DEPTH_ESTIMATORS, DepthEstimator  # Base class and registry.


@DEPTH_ESTIMATORS.register("sgbm")
class SGBMDepthEstimator(DepthEstimator):
    """Depth from OpenCV's semi-global block matcher."""

    def __init__(
        self,
        num_disparities: int = 128,
        block_size: int = 5,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 2,
        min_depth: float = 0.3,
        max_depth: float = 40.0,
    ) -> None:
        """Configure SGBM.

        Args:
            num_disparities: Disparity search range in pixels; must be a multiple of 16.
                Sets the minimum measurable depth: fx * baseline / num_disparities.
            block_size: Matching window size (odd number).
            uniqueness_ratio: Margin (percent) by which the best match must beat the second best.
            speckle_window_size: Maximum size of small disconnected regions treated as noise.
            speckle_range: Maximum disparity variation within a connected region.
            min_depth: Depths below this (metres) are marked invalid.
            max_depth: Depths above this (metres) are marked invalid; far points constrain translation poorly.

        Raises:
            ValueError: If num_disparities is not a positive multiple of 16.
        """
        if num_disparities <= 0 or num_disparities % 16 != 0:  # OpenCV requirement.
            raise ValueError("num_disparities must be a positive multiple of 16")  # Catch config mistakes early.
        channels = 1  # Grayscale input.
        self._matcher = cv2.StereoSGBM.create(  # Build the matcher once and reuse it.
            minDisparity=0,  # Search disparities starting at zero.
            numDisparities=num_disparities,  # Search range.
            blockSize=block_size,  # Window size.
            P1=8 * channels * block_size**2,  # Penalty for disparity changes of 1 (OpenCV's recommended value).
            P2=32 * channels * block_size**2,  # Penalty for larger disparity jumps (recommended value).
            uniquenessRatio=uniqueness_ratio,  # Reject ambiguous matches.
            speckleWindowSize=speckle_window_size,  # Speckle filter window.
            speckleRange=speckle_range,  # Speckle filter range.
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # Faster variant with similar quality.
        )
        self._min_depth = min_depth  # Lower validity bound.
        self._max_depth = max_depth  # Upper validity bound.

    def compute(self, left: ImageArray, right: ImageArray, camera: StereoCamera) -> Float32Array:
        """Compute a depth map for the left image.

        Args:
            left: Rectified left grayscale image.
            right: Rectified right grayscale image.
            camera: Rectified stereo camera model.

        Returns:
            Float32 depth map in metres with NaN for invalid pixels.
        """
        disparity = self._matcher.compute(left, right).astype(np.float32) / 16.0  # SGBM returns fixed-point (x16) disparities.
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)  # Start with everything invalid.
        valid = disparity > 0.5  # Non-positive disparities are failures; tiny ones are too far to trust.
        depth[valid] = (camera.fx * camera.baseline) / disparity[valid]  # Triangulate: Z = fx * B / d.
        depth[(depth < self._min_depth) | (depth > self._max_depth)] = np.nan  # Enforce the validity range.
        return depth  # Metric depth map.
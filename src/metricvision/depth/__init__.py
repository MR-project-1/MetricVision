"""Depth estimators (stereo SGBM, monocular metric networks). Importing this package registers every estimator."""

from metricvision.depth import mono_metric, sgbm  # noqa: F401  # Imported for their registration side effect (torch loads lazily).
from metricvision.depth.base import DEPTH_ESTIMATORS, DepthEstimator  # Public API.

__all__ = ["DEPTH_ESTIMATORS", "DepthEstimator"]  # Names exported by star imports.

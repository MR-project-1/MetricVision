"""Stereo depth estimators. Importing this package registers every estimator."""

from metricvision.depth import sgbm  # noqa: F401  # Imported for its registration side effect.
from metricvision.depth.base import DEPTH_ESTIMATORS, DepthEstimator  # Public API.

__all__ = ["DEPTH_ESTIMATORS", "DepthEstimator"]  # Names exported by star imports.

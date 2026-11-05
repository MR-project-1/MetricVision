"""Stereo calibration: ChArUco/checkerboard detection, joint stereo calibration with uncertainty, rectification.

Ported from the earlier MetricVision prototype. Used by ``scripts/calibrate_charuco.py``, ``scripts/calibrate_stereo.py``,
``scripts/extract_rosbag.py`` and the ``stereo_folder`` dataset (RealSense D455 wheelchair sequences).
"""

from metricvision.calibration.camera import RawStereoCalib, Rectifier

__all__ = ["RawStereoCalib", "Rectifier"]

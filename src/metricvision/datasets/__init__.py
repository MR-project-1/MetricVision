"""Stereo dataset loaders. Importing this package registers every loader in DATASETS."""

from metricvision.datasets import (  # noqa: F401  # Imported for their registration side effect.
    euroc,
    kitti,
    rosbag_stereo,
    synthetic,
)
from metricvision.datasets.base import DATASETS, StereoDataset  # Public API of the package.

__all__ = ["DATASETS", "StereoDataset"]  # Names exported by `from metricvision.datasets import *`.

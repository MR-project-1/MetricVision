"""Abstract interface that every stereo dataset loader implements."""

from __future__ import annotations  # Postponed evaluation of type hints.

from abc import ABC, abstractmethod  # Abstract base class machinery.
from collections.abc import Iterator  # Return type of __iter__.

import numpy as np  # Identity transform for the default body frame.

from metricvision.core.registry import Registry  # Name -> class lookup for configs.
from metricvision.core.types import FloatArray, StereoCamera, StereoFrame, Trajectory  # Shared types.


class StereoDataset(ABC):
    """Base class for rectified stereo sequences.

    Subclasses hide dataset-specific file layouts and calibration formats and
    expose every sequence the same way: a rectified camera model, indexable
    frames, and an optional ground-truth trajectory.
    """

    @property
    @abstractmethod
    def camera(self) -> StereoCamera:
        """Return the rectified stereo camera model.

        Returns:
            Intrinsics and baseline of the rectified pair.
        """

    @property
    @abstractmethod
    def tag(self) -> str:
        """Return a short, filesystem-safe identifier such as ``kitti_00``.

        Returns:
            Identifier used to name output folders.
        """

    @property
    def body_from_camera(self) -> FloatArray:
        """Return T_body_camera: the rectified left camera's pose in the ground-truth body frame.

        The default is identity, which is correct when ground truth is already
        expressed in the left camera frame (as on KITTI).

        Returns:
            4x4 rigid transform.
        """
        return np.eye(4, dtype=np.float64)  # Ground truth and camera frames coincide by default.

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stereo frames in the sequence.

        Returns:
            Frame count.
        """

    @abstractmethod
    def load_frame(self, index: int) -> StereoFrame:
        """Load one rectified stereo frame from disk (or generate it).

        Args:
            index: Zero-based frame index.

        Returns:
            The rectified stereo pair with its timestamp.
        """

    def ground_truth(self) -> Trajectory | None:
        """Return the ground-truth trajectory of the body frame, if the dataset has one.

        Returns:
            Trajectory of T_world_body poses, or None when unavailable.
        """
        return None  # Subclasses override when ground truth exists.

    def __getitem__(self, index: int) -> StereoFrame:
        """Allow ``dataset[i]`` as shorthand for ``dataset.load_frame(i)``.

        Args:
            index: Zero-based frame index.

        Returns:
            The requested frame.
        """
        return self.load_frame(index)  # Delegate to the abstract loader.

    def __iter__(self) -> Iterator[StereoFrame]:
        """Iterate over all frames in order.

        Yields:
            Each stereo frame from first to last.
        """
        for index in range(len(self)):  # Visit every index.
            yield self.load_frame(index)  # Load lazily so memory stays flat.


DATASETS: Registry[StereoDataset] = Registry("dataset")  # Global registry of dataset loaders.

"""Loader for the KITTI odometry benchmark (grayscale stereo).

Expected layout, as extracted from the official archives::

    root/sequences/00/image_0/000000.png   (left, rectified)
    root/sequences/00/image_1/000000.png   (right, rectified)
    root/sequences/00/calib.txt
    root/sequences/00/times.txt
    root/poses/00.txt                      (ground truth, sequences 00-10 only)
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from pathlib import Path  # Filesystem paths.

import cv2  # Image loading.
import numpy as np  # Parsing calibration and poses.

from metricvision.core.types import FloatArray, ImageArray, StereoCamera, StereoFrame, Trajectory  # Shared types.
from metricvision.datasets.base import DATASETS, StereoDataset  # Base class and registry.


def _read_calibration(path: Path) -> dict[str, FloatArray]:
    """Parse a KITTI ``calib.txt`` file.

    Args:
        path: Path to calib.txt.

    Returns:
        Mapping from matrix name (``P0``, ``P1``, ...) to its flattened values.
    """
    calibration: dict[str, FloatArray] = {}  # Output mapping.
    for line in path.read_text().splitlines():  # Each line is "NAME: v1 v2 ...".
        if ":" not in line:  # Skip blank or malformed lines.
            continue  # Nothing to parse.
        key, values = line.split(":", 1)  # Split name from numbers.
        calibration[key.strip()] = np.array([float(v) for v in values.split()], dtype=np.float64)  # Store the numbers.
    return calibration  # All matrices in the file.


def _read_image(path: Path) -> ImageArray:
    """Read one grayscale KITTI image.

    Args:
        path: Image file.

    Returns:
        Grayscale image.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # KITTI odometry grayscale images are 8-bit.
    if image is None:  # OpenCV returns None instead of raising.
        raise FileNotFoundError(f"Could not read {path}")  # Raise a clear error ourselves.
    return np.asarray(image, dtype=np.uint8)  # Already rectified by the dataset authors; asarray fixes the static type.


@DATASETS.register("kitti")
class KittiOdometryDataset(StereoDataset):
    """One sequence of the KITTI odometry benchmark."""

    def __init__(self, root: str | Path, sequence: str | int = "00") -> None:
        """Index the images and read calibration.

        Args:
            root: Folder containing ``sequences/`` (and optionally ``poses/``).
            sequence: Sequence number, e.g. ``"00"`` or ``5``.

        Raises:
            FileNotFoundError: If the sequence folder or its images are missing.
        """
        self._root = Path(root).expanduser()  # Resolve "~" in user-supplied paths.
        self._sequence = f"{int(sequence):02d}"  # Normalise to the two-digit folder name.
        sequence_dir = self._root / "sequences" / self._sequence  # Folder of this sequence.
        if not sequence_dir.is_dir():  # Sanity-check the layout.
            raise FileNotFoundError(f"Missing {sequence_dir}")  # Point the user at the problem.
        self._left_paths = sorted((sequence_dir / "image_0").glob("*.png"))  # Left images in order.
        self._right_paths = sorted((sequence_dir / "image_1").glob("*.png"))  # Right images in order.
        if not self._left_paths or len(self._left_paths) != len(self._right_paths):  # Both cameras must match.
            raise FileNotFoundError(f"Missing or unpaired images in {sequence_dir}")  # Refuse inconsistent data.
        self._timestamps = np.loadtxt(sequence_dir / "times.txt", dtype=np.float64).reshape(-1)  # Seconds since start.
        calibration = _read_calibration(sequence_dir / "calib.txt")  # Projection matrices.
        p0 = calibration["P0"].reshape(3, 4)  # Left grayscale camera projection.
        p1 = calibration["P1"].reshape(3, 4)  # Right grayscale camera projection.
        height, width = _read_image(self._left_paths[0]).shape  # Image size from the first frame.
        self._camera = StereoCamera(  # Rectified model straight from the projection matrices.
            fx=float(p0[0, 0]),  # Focal length x.
            fy=float(p0[1, 1]),  # Focal length y.
            cx=float(p0[0, 2]),  # Principal point x.
            cy=float(p0[1, 2]),  # Principal point y.
            baseline=float(-p1[0, 3] / p1[0, 0]),  # P1 stores -fx * baseline in its last column.
            width=int(width),  # Image width.
            height=int(height),  # Image height.
        )

    @property
    def camera(self) -> StereoCamera:
        """Return the rectified stereo camera model.

        Returns:
            Intrinsics and baseline.
        """
        return self._camera  # Built in __init__.

    @property
    def tag(self) -> str:
        """Return an identifier such as ``kitti_00``.

        Returns:
            Filesystem-safe sequence tag.
        """
        return f"kitti_{self._sequence}"  # Dataset name plus sequence number.

    def __len__(self) -> int:
        """Return the number of stereo pairs.

        Returns:
            Frame count.
        """
        return len(self._left_paths)  # One pair per left image.

    def load_frame(self, index: int) -> StereoFrame:
        """Load one stereo pair.

        Args:
            index: Zero-based frame index.

        Returns:
            The stereo frame (already rectified on disk).
        """
        return StereoFrame(  # Assemble the frame.
            index=index,  # Position in the sequence.
            timestamp=float(self._timestamps[index]),  # Seconds since sequence start.
            left=_read_image(self._left_paths[index]),  # Left image.
            right=_read_image(self._right_paths[index]),  # Right image.
        )

    def ground_truth(self) -> Trajectory | None:
        """Load ground-truth poses of the left camera, when published.

        Returns:
            Trajectory of T_world_cam0 poses, or None for sequences 11-21.
        """
        path = self._root / "poses" / f"{self._sequence}.txt"  # Standard ground-truth location.
        if not path.is_file():  # Test sequences have no public ground truth.
            return None  # Evaluation will be skipped.
        rows = np.loadtxt(path, dtype=np.float64).reshape(-1, 12)  # Each row is a flattened 3x4 matrix.
        trajectory = Trajectory()  # Output container.
        for timestamp, row in zip(self._timestamps, rows, strict=True):  # One pose per frame.
            pose = np.eye(4, dtype=np.float64)  # Homogeneous container.
            pose[:3, :] = row.reshape(3, 4)  # Fill [R | t].
            trajectory.append(float(timestamp), pose)  # Store T_world_cam0.
        return trajectory  # Complete ground truth.

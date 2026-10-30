"""Generic stereo image folder: our RealSense D455 wheelchair recordings (or any synchronised stereo pair).

Layout (written by ``scripts/extract_rosbag.py``)::

    <root>/left/<timestamp_ns>.png
    <root>/right/<timestamp_ns>.png
    <root>/calib.yaml          RawStereoCalib (scripts/calibrate_charuco.py, or factory_calib.yaml from the bag)
    <root>/groundtruth.tum     optional: "t x y z qx qy qz qw" of the (unrectified) left camera

Images are rectified on load with the calibration, unless ``already_rectified`` is set (the D455 IR streams
``infra1/infra2 image_rect_raw`` are rectified by the camera; use the factory calibration with zero distortion).
Sequences that start and end at the same marked spot are evaluated without ground truth by the loop-closure error
(``summary.json``: ``loop_closure_error_m``/``_pct``).
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from pathlib import Path  # Paths.

import cv2  # Image IO.
import numpy as np  # Arrays.
from scipy.spatial.transform import Rotation  # Quaternions.

from metricvision.calibration.camera import RawStereoCalib, Rectifier  # Calibration + rectification.
from metricvision.core.geometry import invert_pose, make_pose  # Poses.
from metricvision.core.types import FloatArray, StereoCamera, StereoFrame, Trajectory  # Shared types.
from metricvision.datasets.base import DATASETS, StereoDataset  # Registry.


@DATASETS.register("stereo_folder")
class StereoFolderDataset(StereoDataset):
    """Stereo PNG folder with a RawStereoCalib YAML."""

    def __init__(
        self,
        root: str | Path,
        calib: str | Path | None = None,
        already_rectified: bool = False,
        start: int = 0,
        stride: int = 1,
        baseline_scale: float = 1.0,
    ) -> None:
        """Index the images and prepare rectification.

        Args:
            root: Sequence folder.
            calib: Calibration YAML (default ``<root>/calib.yaml``).
            already_rectified: Skip rectification (images are rectified; the calibration's K_L and |T| are used).
            start: First frame to use.
            stride: Use every N-th frame (e.g. 2 to emulate 15 Hz from 30 Hz).
            baseline_scale: Multiplies the baseline (deliberate extrinsic error; 1 = off).

        Raises:
            FileNotFoundError: If images or calibration are missing.
        """
        self._root = Path(root).expanduser()
        calib_path = Path(calib).expanduser() if calib else self._root / "calib.yaml"
        if not calib_path.is_file():
            raise FileNotFoundError(f"Missing calibration {calib_path}")
        raw = RawStereoCalib.from_yaml(str(calib_path))
        lefts = sorted((self._root / "left").glob("*.png"))
        rights = {p.name: p for p in (self._root / "right").glob("*.png")}
        pairs = [(p, rights[p.name]) for p in lefts if p.name in rights][start::stride]
        if not pairs:
            raise FileNotFoundError(f"No matching left/right PNGs under {self._root}")
        self._left = [p[0] for p in pairs]
        self._right = [p[1] for p in pairs]
        self._timestamps = np.array([int(p.stem) for p in self._left], dtype=np.float64) * 1e-9  # ns file names -> s.
        if already_rectified:  # Use the calibration as a rectified model.
            self._rectifier = None
            k = raw.K_L
            baseline = float(np.linalg.norm(raw.T))
            self._camera = StereoCamera(
                float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), baseline * baseline_scale, raw.width, raw.height
            )
            self._camera_from_rect = np.eye(4)
        else:
            self._rectifier = Rectifier(raw)
            c = self._rectifier.camera
            self._camera = StereoCamera(c.fx, c.fy, c.cx, c.cy, c.baseline * baseline_scale, c.width, c.height)
            self._camera_from_rect = invert_pose(self._rectifier.T_rect_cam)  # T_cam_rect: rectified -> raw left camera.

    @property
    def camera(self) -> StereoCamera:
        """Return the rectified camera.

        Returns:
            Camera model.
        """
        return self._camera

    @property
    def tag(self) -> str:
        """Return ``folder_<name>``.

        Returns:
            Tag.
        """
        return f"folder_{self._root.name}"

    @property
    def body_from_camera(self) -> FloatArray:
        """Return T_raw-left-camera_rectified-camera (ground truth is given for the raw left camera).

        Returns:
            4x4 transform.
        """
        return self._camera_from_rect

    def __len__(self) -> int:
        """Return the number of pairs.

        Returns:
            Frame count.
        """
        return len(self._left)

    def load_frame(self, index: int) -> StereoFrame:
        """Load and rectify one pair.

        Args:
            index: Frame index.

        Returns:
            Rectified frame.

        Raises:
            FileNotFoundError: If an image cannot be read.
        """
        left = cv2.imread(str(self._left[index]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(self._right[index]), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"Cannot read {self._left[index]} / {self._right[index]}")
        if self._rectifier is not None:
            left, right = self._rectifier(left, right)
        return StereoFrame(index=index, timestamp=float(self._timestamps[index]), left=left, right=right)

    def ground_truth(self) -> Trajectory | None:
        """Read ``groundtruth.tum`` if present.

        Returns:
            Ground truth of the raw left camera, or None.
        """
        path = self._root / "groundtruth.tum"
        if not path.is_file():
            return None
        rows = np.loadtxt(path, comments="#", ndmin=2)
        trajectory = Trajectory()
        for row in rows:
            trajectory.append(float(row[0]), make_pose(Rotation.from_quat(row[4:8]).as_matrix(), row[1:4]))
        return trajectory

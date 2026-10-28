"""Loader for the EuRoC MAV dataset in ASL folder format.

Expected layout (``root`` may point at either the sequence folder or ``mav0``)::

    MH_01_easy/mav0/cam0/data.csv, cam0/data/*.png, cam0/sensor.yaml
    MH_01_easy/mav0/cam1/...
    MH_01_easy/mav0/state_groundtruth_estimate0/data.csv

EuRoC images are distorted and unrectified, so this loader rectifies them with
OpenCV using the calibration shipped in each ``sensor.yaml``.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import csv  # Parsing the image index files.
from pathlib import Path  # Filesystem paths.
from typing import Any  # Loosely typed YAML content.

import cv2  # Image loading and stereo rectification.
import numpy as np  # Matrix maths.
import yaml  # Parsing sensor.yaml calibration files.

from metricvision.core.geometry import make_pose, rotation_from_quaternion_wxyz  # Pose helpers.
from metricvision.core.types import (  # Shared types.
    Float32Array,
    FloatArray,
    ImageArray,
    StereoCamera,
    StereoFrame,
    Trajectory,
)
from metricvision.datasets.base import DATASETS, StereoDataset  # Base class and registry.


def _load_sensor_yaml(path: Path) -> dict[str, Any]:
    """Load an EuRoC ``sensor.yaml`` file.

    The files start with an OpenCV-style ``%YAML:1.0`` directive that PyYAML
    rejects, so that line is stripped before parsing.

    Args:
        path: Path to the sensor.yaml file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ValueError: If the file does not contain a YAML mapping.
    """
    lines = [line for line in path.read_text().splitlines() if not line.startswith("%YAML")]  # Drop the directive.
    data = yaml.safe_load("\n".join(lines))  # Parse the remaining YAML safely.
    if not isinstance(data, dict):  # Calibration files must be key/value mappings.
        raise ValueError(f"Unexpected content in {path}")  # Refuse to continue with bad calibration.
    return data  # Parsed calibration.


def _read_image_index(camera_dir: Path) -> dict[int, Path]:
    """Read a camera's ``data.csv`` into a timestamp -> image path mapping.

    Args:
        camera_dir: Folder such as ``mav0/cam0``.

    Returns:
        Mapping from nanosecond timestamps to image file paths.
    """
    index: dict[int, Path] = {}  # Output mapping.
    with (camera_dir / "data.csv").open(newline="") as handle:  # Open the CSV index.
        for row in csv.reader(handle):  # Each row is "timestamp_ns,filename".
            if not row or row[0].startswith("#"):  # Skip blank lines and the header.
                continue  # Nothing to record.
            index[int(row[0])] = camera_dir / "data" / row[1].strip()  # Store the image path.
    return index  # Completed index.


def _intrinsic_matrix(intrinsics: list[float]) -> FloatArray:
    """Build K from EuRoC's ``[fu, fv, cu, cv]`` intrinsics list.

    Args:
        intrinsics: Focal lengths and principal point.

    Returns:
        3x3 intrinsic matrix.
    """
    fu, fv, cu, cv = (float(v) for v in intrinsics)  # Unpack the four parameters.
    return np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)  # Standard pinhole K.


@DATASETS.register("euroc")
class EuRoCDataset(StereoDataset):
    """EuRoC MAV stereo sequence, rectified on the fly."""

    def __init__(self, root: str | Path) -> None:
        """Index the images and prepare rectification maps.

        Args:
            root: Sequence folder (e.g. ``~/datasets/euroc/MH_01_easy``) or its ``mav0`` subfolder.

        Raises:
            FileNotFoundError: If the expected folders are missing.
        """
        root_path = Path(root).expanduser()  # Resolve "~" in user-supplied paths.
        self._mav0 = root_path / "mav0" if (root_path / "mav0").is_dir() else root_path  # Accept both layouts.
        if not (self._mav0 / "cam0").is_dir():  # Sanity-check the layout.
            raise FileNotFoundError(f"No cam0 folder under {self._mav0}")  # Point the user at the problem.
        left_index = _read_image_index(self._mav0 / "cam0")  # Left camera images.
        right_index = _read_image_index(self._mav0 / "cam1")  # Right camera images.
        self._timestamps_ns = sorted(set(left_index) & set(right_index))  # Keep only synchronised pairs.
        self._left_paths = [left_index[t] for t in self._timestamps_ns]  # Left images in time order.
        self._right_paths = [right_index[t] for t in self._timestamps_ns]  # Right images in time order.
        self._setup_rectification(  # Compute rectification from both cameras' calibration.
            _load_sensor_yaml(self._mav0 / "cam0" / "sensor.yaml"),  # Left calibration.
            _load_sensor_yaml(self._mav0 / "cam1" / "sensor.yaml"),  # Right calibration.
        )

    def _setup_rectification(self, cam0: dict[str, Any], cam1: dict[str, Any]) -> None:
        """Compute rectification maps and the rectified camera model.

        Args:
            cam0: Parsed left camera sensor.yaml.
            cam1: Parsed right camera sensor.yaml.
        """
        width, height = (int(v) for v in cam0["resolution"])  # Image size, identical for both cameras.
        k0, k1 = (
            _intrinsic_matrix(cam0["intrinsics"]),
            _intrinsic_matrix(cam1["intrinsics"]),
        )  # Raw intrinsics.
        d0 = np.asarray(cam0["distortion_coefficients"], dtype=np.float64)  # Left radial-tangential distortion.
        d1 = np.asarray(cam1["distortion_coefficients"], dtype=np.float64)  # Right radial-tangential distortion.
        body_from_cam0 = np.asarray(cam0["T_BS"]["data"], dtype=np.float64).reshape(4, 4)  # Left camera in body (IMU) frame.
        body_from_cam1 = np.asarray(cam1["T_BS"]["data"], dtype=np.float64).reshape(4, 4)  # Right camera in body frame.
        cam1_from_cam0 = np.linalg.inv(body_from_cam1) @ body_from_cam0  # Maps left-camera points into the right camera.
        rectification = cv2.stereoRectify(  # Compute rectifying rotations and projections.
            k0,  # Left raw intrinsics.
            d0,  # Left distortion.
            k1,  # Right raw intrinsics.
            d1,  # Right distortion.
            (width, height),  # Image size.
            cam1_from_cam0[:3, :3],  # Rotation from left to right camera.
            cam1_from_cam0[:3, 3].reshape(3, 1),  # Translation from left to right camera.
            flags=cv2.CALIB_ZERO_DISPARITY,  # Make principal points coincide so disparity is purely horizontal.
            alpha=0.0,  # Crop to valid pixels only (no black borders).
        )
        rect0, rect1, proj0, proj1 = (np.asarray(m, dtype=np.float64) for m in rectification[:4])  # R1, R2, P1, P2 as float64.
        self._map0 = self._rectify_maps(k0, d0, rect0, proj0, width, height)  # Left lookup maps.
        self._map1 = self._rectify_maps(k1, d1, rect1, proj1, width, height)  # Right lookup maps.
        self._camera = StereoCamera(  # Intrinsics of the rectified pair.
            fx=float(proj0[0, 0]),  # Rectified focal length x.
            fy=float(proj0[1, 1]),  # Rectified focal length y.
            cx=float(proj0[0, 2]),  # Rectified principal point x.
            cy=float(proj0[1, 2]),  # Rectified principal point y.
            baseline=float(-proj1[0, 3] / proj1[0, 0]),  # P1 stores -fx * baseline in its last column.
            width=width,  # Image width.
            height=height,  # Image height.
        )
        rect_to_raw = make_pose(np.asarray(rect0, dtype=np.float64).T, np.zeros(3))  # Rectification is a pure rotation: raw_from_rect = R0^T.
        self._body_from_rect = body_from_cam0 @ rect_to_raw  # Rectified left camera expressed in the body frame.

    @staticmethod
    def _rectify_maps(
        k: FloatArray, dist: FloatArray, rect: FloatArray, proj: FloatArray, width: int, height: int
    ) -> tuple[Float32Array, Float32Array]:
        """Build the per-pixel lookup tables that undistort and rectify one camera.

        Args:
            k: Raw intrinsic matrix.
            dist: Distortion coefficients.
            rect: Rectifying rotation from stereoRectify.
            proj: Rectified projection matrix from stereoRectify.
            width: Image width.
            height: Image height.

        Returns:
            (map_x, map_y) float32 lookup tables for cv2.remap.
        """
        map_x, map_y = cv2.initUndistortRectifyMap(k, dist, rect, proj, (width, height), cv2.CV_32FC1)  # Compute the maps.
        return np.asarray(map_x, dtype=np.float32), np.asarray(map_y, dtype=np.float32)  # Plain float32 arrays.

    @property
    def camera(self) -> StereoCamera:
        """Return the rectified stereo camera model.

        Returns:
            Rectified intrinsics and baseline.
        """
        return self._camera  # Computed in _setup_rectification.

    @property
    def tag(self) -> str:
        """Return an identifier such as ``euroc_MH_01_easy``.

        Returns:
            Filesystem-safe sequence tag.
        """
        name = self._mav0.parent.name if self._mav0.name == "mav0" else self._mav0.name  # Sequence folder name.
        return f"euroc_{name}"  # Prefix with the dataset name.

    @property
    def body_from_camera(self) -> FloatArray:
        """Return T_body_camera for the rectified left camera.

        EuRoC ground truth describes the IMU (body) frame, not the camera, so
        estimated camera poses must be converted before evaluation.

        Returns:
            4x4 rigid transform.
        """
        return self._body_from_rect  # Includes the rectification rotation.

    def __len__(self) -> int:
        """Return the number of synchronised stereo pairs.

        Returns:
            Frame count.
        """
        return len(self._timestamps_ns)  # One entry per synchronised pair.

    def _load_rectified(self, path: Path, maps: tuple[Float32Array, Float32Array]) -> ImageArray:
        """Read one grayscale image and rectify it.

        Args:
            path: Image file.
            maps: (map_x, map_y) lookup tables from initUndistortRectifyMap.

        Returns:
            Rectified grayscale image.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # EuRoC images are 8-bit grayscale.
        if image is None:  # OpenCV returns None instead of raising.
            raise FileNotFoundError(f"Could not read {path}")  # Raise a clear error ourselves.
        rectified = cv2.remap(image, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)  # Undistort and rectify.
        return np.asarray(rectified, dtype=np.uint8)  # Fix the static type returned by OpenCV.

    def load_frame(self, index: int) -> StereoFrame:
        """Load and rectify one stereo pair.

        Args:
            index: Zero-based frame index.

        Returns:
            The rectified stereo frame.
        """
        return StereoFrame(  # Assemble the frame.
            index=index,  # Position in the sequence.
            timestamp=self._timestamps_ns[index] * 1e-9,  # Nanoseconds to seconds.
            left=self._load_rectified(self._left_paths[index], self._map0),  # Rectified left image.
            right=self._load_rectified(self._right_paths[index], self._map1),  # Rectified right image.
        )

    def ground_truth(self) -> Trajectory | None:
        """Load the motion-capture / laser-tracker ground truth of the body frame.

        Returns:
            Trajectory of T_world_body poses, or None if the file is missing.
        """
        path = self._mav0 / "state_groundtruth_estimate0" / "data.csv"  # Standard ground-truth location.
        if not path.is_file():  # Some sequences may be distributed without ground truth.
            return None  # Evaluation will be skipped.
        data = np.loadtxt(path, delimiter=",", comments="#")  # Columns: t_ns, p_xyz, q_wxyz, velocities, biases.
        trajectory = Trajectory()  # Output container.
        for row in data:  # One ground-truth sample per row (200 Hz).
            rotation = rotation_from_quaternion_wxyz(row[4:8])  # Orientation of the body in the world.
            trajectory.append(row[0] * 1e-9, make_pose(rotation, row[1:4]))  # Store T_world_body at time t.
        return trajectory  # Complete ground truth.

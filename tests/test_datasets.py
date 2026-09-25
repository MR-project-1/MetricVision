"""Dataset loader tests using tiny fake folders with real EuRoC/KITTI calibration values."""

from __future__ import annotations  # Postponed evaluation of type hints.

from pathlib import Path  # Temporary folders.

import cv2  # Writing fake images.
import numpy as np  # Fake image content and checks.

from metricvision.datasets.euroc import EuRoCDataset  # Loader under test.
from metricvision.datasets.kitti import KittiOdometryDataset  # Loader under test.

EUROC_CAM0 = """%YAML:1.0
sensor_type: camera
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375]
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
"""  # Real MH_01_easy cam0 calibration (EuRoC files start with an OpenCV %YAML directive).

EUROC_CAM1 = """%YAML:1.0
sensor_type: camera
T_BS:
  cols: 4
  rows: 4
  data: [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [457.587, 456.134, 379.999, 255.238]
distortion_model: radial-tangential
distortion_coefficients: [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]
"""  # Real MH_01_easy cam1 calibration.


def _write_euroc(root: Path, num_frames: int = 3) -> None:
    """Create a minimal EuRoC folder with random images.

    Args:
        root: Sequence folder to create.
        num_frames: Number of stereo pairs.
    """
    rng = np.random.default_rng(0)  # Deterministic content.
    for cam, calib in (("cam0", EUROC_CAM0), ("cam1", EUROC_CAM1)):  # Both cameras.
        cam_dir = root / "mav0" / cam  # Camera folder.
        (cam_dir / "data").mkdir(parents=True)  # Image folder.
        (cam_dir / "sensor.yaml").write_text(calib)  # Calibration.
        lines = ["#timestamp [ns],filename"]  # CSV header.
        for i in range(num_frames):  # One image per frame.
            stamp = 1403636579763555584 + i * 50_000_000  # 20 Hz nanosecond timestamps.
            cv2.imwrite(str(cam_dir / "data" / f"{stamp}.png"), rng.integers(0, 256, (480, 752), dtype=np.uint8))  # Random image.
            lines.append(f"{stamp},{stamp}.png")  # Index entry.
        (cam_dir / "data.csv").write_text("\n".join(lines) + "\n")  # Write the index.


def test_euroc_rectification(tmp_path: Path) -> None:
    """Rectified EuRoC camera must have the known ~11 cm baseline and correctly sized images.

    Args:
        tmp_path: Pytest-provided temporary folder.
    """
    _write_euroc(tmp_path / "MH_01_easy")  # Build the fake sequence.
    dataset = EuRoCDataset(tmp_path / "MH_01_easy")  # Load it.
    assert len(dataset) == 3  # All pairs indexed.
    assert dataset.tag == "euroc_MH_01_easy"  # Tag from the folder name.
    assert abs(dataset.camera.baseline - 0.110) < 0.002  # EuRoC baseline is about 11 cm.
    frame = dataset.load_frame(1)  # Load one pair.
    assert frame.left.shape == (480, 752) and frame.right.shape == (480, 752)  # Rectification keeps the size.
    body_from_cam = dataset.body_from_camera  # Camera-to-body extrinsic.
    np.testing.assert_allclose(body_from_cam[:3, :3] @ body_from_cam[:3, :3].T, np.eye(3), atol=1e-9)  # Valid rotation.
    assert dataset.ground_truth() is None  # No ground-truth file in the fake folder.


def test_kitti_calibration(tmp_path: Path) -> None:
    """KITTI loader must recover the ~54 cm baseline from calib.txt.

    Args:
        tmp_path: Pytest-provided temporary folder.
    """
    sequence_dir = tmp_path / "sequences" / "00"  # Sequence folder.
    for camera in ("image_0", "image_1"):  # Left and right image folders.
        (sequence_dir / camera).mkdir(parents=True)  # Create the folder.
        for i in range(2):  # Two frames.
            cv2.imwrite(str(sequence_dir / camera / f"{i:06d}.png"), np.zeros((376, 1241), dtype=np.uint8))  # Blank image.
    p0 = "7.188560e+02 0 6.071928e+02 0 0 7.188560e+02 1.852157e+02 0 0 0 1 0"  # Real sequence 00 P0.
    p1 = "7.188560e+02 0 6.071928e+02 -3.861448e+02 0 7.188560e+02 1.852157e+02 0 0 0 1 0"  # Real sequence 00 P1.
    (sequence_dir / "calib.txt").write_text(f"P0: {p0}\nP1: {p1}\n")  # Calibration file.
    (sequence_dir / "times.txt").write_text("0.0\n0.1\n")  # Timestamps.
    dataset = KittiOdometryDataset(tmp_path, sequence=0)  # Integer sequence ids are accepted.
    assert dataset.tag == "kitti_00"  # Two-digit tag.
    assert abs(dataset.camera.baseline - 0.537) < 0.001  # KITTI baseline is about 54 cm.
    assert dataset.load_frame(1).timestamp == 0.1  # Timestamps come from times.txt.

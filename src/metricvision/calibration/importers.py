"""Import stereo calibrations from standard toolboxes into ``RawStereoCalib``.

* **MC-Calib** (``calibrated_cameras_data.yml``, OpenCV FileStorage): ``camera_<i>`` with ``camera_matrix``,
  ``distortion_vector``, ``distortion_type`` (0 Brown, 1 Kannala fisheye), ``img_width``/``img_height`` and a 4x4
  ``camera_pose_matrix`` "expressed in camera 0 referencial". The documentation does not state the direction of that
  matrix, so it is a parameter (``pose_convention``): ``cam_to_ref`` (X_ref = M X_cam, the pose of the camera in the
  reference frame) or ``ref_to_cam`` (X_cam = M X_ref). ``scripts/import_calibration.py --check-images`` decides it
  (and the left/right assignment) from a real image pair by rectifying with every hypothesis and keeping the one with
  positive disparities and the smallest vertical error, so the direction never has to be guessed.
* **Kalibr** camchain (``camchain.yaml``): ``cam0``/``cam1`` with ``intrinsics [fu, fv, pu, pv]``,
  ``distortion_model`` (radtan | equidistant), ``distortion_coeffs``, ``resolution`` and ``T_cn_cnm1`` (cam1 from cam0,
  X_c1 = T X_c0). ``T_cam_imu`` is kept in ``meta`` for the stereo-inertial extension.
* Our own YAML (``scripts/calibrate_charuco.py``, ``extract_rosbag.py`` factory calibration).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from metricvision.calibration.camera import RawStereoCalib


def _fs_mat(node) -> np.ndarray:  # noqa: ANN001
    """Matrix of an OpenCV FileStorage node."""
    mat = node.mat()
    return np.asarray(mat, dtype=np.float64) if mat is not None else np.zeros(0)


def load_mc_calib(path: str | Path, left: int = 0, right: int = 1, pose_convention: str = "cam_to_ref") -> RawStereoCalib:
    """Read an MC-Calib ``calibrated_cameras_data.yml`` as a stereo pair.

    Args:
        path: MC-Calib camera file.
        left: Camera index used as the left camera.
        right: Camera index used as the right camera (it must sit to the right, +x, of the left camera).
        pose_convention: ``cam_to_ref`` or ``ref_to_cam`` (see the module docstring).

    Returns:
        Calibration with X_R = R X_L + T.

    Raises:
        ValueError: On an unknown convention, missing cameras or mixed distortion models.
        FileNotFoundError: If the file cannot be opened.
    """
    if pose_convention not in ("cam_to_ref", "ref_to_cam"):
        raise ValueError(f"pose_convention must be cam_to_ref or ref_to_cam, got {pose_convention}")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open {path}")
    cams = {}
    for index in (left, right):
        node = fs.getNode(f"camera_{index}")
        if node.empty():
            raise ValueError(f"{path} has no camera_{index}")
        pose = _fs_mat(node.getNode("camera_pose_matrix")).reshape(4, 4)
        ref_from_cam = pose if pose_convention == "cam_to_ref" else np.linalg.inv(pose)
        cams[index] = {
            "K": _fs_mat(node.getNode("camera_matrix")).reshape(3, 3),
            "D": _fs_mat(node.getNode("distortion_vector")).ravel(),
            "type": int(node.getNode("distortion_type").real()),
            "group": int(node.getNode("camera_group").real()) if not node.getNode("camera_group").empty() else 0,
            "size": (int(node.getNode("img_width").real()), int(node.getNode("img_height").real())),
            "ref_from_cam": ref_from_cam,
        }
    fs.release()
    a, b = cams[left], cams[right]
    if a["type"] != b["type"]:
        raise ValueError("left and right cameras use different distortion models")
    if a["group"] != b["group"]:
        raise ValueError("left and right cameras are in different MC-Calib camera groups (no common extrinsic)")
    right_from_left = np.linalg.inv(b["ref_from_cam"]) @ a["ref_from_cam"]  # X_R = T_RL X_L.
    model = "fisheye" if a["type"] == 1 else "radtan"
    meta = {"source": f"MC-Calib {Path(path).name} (cameras {left}->{right}, {pose_convention})", "distortion_model": model}
    width, height = a["size"]
    return RawStereoCalib(a["K"], a["D"], b["K"], b["D"], right_from_left[:3, :3], right_from_left[:3, 3], width, height, meta)


def load_kalibr(path: str | Path, left: str = "cam0", right: str = "cam1") -> RawStereoCalib:
    """Read a Kalibr camchain (pinhole cameras) as a stereo pair.

    Args:
        path: ``camchain.yaml`` or ``camchain-imucam.yaml``.
        left: Left camera key.
        right: Right camera key (its ``T_cn_cnm1`` must be relative to the left camera).

    Returns:
        Calibration with X_R = R X_L + T (``meta`` holds ``T_cam_imu`` of both cameras when present).

    Raises:
        ValueError: For non-pinhole cameras or a missing extrinsic.
    """
    data = yaml.safe_load(Path(path).read_text())
    cams = []
    for key in (left, right):
        cam = data[key]
        if cam.get("camera_model", "pinhole") != "pinhole":
            raise ValueError(f"{key}: only pinhole cameras are supported, got {cam.get('camera_model')}")
        fu, fv, pu, pv = cam["intrinsics"]
        cams.append(
            {
                "K": np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]]),
                "D": np.asarray(cam["distortion_coeffs"], dtype=np.float64),
                "model": cam.get("distortion_model", "radtan"),
                "size": tuple(cam["resolution"]),
            }
        )
    if "T_cn_cnm1" not in data[right]:
        raise ValueError(f"{right} has no T_cn_cnm1 (extrinsic relative to {left})")
    right_from_left = np.asarray(data[right]["T_cn_cnm1"], dtype=np.float64)
    meta: dict = {"source": f"Kalibr {Path(path).name}", "distortion_model": "fisheye" if cams[0]["model"] == "equidistant" else "radtan"}
    for key in (left, right):
        if "T_cam_imu" in data[key]:
            meta[f"T_{key}_imu"] = np.asarray(data[key]["T_cam_imu"], dtype=np.float64).tolist()
    d_left = cams[0]["D"] if meta["distortion_model"] == "fisheye" else np.pad(cams[0]["D"], (0, max(0, 5 - cams[0]["D"].size)))
    d_right = cams[1]["D"] if meta["distortion_model"] == "fisheye" else np.pad(cams[1]["D"], (0, max(0, 5 - cams[1]["D"].size)))
    width, height = cams[0]["size"]
    return RawStereoCalib(cams[0]["K"], d_left, cams[1]["K"], d_right, right_from_left[:3, :3], right_from_left[:3, 3], int(width), int(height), meta)


def detect_format(path: str | Path) -> str:
    """Guess the calibration file format.

    Args:
        path: File.

    Returns:
        ``mc_calib``, ``kalibr`` or ``metricvision``.
    """
    text = Path(path).read_text(errors="ignore")
    if "camera_pose_matrix" in text and "nb_camera" in text:
        return "mc_calib"
    if "T_cn_cnm1" in text or ("cam0:" in text and "intrinsics:" in text):
        return "kalibr"
    return "metricvision"


def load_calibration(path: str | Path, **kwargs) -> RawStereoCalib:  # noqa: ANN003
    """Load any supported stereo calibration file.

    Args:
        path: Calibration file.
        **kwargs: Format-specific options (MC-Calib: left, right, pose_convention; Kalibr: left, right).

    Returns:
        The calibration.
    """
    fmt = detect_format(path)
    if fmt == "mc_calib":
        return load_mc_calib(path, **kwargs)
    if fmt == "kalibr":
        return load_kalibr(path, **{k: v for k, v in kwargs.items() if k in ("left", "right")})
    return RawStereoCalib.from_yaml(str(path))


def compare_calibrations(a: RawStereoCalib, b: RawStereoCalib) -> dict[str, float]:
    """Extrinsic difference between two calibrations (e.g. MC-Calib vs D455 factory): the RQ3 calibration-error input.

    Args:
        a: First calibration.
        b: Second calibration.

    Returns:
        Baseline difference (mm), rotation difference (deg) and per-axis rotation (deg, x=pitch, y=yaw, z=roll).
    """
    delta = a.R @ b.R.T
    rvec, _ = cv2.Rodrigues(delta)
    return {
        "baseline_a_mm": 1000.0 * float(np.linalg.norm(a.T)),
        "baseline_b_mm": 1000.0 * float(np.linalg.norm(b.T)),
        "baseline_diff_mm": 1000.0 * float(np.linalg.norm(a.T) - np.linalg.norm(b.T)),
        "translation_diff_mm": 1000.0 * float(np.linalg.norm(a.T - b.T)),
        "rotation_diff_deg": float(np.degrees(np.linalg.norm(rvec))),
        "pitch_deg": float(np.degrees(rvec[0, 0])),
        "yaw_deg": float(np.degrees(rvec[1, 0])),
        "roll_deg": float(np.degrees(rvec[2, 0])),
    }

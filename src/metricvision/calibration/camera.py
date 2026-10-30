"""Unrectified stereo calibration (pinhole + radial-tangential) and rectification to a ``StereoCamera``.

``RawStereoCalib`` is the YAML format written by ``scripts/calibrate_charuco.py`` (ChArUco, with uncertainty),
``scripts/calibrate_stereo.py`` (checkerboard) and ``scripts/extract_rosbag.py`` (D455 factory calibration), and read
by the ``stereo_folder`` dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from metricvision.calibration._geometry import make_T
from metricvision.core.types import StereoCamera


@dataclass
class RawStereoCalib:
    """Unrectified calibration (pinhole + radial-tangential)."""

    K_L: np.ndarray
    D_L: np.ndarray
    K_R: np.ndarray
    D_R: np.ndarray
    R: np.ndarray  # X_R = R X_L + T
    T: np.ndarray
    width: int
    height: int
    meta: dict = field(default_factory=dict)

    def to_yaml(self, path: str) -> None:
        d = {
            "width": int(self.width),
            "height": int(self.height),
            "K_L": self.K_L.tolist(),
            "D_L": np.asarray(self.D_L).ravel().tolist(),
            "K_R": self.K_R.tolist(),
            "D_R": np.asarray(self.D_R).ravel().tolist(),
            "R": self.R.tolist(),
            "T": np.asarray(self.T).ravel().tolist(),
            "baseline_m": float(np.linalg.norm(self.T)),
        }
        d.update(self.meta)
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    @staticmethod
    def from_yaml(path: str) -> RawStereoCalib:
        with open(path) as f:
            d = yaml.safe_load(f)
        a = lambda k: np.asarray(d[k], dtype=np.float64)
        meta = {k: v for k, v in d.items() if k not in ("width", "height", "K_L", "D_L", "K_R", "D_R", "R", "T")}
        return RawStereoCalib(a("K_L"), a("D_L"), a("K_R"), a("D_R"), a("R"), a("T").reshape(3), int(d["width"]), int(d["height"]), meta)


class Rectifier:
    """Wraps cv2.stereoRectify + remap tables; exposes the rectified ``StereoCamera``."""

    def __init__(self, calib: RawStereoCalib, alpha: float = 0.0):
        size = (calib.width, calib.height)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            calib.K_L, calib.D_L, calib.K_R, calib.D_R, size, calib.R, calib.T.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
        )
        self.R1, self.R2 = R1, R2
        self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(calib.K_L, calib.D_L, R1, P1, size, cv2.CV_32FC1)
        self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(calib.K_R, calib.D_R, R2, P2, size, cv2.CV_32FC1)
        baseline = float(-P2[0, 3] / P2[0, 0])
        self.camera = StereoCamera(
            fx=float(P1[0, 0]), fy=float(P1[1, 1]), cx=float(P1[0, 2]), cy=float(P1[1, 2]), baseline=baseline, width=calib.width, height=calib.height
        )
        self.T_rect_cam = make_T(R1, np.zeros(3))  # X_rect = R1 X_cam_left.

    def __call__(self, imgL: np.ndarray, imgR: np.ndarray):
        L = cv2.remap(imgL, self.mapLx, self.mapLy, cv2.INTER_LINEAR)
        R = cv2.remap(imgR, self.mapRx, self.mapRy, cv2.INTER_LINEAR)
        return L, R

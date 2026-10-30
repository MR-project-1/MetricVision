"""Calibration recovers a known rig from synthetic checkerboard views."""

import cv2
import numpy as np

from metricvision.calibration._geometry import so3_exp
from metricvision.calibration.solver import board_points, calibrate_stereo, project_radtan


def test_projection_matches_opencv():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(2, 5, 50)])
    intr = np.array([500.0, 505.0, 320.0, 240.0])
    dist = np.array([-0.2, 0.05, 0.001, -0.0015, 0.01])
    K = np.array([[500, 0, 320], [0, 505, 240], [0, 0, 1.0]])
    ref, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K, dist)
    assert np.allclose(project_radtan(X, intr, dist), ref.reshape(-1, 2), atol=1e-8)


def test_stereo_calibration_recovers_baseline():
    rng = np.random.default_rng(1)
    obj = board_points(9, 6, 0.04)
    intrL = np.array([600.0, 600.0, 330.0, 245.0])
    intrR = np.array([605.0, 603.0, 318.0, 238.0])
    dL = np.array([-0.15, 0.03, 0.0005, -0.0003, 0.0])
    dR = np.array([-0.14, 0.025, -0.0004, 0.0002, 0.0])
    R_RL = so3_exp([0.004, -0.01, 0.002])
    t_RL = np.array([-0.095, 0.0006, -0.0004])
    ptsL, ptsR = [], []
    center = obj.mean(0)
    while len(ptsL) < 25:
        R = so3_exp(rng.normal(0, 0.35, 3))
        t = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15), rng.uniform(0.6, 1.4)])
        XL = (obj - center) @ R.T + t
        XR = XL @ R_RL.T + t_RL
        uL = project_radtan(XL, intrL, dL)
        uR = project_radtan(XR, intrR, dR)
        inside = lambda u: np.all((u[:, 0] > 5) & (u[:, 0] < 635) & (u[:, 1] > 5) & (u[:, 1] < 475))
        if inside(uL) and inside(uR) and XL[:, 2].min() > 0.2:
            ptsL.append(uL + rng.normal(0, 0.15, uL.shape))
            ptsR.append(uR + rng.normal(0, 0.15, uR.shape))
    calib, rep = calibrate_stereo(obj, ptsL, ptsR, (640, 480))
    assert rep["rms_joint_px"] < 0.3
    assert abs(np.linalg.norm(calib.T) - np.linalg.norm(t_RL)) < 1e-3  # < 1 mm baseline error
    assert abs(calib.K_L[0, 0] - 600) < 3 and abs(calib.K_R[0, 0] - 605) < 3

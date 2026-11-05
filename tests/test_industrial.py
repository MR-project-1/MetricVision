"""Industrial-standard integrations: GTSAM window BA, COLMAP rig RANSAC, Ceres cross-check, MC-Calib/Kalibr import."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from metricvision.backend.lm import LMConfig, refine_pose_lm
from metricvision.backend.pose import PnPConfig, PnPPoseEstimator
from metricvision.backend.window_ba import SlidingWindowBA, WindowBAConfig
from metricvision.calibration.camera import RawStereoCalib, Rectifier
from metricvision.calibration.importers import compare_calibrations, detect_format, load_calibration, load_kalibr, load_mc_calib
from metricvision.core.geometry import invert_pose, make_pose, se3_exp
from metricvision.datasets.synthetic import SyntheticPlaneDataset

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
B = 0.12


def _project(points: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    """(u_L, v_L, u_R) of world points."""
    p = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return np.column_stack([u, v, u - K[0, 0] * B / p[:, 2]])


def _fill_window(ba: SlidingWindowBA, seed: int = 0) -> list[np.ndarray]:
    """Identical noisy 5-keyframe problem for both solvers."""
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-1.5, 1.5, 300), rng.uniform(3, 15, 300)])
    truth = [se3_exp(np.array([0.02 * k, 0.0, -0.3 * k, 0.0, 0.01 * k, 0.0])) for k in range(5)]
    for k, pose in enumerate(truth):
        obs = _project(points, pose) + rng.normal(0, 0.5, (300, 3))
        start = pose if k == 0 else se3_exp(rng.normal(0, 0.01, 6)) @ pose
        ba.add_keyframe(k, start, np.arange(300), obs, points + rng.normal(0, 0.05, points.shape) * points[:, 2:3] / 5)
    return truth


def test_gtsam_window_ba_matches_numpy_solver() -> None:
    """GTSAM (GenericStereoFactor3D) and our Schur LM minimise the same cost: same poses, comparable covariance."""
    pytest.importorskip("gtsam")
    from metricvision.backend.gtsam_window import GtsamWindowBA

    config = WindowBAConfig(window=5, keyframe_every=1, iterations=20)
    ours, theirs = SlidingWindowBA(config, K, B), GtsamWindowBA(config, K, B)
    truth = _fill_window(ours)
    _fill_window(theirs)
    assert ours.optimise(0.5) is not None and theirs.optimise(0.5) is not None
    for a, b, t in zip(ours.keyframes, theirs.keyframes, truth, strict=True):
        assert np.linalg.norm(a.camera_from_world[:3, 3] - b.camera_from_world[:3, 3]) < 5e-4  # Same optimum (sub-mm).
        assert np.linalg.norm((b.camera_from_world @ invert_pose(t))[:3, 3]) < 0.01
    assert ours.newest_covariance is not None and theirs.newest_covariance is not None
    ratio = np.diag(theirs.newest_covariance) / np.diag(ours.newest_covariance)
    assert np.all((ratio > 0.5) & (ratio < 2.0))  # Same uncertainty model, same conventions.


def test_colmap_rig_ransac_rejects_right_only_outliers() -> None:
    """COLMAP generalized absolute pose with the stereo pair as a two-camera rig."""
    pytest.importorskip("pycolmap")
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(-4, 4, 300), rng.uniform(-2, 2, 300), rng.uniform(4, 25, 300)])
    truth = se3_exp(np.array([0.05, -0.02, 0.4, 0.01, 0.03, -0.005]))
    obs = _project(points, truth) + rng.normal(0, 0.3, (300, 3))
    obs[:40, 2] += rng.uniform(8, 20, 40)  # Wrong stereo matches.
    estimator = PnPPoseEstimator(PnPConfig(ransac="colmap_rig", refine="lm"))
    est = estimator.estimate(points, obs[:, :2], K, None, points_2d_right=np.column_stack([obs[:, 2], obs[:, 1]]), baseline=B, pixel_sigma=0.3)
    assert est.success
    assert not est.inlier_mask[:40].any() and est.inlier_mask[40:].mean() > 0.95
    assert np.linalg.norm((est.camera_from_reference @ invert_pose(truth))[:3, 3]) < 0.02


def test_rig_lm_agrees_with_ceres() -> None:
    """The analytic rigid-rig LM and Ceres (pyceres, numeric Python cost, same residual and Huber) reach the same pose."""
    pyceres = pytest.importorskip("pyceres")
    rng = np.random.default_rng(2)
    points = np.column_stack([rng.uniform(-4, 4, 120), rng.uniform(-2, 2, 120), rng.uniform(4, 20, 120)])
    truth = se3_exp(np.array([0.03, 0.01, -0.5, 0.004, 0.01, 0.002]))
    obs = _project(points, truth) + rng.normal(0, 0.5, (120, 3))
    obs[:6] += 15.0  # A few outliers for the robust kernel.
    start = se3_exp(np.array([0.05, -0.05, 0.05, 0.01, -0.01, 0.01])) @ truth
    ours = refine_pose_lm(
        start, points, obs[:, :2], K, np.column_stack([obs[:, 2], obs[:, 1]]), B, None, LMConfig(pixel_sigma=0.5, max_iterations=50)
    )

    class StereoCost(pyceres.CostFunction):
        """r = (u_L, v_L, u_R) residual / sigma of one point; parameters: delta (6) applied as exp(delta) @ start."""

        def __init__(self, point: np.ndarray, measurement: np.ndarray) -> None:
            super().__init__()
            self.set_num_residuals(3)
            self.set_parameter_block_sizes([6])
            self.point, self.measurement = point, measurement

        def _residual(self, delta: np.ndarray) -> np.ndarray:
            return (_project(self.point[None], se3_exp(delta) @ start)[0] - self.measurement) / 0.5

        def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802, ANN001, ANN201
            delta = np.asarray(parameters[0], dtype=np.float64)
            residuals[:] = self._residual(delta)
            if jacobians is not None and jacobians[0] is not None:
                jac = np.zeros((3, 6))
                for i in range(6):  # Central differences: an independent check of our analytic Jacobians.
                    step = np.zeros(6)
                    step[i] = 1e-6
                    jac[:, i] = (self._residual(delta + step) - self._residual(delta - step)) / 2e-6
                jacobians[0][:] = jac.ravel()
            return True

    delta = np.zeros(6)
    problem = pyceres.Problem()
    for point, measurement in zip(points, obs, strict=True):
        problem.add_residual_block(StereoCost(point, measurement), pyceres.HuberLoss(2.0), [delta])
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.max_num_iterations = 100
    options.function_tolerance = 1e-14
    options.gradient_tolerance = 1e-14
    options.parameter_tolerance = 1e-14
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    ceres_pose = se3_exp(delta) @ start
    assert np.linalg.norm(ceres_pose[:3, 3] - ours.camera_from_reference[:3, 3]) < 1e-4
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(ceres_pose[:3, :3] @ ours.camera_from_reference[:3, :3].T)[0])) < 1e-3


def _write_mc_calib(path: Path, right_from_left: np.ndarray, size: tuple[int, int], k: np.ndarray, fisheye: bool = False) -> None:
    """MC-Calib-style file with camera_pose_matrix = pose of the camera in camera 0 (cam_to_ref)."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", 2)
    for index, pose in ((0, np.eye(4)), (1, invert_pose(right_from_left))):
        fs.startWriteStruct(f"camera_{index}", cv2.FILE_NODE_MAP)
        fs.write("camera_matrix", k)
        fs.write("distortion_vector", np.zeros((1, 4 if fisheye else 5)))
        fs.write("distortion_type", 1 if fisheye else 0)
        fs.write("camera_group", 0)
        fs.write("img_width", size[0])
        fs.write("img_height", size[1])
        fs.write("camera_pose_matrix", pose)
        fs.endWriteStruct()
    fs.release()


def _rotated_pair(rotation: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Synthetic stereo pairs whose right camera is rotated by ``rotation`` about its centre (exact homography)."""
    data = SyntheticPlaneDataset(num_frames=3)
    k = data.camera.intrinsic_matrix
    h = k @ rotation @ np.linalg.inv(k)
    pairs = []
    for i in range(3):
        frame = data.load_frame(i)
        right = cv2.warpPerspective(frame.right, h, (data.camera.width, data.camera.height), flags=cv2.INTER_CUBIC)
        pairs.append((frame.left, right))
    return pairs, k


def test_mc_calib_import_and_convention_check(tmp_path: Path) -> None:
    """MC-Calib import: the image check picks the right pose convention and recovers the rotated rig."""
    rotation = cv2.Rodrigues(np.radians(np.array([0.6, -0.4, 0.3]))[:, None])[0]
    pairs, k = _rotated_pair(rotation)
    right_from_left = make_pose(rotation, -rotation @ np.array([0.12, 0.0, 0.0]))  # X_R = R (X_L - b).
    path = tmp_path / "calibrated_cameras_data.yml"
    _write_mc_calib(path, right_from_left, (640, 480), k)
    assert detect_format(path) == "mc_calib"
    good = load_mc_calib(path, 0, 1, "cam_to_ref")
    assert np.allclose(good.R, rotation, atol=1e-9) and np.allclose(good.T, right_from_left[:3, 3], atol=1e-9)
    spec = importlib.util.spec_from_file_location("import_calibration", Path(__file__).resolve().parents[1] / "scripts" / "import_calibration.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    good_check = module.rectification_check(good, pairs)
    bad_check = module.rectification_check(load_mc_calib(path, 0, 1, "ref_to_cam"), pairs)
    assert good_check["median_dv"] < 0.15 and good_check["positive_disparity"] > 0.95
    assert bad_check["median_dv"] > 1.0 or bad_check["positive_disparity"] < 0.5 or bad_check["baseline"] < 0
    rect = Rectifier(good)
    assert abs(rect.camera.baseline - 0.12) < 1e-6


def test_kalibr_import_and_comparison(tmp_path: Path) -> None:
    """Kalibr camchain import (pinhole/radtan) and the factory-vs-calibrated comparison."""
    rotation = cv2.Rodrigues(np.radians(np.array([0.0, 0.2, 0.0]))[:, None])[0]
    t_c1_c0 = make_pose(rotation, np.array([-0.095, 0.001, 0.0]))
    camchain = {
        "cam0": {
            "camera_model": "pinhole",
            "intrinsics": [430.0, 430.0, 424.0, 240.0],
            "distortion_model": "radtan",
            "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
            "resolution": [848, 480],
            "T_cam_imu": np.eye(4).tolist(),
        },
        "cam1": {
            "camera_model": "pinhole",
            "intrinsics": [431.0, 431.0, 425.0, 241.0],
            "distortion_model": "radtan",
            "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
            "resolution": [848, 480],
            "T_cn_cnm1": t_c1_c0.tolist(),
        },
    }
    path = tmp_path / "camchain.yaml"
    path.write_text(yaml.safe_dump(camchain))
    assert detect_format(path) == "kalibr"
    calib = load_kalibr(path)
    assert np.allclose(calib.R, rotation) and np.allclose(calib.T, t_c1_c0[:3, 3]) and calib.D_L.size == 5 and "T_cam0_imu" in calib.meta
    factory = RawStereoCalib(calib.K_L, calib.D_L, calib.K_R, calib.D_R, np.eye(3), np.array([-0.095, 0.0, 0.0]), 848, 480)
    diff = compare_calibrations(calib, factory)
    assert abs(diff["yaw_deg"] - 0.2) < 1e-6 and abs(diff["rotation_diff_deg"] - 0.2) < 1e-6
    out = tmp_path / "calib.yaml"
    calib.to_yaml(str(out))
    assert np.allclose(load_calibration(out).R, rotation)


def test_fisheye_rectifier(tmp_path: Path) -> None:
    """MC-Calib distortion type 1 (Kannala-Brandt) is rectified with cv2.fisheye."""
    path = tmp_path / "calibrated_cameras_data.yml"
    _write_mc_calib(path, make_pose(np.eye(3), np.array([-0.1, 0.0, 0.0])), (640, 480), K, fisheye=True)
    calib = load_mc_calib(path)
    assert calib.meta["distortion_model"] == "fisheye" and calib.D_L.size == 4
    rect = Rectifier(calib)
    left, right = rect(np.zeros((480, 640), np.uint8), np.zeros((480, 640), np.uint8))
    assert left.shape == (480, 640) and abs(rect.camera.baseline - 0.1) < 1e-6

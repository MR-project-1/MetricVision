"""Tests of the method upgrades: stereo RANSAC, LM covariance, anchors, re-alignment, KLT stereo matching, rig modes, --set."""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import replace  # Config variants.
from pathlib import Path  # Temporary folders.

import cv2  # Synthetic images.
import numpy as np  # Array maths.
import pytest  # Parametrised tests.

from metricvision.backend.anchors import (  # Landmark models.
    AnchorState,
    fuse_inverse_depth,
    inverse_depth_points,
    median_chi2,
    pose_uncertainty_covariance,
)
from metricvision.backend.lm import LMConfig, refine_pose_lm  # Rigid-rig LM.
from metricvision.backend.odometry import OdometryConfig  # Back-end settings.
from metricvision.backend.pose import PnPConfig, stereo_ransac  # Stereo RANSAC.
from metricvision.config import ComponentConfig, ExperimentConfig  # Configs.
from metricvision.core.geometry import invert_pose, make_pose, se3_exp  # Poses.
from metricvision.core.types import StereoCamera  # Camera model.
from metricvision.datasets.kitti import KittiOdometryDataset  # Calibration override.
from metricvision.datasets.synthetic import SyntheticPlaneDataset  # Textured plane.
from metricvision.frontend.alignment import align_warped, predict_affine  # Re-alignment.
from metricvision.frontend.stereo_klt import FastOrbKltFrontend, StereoTracks  # KLT stereo matcher.
from metricvision.pipeline import ExperimentRunner  # End to end.

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])  # Test intrinsics.
BASELINE = 0.12  # Test baseline (m).


def _scene(rng: np.random.Generator, n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random points in front of a moved camera and their exact stereo projections.

    Args:
        rng: Random generator.
        n: Point count.

    Returns:
        Reference points, true T_cur_ref, left pixels, right pixels.
    """
    points = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-2, 2, n), rng.uniform(4, 25, n)])  # Reference frame.
    pose = se3_exp(np.array([0.05, -0.02, 0.4, 0.01, 0.03, -0.005]))  # T_cur_ref.
    p = points @ pose[:3, :3].T + pose[:3, 3]  # Current camera.
    left = np.column_stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]])  # Left pixels.
    right = left - np.column_stack([K[0, 0] * BASELINE / p[:, 2], np.zeros(n)])  # Rectified right pixels.
    return points, pose, left, right


def _pose_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Translation (m) and rotation (deg) error.

    Args:
        estimate: Estimated pose.
        truth: True pose.

    Returns:
        (translation error, rotation error).
    """
    delta = estimate @ invert_pose(truth)  # Left error.
    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))  # Rotation angle.
    return float(np.linalg.norm(delta[:3, 3])), float(angle)


def test_stereo_ransac_rejects_left_and_right_only_outliers() -> None:
    """Stereo RANSAC must find the pose and flag outliers that are wrong in the right image only."""
    rng = np.random.default_rng(1)  # Deterministic.
    points, truth, left, right = _scene(rng)  # Exact data.
    left = left + rng.normal(0, 0.3, left.shape)  # Pixel noise.
    right = right + rng.normal(0, 0.3, right.shape)
    left[:60] += rng.uniform(15, 40, (60, 2))  # 20 % gross left outliers.
    right[60:90, 0] += rng.uniform(8, 20, 30)  # 10 % right-only outliers (wrong stereo match).
    config = PnPConfig(ransac="stereo", reprojection_error_px=2.0, iterations=300)  # Settings.
    result = stereo_ransac(points, left, right, K, BASELINE, None, config, np.random.default_rng(0))  # No prior.
    assert result is not None
    pose, inliers = result
    t_err, r_err = _pose_error(pose, truth)
    assert t_err < 0.05 and r_err < 0.3  # Minimal-sample hypothesis is close.
    mask = np.zeros(len(points), dtype=bool)
    mask[inliers] = True
    assert not mask[:90].any()  # Both outlier kinds rejected (right-only ones only a stereo score can catch).
    assert mask[90:].mean() > 0.95  # Nearly all true inliers kept.
    again = stereo_ransac(points, left, right, K, BASELINE, None, config, np.random.default_rng(0))  # Same seed.
    assert again is not None and np.allclose(again[0], pose)  # Deterministic.


def test_lm_covariance_is_consistent() -> None:
    """Average NEES of the rigid-rig LM over Monte Carlo trials must be close to the 6 pose dof."""
    rng = np.random.default_rng(2)  # Deterministic.
    points, truth, left0, right0 = _scene(rng, 200)  # Fixed geometry.
    sigma = 0.5  # Pixel noise.
    config = LMConfig(pixel_sigma=sigma, huber_delta=np.inf)  # Plain least squares.
    nees = []  # Normalised errors.
    for _ in range(150):  # Trials.
        left = left0 + rng.normal(0, sigma, left0.shape)
        right = right0.copy()
        right[:, 0] += rng.normal(0, sigma, len(right))  # u_r noise (v_r is not a residual).
        res = refine_pose_lm(truth, points, left, K, right, BASELINE, None, config)
        assert res.covariance is not None
        delta = res.camera_from_reference @ invert_pose(truth)  # exp(delta^) ~ T_est T_true^-1.
        rvec, _ = cv2.Rodrigues(delta[:3, :3])  # phi.
        err = np.concatenate([delta[:3, 3], rvec.ravel()])  # (rho, phi) to first order.
        nees.append(float(err @ np.linalg.solve(res.covariance, err)))
    assert 4.5 < np.mean(nees) < 7.5  # chi^2(6) mean is 6.
    assert abs(np.median(res.whitened_sq_norms) - median_chi2(3)) < 0.6  # Residuals follow chi^2(3).


def test_inverse_depth_covariance_matches_monte_carlo_and_fusion_shrinks_it() -> None:
    """Inverse-depth anchors: Z covariance matches sampling; fusing a consistent observation reduces variance."""
    camera = StereoCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, baseline=BASELINE, width=640, height=480)  # Camera.
    rng = np.random.default_rng(3)
    depth, sigma = 8.0, 0.5  # Far-ish point.
    disparity = camera.fx * BASELINE / depth  # 7.5 px.
    samples = camera.fx * BASELINE / (disparity + rng.normal(0, sigma * np.sqrt(2), 20000))  # Disparity noise of two pixels.
    rho = 1.0 / depth
    rho_var = (sigma * np.sqrt(2) / (camera.fx * BASELINE)) ** 2  # Linear in disparity: exact.
    assert abs(np.var(1.0 / samples) - rho_var) / rho_var < 0.05  # 1/Z is Gaussian when the disparity is.
    state = AnchorState(
        points=np.full((1, 3), np.nan),
        covariances=np.full((1, 3, 3), np.nan),
        birth_frame=np.array([0]),
        birth_left=np.array([[400.0, 260.0]]),
        inverse_depth=np.array([rho]),
        inverse_depth_var=np.array([rho_var]),
        birth_pose_cov=np.zeros((1, 3, 3)),
        age=np.array([0]),
    )
    poses = {0: np.eye(4)}  # Birth camera = world.
    points, covs = inverse_depth_points(state, poses, camera, sigma)
    assert abs(points[0, 2] - depth) < 1e-9  # On the birth ray at the right depth.
    assert covs[0, 2, 2] > 0
    observation = points.copy()  # A consistent re-observation...
    obs_cov = covs.copy()  # ...with the same uncertainty.
    fused, count = fuse_inverse_depth(state, poses, observation, obs_cov)
    assert count == 1 and fused.inverse_depth_var[0] < 0.6 * rho_var  # Variance roughly halves.
    far = observation + np.array([[0.0, 0.0, 30.0]])  # Inconsistent observation (38 m instead of 8 m).
    _, count = fuse_inverse_depth(state, poses, far, obs_cov)
    assert count == 0  # Gated out.


def test_pose_inflation_grows_with_distance() -> None:
    """Pose uncertainty must inflate far points more than near ones (rotation error x lever arm)."""
    cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])  # 1 mm / 1 mrad.
    near, far = pose_uncertainty_covariance(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 20.0]]), np.eye(4), cov)
    assert np.trace(far) > 10 * np.trace(near)
    assert np.allclose(pose_uncertainty_covariance(np.zeros((2, 3)), np.eye(4), None), 0.0)  # No pose covariance, no inflation.


def test_warped_alignment_recovers_the_true_position() -> None:
    """Re-alignment against the birth template must undo a sub-pixel position error after a scale change."""
    rng = np.random.default_rng(4)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (480, 640)).astype(np.float32), (0, 0), 2.0)  # Smooth texture.
    scale = 1.08  # Camera moves 8 % closer to a fronto-parallel plane.
    warp = np.array([[scale, 0, 320 * (1 - scale)], [0, scale, 240 * (1 - scale)]])  # Zoom about the centre.
    current = cv2.warpAffine(texture, warp, (640, 480), flags=cv2.INTER_CUBIC)  # Current image.
    ref = np.column_stack([rng.uniform(150, 490, 80), rng.uniform(120, 360, 80)])  # Birth pixels.
    truth = ref * scale + warp[:, 2]  # Their true current positions.
    depth = np.full(80, 10.0)  # Plane depth.
    current_from_ref = make_pose(np.eye(3), np.array([0.0, 0.0, 10.0 * (1 / scale - 1)]))  # Zoom by forward motion.
    affine, predicted = predict_affine(K, ref, depth, current_from_ref)
    assert np.allclose(affine[:, 0, 0], scale, atol=1e-3)  # Predicted local scale.
    drifted = truth + rng.uniform(-0.8, 0.8, truth.shape)  # KLT drift.
    aligned, ok = align_warped(texture.astype(np.uint8), current.astype(np.uint8), ref, affine, drifted)
    assert ok.mean() > 0.95
    assert np.median(np.linalg.norm(aligned[ok] - truth[ok], axis=1)) < 0.1  # Back on the physical point.


def test_klt_stereo_matcher_gives_accurate_disparity() -> None:
    """KLT stereo matching on the synthetic plane must recover the analytic disparity (fx B / Z = 13.5 px)."""
    dataset = SyntheticPlaneDataset(num_frames=2)  # Plane at 4 m, f = 450, B = 0.12.
    frame = dataset.load_frame(0)
    frontend = FastOrbKltFrontend(stereo_matcher="klt", corner_score="min_eig", fast_threshold_min=5)
    tracks = frontend.detect(frame.left, frame.right, StereoTracks.empty())
    assert len(tracks) > 300
    assert abs(float(np.median(tracks.disparity)) - 13.5) < 0.1
    assert np.percentile(np.abs(tracks.disparity - 13.5), 90) < 0.3
    assert np.abs(tracks.left[:, 1] - tracks.right[:, 1]).max() <= 1.5  # Epipolar constraint held.


def _sparse_config(tmp_path: Path, **odometry: object) -> ExperimentConfig:
    """Short sparse synthetic run.

    Args:
        tmp_path: Output folder.
        **odometry: OdometryConfig overrides.

    Returns:
        Experiment configuration.
    """
    base = OdometryConfig(sparse_point_anchor="fused", pnp=PnPConfig(refine="lm"))  # Report configuration.
    return ExperimentConfig(
        dataset=ComponentConfig("synthetic", {"num_frames": 25}),
        matcher=ComponentConfig("orb"),
        depth=ComponentConfig("sgbm"),
        odometry=replace(base, **odometry),
        output_dir=tmp_path,
        pipeline="sparse",
    )


@pytest.mark.parametrize(
    "options",
    [
        {"pnp": PnPConfig(refine="lm", ransac="stereo")},
        {"anchor_param": "inverse_depth", "anchor_pose_inflation": True},
        {"noise_model": "adaptive"},
        {"realign_every": 3},
        {"rig_mode": "averaged"},
    ],
)
def test_sparse_pipeline_with_new_options(options: dict, tmp_path: Path) -> None:
    """Every new option must keep the synthetic trajectory accurate.

    Args:
        options: OdometryConfig overrides.
        tmp_path: Output folder.
    """
    summary = ExperimentRunner(_sparse_config(tmp_path, **options)).run(show_progress=False)
    assert summary.num_tracking_failures == 0
    assert summary.errors is not None and summary.errors.ate_rmse_m < 0.02
    assert abs(summary.mean_vertical_disparity_px) < 0.2  # Rectified synthetic rig: no vertical disparity.


def test_rig_modes_report_extrinsic_deviation(tmp_path: Path) -> None:
    """Independent estimation reports a small implied extrinsic deviation; the rigid rig reports none."""
    rigid = ExperimentRunner(_sparse_config(tmp_path / "r")).run(show_progress=False)
    independent = ExperimentRunner(_sparse_config(tmp_path / "i", rig_mode="independent")).run(show_progress=False)
    assert np.isnan(rigid.mean_extrinsic_dev_trans_mm)
    assert 0.0 < independent.mean_extrinsic_dev_trans_mm < 20.0


def test_generic_set_overrides() -> None:
    """``--set`` must reach nested odometry settings and component params, and reject unknown keys."""
    config = _sparse_config(Path("unused")).with_settings(
        [
            "odometry.pnp.ransac=stereo",
            "odometry.noise_bounds_px=[0.2, 2.0]",
            "odometry.rig_mode=independent",
            "stereo_frontend.stereo_matcher=klt",
            "dataset.num_frames=7",
            "max_frames=5",
        ]
    )
    assert config.odometry.pnp.ransac == "stereo" and config.odometry.pnp.refine == "lm"  # Siblings kept.
    assert config.odometry.noise_bounds_px == (0.2, 2.0)
    assert config.odometry.rig_mode == "independent"
    assert config.stereo_frontend.params["stereo_matcher"] == "klt"
    assert config.dataset.params["num_frames"] == 7 and config.max_frames == 5
    with pytest.raises(ValueError):
        config.with_settings(["odometry.not_a_key=1"])
    with pytest.raises(ValueError):
        config.with_settings(["odometry.rig_mode=wobbly"])


def test_kitti_calibration_override(tmp_path: Path) -> None:
    """A replacement calib.txt (e.g. the Cvisic et al. recalibration) and a baseline scale must be honoured.

    Args:
        tmp_path: Output folder.
    """
    sequence_dir = tmp_path / "sequences" / "03"
    for camera in ("image_0", "image_1"):
        (sequence_dir / camera).mkdir(parents=True)
        cv2.imwrite(str(sequence_dir / camera / "000000.png"), np.zeros((376, 1241), dtype=np.uint8))
    (sequence_dir / "times.txt").write_text("0.0\n")
    (sequence_dir / "calib.txt").write_text("P0: 700 0 600 0 0 700 185 0 0 0 1 0\nP1: 700 0 600 -378 0 700 185 0 0 0 1 0\n")  # B = 0.54.
    alt = tmp_path / "recalib_03.txt"
    alt.write_text("P0: 705 0 601 0 0 705 186 0 0 0 1 0\nP1: 705 0 601 -380.7 0 705 186 0 0 0 1 0\n")  # B = 0.54, other K.
    default = KittiOdometryDataset(tmp_path, sequence="03")
    override = KittiOdometryDataset(tmp_path, sequence="03", calib_file=str(tmp_path / "recalib_{seq}.txt"), baseline_scale=1.01)
    assert default.camera.fx == 700.0 and override.camera.fx == 705.0
    assert abs(override.camera.baseline - 0.54 * 1.01) < 1e-9
    with pytest.raises(FileNotFoundError):
        KittiOdometryDataset(tmp_path, sequence="03", calib_file=str(tmp_path / "missing.txt"))

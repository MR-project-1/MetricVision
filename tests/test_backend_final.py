"""Tests of the final back-end additions: sliding-window BA, alternative estimators, consistency and paired statistics."""

from __future__ import annotations  # Postponed evaluation of type hints.

from pathlib import Path  # Temporary folders.

import numpy as np  # Arrays.
import pytest  # Parametrisation.

from metricvision.backend.lm import LMConfig, refine_pose_lm  # Rig LM.
from metricvision.backend.motion_alternatives import mahalanobis_3d, refine_translation, rotation_from_essential, umeyama  # Baselines.
from metricvision.backend.odometry import OdometryConfig  # Settings.
from metricvision.backend.pose import PnPConfig  # Settings.
from metricvision.backend.window_ba import SlidingWindowBA, WindowBAConfig  # Unit under test.
from metricvision.config import ComponentConfig, ExperimentConfig  # Configs.
from metricvision.core.geometry import invert_pose, se3_exp  # Poses.
from metricvision.core.types import Trajectory  # Trajectories.
from metricvision.evaluation.consistency import relative_nees  # NEES.
from metricvision.evaluation.statistics import paired_comparison  # Paired tests.
from metricvision.pipeline import ExperimentRunner  # End to end.

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])  # Intrinsics.
B = 0.12  # Baseline.


def _project(points: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    """(u_L, v_L, u_R) of world points."""
    p = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return np.column_stack([u, v, u - K[0, 0] * B / p[:, 2]])


def _window(rng: np.random.Generator, n_points: int = 400, n_kf: int = 5):  # noqa: ANN202
    """Noisy window problem with known truth."""
    points = np.column_stack([rng.uniform(-3, 3, n_points), rng.uniform(-1.5, 1.5, n_points), rng.uniform(3, 15, n_points)])
    truth = [se3_exp(np.array([0.02 * k, 0.0, -0.3 * k, 0.0, 0.01 * k, 0.0])) for k in range(n_kf)]
    ba = SlidingWindowBA(WindowBAConfig(window=n_kf, keyframe_every=1), K, B)
    for k, pose in enumerate(truth):
        obs = _project(points, pose) + rng.normal(0, 0.5, (n_points, 3))
        noisy = pose if k == 0 else se3_exp(rng.normal(0, 0.01, 6)) @ pose
        init = points + rng.normal(0, 0.05, points.shape) * points[:, 2:3] / 5
        ba.add_keyframe(k, noisy, np.arange(n_points), obs, init)
    return ba, truth, points


def test_window_ba_reduces_pose_and_landmark_error() -> None:
    """BA must pull perturbed keyframes back towards the truth and keep the oldest keyframe fixed."""
    rng = np.random.default_rng(0)
    ba, truth, points = _window(rng)
    before = [np.linalg.norm((kf.camera_from_world @ invert_pose(t))[:3, 3]) for kf, t in zip(ba.keyframes, truth, strict=True)]
    first = ba.keyframes[0].camera_from_world.copy()
    result = ba.optimise(0.5)
    assert result is not None and result.final_cost < 0.1 * result.initial_cost
    after = [np.linalg.norm((kf.camera_from_world @ invert_pose(t))[:3, 3]) for kf, t in zip(ba.keyframes, truth, strict=True)]
    assert np.allclose(ba.keyframes[0].camera_from_world, first)  # Gauge: oldest fixed.
    assert max(after[1:]) < 0.3 * min(before[1:])  # Centimetres -> millimetres.
    estimated = np.stack([ba.landmarks[i] for i in range(len(points))])
    assert np.median(np.linalg.norm(estimated - points, axis=1)) < 0.2


def test_window_ba_flags_outliers_in_newest_keyframe() -> None:
    """Gross errors in the newest keyframe must be reported as outliers."""
    rng = np.random.default_rng(1)
    ba, _, _ = _window(rng)
    kf = ba.keyframes[-1]
    kf.observations[:10, :2] += 25.0  # Corrupt ten observations.
    result = ba.optimise(0.5)
    assert result is not None
    assert set(range(10)) <= set(result.outlier_ids.tolist())
    assert len(result.outlier_ids) < 20


def test_window_ba_policy_and_bookkeeping() -> None:
    """Keyframe policy and landmark pruning."""
    ba = SlidingWindowBA(WindowBAConfig(window=2, keyframe_every=3, min_translation_m=10.0), K, B)
    pose = np.eye(4)
    assert ba.wants_keyframe(pose)  # First frame.
    ba.add_keyframe(0, pose, np.array([0, 1]), np.ones((2, 3)), np.ones((2, 3)))
    assert not ba.wants_keyframe(pose) and not ba.wants_keyframe(pose) and ba.wants_keyframe(pose)  # Every 3rd.
    ba.add_keyframe(3, pose, np.array([1, 2]), np.ones((2, 3)), np.ones((2, 3)))
    ba.add_keyframe(6, pose, np.array([2, 3]), np.ones((2, 3)), np.ones((2, 3)))  # Window of 2: frame 0 drops out.
    assert set(ba.landmarks) == {1, 2, 3}  # Landmark 0 left the window with keyframe 0.


def test_alternative_estimators_recover_motion() -> None:
    """Umeyama, Mahalanobis 3D-3D, essential-matrix rotation and translation-only LM on clean data."""
    rng = np.random.default_rng(2)
    points = np.column_stack([rng.uniform(-4, 4, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 20, 200)])
    truth = se3_exp(np.array([0.05, -0.02, 0.5, 0.01, 0.03, -0.01]))  # T_cur_world (world = previous camera).
    cam = points @ truth[:3, :3].T + truth[:3, 3]
    assert np.allclose(umeyama(points, cam), truth, atol=1e-9)
    cov = np.tile(np.eye(3) * 1e-4, (200, 1, 1))
    assert np.allclose(mahalanobis_3d(points, cov, cam, cov, np.eye(4)), truth, atol=1e-6)
    obs = _project(points, truth)
    prev = _project(points, np.eye(4))
    rotation = rotation_from_essential(prev[:, :2], obs[:, :2], K)
    assert rotation is not None and np.degrees(np.arccos(np.clip((np.trace(rotation @ truth[:3, :3].T) - 1) / 2, -1, 1))) < 0.05
    start = truth.copy()
    start[:3, 3] += 0.2
    refined = refine_translation(start, points, obs[:, :2], np.column_stack([obs[:, 2], obs[:, 1]]), K, B, None, 0.5)
    assert np.linalg.norm(refined[:3, 3] - truth[:3, 3]) < 1e-6


def test_relative_nees_is_six_for_a_consistent_estimator() -> None:
    """Frame-to-frame NEES of LM estimates with their own covariance averages ~6."""
    rng = np.random.default_rng(3)
    gt, est, covs = Trajectory(), Trajectory(), [None]
    gt.append(0.0, np.eye(4))
    est.append(0.0, np.eye(4))
    for k in range(1, 120):
        pts_prev = np.column_stack([rng.uniform(-4, 4, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 20, 200)])  # Previous camera.
        rel_true = se3_exp(np.array([0.01, 0.0, -0.5, 0.0, 0.002, 0.0]))  # T_cur_prev.
        obs = _project(pts_prev, rel_true) + rng.normal(0, 0.5, (200, 3))
        res = refine_pose_lm(
            rel_true, pts_prev, obs[:, :2], K, np.column_stack([obs[:, 2], obs[:, 1]]), B, None, LMConfig(pixel_sigma=0.5, huber_delta=np.inf)
        )
        gt.append(0.1 * k, gt.poses[-1] @ invert_pose(rel_true))  # T_world_cur.
        est.append(0.1 * k, est.poses[-1] @ invert_pose(res.camera_from_reference))  # Odometry chains its own estimates.
        covs.append(res.covariance)
    summary, nees = relative_nees(est, gt, covs)
    assert summary.num_frames == 119 and np.isnan(nees[0])
    assert 4.8 < summary.mean_nees < 7.2


def test_paired_comparison() -> None:
    """Wilcoxon and bootstrap detect a systematic improvement and not a null difference."""
    rng = np.random.default_rng(4)
    a = rng.uniform(1.0, 2.0, 200)
    better = paired_comparison(a, a - 0.1 + rng.normal(0, 0.02, 200), blocks=np.arange(200) // 5)
    assert better.mean_diff < 0 and better.ci_high < 0 and better.p_value < 1e-6 and better.win_rate > 0.9
    null = paired_comparison(a, a + rng.normal(0, 0.02, 200))
    assert null.ci_low < 0 < null.ci_high


@pytest.mark.parametrize(
    "settings",
    [
        {"ba_window": 5},
        {"ba_window": 5, "anchor_param": "inverse_depth"},
        {"pose_method": "soft"},
        {"pose_method": "mahalanobis3d"},
    ],
)
def test_pipeline_with_ba_and_alternatives(settings: dict, tmp_path: Path) -> None:
    """End to end on the noisy synthetic sequence; summaries carry the new evaluation fields.

    Args:
        settings: OdometryConfig overrides.
        tmp_path: Output folder.
    """
    config = ExperimentConfig(
        dataset=ComponentConfig("synthetic", {"num_frames": 40, "image_noise_std": 4.0}),
        matcher=ComponentConfig("orb"),
        depth=ComponentConfig("sgbm"),
        odometry=OdometryConfig(sparse_point_anchor="fused", pnp=PnPConfig(refine="lm"), **settings),
        output_dir=tmp_path,
        pipeline="sparse",
    )
    runner = ExperimentRunner(config)
    summary = runner.run(show_progress=False)
    assert summary.num_tracking_failures == 0
    assert summary.errors is not None and summary.errors.ate_rmse_m < 0.03
    assert np.isfinite(summary.errors.rpe_1s_trans_rmse_m) and np.isfinite(summary.loop_closure_error_m)
    assert (runner.output_dir / "pose_covariances.npz").is_file()
    if settings.get("pose_method", "pnp") == "pnp":
        assert summary.consistency is not None and summary.consistency["num_frames"] > 30

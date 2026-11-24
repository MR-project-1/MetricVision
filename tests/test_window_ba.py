"""Tests for the sliding-window rigid-rig bundle adjustment."""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Numerical checks.
from scipy.spatial.transform import Rotation  # Random rotations.

from metricvision.backend.window_ba import WindowBAConfig, WindowBAProblem, _linearise, windowed_ba  # Units under test.
from metricvision.core.geometry import invert_pose, make_pose, se3_exp  # Pose helpers.
from metricvision.core.types import StereoCamera  # Camera model.

CAMERA = StereoCamera(fx=718.856, fy=718.856, cx=607.19, cy=185.22, baseline=0.537, width=1241, height=376)  # KITTI-like rig.
K = CAMERA.intrinsic_matrix  # Intrinsics.


def _window(seed: int, n_points: int = 300, far: bool = False, noise_px: float = 0.5) -> tuple[WindowBAProblem, np.ndarray, np.ndarray]:
    """Build a 5-keyframe forward-driving window with exact ground truth and noisy stereo observations.

    Args:
        seed: Random seed.
        n_points: Number of landmarks.
        far: Use distant landmarks (poorly constrained by stereo).
        noise_px: Pixel noise on the observations.

    Returns:
        The problem (with perturbed initial values), true poses (T_cam,world) and true points.
    """
    rng = np.random.default_rng(seed)  # Seeded generator.
    world_from_cam = [
        make_pose(Rotation.from_euler("y", 2 * k, degrees=True).as_matrix(), [0.2 * k, 0.0, 1.0 * k]) for k in range(5)
    ]  # Gentle curve.
    true_poses = np.array([invert_pose(t) for t in world_from_cam])  # T_cam,world.
    depth_range = (30, 80) if far else (5, 40)  # Landmark depths.
    points = np.column_stack([rng.uniform(-15, 15, n_points), rng.uniform(-2, 2, n_points), rng.uniform(*depth_range, n_points)])  # World points.
    obs_k, obs_j, obs_px = [], [], []  # Observations.
    for k, pose in enumerate(true_poses):  # Project into every keyframe.
        p = points @ pose[:3, :3].T + pose[:3, 3]  # Camera frame.
        u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]  # Left column.
        v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]  # Row.
        ur = K[0, 0] * (p[:, 0] - CAMERA.baseline) / p[:, 2] + K[0, 2]  # Right column.
        visible = (p[:, 2] > 1) & (u > 0) & (u < CAMERA.width) & (v > 0) & (v < CAMERA.height)  # In view.
        idx = np.flatnonzero(visible)  # Visible landmarks.
        obs_k += [k] * idx.size  # Keyframe ids.
        obs_j += idx.tolist()  # Landmark ids.
        obs_px.append(np.column_stack([u, v, ur])[idx] + rng.normal(scale=noise_px, size=(idx.size, 3)))  # Noisy pixels.
    obs_px_all = np.vstack(obs_px)  # (E, 3).
    obs_px_all[3] += 40.0  # One gross outlier.
    poses0 = true_poses.copy()  # Perturbed initial poses (keyframe 0 stays exact: it is the gauge).
    for k in range(1, 5):  # Perturb the free poses.
        poses0[k] = se3_exp(np.concatenate([rng.normal(scale=0.1, size=3), rng.normal(scale=0.005, size=3)])) @ poses0[k]  # ~0.17 m, ~0.5 deg.
    points0 = points + rng.normal(scale=0.3, size=points.shape)  # Perturbed landmarks.
    problem = WindowBAProblem(
        poses=poses0,
        fixed=np.array([True, False, False, False, False]),
        points=points0,
        obs_keyframe=np.array(obs_k),
        obs_point=np.array(obs_j),
        obs_pixels=obs_px_all,
        prior_keyframe=np.zeros(0, dtype=np.int64),
        prior_point=np.zeros(0, dtype=np.int64),
        prior_depth=np.zeros(0),
        prior_sigma=np.zeros(0),
    )
    return problem, true_poses, points


def _pose_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Mean translation error of camera centres over the window.

    Args:
        estimate: (K, 4, 4) T_cam,world.
        truth: (K, 4, 4) T_cam,world.

    Returns:
        Mean centre error in metres.
    """
    return float(np.mean([np.linalg.norm(invert_pose(e)[:3, 3] - invert_pose(t)[:3, 3]) for e, t in zip(estimate, truth, strict=True)]))  # Centres.


def test_window_ba_recovers_poses_and_flags_outlier() -> None:
    """BA must pull perturbed poses back to the truth and flag the corrupted observation."""
    problem, true_poses, _ = _window(0)  # Window.
    result = windowed_ba(problem, K, CAMERA.baseline, WindowBAConfig(pixel_sigma=0.5))  # Optimise.
    assert _pose_error(problem.poses, true_poses) > 0.1  # Start is clearly wrong.
    assert _pose_error(result.poses, true_poses) < 0.02  # Recovered to 2 cm.
    assert result.final_cost < result.initial_cost  # Cost decreased.
    assert not result.obs_inliers[3]  # The outlier is flagged.
    np.testing.assert_allclose(result.poses[0], problem.poses[0])  # The gauge pose is untouched.


def test_correct_depth_priors_help_distant_landmarks() -> None:
    """With far (weakly triangulated) landmarks and noisy pixels, exact depth priors must reduce pose error."""
    problem, true_poses, true_points = _window(1, far=True, noise_px=1.5)  # Hard window.
    k, j = problem.obs_keyframe, problem.obs_point  # Put a prior on every observation.
    depth = np.einsum("nij,nj->ni", true_poses[k, :3, :3], true_points[j])[:, 2] + true_poses[k, 2, 3]  # True depths.
    with_priors = WindowBAProblem(
        **{**problem.__dict__, "prior_keyframe": k, "prior_point": j, "prior_depth": depth, "prior_sigma": 0.02 * depth}
    )  # 2% priors.
    config = WindowBAConfig(pixel_sigma=1.5, max_iterations=15)  # Matching noise model.
    plain = windowed_ba(problem, K, CAMERA.baseline, config)  # Stereo only.
    guided = windowed_ba(with_priors, K, CAMERA.baseline, config)  # Stereo + priors.
    assert _pose_error(guided.poses, true_poses) < _pose_error(plain.poses, true_poses)  # Priors help.


def test_factor_jacobians_match_finite_differences() -> None:
    """Analytic pose and landmark Jacobians of stereo and prior factors must match central differences."""
    problem, true_poses, true_points = _window(2, n_points=20)  # Small window.
    k, j = problem.obs_keyframe, problem.obs_point  # Prior on every observation too.
    problem = WindowBAProblem(
        **{**problem.__dict__, "prior_keyframe": k, "prior_point": j, "prior_depth": np.full(k.size, 10.0), "prior_sigma": np.full(k.size, 1.0)}
    )
    config = WindowBAConfig(huber_delta=1e9)  # Pure least squares so the weights are 1.
    blocks, _, _ = _linearise(problem, problem.poses, problem.points, K, CAMERA.baseline, config)  # Analytic.
    eps = 1e-6  # Step.
    for block_index in range(2):  # Stereo block, then prior block.
        residuals, jac_pose, jac_point, kk, jj = blocks[block_index]  # Factors.
        for axis in range(6):  # Pose perturbations (applied to every pose; each factor sees its own).
            delta = np.zeros(6)
            delta[axis] = eps
            plus = np.array([se3_exp(delta) @ p for p in problem.poses])  # Forward.
            minus = np.array([se3_exp(-delta) @ p for p in problem.poses])  # Backward.
            r_plus = _linearise(problem, plus, problem.points, K, CAMERA.baseline, config)[0][block_index][0]
            r_minus = _linearise(problem, minus, problem.points, K, CAMERA.baseline, config)[0][block_index][0]
            np.testing.assert_allclose(jac_pose[:, :, axis], (r_plus - r_minus) / (2 * eps), rtol=1e-4, atol=1e-4)  # Column.
        for axis in range(3):  # Landmark perturbations.
            offset = np.zeros(3)
            offset[axis] = eps
            r_plus = _linearise(problem, problem.poses, problem.points + offset, K, CAMERA.baseline, config)[0][block_index][0]
            r_minus = _linearise(problem, problem.poses, problem.points - offset, K, CAMERA.baseline, config)[0][block_index][0]
            np.testing.assert_allclose(jac_point[:, :, axis], (r_plus - r_minus) / (2 * eps), rtol=1e-4, atol=1e-4)  # Column.

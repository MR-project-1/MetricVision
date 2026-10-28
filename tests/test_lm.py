"""Tests for the rigid-rig Levenberg-Marquardt refinement and its analytic Jacobians."""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Numerical checks.
from scipy.spatial.transform import Rotation  # Random rotations.

from metricvision.backend.lm import LMConfig, refine_pose_lm, stereo_residuals  # Units under test.
from metricvision.core.geometry import make_pose, se3_exp  # Pose helpers.
from metricvision.core.stereo import triangulate_rectified, triangulation_covariance  # Covariance source.
from metricvision.core.types import StereoCamera  # Camera model.

CAMERA = StereoCamera(fx=718.856, fy=718.856, cx=607.19, cy=185.22, baseline=0.537, width=1241, height=376)  # KITTI-like rig.
K = CAMERA.intrinsic_matrix  # Intrinsics.


def _scene(seed: int, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Random points in front of the camera and a small ground-truth motion.

    Args:
        seed: Random seed.
        n: Number of points.

    Returns:
        Points in the reference frame and the true T_cur_ref.
    """
    rng = np.random.default_rng(seed)  # Seeded generator.
    points = np.column_stack([rng.uniform(-10, 10, n), rng.uniform(-2, 2, n), rng.uniform(4, 40, n)])  # Visible volume.
    truth = make_pose(Rotation.from_rotvec(rng.normal(scale=0.02, size=3)).as_matrix(), rng.normal(scale=0.3, size=3))  # Small motion.
    return points, truth  # Scene.


def _project(pose: np.ndarray, points: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Project reference points into the (optionally shifted) current camera.

    Args:
        pose: T_cur_ref.
        points: (N, 3) reference points.
        shift: Baseline offset along x (B for the right camera).

    Returns:
        (N, 2) pixel coordinates.
    """
    p = points @ pose[:3, :3].T + pose[:3, 3] - np.array([shift, 0.0, 0.0])  # Camera-frame points.
    return np.column_stack([CAMERA.fx * p[:, 0] / p[:, 2] + CAMERA.cx, CAMERA.fy * p[:, 1] / p[:, 2] + CAMERA.cy])  # Pinhole.


def test_se3_exp_small_and_large_angles() -> None:
    """exp must give a valid rotation and reduce to a pure translation for zero rotation."""
    np.testing.assert_allclose(se3_exp(np.array([1.0, 2.0, 3.0, 0, 0, 0]))[:3, 3], [1, 2, 3])  # Pure translation.
    pose = se3_exp(np.array([0.1, -0.2, 0.3, 0.5, -0.4, 0.2]))  # Generic twist.
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)  # Orthonormal.
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_rotvec([0.5, -0.4, 0.2]).as_matrix(), atol=1e-12)  # Matches Rodrigues.


def test_analytic_jacobians_match_numerical() -> None:
    """Pose and point Jacobians must agree with central finite differences."""
    points, pose = _scene(0, n=20)  # Small scene.
    obs_l, obs_r = _project(pose, points), _project(pose, points, CAMERA.baseline)  # Exact observations.
    r0, jac_pose, jac_point, _ = stereo_residuals(pose, points, obs_l, obs_r, K, CAMERA.baseline)  # Analytic.
    eps = 1e-6  # Step size.
    for k in range(6):  # Each twist component.
        delta = np.zeros(6)  # Perturbation.
        delta[k] = eps  # One coordinate.
        r_plus = stereo_residuals(se3_exp(delta) @ pose, points, obs_l, obs_r, K, CAMERA.baseline)[0]  # Forward.
        r_minus = stereo_residuals(se3_exp(-delta) @ pose, points, obs_l, obs_r, K, CAMERA.baseline)[0]  # Backward.
        np.testing.assert_allclose(jac_pose[:, :, k], (r_plus - r_minus) / (2 * eps), rtol=1e-4, atol=1e-4)  # Column k.
    for k in range(3):  # Each point coordinate.
        offset = np.zeros(3)  # Perturbation.
        offset[k] = eps  # One coordinate.
        r_plus = stereo_residuals(pose, points + offset, obs_l, obs_r, K, CAMERA.baseline)[0]  # Forward.
        r_minus = stereo_residuals(pose, points - offset, obs_l, obs_r, K, CAMERA.baseline)[0]  # Backward.
        np.testing.assert_allclose(jac_point[:, :, k], (r_plus - r_minus) / (2 * eps), rtol=1e-4, atol=1e-4)  # Column k.
    assert np.abs(r0).max() < 1e-9  # Exact observations give zero residuals.


def test_rig_lm_recovers_pose_with_noise_and_outliers() -> None:
    """Stereo LM must converge from a perturbed start despite noise and 10% gross outliers."""
    points, truth = _scene(1)  # Scene and true motion.
    rng = np.random.default_rng(2)  # Noise generator.
    obs_l = _project(truth, points) + rng.normal(scale=0.5, size=(len(points), 2))  # Noisy left observations.
    obs_r = _project(truth, points, CAMERA.baseline) + rng.normal(scale=0.5, size=(len(points), 2))  # Noisy right observations.
    outliers = rng.choice(len(points), 20, replace=False)  # 10% gross outliers.
    obs_l[outliers] += rng.uniform(20, 60, size=(20, 2))  # Corrupt them.
    start = se3_exp(np.array([0.2, -0.1, 0.3, 0.02, -0.03, 0.01])) @ truth  # Perturbed initial guess.
    result = refine_pose_lm(start, points, obs_l, K, obs_r, CAMERA.baseline, config=LMConfig(pixel_sigma=0.5))  # Refine.
    error = np.linalg.inv(truth) @ result.camera_from_reference  # Residual transform.
    assert np.linalg.norm(error[:3, 3]) < 0.02  # Translation within 2 cm.
    assert result.final_cost < result.initial_cost  # Cost decreased.
    assert not result.inlier_mask[outliers].any()  # Outliers are flagged.
    assert result.inlier_mask.sum() > 160  # Most inliers kept.


def test_triangulation_covariance_matches_monte_carlo() -> None:
    """First-order covariance must match the sample covariance for small pixel noise."""
    point = np.array([[700.0, 200.0]])  # Left pixel.
    disparity = np.array([20.0])  # About 19 m away.
    sigma = 0.3  # Pixel noise.
    analytic = triangulation_covariance(point, disparity, CAMERA, sigma)[0]  # Linearised covariance.
    rng = np.random.default_rng(3)  # Sampler.
    noise = rng.normal(scale=sigma, size=(200_000, 3))  # Noise on (u_l, v, u_r).
    samples, _ = triangulate_rectified(point + noise[:, :2], disparity + noise[:, 0] - noise[:, 2], CAMERA)  # Perturbed triangulation.
    np.testing.assert_allclose(np.cov(samples.T), analytic, rtol=0.1, atol=1e-6)  # Agree within 10%.

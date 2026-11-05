"""Alternative motion estimators on a fixed inlier set (RQ2 baselines for covariance-weighted reprojection).

All three take the same inputs as the rigid-rig LM, so a comparison isolates the *estimator*, not outlier rejection:

* ``umeyama3d``: 3D-3D alignment of the landmark positions to the current stereo triangulations (closed form,
  unweighted; Umeyama/Kabsch). The classic 3D-3D baseline that minimises error in noisy triangulated depth.
* ``mahalanobis3d``: the same alignment, but Gauss-Newton on the Mahalanobis distance with the propagated
  covariances of both point sets (weighted 3D-3D, as in Matthies & Shafer).
* ``soft``: SOFT-style decoupling (Cvisic & Petrovic). Rotation from the 5-point essential matrix on the temporal
  left-image correspondences (independent of depth error), then translation only, by LM on the covariance-weighted
  stereo reprojection residual with the rotation held fixed.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import cv2  # Essential matrix.
import numpy as np  # Array maths.

from metricvision.backend.lm import _robust_cost, _whitening, stereo_residuals  # Shared residual model.
from metricvision.core.geometry import make_pose, se3_exp, skew  # Pose helpers.
from metricvision.core.types import FloatArray  # Shared types.

METHODS = ("pnp", "umeyama3d", "mahalanobis3d", "soft")  # Valid odometry.pose_method values.


def umeyama(source: FloatArray, target: FloatArray, weights: FloatArray | None = None) -> FloatArray:
    """Rigid transform T (no scale) minimising sum w ||target - T source||^2.

    Args:
        source: (N, 3) points.
        target: (N, 3) points.
        weights: (N,) optional non-negative weights.

    Returns:
        4x4 T mapping source to target.
    """
    w = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu_s, mu_t = w @ source, w @ target  # Weighted centroids.
    cov = ((target - mu_t) * w[:, None]).T @ (source - mu_s)  # Cross-covariance.
    u, _, vt = np.linalg.svd(cov)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])  # Proper rotation.
    rotation = u @ d @ vt
    return make_pose(rotation, mu_t - rotation @ mu_s)


def mahalanobis_3d(
    points_world: FloatArray,
    cov_world: FloatArray,
    points_cam: FloatArray,
    cov_cam: FloatArray,
    initial: FloatArray,
    iterations: int = 10,
    huber_delta: float = 2.0,
) -> FloatArray:
    """Gauss-Newton on sum rho( e^T (Sigma_c + R Sigma_w R^T)^-1 e ), e = X_c - T X_w.

    Args:
        points_world: (N, 3) landmarks.
        cov_world: (N, 3, 3) their covariances.
        points_cam: (N, 3) current triangulations (camera frame).
        cov_cam: (N, 3, 3) their covariances.
        initial: Initial T_cam_world.
        iterations: Iterations.
        huber_delta: Huber threshold (sigma units).

    Returns:
        Refined T_cam_world.
    """
    pose = initial.copy()
    for _ in range(iterations):
        rotation, translation = pose[:3, :3], pose[:3, 3]
        predicted = points_world @ rotation.T + translation  # T X_w.
        e = points_cam - predicted  # Residual.
        cov = cov_cam + rotation @ cov_world @ rotation.T  # Combined covariance.
        info = np.linalg.inv(cov)
        chol = np.linalg.cholesky(info)  # info = L L^T, whitened residual L^T e.
        jac = np.zeros((len(e), 3, 6))  # d e / d delta for T <- exp(delta) T: -[I | -[p]x].
        jac[:, :, :3] = -np.eye(3)
        jac[:, :, 3:] = skew_batch(predicted)
        we = np.einsum("nji,nj->ni", chol, e)  # L^T e.
        wj = np.einsum("nji,njk->nik", chol, jac)  # L^T J.
        _, weights = _robust_cost(we, huber_delta)
        h = np.einsum("n,nki,nkj->ij", weights, wj, wj)
        g = np.einsum("n,nki,nk->i", weights, wj, we)
        delta = np.linalg.solve(h + 1e-9 * np.eye(6), -g)
        pose = se3_exp(delta) @ pose
        if np.max(np.abs(delta)) < 1e-9:
            break
    return pose


def skew_batch(vectors: FloatArray) -> FloatArray:
    """Stack of skew-symmetric matrices.

    Args:
        vectors: (N, 3).

    Returns:
        (N, 3, 3) [v]x.
    """
    return np.stack([skew(v) for v in vectors]) if len(vectors) else np.zeros((0, 3, 3))


def rotation_from_essential(previous_left: FloatArray, current_left: FloatArray, camera_matrix: FloatArray) -> FloatArray | None:
    """Relative rotation R_cur_prev from the 5-point essential matrix (RANSAC, 1 px).

    Args:
        previous_left: (N, 2) pixels in the previous left image.
        current_left: (N, 2) the same tracks in the current left image.
        camera_matrix: 3x3 intrinsics.

    Returns:
        Rotation, or None when the estimate is degenerate (e.g. too little parallax for recoverPose).
    """
    if len(previous_left) < 8:
        return None
    essential, mask = cv2.findEssentialMat(previous_left, current_left, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    if essential is None or essential.shape != (3, 3):
        return None
    count, rotation, _, _ = cv2.recoverPose(essential, previous_left, current_left, camera_matrix, mask=mask)
    return rotation if count >= 8 else None


def refine_translation(
    initial: FloatArray,
    points_world: FloatArray,
    left: FloatArray,
    right: FloatArray | None,
    camera_matrix: FloatArray,
    baseline: float,
    covariances: FloatArray | None,
    sigma: float,
    iterations: int = 10,
    huber_delta: float = 2.0,
) -> FloatArray:
    """Translation-only Gauss-Newton on the covariance-weighted stereo residual (rotation held fixed).

    Args:
        initial: Initial T_cam_world (its rotation is kept).
        points_world: (N, 3) landmarks.
        left: (N, 2) left observations.
        right: (N, 2) right observations or None.
        camera_matrix: Intrinsics.
        baseline: Baseline.
        covariances: (N, 3, 3) landmark covariances or None.
        sigma: Pixel noise.
        iterations: Iterations.
        huber_delta: Huber threshold.

    Returns:
        T_cam_world with refined translation.
    """
    pose = initial.copy()
    _, _, jac_point, _ = stereo_residuals(pose, points_world, left, right, camera_matrix, baseline)
    whiten = _whitening(jac_point, covariances, sigma)
    for _ in range(iterations):
        r, jp, _, ok = stereo_residuals(pose, points_world, left, right, camera_matrix, baseline)
        e = np.einsum("nij,nj->ni", whiten, r)
        jt = whiten @ jp[:, :, :3]  # Translation columns (left perturbation: d/d rho is the translation part).
        e[~ok], jt[~ok] = 0.0, 0.0
        _, weights = _robust_cost(e, huber_delta)
        h = np.einsum("n,nki,nkj->ij", weights, jt, jt)
        g = np.einsum("n,nki,nk->i", weights, jt, e)
        step = np.linalg.solve(h + 1e-9 * np.eye(3), -g)
        pose[:3, 3] += step  # exp([rho, 0]) T = (R, t + rho).
        if np.max(np.abs(step)) < 1e-9:
            break
    return pose

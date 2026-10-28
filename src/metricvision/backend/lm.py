"""Levenberg-Marquardt pose refinement for a rigid stereo rig with analytic Jacobians.

Problem: given 3D points X_i in a reference frame (with covariance Sigma_i from
stereo triangulation) and their observations in the current left and,
optionally, right image, find T = T_cur_ref minimising

    sum_i rho( || L_i^-1 r_i(T) ||^2 ),   r_i = [pi(T X_i) - z_l; pi_u(T X_i - b) - u_r]

where b = (B, 0, 0) is the rigid baseline, pi_u is the column of the projection (in a rectified
rig the right row equals the left row, so v_r adds no depth information and would only import
residual rectification error; this is the (u_l, v_l, u_r) stereo residual of ORB-SLAM2), rho is the Huber loss and
L_i L_i^T = J_X Sigma_i J_X^T + sigma_px^2 I is the propagated residual covariance,
so uncertain (far) points are down-weighted.

The pose is updated on the left: T <- exp(delta^) T with delta = (rho, phi), giving
the analytic Jacobian  d p / d delta = [I | -[p]x]  for p = T X, and
d pi / d p = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]].
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import dataclass  # Config and result containers.

import numpy as np  # Vectorised linear algebra.

from metricvision.core.geometry import se3_exp  # Exponential map for the update.
from metricvision.core.types import BoolArray, FloatArray  # Shared types.


@dataclass(frozen=True)
class LMConfig:
    """Settings for the Levenberg-Marquardt refinement.

    Attributes:
        max_iterations: Maximum number of LM iterations.
        initial_lambda: Initial damping factor (multiplies diag(H)).
        huber_delta: Huber threshold on the whitened residual norm (in standard deviations); ``inf`` gives plain least squares.
        pixel_sigma: Observation noise of each image coordinate, pixels.
        step_tolerance: Stop when the update norm falls below this.
        inlier_chi2_stereo: Whitened squared-residual threshold for stereo (3-dof: u_l, v_l, u_r) residuals.
        inlier_chi2_mono: Whitened squared-residual threshold for monocular (2-dof) residuals.
    """

    max_iterations: int = 20  # Iteration budget.
    initial_lambda: float = 1e-3  # Starting damping.
    huber_delta: float = 2.0  # Robust kernel threshold (sigma units).
    pixel_sigma: float = 1.0  # Pixel noise.
    step_tolerance: float = 1e-8  # Convergence threshold.
    inlier_chi2_stereo: float = 11.34  # chi^2(3 dof, 99%).
    inlier_chi2_mono: float = 9.21  # chi^2(2 dof, 99%).


@dataclass(frozen=True)
class LMResult:
    """Outcome of one refinement.

    Attributes:
        camera_from_reference: Refined T_cur_ref.
        initial_cost: Robust cost at the initial pose.
        final_cost: Robust cost at the refined pose.
        iterations: Iterations performed.
        cost_history: Robust cost after every accepted step, starting with the initial cost.
        inlier_mask: (N,) points whose whitened residual passes the chi^2 test at the refined pose.
        covariance: (6, 6) pose covariance from the Gauss-Newton information at the solution (inliers only,
            IRLS-weighted); None if the information matrix is singular.
        whitened_sq_norms: (N,) squared whitened residual norms at the solution (NaN for invalid points).
        residual_dof: Residual dimension per point.
    """

    camera_from_reference: FloatArray  # Refined pose.
    initial_cost: float  # Cost before.
    final_cost: float  # Cost after.
    iterations: int  # Iterations used.
    inlier_mask: BoolArray  # Post-refinement inliers.
    cost_history: tuple[float, ...] = ()  # Convergence trace (for plots).
    covariance: FloatArray | None = None  # (6, 6) covariance of the left-perturbation delta of T_cur_ref (inverse Gauss-Newton information).
    whitened_sq_norms: FloatArray | None = None  # (N,) squared Mahalanobis norm of each residual at the solution (noise calibration).
    residual_dof: int = 3  # Dimension of each residual block (3 stereo, 2 mono).


def projection_jacobian(points: FloatArray, fx: float, fy: float) -> FloatArray:
    """Jacobian of the pinhole projection with respect to the camera-frame point.

    Args:
        points: (N, 3) points in the camera frame.
        fx: Horizontal focal length.
        fy: Vertical focal length.

    Returns:
        (N, 2, 3) Jacobians d(u, v) / d(x, y, z).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]  # Components.
    jac = np.zeros((points.shape[0], 2, 3), dtype=np.float64)  # Output.
    jac[:, 0, 0] = fx / z  # du/dx.
    jac[:, 0, 2] = -fx * x / z**2  # du/dz.
    jac[:, 1, 1] = fy / z  # dv/dy.
    jac[:, 1, 2] = -fy * y / z**2  # dv/dz.
    return jac  # Stacked 2x3 blocks.


def point_jacobian_se3(points: FloatArray) -> FloatArray:
    """Jacobian of p = exp(delta) T X with respect to delta at delta = 0: [I | -[p]x].

    Args:
        points: (N, 3) transformed points p = T X.

    Returns:
        (N, 3, 6) Jacobians.
    """
    jac = np.zeros((points.shape[0], 3, 6), dtype=np.float64)  # Output.
    jac[:, :, :3] = np.eye(3)  # Translation part.
    x, y, z = points[:, 0], points[:, 1], points[:, 2]  # Components for -[p]x = [[0, z, -y], [-z, 0, x], [y, -x, 0]].
    jac[:, 0, 4], jac[:, 0, 5] = z, -y  # Row 1.
    jac[:, 1, 3], jac[:, 1, 5] = -z, x  # Row 2.
    jac[:, 2, 3], jac[:, 2, 4] = y, -x  # Row 3.
    return jac  # d p / d delta.


def stereo_residuals(
    camera_from_reference: FloatArray,
    points_3d: FloatArray,
    observations_left: FloatArray,
    observations_right: FloatArray | None,
    camera_matrix: FloatArray,
    baseline: float,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """Evaluate reprojection residuals and their analytic Jacobians.

    Args:
        camera_from_reference: Current estimate T_cur_ref.
        points_3d: (N, 3) points in the reference frame.
        observations_left: (N, 2) observations in the current left image.
        observations_right: (N, 2) observations in the current right image, or None (monocular).
        camera_matrix: 3x3 intrinsics shared by both rectified cameras.
        baseline: Stereo baseline B in metres.

    Returns:
        A tuple (r, J_pose, J_point, valid):
        residuals (N, m), pose Jacobians (N, m, 6), point Jacobians (N, m, 3) and a
        mask of points in front of both cameras; m = 3 for stereo (u_l, v_l, u_r), 2 for mono.
    """
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]  # Intrinsics.
    rotation, translation = camera_from_reference[:3, :3], camera_from_reference[:3, 3]  # Pose parts.
    p_left = points_3d @ rotation.T + translation  # Points in the current left camera.
    valid = p_left[:, 2] > 1e-6  # In front of the camera.
    z = np.where(valid, p_left[:, 2], 1.0)  # Safe denominator for invalid points.
    p_safe = np.column_stack([p_left[:, 0], p_left[:, 1], z])  # Guarded points.
    d_pose = point_jacobian_se3(p_safe)  # (N, 3, 6).
    blocks_r = [np.column_stack([fx * p_safe[:, 0] / z + cx, fy * p_safe[:, 1] / z + cy]) - observations_left]  # Left residual.
    jac_proj = projection_jacobian(p_safe, fx, fy)  # (N, 2, 3).
    blocks_pose = [jac_proj @ d_pose]  # Left pose Jacobian.
    blocks_point = [jac_proj @ rotation]  # Left point Jacobian (dp/dX = R).
    if observations_right is not None:  # Rigid rig: the right camera is the left one shifted by -B along x.
        p_right = p_safe - np.array([baseline, 0.0, 0.0])  # Same depth, shifted x.
        blocks_r.append((fx * p_right[:, 0] / z + cx - observations_right[:, 0])[:, None])  # Right column residual only (u_r).
        jac_proj_r = projection_jacobian(p_right, fx, fy)  # (N, 2, 3).
        blocks_pose.append(jac_proj_r[:, :1] @ d_pose)  # d p_right / d delta = d p_left / d delta; u row only.
        blocks_point.append(jac_proj_r[:, :1] @ rotation)  # Right point Jacobian, u row only.
    residuals = np.concatenate(blocks_r, axis=1)  # (N, m).
    jac_pose = np.concatenate(blocks_pose, axis=1)  # (N, m, 6).
    jac_point = np.concatenate(blocks_point, axis=1)  # (N, m, 3).
    return residuals, jac_pose, jac_point, valid  # Everything LM needs.


def _whitening(jac_point: FloatArray, covariances: FloatArray | None, pixel_sigma: float) -> FloatArray:
    """Return per-point matrices W_i with W_i^T W_i = (J_X Sigma_i J_X^T + sigma^2 I)^-1.

    Args:
        jac_point: (N, m, 3) residual Jacobians with respect to the 3D point.
        covariances: (N, 3, 3) point covariances, or None for pixel noise only.
        pixel_sigma: Observation noise per coordinate.

    Returns:
        (N, m, m) whitening matrices.
    """
    n, m = jac_point.shape[:2]  # Sizes.
    cov = np.broadcast_to(pixel_sigma**2 * np.eye(m), (n, m, m)).copy()  # Observation noise.
    if covariances is not None:  # Add the propagated triangulation uncertainty.
        cov += jac_point @ covariances @ jac_point.transpose(0, 2, 1)  # J Sigma J^T.
    chol = np.linalg.cholesky(cov)  # cov = C C^T.
    return np.linalg.inv(chol)  # W = C^-1, so ||W r||^2 = r^T cov^-1 r.


def _robust_cost(whitened: FloatArray, delta: float) -> tuple[float, FloatArray]:
    """Huber cost of whitened residuals and the IRLS weights.

    Args:
        whitened: (N, m) whitened residuals.
        delta: Huber threshold on the per-point residual norm.

    Returns:
        Total cost and (N,) weights.
    """
    norms = np.linalg.norm(whitened, axis=1)  # Per-point Mahalanobis distance.
    if not np.isfinite(delta):  # Robust kernel disabled (ablation): ordinary least squares.
        return float(0.5 * np.sum(norms**2)), np.ones_like(norms)  # Quadratic cost, unit weights.
    quadratic = norms <= delta  # Inside the quadratic region.
    cost = np.where(quadratic, 0.5 * norms**2, delta * (norms - 0.5 * delta))  # Huber loss.
    weights = np.where(quadratic, 1.0, delta / np.maximum(norms, 1e-12))  # IRLS weights.
    return float(cost.sum()), weights  # Scalar cost plus weights.


def refine_pose_lm(
    camera_from_reference: FloatArray,
    points_3d: FloatArray,
    observations_left: FloatArray,
    camera_matrix: FloatArray,
    observations_right: FloatArray | None = None,
    baseline: float = 0.0,
    covariances: FloatArray | None = None,
    config: LMConfig | None = None,
) -> LMResult:
    """Refine T_cur_ref with Levenberg-Marquardt on (stereo) reprojection error.

    Args:
        camera_from_reference: Initial T_cur_ref (e.g. from PnP + RANSAC).
        points_3d: (N, 3) points in the reference frame.
        observations_left: (N, 2) current left-image observations.
        camera_matrix: 3x3 intrinsics.
        observations_right: (N, 2) current right-image observations, or None for monocular refinement.
        baseline: Stereo baseline (metres); used only with right observations.
        covariances: (N, 3, 3) point covariances for weighting, or None.
        config: LM settings; defaults when omitted.

    Returns:
        The refined pose with costs and post-refinement inliers.
    """
    config = config or LMConfig()  # Defaults.
    pose = np.asarray(camera_from_reference, dtype=np.float64).copy()  # Working estimate.
    _, _, jac_point, valid = stereo_residuals(pose, points_3d, observations_left, observations_right, camera_matrix, baseline)  # Initial evaluation.
    whiten = _whitening(jac_point, covariances, config.pixel_sigma)  # Fixed weights from the initial linearisation.
    whiten[~valid] = 0.0  # Points behind the camera contribute nothing.

    def evaluate(candidate: FloatArray) -> tuple[float, FloatArray, FloatArray, FloatArray]:
        """Whitened residuals, Jacobians, cost and weights at a candidate pose."""
        r, jp, _, ok = stereo_residuals(candidate, points_3d, observations_left, observations_right, camera_matrix, baseline)  # Raw terms.
        e = np.einsum("nij,nj->ni", whiten, r)  # Whitened residuals.
        e[~ok] = 0.0  # Drop points that moved behind the camera.
        cost, w = _robust_cost(e, config.huber_delta)  # Robust cost and weights.
        return cost, e, whiten @ jp, w  # Whitened Jacobian J_w = W J.

    cost, e, jw, w = evaluate(pose)  # Start.
    initial_cost = cost  # Remember for reporting.
    history = [cost]  # Convergence trace.
    lam = config.initial_lambda  # Damping.
    iterations = 0  # Counter.
    while iterations < config.max_iterations:  # LM loop.
        iterations += 1  # Count this iteration.
        hessian = np.einsum("n,nki,nkj->ij", w, jw, jw)  # H = sum w J^T J (Gauss-Newton approximation).
        gradient = np.einsum("n,nki,nk->i", w, jw, e)  # g = sum w J^T e.
        improved = False  # Whether this iteration found a better pose.
        while lam < 1e10:  # Increase damping until a step reduces the cost.
            damped = hessian + lam * np.diag(np.diag(hessian) + 1e-12)  # Marquardt scaling by diag(H).
            step = np.linalg.solve(damped, -gradient)  # LM step.
            candidate = se3_exp(step) @ pose  # Left-multiplicative update.
            new_cost, new_e, new_jw, new_w = evaluate(candidate)  # Try it.
            if new_cost < cost:  # Accept: move towards Gauss-Newton.
                pose, cost, e, jw, w = candidate, new_cost, new_e, new_jw, new_w  # Commit.
                history.append(cost)  # Record progress.
                lam = max(lam / 10.0, 1e-12)  # Trust the quadratic model more.
                improved = True  # Progress made.
                break  # Next iteration.
            lam *= 10.0  # Reject: move towards gradient descent.
        if not improved or float(np.linalg.norm(step)) < config.step_tolerance:  # Converged or stuck.
            break  # Done.
    threshold = config.inlier_chi2_stereo if observations_right is not None else config.inlier_chi2_mono  # Per-dimension chi^2 test.
    sq_norms = np.sum(e**2, axis=1)  # Squared Mahalanobis norm per point.
    inliers = (sq_norms < threshold) & valid  # Post-refinement inliers.
    information = np.einsum("n,nki,nkj->ij", w * inliers, jw, jw)  # Robust GN information of the inliers at the solution.
    try:  # Covariance of the left-perturbation increment; singular for degenerate geometry.
        covariance: FloatArray | None = np.linalg.inv(information)  # Sigma_delta = H^-1 (residuals are whitened).
    except np.linalg.LinAlgError:  # Rank-deficient information.
        covariance = None  # Unknown uncertainty.
    sq_norms = np.where(valid, sq_norms, np.nan)  # Invalid points carry no information.
    dof = 3 if observations_right is not None else 2  # Residual dimension.
    return LMResult(pose, initial_cost, cost, iterations, inliers, tuple(history), covariance, sq_norms, dof)  # Report.

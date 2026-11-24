"""Sliding-window bundle adjustment for a rigid stereo rig, with optional gated monocular depth priors.

Unknowns: the poses T_k = T_cam_k,world of the keyframes in the window (the oldest
one is held fixed to remove the gauge freedom) and the world positions P_j of the
landmarks they observe. Each stereo observation contributes the rigid-rig residual

    r_kj = [ pi(T_k P_j) - (u_l, v_l) ;  pi_u(T_k P_j - b) - u_r ] / sigma_px        (3 rows)

and, optionally, a monocular depth prior in the spirit of RoMeO (Cheng et al., 2024)

    r_kj^prior = ( z(T_k P_j) - d*_kj ) / sigma*_kj                                   (1 row)

where d* is a network's metric depth at the observed pixel. Which priors enter is
decided by the caller (see ``SparseStereoOdometry``); this module only optimises.
All factors use a Huber kernel. The normal equations are solved with the Schur
complement on the 3x3 landmark blocks, so the dense system is only 6x(free poses).
Pose updates are left-multiplicative, T <- exp(delta^) T, with the same analytic
Jacobians as ``backend/lm.py``.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import dataclass  # Config and result containers.

import numpy as np  # Vectorised linear algebra.
import numpy.typing as npt  # Integer index arrays.

from metricvision.backend.lm import point_jacobian_se3, projection_jacobian  # Shared analytic Jacobians.
from metricvision.core.geometry import se3_exp  # Pose update.
from metricvision.core.types import BoolArray, FloatArray  # Shared types.

IndexArray = npt.NDArray[np.int64]  # Integer index arrays (keyframe / landmark ids per factor).


@dataclass(frozen=True)
class WindowBAConfig:
    """Settings for the windowed bundle adjustment.

    Attributes:
        max_iterations: Maximum LM iterations per window.
        pixel_sigma: Observation noise per image coordinate, pixels.
        huber_delta: Huber threshold on each factor's whitened residual norm.
        initial_lambda: Initial LM damping (multiplies the diagonal).
        inlier_chi2: chi^2(3 dof, 99%) threshold for flagging outlier observations after the solve.
    """

    max_iterations: int = 8  # Iteration budget.
    pixel_sigma: float = 1.0  # Pixel noise.
    huber_delta: float = 2.0  # Robust threshold (sigma units).
    initial_lambda: float = 1e-3  # Starting damping.
    inlier_chi2: float = 11.34  # Outlier test.


@dataclass(frozen=True)
class WindowBAProblem:
    """One window: keyframe poses, landmarks and the factors linking them.

    Attributes:
        poses: (K, 4, 4) keyframe poses T_cam,world.
        fixed: (K,) poses held constant (at least one, for the gauge).
        points: (M, 3) landmark positions in the world frame.
        obs_keyframe: (E,) keyframe index of each stereo observation.
        obs_point: (E,) landmark index of each stereo observation.
        obs_pixels: (E, 3) observed (u_l, v_l, u_r).
        prior_keyframe: (P,) keyframe index of each depth prior.
        prior_point: (P,) landmark index of each depth prior.
        prior_depth: (P,) prior depth d* in the keyframe camera, metres.
        prior_sigma: (P,) standard deviation of each prior, metres.
    """

    poses: FloatArray  # Keyframe poses.
    fixed: BoolArray  # Gauge mask.
    points: FloatArray  # Landmarks.
    obs_keyframe: IndexArray  # Observation -> keyframe.
    obs_point: IndexArray  # Observation -> landmark.
    obs_pixels: FloatArray  # (u_l, v_l, u_r).
    prior_keyframe: IndexArray  # Prior -> keyframe.
    prior_point: IndexArray  # Prior -> landmark.
    prior_depth: FloatArray  # d*.
    prior_sigma: FloatArray  # sigma*.


@dataclass(frozen=True)
class WindowBAResult:
    """Outcome of one windowed bundle adjustment.

    Attributes:
        poses: (K, 4, 4) optimised keyframe poses T_cam,world.
        points: (M, 3) optimised landmarks.
        initial_cost: Robust cost before optimisation.
        final_cost: Robust cost after optimisation.
        iterations: LM iterations performed.
        obs_inliers: (E,) stereo observations passing the chi^2 test at the solution.
    """

    poses: FloatArray  # Poses.
    points: FloatArray  # Landmarks.
    initial_cost: float  # Cost before.
    final_cost: float  # Cost after.
    iterations: int  # Iterations used.
    obs_inliers: BoolArray  # Outlier flags.


def _linearise(
    problem: WindowBAProblem, poses: FloatArray, points: FloatArray, camera_matrix: FloatArray, baseline: float, config: WindowBAConfig
) -> tuple[list[tuple[FloatArray, FloatArray, FloatArray, IndexArray, IndexArray]], float, FloatArray]:
    """Evaluate whitened residuals and Jacobians of every factor.

    Args:
        problem: Factor graph.
        poses: Current poses.
        points: Current landmarks.
        camera_matrix: Rectified intrinsics.
        baseline: Stereo baseline in metres.
        config: BA settings.

    Returns:
        A list of factor blocks (residuals (F, m), pose Jacobians (F, m, 6), point Jacobians (F, m, 3),
        keyframe ids, landmark ids) with IRLS weights folded in, the robust cost, and the stereo
        observations' squared whitened residuals (for the inlier test).
    """
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]  # Intrinsics.
    blocks = []  # Factor blocks.
    cost = 0.0  # Robust cost.

    def robust(residuals: FloatArray) -> tuple[float, FloatArray]:
        """Huber cost and sqrt IRLS weights of whitened residual blocks."""
        norms = np.linalg.norm(residuals, axis=1)  # Per-factor Mahalanobis distance.
        inside = norms <= config.huber_delta  # Quadratic region.
        rho = np.where(inside, 0.5 * norms**2, config.huber_delta * (norms - 0.5 * config.huber_delta))  # Huber loss.
        weights = np.where(inside, 1.0, config.huber_delta / np.maximum(norms, 1e-12))  # IRLS weights.
        return float(rho.sum()), np.sqrt(weights)  # Cost and sqrt weights.

    k, j = problem.obs_keyframe, problem.obs_point  # Stereo factors.
    rotations, translations = poses[k, :3, :3], poses[k, :3, 3]  # Per-factor pose parts.
    p = np.einsum("nij,nj->ni", rotations, points[j]) + translations  # Points in the left camera.
    z = np.maximum(p[:, 2], 1e-3)  # Guard against points behind the camera.
    p_safe = np.column_stack([p[:, 0], p[:, 1], z])  # Guarded points.
    p_right = p_safe - np.array([baseline, 0.0, 0.0])  # Right camera.
    predicted = np.column_stack([fx * p_safe[:, 0] / z + cx, fy * p_safe[:, 1] / z + cy, fx * p_right[:, 0] / z + cx])  # (u_l, v_l, u_r).
    residuals = (predicted - problem.obs_pixels) / config.pixel_sigma  # Whitened stereo residuals.
    jac_proj = projection_jacobian(p_safe, fx, fy)  # (E, 2, 3).
    jac_proj = np.concatenate([jac_proj, projection_jacobian(p_right, fx, fy)[:, :1]], axis=1) / config.pixel_sigma  # (E, 3, 3): add the u_r row.
    jac_pose = jac_proj @ point_jacobian_se3(p_safe)  # (E, 3, 6).
    jac_point = jac_proj @ rotations  # (E, 3, 3).
    stereo_cost, sqrt_w = robust(residuals)  # Robust weighting.
    cost += stereo_cost  # Accumulate.
    chi2 = np.sum(residuals**2, axis=1)  # For the inlier test.
    blocks.append((residuals * sqrt_w[:, None], jac_pose * sqrt_w[:, None, None], jac_point * sqrt_w[:, None, None], k, j))  # Weighted block.

    if problem.prior_depth.size:  # Monocular depth priors on the landmark depth in the keyframe camera.
        k, j = problem.prior_keyframe, problem.prior_point  # Prior factors.
        rotations, translations = poses[k, :3, :3], poses[k, :3, 3]  # Pose parts.
        p = np.einsum("nij,nj->ni", rotations, points[j]) + translations  # Camera-frame points.
        sigma = problem.prior_sigma[:, None]  # (P, 1).
        residuals = (p[:, 2:3] - problem.prior_depth[:, None]) / sigma  # (P, 1).
        jac_pose = point_jacobian_se3(p)[:, 2:3, :] / sigma[:, :, None]  # dz/ddelta = row 3 of [I | -[p]x].
        jac_point = rotations[:, 2:3, :] / sigma[:, :, None]  # dz/dP = third row of R.
        prior_cost, sqrt_w = robust(residuals)  # Robust weighting.
        cost += prior_cost  # Accumulate.
        blocks.append((residuals * sqrt_w[:, None], jac_pose * sqrt_w[:, None, None], jac_point * sqrt_w[:, None, None], k, j))  # Weighted block.
    return blocks, cost, chi2  # Everything the solver needs.


def _cost(
    problem: WindowBAProblem, poses: FloatArray, points: FloatArray, camera_matrix: FloatArray, baseline: float, config: WindowBAConfig
) -> float:
    """Robust cost only (for LM step acceptance).

    Args:
        problem: Factor graph.
        poses: Candidate poses.
        points: Candidate landmarks.
        camera_matrix: Intrinsics.
        baseline: Baseline.
        config: Settings.

    Returns:
        Total robust cost.
    """
    return _linearise(problem, poses, points, camera_matrix, baseline, config)[1]  # Reuse the full evaluation (cheap at window size).


def windowed_ba(problem: WindowBAProblem, camera_matrix: FloatArray, baseline: float, config: WindowBAConfig | None = None) -> WindowBAResult:
    """Jointly refine window keyframe poses and landmarks with Schur-complement Levenberg-Marquardt.

    Args:
        problem: Keyframes, landmarks, stereo observations and depth priors.
        camera_matrix: 3x3 rectified intrinsics.
        baseline: Stereo baseline in metres.
        config: BA settings; defaults when omitted.

    Returns:
        Optimised poses and landmarks with costs and observation inlier flags.
    """
    config = config or WindowBAConfig()  # Defaults.
    poses, points = problem.poses.copy(), problem.points.copy()  # Working copies.
    free = np.flatnonzero(~problem.fixed)  # Optimised keyframes.
    slot = np.full(len(poses), -1, dtype=np.int64)  # Keyframe -> position in the reduced system (-1 = fixed).
    slot[free] = np.arange(free.size)  # Free poses get consecutive slots.
    n_pose, n_point = 6 * free.size, len(points)  # Unknown counts.
    blocks, cost, chi2 = _linearise(problem, poses, points, camera_matrix, baseline, config)  # Initial state.
    initial_cost, lam, iterations = cost, config.initial_lambda, 0  # LM state.
    while iterations < config.max_iterations and n_pose > 0:  # LM loop.
        iterations += 1  # Count.
        h_pp = np.zeros((n_pose, n_pose))  # Pose-pose block.
        h_pl = np.zeros((n_pose, n_point, 3))  # Pose-landmark coupling (dense: small window).
        h_ll = np.zeros((n_point, 3, 3))  # Block-diagonal landmark Hessian.
        g_p, g_l = np.zeros(n_pose), np.zeros((n_point, 3))  # Gradients.
        for residuals, jac_pose, jac_point, k, j in blocks:  # Accumulate J^T J and J^T r.
            np.add.at(h_ll, j, np.einsum("fmi,fmj->fij", jac_point, jac_point))  # Landmark blocks.
            np.add.at(g_l, j, np.einsum("fmi,fm->fi", jac_point, residuals))  # Landmark gradient.
            on_free = slot[k] >= 0  # Factors touching a free pose.
            if not on_free.any():  # Only fixed poses.
                continue
            s, jp, jl, r, jj = slot[k][on_free], jac_pose[on_free], jac_point[on_free], residuals[on_free], j[on_free]  # Free subset.
            pp = np.einsum("fmi,fmj->fij", jp, jp)  # (F, 6, 6).
            for a in range(free.size):  # Few poses: loop over slots.
                sel = s == a  # Factors on pose a.
                if sel.any():
                    h_pp[6 * a : 6 * a + 6, 6 * a : 6 * a + 6] += pp[sel].sum(axis=0)  # Diagonal pose block.
                    g_p[6 * a : 6 * a + 6] += np.einsum("fmi,fm->i", jp[sel], r[sel])  # Pose gradient.
                    coupling = np.zeros((n_point, 6, 3))  # Pose a's coupling with every landmark.
                    np.add.at(coupling, jj[sel], np.einsum("fmi,fmj->fij", jp[sel], jl[sel]))  # Sum factors per landmark.
                    h_pl[6 * a : 6 * a + 6] += coupling.transpose(1, 0, 2)  # Store as (6, M, 3).
        improved, previous_cost = False, cost  # Whether a step was accepted; cost before it.
        while lam < 1e6:  # Damping loop (bounded so a bad window cannot stall the pipeline).
            h_ll_d = h_ll + lam * (np.einsum("nii->ni", h_ll)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3))  # Damped landmark blocks.
            v_inv = np.linalg.inv(h_ll_d)  # (M, 3, 3) block inverses.
            w_vinv = np.einsum("pmi,mij->pmj", h_pl, v_inv)  # W V^-1, (6F, M, 3).
            schur = h_pp + lam * np.diag(np.diag(h_pp) + 1e-9) - w_vinv.reshape(n_pose, -1) @ h_pl.reshape(n_pose, -1).T  # Reduced system.
            rhs = -g_p + w_vinv.reshape(n_pose, -1) @ g_l.reshape(-1)  # Reduced right-hand side.
            try:
                delta_pose = np.linalg.solve(schur, rhs)  # Pose step.
            except np.linalg.LinAlgError:  # Singular (degenerate window).
                lam *= 10.0  # Damp harder.
                continue
            delta_point = -np.einsum("mij,mj->mi", v_inv, g_l + np.einsum("pmi,p->mi", h_pl, delta_pose))  # Back-substitution.
            candidate_poses = poses.copy()  # Apply the step.
            for a, k in enumerate(free):  # Left-multiplicative pose update.
                candidate_poses[k] = se3_exp(delta_pose[6 * a : 6 * a + 6]) @ poses[k]  # T <- exp(delta) T.
            candidate_points = points + delta_point  # Landmark update.
            new_cost = _cost(problem, candidate_poses, candidate_points, camera_matrix, baseline, config)  # Try it.
            if new_cost < cost:  # Accept.
                poses, points, lam, improved = candidate_poses, candidate_points, max(lam / 10.0, 1e-9), True  # Commit.
                blocks, cost, chi2 = _linearise(problem, poses, points, camera_matrix, baseline, config)  # Relinearise.
                break
            lam *= 10.0  # Reject.
        if not improved or previous_cost - cost < 1e-6 * previous_cost:  # Stuck, or converged (relative decrease below 1e-6).
            break
    return WindowBAResult(poses, points, initial_cost, cost, iterations, chi2 < config.inlier_chi2)  # Report.

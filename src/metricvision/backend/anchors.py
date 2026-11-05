"""Landmark anchors for the sparse pipeline: per-track state, inverse-depth model and uncertainty fusion.

Each track keeps the frame where it was born, its birth pixel and its inverse depth rho = 1/Z in the birth
camera. Two landmark parameterisations are supported:

* ``xyz``: a Gaussian world point (X, Sigma), fused in information form (the mid-report method).
* ``inverse_depth``: the point is fixed to the birth ray and only rho is estimated (1-D fusion). Stereo depth
  error is strongly non-Gaussian and biased at long range (Z = f B / d), while inverse depth is close to
  linear in disparity, so far points are represented and fused far more faithfully.

Optionally the uncertainty of the camera pose that triangulated an observation is propagated into the point
covariance ("pose inflation"). Without it, re-observations shrink the anchor covariance as 1/n even though
every observation carries the same pose error, and long tracks become over-trusted.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import dataclass  # Track-state container.

import numpy as np  # Vectorised maths.

from metricvision.core.types import FloatArray, StereoCamera  # Shared types.

INVERSE_DEPTH_GATE = 9.0  # chi^2(1 dof) gate (3 sigma) for fusing an inverse-depth observation.
MIN_INVERSE_DEPTH = 1e-4  # 10 km: keeps far points finite and in front of the camera.


@dataclass(frozen=True)
class AnchorState:
    """Per-track landmark state; row i belongs to track i.

    Attributes:
        points: (N, 3) world points of the ``xyz`` model (NaN = no valid depth).
        covariances: (N, 3, 3) their world covariances.
        birth_frame: (N,) frame index where each track was born.
        birth_left: (N, 2) birth pixel in the left image (template centre for re-alignment, ray of the
            inverse-depth model).
        inverse_depth: (N,) rho = 1/Z in the birth camera.
        inverse_depth_var: (N,) variance of rho.
        birth_pose_cov: (N, 3, 3) extra world covariance from the birth pose uncertainty (zeros without inflation).
        age: (N,) frames since birth.
    """

    points: FloatArray
    covariances: FloatArray
    birth_frame: np.ndarray
    birth_left: FloatArray
    inverse_depth: FloatArray
    inverse_depth_var: FloatArray
    birth_pose_cov: FloatArray
    age: np.ndarray
    track_id: np.ndarray | None = None  # (N,) persistent identifiers (keyframe bundle adjustment); None = not tracked.

    def __post_init__(self) -> None:
        """Give every row an identifier (-1 = unassigned) so all fields have the same length."""
        if self.track_id is None:
            object.__setattr__(self, "track_id", np.full(self.points.shape[0], -1, dtype=np.int64))

    @classmethod
    def empty(cls) -> AnchorState:
        """Create a state with no tracks.

        Returns:
            Empty state.
        """
        return cls(
            np.zeros((0, 3)),
            np.zeros((0, 3, 3)),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 2)),
            np.zeros(0),
            np.zeros(0),
            np.zeros((0, 3, 3)),
            np.zeros(0, dtype=np.int64),
        )

    def __len__(self) -> int:
        """Return the number of tracks.

        Returns:
            Row count.
        """
        return int(self.points.shape[0])

    def subset(self, index: np.ndarray) -> AnchorState:
        """Select rows by boolean mask or integer index.

        Args:
            index: Mask or indices.

        Returns:
            Selected state.
        """
        return AnchorState(**{name: getattr(self, name)[index] for name in self.__dataclass_fields__})

    def concatenate(self, other: AnchorState) -> AnchorState:
        """Append another state.

        Args:
            other: Rows to append.

        Returns:
            Combined state.
        """
        return AnchorState(**{name: np.concatenate([getattr(self, name), getattr(other, name)]) for name in self.__dataclass_fields__})

    def replace(self, **changes: np.ndarray) -> AnchorState:
        """Return a copy with some fields replaced.

        Args:
            **changes: Field name -> new array.

        Returns:
            Updated state.
        """
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}  # Current fields.
        values.update(changes)  # Apply changes.
        return AnchorState(**values)  # New state.


def pose_uncertainty_covariance(points_camera: FloatArray, world_from_camera: FloatArray, pose_covariance: FloatArray | None) -> FloatArray:
    """World covariance that a camera's pose uncertainty induces on points triangulated by that camera.

    With the pose estimated as T_cw <- exp(delta^) T_cw, a camera-frame point maps to the world as
    X_w = T_cw^-1 X_c, so d X_w / d delta = -R_cw^T [I | -[X_c]x] and Sigma_w = J Sigma_delta J^T.

    Args:
        points_camera: (N, 3) points in the camera frame.
        world_from_camera: T_world_cam of that camera.
        pose_covariance: (6, 6) covariance of delta (translation first), or None.

    Returns:
        (N, 3, 3) world-frame covariances (zeros without a pose covariance).
    """
    n = points_camera.shape[0]  # Point count.
    if pose_covariance is None or n == 0 or not np.isfinite(pose_covariance).all():  # Nothing to add.
        return np.zeros((n, 3, 3))  # No inflation.
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]  # Components.
    jac = np.zeros((n, 3, 6))  # d X_c' / d delta = [I | -[X_c]x].
    jac[:, :, :3] = np.eye(3)
    jac[:, 0, 4], jac[:, 0, 5] = z, -y
    jac[:, 1, 3], jac[:, 1, 5] = -z, x
    jac[:, 2, 3], jac[:, 2, 4] = y, -x
    jac = -world_from_camera[:3, :3] @ jac  # Map into the world: -R_wc [I | -[X_c]x].
    return jac @ pose_covariance @ jac.transpose(0, 2, 1)  # J Sigma J^T.


def inverse_depth_points(state: AnchorState, poses: dict[int, FloatArray], camera: StereoCamera, pixel_sigma: float) -> tuple[FloatArray, FloatArray]:
    """World points and covariances of the inverse-depth model.

    X_b = (x_n / rho, y_n / rho, 1 / rho) with (x_n, y_n) the normalised birth pixel; the covariance follows from
    the Jacobian with respect to (u, v, rho) with pixel noise sigma on (u, v), then rotated into the world.

    Args:
        state: Track state.
        poses: Birth-frame poses T_world_cam keyed by frame index.
        camera: Rectified camera.
        pixel_sigma: Pixel noise of the birth observation.

    Returns:
        (N, 3) world points (NaN when rho is invalid) and (N, 3, 3) covariances.
    """
    n = len(state)  # Track count.
    points = np.full((n, 3), np.nan)  # Output.
    covariances = np.full((n, 3, 3), np.nan)
    if n == 0:  # Nothing to do.
        return points, covariances
    rho = state.inverse_depth  # Inverse depths.
    ok = np.isfinite(rho) & (rho > 0)  # Usable rows.
    xn = (state.birth_left[:, 0] - camera.cx) / camera.fx  # Normalised ray.
    yn = (state.birth_left[:, 1] - camera.cy) / camera.fy
    safe = np.where(ok, rho, 1.0)  # Avoid division by zero.
    local = np.column_stack([xn / safe, yn / safe, 1.0 / safe])  # Birth-camera points.
    jac = np.zeros((n, 3, 3))  # d X_b / d (u, v, rho).
    jac[:, 0, 0] = 1.0 / (camera.fx * safe)
    jac[:, 1, 1] = 1.0 / (camera.fy * safe)
    jac[:, 0, 2] = -xn / safe**2
    jac[:, 1, 2] = -yn / safe**2
    jac[:, 2, 2] = -1.0 / safe**2
    measurement_cov = np.zeros((n, 3, 3))  # Covariance of (u, v, rho).
    measurement_cov[:, 0, 0] = measurement_cov[:, 1, 1] = pixel_sigma**2
    measurement_cov[:, 2, 2] = state.inverse_depth_var
    local_cov = jac @ measurement_cov @ jac.transpose(0, 2, 1)  # Birth-frame covariance.
    for frame in np.unique(state.birth_frame[ok]):  # One rotation per birth frame.
        rows = ok & (state.birth_frame == frame)
        pose = poses[int(frame)]  # T_world_birth.
        points[rows] = local[rows] @ pose[:3, :3].T + pose[:3, 3]
        covariances[rows] = pose[:3, :3] @ local_cov[rows] @ pose[:3, :3].T + state.birth_pose_cov[rows]
    return points, covariances


def fuse_inverse_depth(
    state: AnchorState,
    poses: dict[int, FloatArray],
    observed_points_world: FloatArray,
    observed_covariances_world: FloatArray,
) -> tuple[AnchorState, int]:
    """Fuse new stereo observations (world frame) into the inverse depths (1-D Kalman update with a gate).

    Args:
        state: Track state (one row per observation).
        poses: Birth-frame poses keyed by frame index.
        observed_points_world: (N, 3) fresh triangulations in the world (NaN = invalid).
        observed_covariances_world: (N, 3, 3) their covariances (including pose inflation, if enabled).

    Returns:
        Updated state and the number of fused observations.
    """
    rho, var = state.inverse_depth.copy(), state.inverse_depth_var.copy()  # Working copies.
    ok = np.isfinite(observed_points_world).all(axis=1) & np.isfinite(rho) & (rho > 0)  # Fusable rows.
    fused = 0  # Counter.
    for frame in np.unique(state.birth_frame[ok]):  # Express observations in each birth camera.
        rows = np.flatnonzero(ok & (state.birth_frame == frame))
        pose = poses[int(frame)]  # T_world_birth.
        rotation_bw = pose[:3, :3].T  # Birth from world rotation.
        local = (observed_points_world[rows] - pose[:3, 3]) @ rotation_bw.T  # Points in the birth camera.
        local_cov = rotation_bw @ observed_covariances_world[rows] @ rotation_bw.T  # Covariances there.
        z = local[:, 2]  # Depth along the birth axis.
        good = z > 1e-3  # In front of the birth camera.
        rho_obs = np.where(good, 1.0 / np.where(good, z, 1.0), np.nan)  # Observed inverse depth.
        var_obs = local_cov[:, 2, 2] / np.where(good, z, 1.0) ** 4  # var(1/z) = var(z) / z^4.
        innovation = rho_obs - rho[rows]  # Disagreement.
        gate = good & (innovation**2 < INVERSE_DEPTH_GATE * (var[rows] + var_obs))  # Reject inconsistent observations.
        k = var[rows] / (var[rows] + var_obs)  # Kalman gain.
        new_rho = rho[rows] + k * innovation  # Update.
        new_var = (1.0 - k) * var[rows]
        rho[rows] = np.where(gate, np.maximum(new_rho, MIN_INVERSE_DEPTH), rho[rows])
        var[rows] = np.where(gate, new_var, var[rows])
        fused += int(gate.sum())
    return state.replace(inverse_depth=rho, inverse_depth_var=var), fused


def fuse_xyz(
    points: FloatArray, covariances: FloatArray, observed_points: FloatArray, observed_covariances: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Information-form fusion of Gaussian world points (the mid-report ``fused`` anchors).

    X = (S1^-1 + S2^-1)^-1 (S1^-1 X1 + S2^-1 X2).

    Args:
        points: (N, 3) prior anchors.
        covariances: (N, 3, 3) their covariances.
        observed_points: (N, 3) fresh triangulations (NaN = invalid).
        observed_covariances: (N, 3, 3) their covariances.

    Returns:
        Fused points and covariances.
    """
    ok = np.isfinite(observed_points).all(axis=1) & np.isfinite(points).all(axis=1)  # Both valid.
    points, covariances = points.copy(), covariances.copy()  # Do not modify the inputs.
    if not ok.any():  # Nothing to fuse.
        return points, covariances
    info_old, info_new = np.linalg.inv(covariances[ok]), np.linalg.inv(observed_covariances[ok])  # Information matrices.
    fused_cov = np.linalg.inv(info_old + info_new)  # Combined covariance.
    fused = np.einsum("nij,nj->ni", fused_cov, np.einsum("nij,nj->ni", info_old, points[ok]) + np.einsum("nij,nj->ni", info_new, observed_points[ok]))
    points[ok], covariances[ok] = fused, fused_cov  # Update.
    return points, covariances


def median_chi2(dof: int) -> float:
    """Median of a chi^2 distribution (Wilson-Hilferty approximation, < 0.5 % error for dof >= 1).

    Args:
        dof: Degrees of freedom.

    Returns:
        Median value.
    """
    return float(dof * (1.0 - 2.0 / (9.0 * dof)) ** 3)  # k (1 - 2/(9k))^3.

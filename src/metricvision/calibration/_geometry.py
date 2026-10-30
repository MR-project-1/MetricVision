"""SE(3)/SO(3) utilities, pinhole projection and its Jacobians.

Conventions
-----------
* A transform T (4x4) maps points: X_a = T_ab @ X_b.
* Twist ordering is xi = [rho; phi] (translation first, rotation second),
  matching the left-perturbation update T <- exp(delta^) T used in the report.
* Camera pose C_k is camera-to-world. The relative motion T_{k+1,k} maps points
  expressed in camera k into camera k+1, so C_{k+1} = C_k @ inv(T_{k+1,k}).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

EPS = 1e-10


def hat(v: np.ndarray) -> np.ndarray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def batch_hat(V: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,3,3) skew-symmetric matrices."""
    N = V.shape[0]
    S = np.zeros((N, 3, 3))
    S[:, 0, 1] = -V[:, 2]
    S[:, 0, 2] = V[:, 1]
    S[:, 1, 0] = V[:, 2]
    S[:, 1, 2] = -V[:, 0]
    S[:, 2, 0] = -V[:, 1]
    S[:, 2, 1] = V[:, 0]
    return S


def so3_exp(phi: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(phi).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(phi)
    if th < 1e-8:
        return np.eye(3) + 0.5 * hat(phi)
    a = phi / th
    s, c = np.sin(th), np.cos(th)
    return (s / th) * np.eye(3) + (1 - s / th) * np.outer(a, a) + ((1 - c) / th) * hat(a)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = so3_left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    phi = so3_log(T[:3, :3])
    rho = np.linalg.solve(so3_left_jacobian(phi), T[:3, 3])
    return np.concatenate([rho, phi])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply T to (N,3) points."""
    return P @ T[:3, :3].T + T[:3, 3]


def rot_angle(R: np.ndarray) -> float:
    """Rotation angle in radians."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(c))


def project(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pinhole projection of (N,3) camera-frame points -> (N,2) pixels."""
    z = X[:, 2]
    u = K[0, 0] * X[:, 0] / z + K[0, 2]
    v = K[1, 1] * X[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def project_jacobian(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """d pi / d X  -> (N,2,3)   (Eq. 9 of the timeline document)."""
    fx, fy = K[0, 0], K[1, 1]
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    iz = 1.0 / z
    iz2 = iz * iz
    J = np.zeros((X.shape[0], 2, 3))
    J[:, 0, 0] = fx * iz
    J[:, 0, 2] = -fx * x * iz2
    J[:, 1, 1] = fy * iz
    J[:, 1, 2] = -fy * y * iz2
    return J


def chordal_mean_rotation(Rs) -> np.ndarray:
    """Chordal L2 mean of rotations (projection of the arithmetic mean onto SO(3))."""
    M = np.sum(np.stack(Rs), axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def T_to_tum_row(t: float, T: np.ndarray) -> str:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
    p = T[:3, 3]
    return f"{t:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"


def tum_row_to_T(vals) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = vals[1:4]
    T[:3, :3] = Rotation.from_quat(vals[4:8]).as_matrix()
    return T


def interpolate_pose(t0, T0, t1, T1, t):
    """Linear translation + slerp interpolation between two poses."""
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    T = np.eye(4)
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    dR = T0[:3, :3].T @ T1[:3, :3]
    T[:3, :3] = T0[:3, :3] @ so3_exp(a * so3_log(dR))
    return T

"""Render synthetic stereo views of a ChArUco board through a distorted pinhole rig.

Used by the tests and by `tools/calibrate_charuco.py --selftest`: the full chain
(detection -> partial-view joint calibration) must recover the known rig.
"""

from __future__ import annotations

import cv2
import numpy as np

from metricvision.calibration._geometry import make_T, so3_exp
from metricvision.calibration.charuco import BoardSpec


def undistort_normalized(uv, K, D, iters=30):
    """Pixels -> undistorted normalised coordinates (fixed-point inversion of radtan)."""
    k1, k2, p1, p2, k3 = np.pad(np.asarray(D, float).ravel(), (0, 5))[:5]
    xd = (uv[:, 0] - K[0, 2]) / K[0, 0]
    yd = (uv[:, 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    return np.stack([x, y], 1)


class CharucoRenderer:
    def __init__(self, spec: BoardSpec, px_per_square: int = 120, legacy: bool = False):
        self.spec = spec
        self.margin = px_per_square // 2
        self.img = spec.render(px_per_square, self.margin, legacy=legacy).astype(np.float32)
        self.ppm = px_per_square / spec.square_length
        self._rays = {}

    def render(self, K, D, T_cam_board, size, noise=1.5, blur=0.6, background=110.0, rng=None, ss=3):
        """ss x ss supersampling then area averaging = pixel integration (no aliasing)."""
        w0, h0 = size
        w, h = w0 * ss, h0 * ss
        key = (tuple(np.ravel(K)), tuple(np.ravel(D)), w0, h0, ss)
        if key not in self._rays:  # undistorted ray grid depends only on the camera
            g = (np.arange(w, dtype=np.float64) + 0.5) / ss - 0.5
            gy = (np.arange(h, dtype=np.float64) + 0.5) / ss - 0.5
            u, v = np.meshgrid(g, gy)
            n = undistort_normalized(np.stack([u.ravel(), v.ravel()], 1), K, D)
            self._rays[key] = np.column_stack([n, np.ones(len(n))])
        rays = self._rays[key]
        H = np.column_stack([T_cam_board[:3, 0], T_cam_board[:3, 1], T_cam_board[:3, 3]])
        b = rays @ np.linalg.inv(H).T
        front = b[:, 2] > 0
        bx, by = b[:, 0] / b[:, 2], b[:, 1] / b[:, 2]
        mx = (bx * self.ppm + self.margin - 0.5).reshape(h, w).astype(np.float32)
        my = (by * self.ppm + self.margin - 0.5).reshape(h, w).astype(np.float32)
        mx[~front.reshape(h, w)] = -1e4
        out = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background)
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_AREA)
        if blur > 0:
            out = cv2.GaussianBlur(out, (0, 0), blur)
        rng = rng or np.random.default_rng(0)
        out = out * 0.75 + 30 + rng.normal(0, noise, out.shape)  # contrast < 1, like a real camera
        return np.clip(out, 0, 255).astype(np.uint8)


def random_board_pose(rng, spec: BoardSpec, dist=(0.45, 1.0), max_tilt_deg=50.0, lateral=0.35):
    """Board -> left camera; board centre roughly on the optical axis, tilted, sometimes off-image."""
    bw, bh = spec.size_m
    c = np.array([bw / 2, bh / 2, 0.0])
    tilt = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2))
    R = so3_exp([tilt[0], tilt[1], 0.0]) @ so3_exp([0, 0, rng.uniform(-0.3, 0.3)])
    z = rng.uniform(*dist)
    p = np.array([rng.uniform(-lateral, lateral) * z, rng.uniform(-lateral, lateral) * z * 0.7, z])
    return make_T(R, p - R @ c)


def default_rig(width=640, height=480):
    K_L = np.array([[452.0, 0, 322.5], [0, 451.0, 238.0], [0, 0, 1]])
    K_R = np.array([[449.0, 0, 318.0], [0, 448.5, 241.5], [0, 0, 1]])
    D_L = np.array([-0.06, 0.025, 0.0008, -0.0006, 0.0])
    D_R = np.array([-0.055, 0.02, -0.0005, 0.0004, 0.0])
    T_RL = make_T(so3_exp(np.radians([0.15, -0.3, 0.1])), [-0.0950, 0.0006, -0.0004])
    return K_L, D_L, K_R, D_R, T_RL, (width, height)

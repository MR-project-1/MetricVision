"""Stereo self-calibration from live images: re-estimate the left->right rotation (and baseline direction).

Why: on the wheelchair the camera mount moves between sessions (the stereo rotation changed by 8.7 deg after the
bags, then again by ~2.5 deg within one afternoon). A rectification from a stale extrinsic leaves the rows
misaligned (20-50 px), the stereo matcher then finds no partners and the VO cannot track at all.

The intrinsics (factory camera_info) and the baseline *length* are trusted; the rotation and the baseline *direction*
are estimated from SIFT matches between the raw left and right images of a few frames: essential matrix (RANSAC),
then least squares on the Sampson error. The scale is not observable from one stereo pair, hence the fixed length.

Used by the live ROS node at startup and whenever tracking fails (see ``metricvision.ros.vo_node``); the offline
equivalent for recorded bags is ``scripts/calibrate_rig_from_bags.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from metricvision.core.geometry import make_pose
from metricvision.core.types import FloatArray, ImageArray
from metricvision.datasets.rosbag_stereo import RigCalibration


def stereo_matches(left: ImageArray, right: ImageArray, calib: RigCalibration, max_features: int = 3000) -> tuple[FloatArray, FloatArray]:
    """SIFT left<->right matches (ratio test) on raw images, undistorted to normalised coordinates.

    Returns:
        (N, 2) left and (N, 2) right normalised points.
    """
    sift = cv2.SIFT_create(max_features)  # type: ignore[attr-defined]
    ka, da = sift.detectAndCompute(left, None)
    kb, db = sift.detectAndCompute(right, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    xa = np.array([ka[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    xb = np.array([kb[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    return (
        cv2.undistortPoints(xa, calib.k_left, calib.d_left).reshape(-1, 2).astype(np.float64),
        cv2.undistortPoints(xb, calib.k_right, calib.d_right).reshape(-1, 2).astype(np.float64),
    )


def sampson(params: FloatArray, xl: FloatArray, xr: FloatArray) -> FloatArray:
    """Sampson epipolar residuals (normalised units) for x_r = R x_l + t.

    Args:
        params: rotvec (3) + spherical angles (theta, phi) of the unit baseline direction t.
        xl: (N, 2) normalised left points.
        xr: (N, 2) normalised right points.
    """
    rot = Rotation.from_rotvec(params[:3]).as_matrix()
    th, ph = params[3], params[4]
    t = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    e = tx @ rot
    hl = np.column_stack([xl, np.ones(len(xl))])
    hr = np.column_stack([xr, np.ones(len(xr))])
    el, er = hl @ e.T, hr @ e
    num = np.einsum("ij,ij->i", hr, el)
    den = np.sqrt(el[:, 0] ** 2 + el[:, 1] ** 2 + er[:, 0] ** 2 + er[:, 1] ** 2)
    return np.asarray(num / den)


def vertical_disparity(calib: RigCalibration, xl: FloatArray, xr: FloatArray) -> FloatArray:
    """Vertical disparity (px) of normalised matches after rectifying with ``calib`` (0 = rows aligned)."""
    r1, r2, p1, p2, *_ = cv2.stereoRectify(
        calib.k_left,
        calib.d_left,
        calib.k_right,
        calib.d_right,
        (calib.width, calib.height),
        calib.right_from_left[:3, :3],
        calib.right_from_left[:3, 3].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    hl = (r1 @ np.column_stack([xl, np.ones(len(xl))]).T).T
    hr = (r2 @ np.column_stack([xr, np.ones(len(xr))]).T).T
    yl = p1[1, 1] * hl[:, 1] / hl[:, 2] + p1[1, 2]
    yr = p2[1, 1] * hr[:, 1] / hr[:, 2] + p2[1, 2]
    return np.asarray(yr - yl)


@dataclass(frozen=True)
class SelfCalResult:
    """Outcome of :func:`estimate_extrinsic`.

    Attributes:
        calibration: ``calib`` with the re-estimated ``right_from_left`` (same baseline length).
        matches: Matches used.
        inliers: RANSAC inliers.
        sampson_rms_px: Epipolar RMS of the inliers after refinement (pixels).
        dy_before_px: Median |vertical disparity| with the input calibration.
        dy_after_px: Median |vertical disparity| with the new one.
        rotation_change_deg: Angle between the old and the new stereo rotation.
    """

    calibration: RigCalibration
    matches: int
    inliers: int
    sampson_rms_px: float
    dy_before_px: float
    dy_after_px: float
    rotation_change_deg: float

    @property
    def reliable(self) -> bool:
        """Enough well-fitting matches and rows actually aligned afterwards."""
        return self.inliers >= 150 and self.sampson_rms_px < 1.0 and self.dy_after_px < 1.0


def estimate_extrinsic(calib: RigCalibration, xl: FloatArray, xr: FloatArray) -> SelfCalResult | None:
    """Re-estimate the stereo rotation and baseline direction of ``calib`` from normalised matches.

    Returns:
        The result (check :attr:`SelfCalResult.reliable`), or None with too few matches.
    """
    if len(xl) < 30:
        return None
    fx = float(calib.k_left[0, 0])
    baseline = float(np.linalg.norm(calib.right_from_left[:3, 3]))
    e, mask = cv2.findEssentialMat(xl, xr, np.eye(3), cv2.RANSAC, 0.9999, 1.0 / fx)
    if e is None or e.shape != (3, 3):
        return None
    _, rot, t, mask = cv2.recoverPose(e, xl, xr, np.eye(3), mask=mask)
    inl = mask.ravel() > 0
    if inl.sum() < 30:
        return None
    t = t.ravel() / np.linalg.norm(t)
    old_t = calib.right_from_left[:3, 3] / baseline
    if np.dot(t, old_t) < 0:  # recoverPose picks the sign of t only up to the cheirality of the points.
        t = -t
    x0 = np.concatenate([Rotation.from_matrix(rot).as_rotvec(), [np.arccos(np.clip(t[2], -1, 1)), np.arctan2(t[1], t[0])]])
    res = least_squares(sampson, x0, args=(xl[inl], xr[inl]), loss="huber", f_scale=1.0 / fx)
    rot = Rotation.from_rotvec(res.x[:3]).as_matrix()
    th, ph = res.x[3], res.x[4]
    t = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    new = replace(calib, right_from_left=make_pose(rot, np.asarray(t * baseline, dtype=np.float64)))
    rms_px = float(np.sqrt(np.mean(sampson(res.x, xl[inl], xr[inl]) ** 2)) * fx)
    change = float(np.degrees(np.linalg.norm(Rotation.from_matrix(rot @ calib.right_from_left[:3, :3].T).as_rotvec())))
    return SelfCalResult(
        calibration=new,
        matches=len(xl),
        inliers=int(inl.sum()),
        sampson_rms_px=rms_px,
        dy_before_px=float(np.median(np.abs(vertical_disparity(calib, xl[inl], xr[inl])))),
        dy_after_px=float(np.median(np.abs(vertical_disparity(new, xl[inl], xr[inl])))),
        rotation_change_deg=change,
    )

"""Re-estimate the wheelchair stereo extrinsic from recorded bags and write a rig calibration YAML.

Why: the MC-Calib extrinsic (ChArUco, 1280x720) leaves ~20 px vertical disparity on the 640x480
bags even when the chair stands still, so the cameras have moved (or were re-mounted) since that
calibration. The *metric baseline* is kept from MC-Calib; the rotation and the baseline *direction*
are re-estimated from the bags:

1. take frames where wheel odometry says the chair is still (no left/right time-offset effects),
2. match SIFT left<->right on the raw images, undistort with the RealSense factory intrinsics (camera_info),
3. pool all bags, essential matrix (RANSAC) + recoverPose, then refine R, t_dir by least squares on the
   Sampson error of all inliers,
4. scale t to |t| of MC-Calib, write ``right_from_left`` and ``body_from_left`` (from /tf_static).

Usage:
    uv run python scripts/calibrate_rig_from_bags.py ros_bags/small_run_01 ros_bags/small_run_02 ros_bags/run_01 \
        --mc-calib calib.yaml --out configs/wheelchair/rig_640x480.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from metricvision.core.geometry import invert_pose, make_pose
from metricvision.datasets.rosbag_stereo import (
    RigCalibration,
    Rosbag2Sqlite,
    StereoRectifier,
    image_message_to_gray,
    pair_by_timestamp,
    stamp_seconds,
)

# RealSense optical frame (z forward, x right, y down) in its mount link (x forward, y left, z up).
MOUNT_FROM_OPTICAL = make_pose(np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), np.zeros(3))


def read_mc_calib(path: Path) -> FloatArrayLike:
    """Return MC-Calib's camera_1 pose matrix (T_cam0_cam1 in MC-Calib's convention).

    Args:
        path: MC-Calib output YAML (OpenCV FileStorage format).

    Returns:
        4x4 matrix.
    """
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    pose = np.asarray(fs.getNode("camera_1").getNode("camera_pose_matrix").mat(), dtype=np.float64)
    fs.release()
    return pose


FloatArrayLike = np.ndarray


def camera_info(bag: Rosbag2Sqlite, topic: str) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Return (K, D, width, height) of the first CameraInfo on ``topic``, or None.

    Args:
        bag: Open bag.
        topic: CameraInfo topic.

    Returns:
        Intrinsics or None if absent.
    """
    ids = bag.message_ids(topic)
    if not ids:
        return None
    msg = bag.read(topic, ids[0])
    return np.asarray(msg.k, dtype=np.float64).reshape(3, 3), np.asarray(msg.d, dtype=np.float64), int(msg.width), int(msg.height)


def static_transforms(bag: Rosbag2Sqlite) -> dict[tuple[str, str], np.ndarray]:
    """Return every (parent, child) -> T_parent_child in /tf_static.

    Args:
        bag: Open bag.

    Returns:
        Transforms.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    for msg in bag.messages("/tf_static"):
        for tr in msg.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            out[(tr.header.frame_id, tr.child_frame_id)] = make_pose(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([t.x, t.y, t.z]))
    return out


def still_frame_matches(
    bag: Rosbag2Sqlite, left_topic: str, right_topic: str, intr: dict[str, Any], step: int, max_speed: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect undistorted, normalised left/right matches from frames where the chair is still.

    Args:
        bag: Open bag.
        left_topic: Left image topic.
        right_topic: Right image topic.
        intr: {"left": (K, D), "right": (K, D)}.
        step: Use every ``step``-th still pair.
        max_speed: Still = |v| < max_speed m/s and |w| < max_speed rad/s.

    Returns:
        (N, 2) left and (N, 2) right normalised points, and the number of frames used.
    """
    odom = bag.messages("/odom")
    odom_t = np.array([stamp_seconds(m) for m in odom])
    speed = np.array([max(abs(m.twist.twist.linear.x), abs(m.twist.twist.angular.z)) for m in odom])
    lid, lt = bag.header_stamps(left_topic)
    rid, rt = bag.header_stamps(right_topic)
    pairs = pair_by_timestamp(lt, rt, 0.035)
    still = []
    for i, j in pairs:
        window = np.abs(odom_t - lt[i]) < 0.3  # Odometry around the frame.
        if window.any() and speed[window].max() < max_speed:
            still.append((i, j))
    still = still[::step]
    sift = cv2.SIFT_create(4000)
    matcher = cv2.BFMatcher()
    pl, pr = [], []
    for i, j in still:
        a = image_message_to_gray(bag.read(left_topic, lid[i]))
        b = image_message_to_gray(bag.read(right_topic, rid[j]))
        ka, da = sift.detectAndCompute(a, None)
        kb, db = sift.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
        if len(good) < 20:
            continue
        xa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        xb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        pl.append(cv2.undistortPoints(xa, *intr["left"]).reshape(-1, 2))
        pr.append(cv2.undistortPoints(xb, *intr["right"]).reshape(-1, 2))
    if not pl:
        return np.zeros((0, 2)), np.zeros((0, 2)), 0
    return np.concatenate(pl), np.concatenate(pr), len(pl)


def sampson(params: np.ndarray, xl: np.ndarray, xr: np.ndarray) -> np.ndarray:
    """Sampson epipolar residuals for x_r = R x_l + t.

    Args:
        params: rotvec (3) + spherical (theta, phi) of t.
        xl: (N, 2) normalised left points.
        xr: (N, 2) normalised right points.

    Returns:
        (N,) residuals in normalised units.
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
    return num / den


def vertical_disparity(rectifier: StereoRectifier, calib: RigCalibration, xl: np.ndarray, xr: np.ndarray) -> np.ndarray:
    """Vertical disparity (px) of normalised matches after rectification.

    Args:
        rectifier: Rectifier under test.
        calib: Its calibration.
        xl: Normalised left points.
        xr: Normalised right points.

    Returns:
        (N,) y_right - y_left in rectified pixels.
    """
    size = (calib.width, calib.height)
    r1, r2, p1, p2, *_ = cv2.stereoRectify(
        calib.k_left,
        calib.d_left,
        calib.k_right,
        calib.d_right,
        size,
        calib.right_from_left[:3, :3],
        calib.right_from_left[:3, 3].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    hl = (r1 @ np.column_stack([xl, np.ones(len(xl))]).T).T
    hr = (r2 @ np.column_stack([xr, np.ones(len(xr))]).T).T
    yl = p1[1, 1] * hl[:, 1] / hl[:, 2] + p1[1, 2]
    yr = p2[1, 1] * hr[:, 1] / hr[:, 2] + p2[1, 2]
    return yr - yl


def main() -> None:
    """Run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--mc-calib", type=Path, required=True, help="MC-Calib YAML (baseline length is taken from it).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--left-topic", default="/cam1/cam1/color/image_raw")
    parser.add_argument("--right-topic", default="/cam2/cam2/color/image_raw")
    parser.add_argument("--left-info", default="/cam1/cam1/color/camera_info")
    parser.add_argument("--right-info", default="/cam2/cam2/color/camera_info")
    parser.add_argument("--left-mount", default="camera2_mount_link", help="tf_static frame of the physically LEFT camera.")
    parser.add_argument("--body-frame", default="base_footprint", help="Child frame of /odom.")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--max-speed", type=float, default=0.005)
    args = parser.parse_args()

    mc = read_mc_calib(args.mc_calib)  # T_cam1_cam2 (verified against the images: its inverse maps cam1 -> cam2).
    mc_right_from_left = invert_pose(mc)
    baseline = float(np.linalg.norm(mc[:3, 3]))

    bags = [Rosbag2Sqlite(b) for b in args.bags]
    infos = [(camera_info(b, args.left_info), camera_info(b, args.right_info)) for b in bags]
    with_info = [i for i in infos if i[0] is not None and i[1] is not None]
    if not with_info:
        raise SystemExit("No camera_info in any bag: intrinsics unknown.")
    (kl, dl, w, h), (kr, dr, _, _) = with_info[0]
    for a, b in with_info[1:]:
        assert np.allclose(a[0], kl) and np.allclose(b[0], kr), "camera_info differs between bags"
    intr = {"left": (kl, dl), "right": (kr, dr)}
    print(f"intrinsics from camera_info ({len(with_info)}/{len(bags)} bags have it): {w}x{h}, fx_l={kl[0, 0]:.1f} fx_r={kr[0, 0]:.1f}")

    per_bag = {}
    for path, bag in zip(args.bags, bags, strict=True):
        xl, xr, n = still_frame_matches(bag, args.left_topic, args.right_topic, intr, args.step, args.max_speed)
        per_bag[path.name] = (xl, xr)
        print(f"{path.name}: {n} still frames, {len(xl)} matches")
    xl = np.concatenate([v[0] for v in per_bag.values()])
    xr = np.concatenate([v[1] for v in per_bag.values()])

    e, mask = cv2.findEssentialMat(xl, xr, np.eye(3), cv2.RANSAC, 0.9999, 1.0 / kl[0, 0])
    _, rot, t, mask = cv2.recoverPose(e, xl, xr, np.eye(3), mask=mask)
    inl = mask.ravel() > 0
    t = t.ravel() / np.linalg.norm(t)
    x0 = np.concatenate([Rotation.from_matrix(rot).as_rotvec(), [np.arccos(t[2]), np.arctan2(t[1], t[0])]])
    res = least_squares(sampson, x0, args=(xl[inl], xr[inl]), loss="huber", f_scale=1.0 / kl[0, 0])
    rot = Rotation.from_rotvec(res.x[:3]).as_matrix()
    th, ph = res.x[3], res.x[4]
    t = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    right_from_left = make_pose(rot, t * baseline)
    rms_px = float(np.sqrt(np.mean(sampson(res.x, xl[inl], xr[inl]) ** 2)) * kl[0, 0])
    print(f"pooled: {inl.sum()}/{len(xl)} inliers, Sampson RMS {rms_px:.2f} px")

    delta = Rotation.from_matrix(rot @ mc_right_from_left[:3, :3].T).as_rotvec()
    print(
        "rotation  (rotvec deg) bags:",
        np.round(np.degrees(res.x[:3]), 2),
        " MC-Calib:",
        np.round(np.degrees(Rotation.from_matrix(mc_right_from_left[:3, :3]).as_rotvec()), 2),
        f" difference {np.degrees(np.linalg.norm(delta)):.2f} deg",
    )
    print(
        "baseline dir bags:", np.round(t, 3), " MC-Calib:", np.round(mc_right_from_left[:3, 3] / baseline, 3), f" |t| = {baseline:.4f} m (MC-Calib)"
    )

    tf = static_transforms(bags[0])
    body_from_base = tf.get((args.body_frame, "base_link"), np.eye(4))
    base_from_mount = tf[("base_link", args.left_mount)]
    body_from_left = body_from_base @ base_from_mount @ MOUNT_FROM_OPTICAL

    def calib(extrinsic: np.ndarray) -> RigCalibration:
        return RigCalibration(kl, dl, kr, dr, w, h, extrinsic, body_from_left)

    new, old = calib(right_from_left), calib(mc_right_from_left)
    print("\nvertical disparity after rectification on still frames, median |dy| (px):")
    for name, (a, b) in per_bag.items():
        dn = vertical_disparity(StereoRectifier(new), new, a, b)
        do = vertical_disparity(StereoRectifier(old), old, a, b)
        print(
            f"  {name:14s} MC-Calib {np.median(np.abs(do)):6.2f}   re-estimated {np.median(np.abs(dn)):5.2f}"
            f"   (re-est. p90 {np.percentile(np.abs(dn), 90):.2f})"
        )
    rect = StereoRectifier(new)
    print(f"rectified camera: fx={rect.camera.fx:.1f} cx={rect.camera.cx:.1f} cy={rect.camera.cy:.1f} baseline={rect.camera.baseline:.4f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Wheelchair stereo rig, RealSense colour 640x480 (camera_info factory intrinsics).\n"
        "# left = /cam1 (images prove cam1 is LEFT in the optical frame), right = /cam2.\n"
        f"# right_from_left: rotation + baseline direction re-estimated from still frames of {', '.join(per_bag)}\n"
        f"#   (Sampson RMS {rms_px:.2f} px); |baseline| = {baseline:.4f} m from MC-Calib.\n"
        f"# body_from_left: {args.body_frame} <- base_link <- {args.left_mount} (tf_static) <- RealSense optical convention.\n"
        "# Written by scripts/calibrate_rig_from_bags.py\n"
    )
    args.out.write_text(header + yaml.safe_dump(new.to_dict(), sort_keys=False, default_flow_style=None))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
